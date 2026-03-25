import os
import time
import argparse
import json
import subprocess
import sys
import vertexai
import concurrent.futures
import threading
from gemini_api_utils import (
    process_gcs_file, start_chat_session, send_chat_message, 
    init_generation_model, check_gcs_files_exist
)

def main():
    parser = argparse.ArgumentParser(description="Generate Responses using Gemini models")
    parser.add_argument("--json_file", default="output/query_generated.jsonl", help="질문 목록 JSONL 파일 경로")
    parser.add_argument("--output_file", default="output/responses.jsonl", help="통합 답변 목록을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="사용할 생성 모델명")
    parser.add_argument("--location", default="us-central1", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")
    parser.add_argument("--max_workers", type=int, default=3, help="동시 실행할 비디오 개수 (기본값: 3)")

    args = parser.parse_args()

    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                args.gcp_project_id = args.gcp_project_id or config.get("gcp_project_id")
                args.gs_bucket_name = args.gs_bucket_name or config.get("gs_bucket_name")
            except json.JSONDecodeError:
                pass

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}")
    vertexai.init(project=args.gcp_project_id, location=args.location)
    
    # 출력 폴더 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "=" * 50)
    print("Gemini Inference 프로세스를 시작합니다 (Session-based, JSONL Pipeline).")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다. 다른 터미널의 출력을 기다리며 지속 처리합니다.")
    print("=" * 50)

    try:
        while True:
            # 1. Output 진행률 읽기
            processed_ids = set()
            existing_answers_dict = {}
            if os.path.exists(args.output_file):
                with open(args.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            ans = json.loads(line)
                            c_id = ans["content_id"]
                            existing_answers_dict[c_id] = ans
                            # 부분 평가가 아니라 전체 완료인지 검증 (JSONL이 쓰여졌으면 완료로 보는 것이 일반적이나 안전하게 체크)
                            is_complete = True
                            for q in ans.get("queries", []):
                                answers = q.get("answers", {})
                                for mode in ["video", "full", "part"]:
                                    m_ans = answers.get(mode, "")
                                    if not m_ans or str(m_ans).startswith("Error"):
                                        is_complete = False
                            if is_complete:
                                processed_ids.add(c_id)
                        except json.JSONDecodeError:
                            pass

            # 2. Input 읽기
            query_list = []
            if os.path.exists(args.json_file):
                with open(args.json_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                query_list.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            else:
                if not args.continuous:
                    print(f"Error: {args.json_file} 파일이 존재하지 않습니다.")
                    return

            new_data_processed = False

            # Resume Plan 계산 및 출력
            pending_work = {}
            for item in query_list:
                c_id = item["content_id"]
                if c_id in processed_ids: continue
                
                existing_ans_data = existing_answers_dict.get(c_id, {})
                existing_queries = existing_ans_data.get("queries", [])
                existing_query_map = {q["query"]: q.get("answers", {}) for q in existing_queries}
                
                c_pending = {}
                for q_str in item.get("queries", []):
                    missing_modes = []
                    existing_answers = existing_query_map.get(q_str, {})
                    for m in ["video", "full", "part"]:
                        ans = existing_answers.get(m, "")
                        if not ans or str(ans).startswith("Error"):
                            missing_modes.append(m)
                    if missing_modes:
                        c_pending[q_str] = missing_modes
                if c_pending:
                    pending_work[c_id] = c_pending
                    
            if pending_work:
                print("\n[TODO] 작업 목록:")
                for c_id, queries_dict in pending_work.items():
                    print(f"- content_id '{c_id}':")
                    for q, modes in queries_dict.items():
                        print(f"    - query \"{q}\" : {', '.join(modes)}")
                print("-" * 50)

            file_write_lock = threading.Lock()

            def process_item(item):
                content_id = item["content_id"]
                if content_id not in pending_work:
                    return False
                    
                queries = item["queries"]
                
                print(f"\nProcessing Content: '{content_id}'")
                
                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    return False
                    
                parts = {
                    "video": process_gcs_file(args.gs_bucket_name, content_id, mode="video"),
                    "full": process_gcs_file(args.gs_bucket_name, content_id, mode="full"),
                    "part": process_gcs_file(args.gs_bucket_name, content_id, mode="part"),
                }

                print(f"[{content_id}] Initializing Generation models ({args.response_gen_model})...")
                gen_chats = {}
                for mode in ["video", "full", "part"]:
                    gen_model = init_generation_model(mode=mode, model_name=args.response_gen_model)
                    gen_chats[mode] = start_chat_session(gen_model)

                is_first_turn_for_mode = {"video": True, "full": True, "part": True}
                
                existing_ans_data = existing_answers_dict.get(content_id, {})
                existing_queries = existing_ans_data.get("queries", [])
                
                # Copy existing fully or partially completed queries so they are not lost
                answers_dict = {"content_id": content_id, "queries": existing_queries.copy()}
                existing_query_map = {q["query"]: q.get("answers", {}) for q in existing_queries}

                for user_prompt in queries:
                    print(f"[{content_id}] Processing Query: '{user_prompt[:30]}...'")
                    answers_for_query = {}
                    existing_answers = existing_query_map.get(user_prompt, {})
                    
                    for mode in ["video", "full", "part"]:
                        prev_ans = existing_answers.get(mode, "")
                        if prev_ans and not str(prev_ans).startswith("Error"):
                            print(f"[{content_id}]  [{mode}] already completed (skip)")
                            answers_for_query[mode] = prev_ans
                            continue
                        
                        print(f"[{content_id}]  Generating [{mode}]...")
                        file_part = parts[mode] if is_first_turn_for_mode[mode] else None
                        
                        try:
                            time.sleep(2) # API Rate Limit 과부하 방지 (각 생성마다 2초 대기)
                            response = send_chat_message(gen_chats[mode], user_prompt, file_part=file_part)
                            answers_for_query[mode] = response.text
                            is_first_turn_for_mode[mode] = False
                        except Exception as e: 
                            print(f"[{content_id}]  Generating [{mode}] Error: {e}")
                            answers_for_query[mode] = f"Error: {str(e)}"

                    answers_dict["queries"].append({
                        "query": user_prompt,
                        "answers": answers_for_query
                    })
                    print("-" * 50)
                    
                    # 쿼리 하나 끝날 때마다 JSONL로 Append 저장 (부분 저장)
                    with file_write_lock:
                        with open(args.output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(answers_dict, ensure_ascii=False) + "\n")
                    print(f"[{content_id}]  -> 진행중 쿼리 임시 저장 완료: {args.output_file}")
                    
                processed_ids.add(content_id)
                return True

            items_to_process = [item for item in query_list if item["content_id"] in pending_work]
            if items_to_process:
                new_data_processed = True
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    executor.map(process_item, items_to_process)

            if not args.continuous:
                break
            
            if not new_data_processed:
                # 새 데이터가 없으면 잠시 대기
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 모니터링 루프가 중단되었습니다.")

    if not args.continuous:
        print("\n[Aggregation] JSONL 결과를 분석용 JSON 형식으로 병합합니다...")
        output_dir = os.path.dirname(args.output_file) or "output"
        subprocess.run([sys.executable, "jsonl_to_json.py", "--input_dir", output_dir])

    print("\n생성 프로세스가 완료/종료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
