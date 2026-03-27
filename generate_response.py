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
    init_generation_model, init_reference_model, check_gcs_files_exist, load_config
)

def main():
    parser = argparse.ArgumentParser(description="Generate Responses using Gemini models")
    parser.add_argument("--json_file", default="output/query_generated.jsonl", help="질문 목록 JSONL 파일 경로")
    parser.add_argument("--output_file", default="output/responses.jsonl", help="통합 답변 목록을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="사용할 생성 모델명")
    parser.add_argument("--reference_model", default="gemini-2.5-pro", help="Reference Answer 생성 모델명")
    parser.add_argument("--no-reference-ref", dest="reference_use_ref", action="store_false", help="Reference 생성 시 Ref JSONL 미참조 (Video만 사용)")
    parser.set_defaults(reference_use_ref=True)
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")

    args = parser.parse_args()
    args = load_config(args)

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
            # 1. Output 진행률 읽기 - (content_id, query) 쌍 단위로 추적
            processed_pairs = set()  # (content_id, query) 튜플
            if os.path.exists(args.output_file):
                with open(args.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            ans = json.loads(line)
                            c_id = ans.get("content_id")
                            query = ans.get("query")
                            if c_id and query:
                                # 3개 모드에 유효한 답변이 모두 있으면 완료로 간주
                                answers = ans.get("answers", {})
                                is_complete = all(
                                    answers.get(m) and not str(answers.get(m, "")).startswith("Error")
                                    for m in ["video", "full", "part"]
                                )
                                if is_complete:
                                    processed_pairs.add((c_id, query))
                        except json.JSONDecodeError:
                            pass

            # 2. Input 읽기
            query_dict = {}
            if os.path.exists(args.json_file):
                with open(args.json_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "content_id" in data:
                                    query_dict[data["content_id"]] = data
                            except json.JSONDecodeError:
                                pass
                query_list = list(query_dict.values())
            else:
                query_list = []
                if not args.continuous:
                    print(f"Error: {args.json_file} 파일이 존재하지 않습니다.")
                    return

            new_data_processed = False

            # Resume Plan 계산 및 출력 - (content_id, query) 쌍 단위
            pending_work = {}
            for item in query_list:
                c_id = item["content_id"]
                c_pending = [
                    q_str for q_str in item.get("queries", [])
                    if (c_id, q_str) not in processed_pairs
                ]
                if c_pending:
                    pending_work[c_id] = c_pending
                    
            if pending_work:
                print("\n[TODO] 작업 목록:")
                for c_id, queries in pending_work.items():
                    print(f"- content_id '{c_id}':")
                    for q in queries:
                        print(f"    - query \"{q}\"")
                print("-" * 50)

            file_write_lock = threading.Lock()

            def process_item(item):
                content_id = item["content_id"]
                if content_id not in pending_work:
                    return False
                    
                queries = item["queries"]
                pending_queries = pending_work[content_id]
                
                print(f"\nProcessing Content: '{content_id}'")
                
                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    return False
                    
                parts = {
                    "video": process_gcs_file(args.gs_bucket_name, content_id, mode="video"),
                    "full": process_gcs_file(args.gs_bucket_name, content_id, mode="full"),
                    "part": process_gcs_file(args.gs_bucket_name, content_id, mode="part"),
                }
                ref_part = process_gcs_file(args.gs_bucket_name, content_id, mode="ref")

                print(f"[{content_id}] Initializing Generation models ({args.response_gen_model})...")
                gen_chats = {}
                for mode in ["video", "full", "part"]:
                    gen_model = init_generation_model(mode=mode, model_name=args.response_gen_model)
                    gen_chats[mode] = start_chat_session(gen_model)

                print(f"[{content_id}] Initializing Reference model ({args.reference_model}, Ref={'ON' if args.reference_use_ref else 'OFF'})...")
                ref_model = init_reference_model(model_name=args.reference_model)
                ref_chat = start_chat_session(ref_model)
                is_first_ref_turn = True

                is_first_turn_for_mode = {"video": True, "full": True, "part": True}

                for user_prompt in queries:
                    # 이미 처리된 (content_id, query) 쌍이면 건너뜀
                    if user_prompt not in pending_queries:
                        print(f"[{content_id}] Processing Query: '{user_prompt}' -> already completed (skip)")
                        continue

                    print(f"[{content_id}] Processing Query: '{user_prompt}'")

                    # 1. Reference Answer 생성 (Pro + Video [+ Ref JSONL])
                    ref_label = "Video+Ref" if args.reference_use_ref else "Video only"
                    print(f"[{content_id}]  Generating [reference] ({ref_label})...")
                    try:
                        if is_first_ref_turn:
                            ref_file_part = [parts["video"], ref_part] if args.reference_use_ref else [parts["video"]]
                        else:
                            ref_file_part = None
                        ref_contents = (ref_file_part or []) + [user_prompt]
                        ref_response = ref_chat.send_message(ref_contents)
                        reference_answer = ref_response.text
                        is_first_ref_turn = False
                        print(f"[{content_id}]  Reference answer generated ({len(reference_answer)} chars)")
                    except Exception as e:
                        print(f"[{content_id}]  Generating [reference] Error: {e}")
                        reference_answer = f"Error: {str(e)}"

                    # 2. 3개 Mode 답변 생성 (기존 로직)
                    answers_for_query = {}
                    
                    def generate_for_mode(mode):
                        print(f"[{content_id}]  Generating [{mode}]...")
                        file_part = parts[mode] if is_first_turn_for_mode[mode] else None
                        
                        try:
                            time.sleep(1) # 동시 호출 시 약간의 지연
                            response = send_chat_message(gen_chats[mode], user_prompt, file_part=file_part)
                            return mode, response.text
                        except Exception as e: 
                            print(f"[{content_id}]  Generating [{mode}] Error: {e}")
                            return mode, f"Error: {str(e)}"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as mode_executor:
                        futures = [mode_executor.submit(generate_for_mode, m) for m in ["video", "full", "part"]]
                        for future in concurrent.futures.as_completed(futures):
                            m, text = future.result()
                            answers_for_query[m] = text
                            is_first_turn_for_mode[m] = False

                    # 쿼리 하나 끝나면 (content_id, query) 단위로 1줄 append
                    # mode 순서 정렬 (video, full, part)
                    ordered_answers = {m: answers_for_query[m] for m in ["video", "full", "part"] if m in answers_for_query}
                    response_record = {
                        "content_id": content_id,
                        "query": user_prompt,
                        "reference": reference_answer,
                        "answers": ordered_answers
                    }
                    with file_write_lock:
                        with open(args.output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(response_record, ensure_ascii=False) + "\n")
                    processed_pairs.add((content_id, user_prompt))
                    print(f"[{content_id}]  -> Response 저장 완료: {args.output_file}")
                    print("-" * 50)

                return True

            for item in query_list:
                if process_item(item):
                    new_data_processed = True

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
