import os
import argparse
import json
import time
import subprocess
import sys
import vertexai
import concurrent.futures
import threading
from gemini_api_utils import (
    process_gcs_file, start_chat_session, 
    evaluate_answer_session, init_judge_model, check_gcs_files_exist
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Responses using Judge model")
    parser.add_argument("--answers_file", default="output/responses.jsonl", help="답변 목록 JSONL 파일 경로")
    parser.add_argument("--output_file", default="output/scores.jsonl", help="최종 평가 결과 저장 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--judge_model", default="gemini-2.5-pro", help="사용할 평가 모델명")
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

    vertexai.init(project=args.gcp_project_id, location=args.location)
    
    # 출력 폴더 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "=" * 50)
    print("Gemini Evaluation 프로세스를 시작합니다 (Session-based, JSONL Pipeline).")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다. 다른 터미널의 출력을 기다리며 지속 처리합니다.")
    print("=" * 50)

    try:
        while True:
            # 1. Output (진행률) 읽기
            processed_ids = set()
            existing_scores_dict = {}
            if os.path.exists(args.output_file):
                with open(args.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            sc = json.loads(line)
                            c_id = sc.get("content_id")
                            if c_id:
                                existing_scores_dict[c_id] = sc
                                processed_ids.add(c_id)
                        except json.JSONDecodeError:
                            pass

            # 2. Input 읽기
            content_answers_list = []
            if os.path.exists(args.answers_file):
                with open(args.answers_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                content_answers_list.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            else:
                if not args.continuous:
                    print(f"Error: {args.answers_file} 파일이 존재하지 않습니다. 먼저 generate_response.py를 실행하세요.")
                    return

            new_data_processed = False

            # Resume Plan 계산 및 출력
            pending_work = {}
            for content_answers in content_answers_list:
                c_id = content_answers["content_id"]
                if c_id in processed_ids: continue
                
                existing_sc_data = existing_scores_dict.get(c_id, {})
                old_content_scores = existing_sc_data.get("scores", [])
                existing_score_map = {}
                for entry in old_content_scores:
                    q_text = entry.get("query")
                    m = entry.get("mode")
                    judge_data = entry.get("judge", {})
                    if "raw_response" not in judge_data and "scores" in judge_data:
                        existing_score_map.setdefault(q_text, {})[m] = entry
                
                c_pending = {}
                for query_item in content_answers.get("queries", []):
                    q_str = query_item["query"]
                    answers = query_item.get("answers", {})
                    missing_modes = []
                    for m in ["video", "full", "part"]:
                        gen_ans = answers.get(m)
                        if not gen_ans or str(gen_ans).startswith("Error"):
                            continue # 평가할 답변이 없으면 우선 건너뜀
                        if m not in existing_score_map.get(q_str, {}):
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

            def process_item(content_answers):
                content_id = content_answers["content_id"]
                if content_id not in pending_work:
                    return False
                    
                print(f"\nEvaluating Content: '{content_id}'")
                
                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    return False
                    
                video_part = process_gcs_file(args.gs_bucket_name, content_id, mode="video")
                gt_part = process_gcs_file(args.gs_bucket_name, content_id, mode="gt")

                print(f"[{content_id}] Initializing Judge model ({args.judge_model})...")
                judge_model = init_judge_model(model_name=args.judge_model)
                
                judge_chats = {}
                for mode in ["video", "full", "part"]:
                    judge_chats[mode] = start_chat_session(judge_model)
                
                is_first_turn_for_mode = {"video": True, "full": True, "part": True}
                
                # BUGFIX: 기존에는 루프 밖에서 스코어를 읽어버려 여러개의 content_id 처리 시 앞쪽의 스코어 맵이 덮어써지는 문제가 있었음
                # process_item 단위(content_id 단위)에서 기존 스코어를 다시 가져옴
                existing_sc_data = existing_scores_dict.get(content_id, {})
                old_content_scores = existing_sc_data.get("scores", [])
                item_existing_score_map = {}
                for entry in old_content_scores:
                    q_text = entry.get("query")
                    m = entry.get("mode")
                    judge_data = entry.get("judge", {})
                    if "raw_response" not in judge_data and "scores" in judge_data:
                        item_existing_score_map.setdefault(q_text, {})[m] = entry

                # 기존에 평가가 완료되었던 쿼리 점수들이 있다면 먼저 복사해서 누적합니다
                content_scores = []
                if old_content_scores:
                    for entry in old_content_scores:
                        judge_data = entry.get("judge", {})
                        if "raw_response" not in judge_data and "scores" in judge_data:
                            content_scores.append(entry)
                            
                scores_dict = {"content_id": content_id, "scores": content_scores}

                for query_item in content_answers.get("queries", []):
                    user_prompt = query_item["query"]
                    answers = query_item.get("answers", {})
                    
                    print(f"[{content_id}] Scoring Query: '{user_prompt[:30]}...'")
                    
                    for mode in ["video", "full", "part"]:
                        generated_answer = answers.get(mode)
                        if not generated_answer or str(generated_answer).startswith("Error"):
                            print(f"[{content_id}]  Evaluating [{mode}] skipped (no valid answer).")
                            continue
                            
                        # 기존에 이미 평가 완료된 모드라면 다시 API를 부르지 않음
                        if mode in item_existing_score_map.get(user_prompt, {}):
                            print(f"[{content_id}]  Evaluating [{mode}] already completed (skip).")
                            continue
                        
                        print(f"[{content_id}]  Evaluating [{mode}]...")
                        time.sleep(1) # 평가 루프 과부하 방지
                        
                        max_parse_retries = 3
                        parse_success = False
                        current_is_first = is_first_turn_for_mode[mode]
                        score_dict = None
                        
                        for attempt in range(max_parse_retries):
                            try:
                                score_text = evaluate_answer_session(
                                    judge_chat=judge_chats[mode], 
                                    user_prompt=user_prompt, 
                                    generated_answer=generated_answer,
                                    is_first_turn=current_is_first,
                                    video_part=video_part,
                                    gt_json_part=gt_part
                                )
                                
                                # JSON 파싱 정제부
                                clean_text = score_text.strip()
                                if clean_text.startswith("```json"):
                                    clean_text = clean_text[7:]
                                if clean_text.endswith("```"):
                                    clean_text = clean_text[:-3]
                                    
                                score_dict = json.loads(clean_text)
                                parse_success = True
                                break # 파싱 성공 시 루프 탈출
                                
                            except json.JSONDecodeError:
                                print(f"[{content_id}]  [Warning] JSON 파싱 실패 (시도 {attempt+1}/{max_parse_retries}). 잠시 후 재시도합니다.")
                                print(f"[{content_id}]  [Raw Text]: {score_text[:100]}...")
                                # 재시도 시에는 비디오 파트를 다시 업로드하지 않기 위해 첫 턴 플래그 해제
                                current_is_first = False
                                time.sleep(2)
                                
                            except Exception as e:
                                print(f"[{content_id}]  Evaluating [{mode}] error: {e}")
                                break # 기타 API 오류는 중단
                                
                        if not parse_success:
                            print(f"[{content_id}]  [Error] {max_parse_retries}번의 재시도 끝에 JSON 파싱에 실패했습니다.")
                            score_dict = {"raw_response": score_text if 'score_text' in locals() else "Error"}
                            
                        score_entry = {
                            "query": user_prompt,
                            "mode": mode,
                            "judge": score_dict 
                        }   
                        
                        content_scores.append(score_entry)
                        is_first_turn_for_mode[mode] = False
                    
                    # 쿼리 한 개 평가가 끝날 때마다 JSONL로 Append 저장 (부분 저장)
                    with file_write_lock:
                        with open(args.output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(scores_dict, ensure_ascii=False) + "\n")
                    print(f"[{content_id}]  -> 진행중 쿼리 임시 저장 완료: {args.output_file}")
                    print("-" * 50)

                processed_ids.add(content_id)
                return True

            items_to_process = [ca for ca in content_answers_list if ca["content_id"] in pending_work]
            if items_to_process:
                new_data_processed = True
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    executor.map(process_item, items_to_process)

            if not args.continuous:
                break
                
            if not new_data_processed:
                # 새 데이터가 없으면 5초 대기
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 모니터링 루프가 중단되었습니다.")
        
    if not args.continuous:
        print("\n[Aggregation] JSONL 결과를 분석용 JSON 형식으로 병합합니다...")
        output_dir = os.path.dirname(args.output_file) or "output"
        subprocess.run([sys.executable, "jsonl_to_json.py", "--input_dir", output_dir])

    print("\n모든 평가 처리가 완료/종료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
