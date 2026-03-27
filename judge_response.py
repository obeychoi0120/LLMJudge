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
    start_chat_session, 
    evaluate_answer_session, init_judge_model,
    load_config, parse_json_response
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Responses using Judge model")
    parser.add_argument("--answers_file", default="output/responses.jsonl", help="답변 목록 JSONL 파일 경로")
    parser.add_argument("--output_file", default="output/scores.jsonl", help="최종 평가 결과 저장 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--judge_model", default="gemini-2.5-pro", help="사용할 평가 모델명")
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")

    args = parser.parse_args()

    args = load_config(args)
                
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
            # 1. Output (진행률) 읽기 - (content_id, query) 쌍 단위로 추적
            processed_pairs = set()  # (content_id, query) 튜플
            if os.path.exists(args.output_file):
                with open(args.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            sc = json.loads(line)
                            c_id = sc.get("content_id")
                            query = sc.get("query")
                            if c_id and query:
                                processed_pairs.add((c_id, query))
                        except json.JSONDecodeError:
                            pass

            # 2. Input 읽기 - 새 포맷: 각 줄 = {"content_id", "query", "answers"}
            #    content_id별로 queries 리스트로 재그룹핑
            content_answers_dict = {}  # content_id -> {"content_id": ..., "queries": [...]}
            content_query_order = {}   # content_id -> [query, ...] (순서 보존)
            if os.path.exists(args.answers_file):
                with open(args.answers_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                c_id = data.get("content_id")
                                query = data.get("query")
                                answers = data.get("answers")
                                if c_id and query and answers:
                                    # 새 포맷: (content_id, query) 단위 레코드
                                    if c_id not in content_answers_dict:
                                        content_answers_dict[c_id] = {"content_id": c_id, "queries": []}
                                        content_query_order[c_id] = []
                                    if query not in content_query_order[c_id]:
                                        content_query_order[c_id].append(query)
                                        content_answers_dict[c_id]["queries"].append({
                                            "query": query, 
                                            "reference": data.get("reference", ""),
                                            "answers": answers
                                        })
                                elif c_id and "queries" in data:
                                    # 구 포맷 호환: {"content_id", "queries": [...]} 단위 레코드
                                    content_answers_dict[c_id] = data
                            except json.JSONDecodeError:
                                pass
                content_answers_list = list(content_answers_dict.values())
            else:
                content_answers_list = []
                if not args.continuous:
                    print(f"Error: {args.answers_file} 파일이 존재하지 않습니다. 먼저 generate_response.py를 실행하세요.")
                    return

            new_data_processed = False

            # Resume Plan 계산 및 출력 - (content_id, query) 쌍 단위
            pending_work = {}
            for content_answers in content_answers_list:
                c_id = content_answers["content_id"]
                
                c_pending = []
                for query_item in content_answers.get("queries", []):
                    q_str = query_item["query"]
                    answers = query_item.get("answers", {})
                    # 평가 가능한 유효 답변이 하나라도 있고, 아직 처리 안 된 쌍이면 pending
                    has_valid_answer = any(
                        answers.get(m) and not str(answers.get(m, "")).startswith("Error")
                        for m in ["video", "full", "part"]
                    )
                    if has_valid_answer and (c_id, q_str) not in processed_pairs:
                        c_pending.append(q_str)
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

            def process_item(content_answers):
                content_id = content_answers["content_id"]
                if content_id not in pending_work:
                    return False
                    
                print(f"\nEvaluating Content: '{content_id}'")

                pending_queries = pending_work[content_id]
                
                for query_item in content_answers.get("queries", []):
                    user_prompt = query_item["query"]
                    answers = query_item.get("answers", {})
                    reference_answer = query_item.get("reference", "")
                    
                    # 이미 처리된 (content_id, query) 쌍이면 건너뜀
                    if user_prompt not in pending_queries:
                        print(f"[{content_id}] Scoring Query: '{user_prompt[:30]}...' -> already completed (skip)")
                        continue
                    
                    print(f"[{content_id}] Scoring Query: '{user_prompt[:30]}...'")
                    
                    if not reference_answer or str(reference_answer).startswith("Error"):
                        print(f"[{content_id}]  [Warning] Reference answer가 없거나 오류입니다. 이 쿼리를 건너뜁니다.")
                        continue
                    
                    judge_results = {}  # mode -> score_dict
                    
                    def judge_for_mode(mode):
                        generated_answer = answers.get(mode)
                        if not generated_answer or not str(generated_answer).strip() or str(generated_answer).startswith("Error"):
                            print(f"[{content_id}]  Evaluating [{mode}] skipped (no valid answer).")
                            return mode, None
                        
                        print(f"[{content_id}]  Evaluating [{mode}]...")
                        time.sleep(1)
                        
                        # 독립 세션: (query, mode)별 완전 격리
                        judge_model = init_judge_model(model_name=args.judge_model)
                        judge_chat = start_chat_session(judge_model)
                        
                        max_parse_retries = 3
                        parse_success = False
                        score_dict = None
                        
                        for attempt in range(max_parse_retries):
                            try:
                                score_text = evaluate_answer_session(
                                    judge_chat=judge_chat, 
                                    user_prompt=user_prompt, 
                                    generated_answer=generated_answer,
                                    reference_answer=reference_answer
                                )
                                
                                score_dict = parse_json_response(score_text)
                                parse_success = True
                                break
                                
                            except json.JSONDecodeError:
                                print(f"[{content_id}]  [Warning] JSON 파싱 실패 (시도 {attempt+1}/{max_parse_retries}). 잠시 후 재시도합니다.")
                                print(f"[{content_id}]  [Raw Text]: {score_text[:100]}...")
                                time.sleep(2)
                                
                            except Exception as e:
                                print(f"[{content_id}]  Evaluating [{mode}] error: {e}")
                                break
                                
                        if not parse_success:
                            print(f"[{content_id}]  [Error] JSON 파싱 최종 실패.")
                            score_dict = {"raw_response": score_text if 'score_text' in locals() else "Error"}
                            
                        return mode, score_dict

                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as mode_executor:
                        futures = [mode_executor.submit(judge_for_mode, m) for m in ["video", "full", "part"]]
                        for future in concurrent.futures.as_completed(futures):
                            mode, score_dict = future.result()
                            if score_dict is not None:
                                judge_results[mode] = score_dict
                    
                    # 쿼리 한 개 평가가 끝나면 (content_id, query) 단위로 1줄 append
                    if judge_results:
                        # mode 순서 정렬 (video, full, part)
                        ordered_judge = {m: judge_results[m] for m in ["video", "full", "part"] if m in judge_results}
                        score_record = {
                            "content_id": content_id,
                            "query": user_prompt,
                            "judge": ordered_judge
                        }
                        with file_write_lock:
                            with open(args.output_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps(score_record, ensure_ascii=False) + "\n")
                        processed_pairs.add((content_id, user_prompt))
                    print(f"[{content_id}]  -> Score 저장 완료: {args.output_file}")
                    print("-" * 50)

                return True

            for content_answers in content_answers_list:
                if process_item(content_answers):
                    new_data_processed = True

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
