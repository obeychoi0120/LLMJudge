import os
import argparse
import json
import time
import subprocess
import sys
import concurrent.futures
import threading
from gemini_api_utils import (
    create_client, make_generate_config,
    start_chat_session,
    load_config, parse_json_response,
    _retry_api_call,
    ensure_output_dir, load_processed_pairs,
)

# ============================================================
# Judge Prompts
# ============================================================

_JUDGE_PROMPT = """\
당신은 AI 모델이 특정한 영상에 대해 생성한 답변의 품질을 평가하는 객관적이고 전문적인 평가자입니다.
해당 AI 모델은 원본 영상에서 추출한 메타데이터 기반으로 답변을 생성합니다.

당신의 목표는 원본 영상에 대한 [사용자 질문]에 대해 [평가 대상 답변]이 얼마나 훌륭한지,
[기준 답변(Reference Answer)]과 비교하여 평가하는 것입니다.
[기준 답변]은 원본 영상과 Reference 메타데이터를 모두 참조하여 생성된 고품질 정답입니다.
외부 검색은 허용하지 않습니다.

[데이터 목록]
- 기준 답변 (Reference Answer): 원본 영상 + Reference 메타데이터를 기반으로 생성된 정답 답변
- 사용자 질문
- 평가 대상 답변

[평가 기준]
아래 세 가지 항목에 대해 1점부터 5점까지 점수를 매겨주세요. (1점: 매우 나쁨, 3점: 보통/수용 가능함, 5점: 완벽함)
1. 정확성 (Accuracy): 평가 대상 답변이 기준 답변의 핵심 사실과 일치하는가? 기준 답변에 언급된 정보와 모순되거나 사실과 다른 내용(환각)이 포함되어 있지는 않은가?
2. 포괄성 (Completeness): 기준 답변에 포함된 핵심 단서(대사, 텍스트 내용, 행동, 맥락 등)를 평가 대상 답변도 누락 없이 포함했는가?
3. 가독성 (Helpfulness): 정보가 장황하게 나열되지 않고, 시간의 흐름이나 인과관계에 맞게 자연스럽고 이해하기 쉽게 작성되었는가? (만약 평가 대상 답변이 부자연스럽게 메타데이터 구조나 필드명 등을 직접 언급했다면 이 항목에서 감점을 고려하세요.)"""

_JUDGE_FORMAT_PROMPT = """\
[출력 형식]
반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명은 덧붙이지 마십시오.
{
    "rationale": "<각 점수를 부여한 논리적인 이유. 감점 요인이 있다면 명확히 서술하세요.>",
    "scores": {
        "accuracy": <1~5 사이의 정수>,
        "completeness": <1~5 사이의 정수>,
        "helpfulness": <1~5 사이의 정수>
    },
    "total_score": <세 항목 점수의 합계, 최대 15점>
}"""

def make_judge_config(thinking_budget=None):
    return make_generate_config(system_instruction=_JUDGE_PROMPT, thinking_budget=thinking_budget)


def evaluate_answer_session(client, model_name, judge_config, user_prompt, generated_answer, reference_answer):
    """
    Judge 모델이 Reference Answer를 기준으로 generated_answer를 비교 평가합니다.
    """
    user_content = (
        f"[사용자 질문]\n{user_prompt}\n\n"
        f"[기준 답변 (Reference Answer)]\n{reference_answer}\n\n"
        f"[평가 대상 답변]\n{generated_answer}\n\n"
        f"{_JUDGE_FORMAT_PROMPT}"
    )

    judge_chat = start_chat_session(client, model_name, judge_config)
    return _retry_api_call(
        lambda: judge_chat.send_message([user_content]).text,
        label="Judge API",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Responses using Judge model")
    parser.add_argument("--answers_file", default="assets/uq_responses.jsonl", help="답변 목록 JSONL 파일 경로")
    parser.add_argument("--references_file", default="assets/uq_references.jsonl", help="Reference 답변 목록 JSONL 파일 경로")
    parser.add_argument("--output_file", default="assets/uq_response_scores.jsonl", help="최종 평가 결과 저장 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--uq_judge_model", default="gemini-2.5-pro", help="사용할 평가 모델명")
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")
    parser.add_argument("--uq_judge_thinking_budget", type=int, default=2048,
                        help="UQ Judge 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args = parser.parse_args()

    args = load_config(args)
                
    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    client = create_client(args.gcp_project_id, args.location)
    judge_config = make_judge_config(thinking_budget=args.uq_judge_thinking_budget)
    
    # 출력 폴더 생성
    ensure_output_dir(args.output_file)

    print("\n" + "=" * 50)
    print("Gemini Evaluation 프로세스를 시작합니다 (Session-based, JSONL Pipeline).")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다. 다른 터미널의 출력을 기다리며 지속 처리합니다.")
    print("=" * 50)

    try:
        while True:
            # 1. Output (진행률) 읽기 - (content_id, query) 쌍 단위로 추적
            processed_pairs = load_processed_pairs(args.output_file)

            # 2-1. Reference 읽기
            reference_map = {} # (content_id, query) -> reference_text
            if os.path.exists(args.references_file):
                with open(args.references_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                ref_data = json.loads(line)
                                r_cid = ref_data.get("content_id")
                                r_query = ref_data.get("query")
                                r_text = ref_data.get("reference")
                                if r_cid and r_query and r_text:
                                    reference_map[(r_cid, r_query)] = r_text
                            except json.JSONDecodeError:
                                pass

            # 2-2. Input 읽기 - 새 포맷: 각 줄 = {"content_id", "query", "answers"}
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
                                        
                                        # reference_map에서 참조 가져오기
                                        ref_text = reference_map.get((c_id, query), data.get("reference", ""))
                                        
                                        content_answers_dict[c_id]["queries"].append({
                                            "query": query, 
                                            "reference": ref_text,
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
                        
                        # 독립 세션: evaluate_answer_session 내부에서 생성
                        
                        max_parse_retries = 3
                        parse_success = False
                        score_dict = None
                        
                        for attempt in range(max_parse_retries):
                            try:
                                score_text = evaluate_answer_session(
                                    client=client,
                                    model_name=args.uq_judge_model,
                                    judge_config=judge_config,
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
        output_dir = os.path.dirname(args.output_file) or "assets"
        subprocess.run([sys.executable, "jsonl_to_json.py", "--input_dir", output_dir])
        
        # 추가 집계: aggregate_scores.py 호출
        scores_json = os.path.join(output_dir, "scores.json")
        if os.path.exists(scores_json):
            subprocess.run([sys.executable, "aggregate_scores.py", "--scores_file", scores_json])
            # 엑셀 변환 추가
            subprocess.run([sys.executable, "export_to_excel.py"])

    print("\n모든 평가 처리가 완료/종료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
