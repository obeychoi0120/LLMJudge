import os
import argparse
import json
import time
import subprocess
import sys
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    start_chat_session,
    parse_json_response,
    _retry_api_call, retry_parse_json,
    ensure_output_dir, load_processed_pairs,
    init_pipeline, load_jsonl, append_jsonl,
    load_summary_map,
    print_pipeline_banner, print_pipeline_done,
)

# ============================================================
# Judge Prompts
# ============================================================

_JUDGE_PROMPT = """\
당신은 AI 모델이 특정한 영상에 대해 생성한 답변의 품질을 평가하는 객관적이고 전문적인 평가자입니다.
해당 AI 모델은 원본 영상의 시각적, 청각적 정보를 바탕으로 답변을 생성합니다.

당신의 목표는 영상의 실제 내용인 [영상 컨텍스트 (KeyScene Summary)]를 유일한 사실적 근거(Ground Truth Anchor)로 삼아, 
[사용자 질문]에 대한 [평가 대상 답변]이 얼마나 정확하고 훌륭한지 평가하는 것입니다.
[영상 컨텍스트]는 과거 사건의 요약과 현재 장면의 상세한 묘사를 포함하고 있습니다.
외부 검색은 허용하지 않습니다.

[데이터 목록]
- 영상 컨텍스트 (KeyScene Summary): 평가의 기준이 되는 영상의 실제 내용 (과거 장면 요약 및 현재 장면 묘사)
- 사용자 질문
- 평가 대상 답변

[평가 기준]
아래 세 가지 항목에 대해 1점부터 5점까지 점수를 매겨주세요. (1점: 매우 나쁨, 3점: 보통/수용 가능함, 5점: 완벽함)
1. 정확성 (Accuracy): 평가 대상 답변이 [영상 컨텍스트]의 핵심 사실과 일치하는가? [영상 컨텍스트]에 언급된 정보와 모순되거나 사실과 다른 내용(환각)이 포함되어 있지는 않은가?
2. 포괄성 (Completeness): [사용자 질문]에 대답하기 위해 [영상 컨텍스트]에서 반드시 언급되어야 할 핵심 단서를 평가 대상 답변도 누락 없이 포함했는가?
3. 가독성 (Helpfulness): 정보가 장황하게 나열되지 않고, 시간의 흐름이나 인과관계에 맞게 자연스럽고 이해하기 쉽게 작성되었는가? 시청자 관점에서 자연스러운 문장인가? (만약 평가 대상 답변이 부자연스럽게 메타데이터 구조나 필드명, '현재 장면', '과거 장면' 등의 시스템적인 용어를 직접 언급했다면 이 항목에서 감점을 고려하세요.)"""

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

def make_judge_config(thinking_level=None):
    return make_generate_config(system_instruction=_JUDGE_PROMPT, thinking_level=thinking_level)


def evaluate_answer_session(client, model_name, judge_config, user_prompt, generated_answer, keyscene_summary):
    """
    Judge 모델이 KeyScene Summary를 기준으로 generated_answer를 비교 평가합니다.
    """
    user_content = (
        f"[평가 대상 답변]\n{generated_answer}\n\n"
        f"[Reference: 영상 컨텍스트 (KeyScene Summary)]\n{keyscene_summary}\n\n"
        f"위 영상 컨텍스트를 근거로 삼아, 아래 [사용자 질문]에 대한 평가 대상 답변의 품질을 평가하세요.\n"
        f"[사용자 질문]: {user_prompt}\n\n"
        f"{_JUDGE_FORMAT_PROMPT}"
    )

    judge_chat = start_chat_session(client, model_name, judge_config)
    return _retry_api_call(
        lambda: judge_chat.send_message([user_content]).text,
        label="Judge API",
    )


def main():
    parser = get_common_argparser(description="Evaluate Responses using Judge model")
    parser.add_argument("--answers_file", default="assets/vh_responses.jsonl", help="답변 목록 JSONL 파일 경로 (generate_vh_response.py 출력)")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 파일 경로")
    parser.add_argument("--output_file", default="assets/vh_response_scores.jsonl", help="최종 평가 결과 저장 경로 (.jsonl)")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")
    parser.add_argument("--skip_aggregate", action="store_true", help="수행 완료 후 자동 집계 로직을 건너뜁니다.")

    args, client = init_pipeline(parser.parse_args())
    judge_config = make_judge_config(thinking_level=args.uq_judge_thinking_level)
    
    # 출력 폴더 생성
    ensure_output_dir(args.output_file)

    print_pipeline_banner("Gemini Evaluation 프로세스를 시작합니다 (Session-based, JSONL Pipeline).")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다. 다른 터미널의 출력을 기다리며 지속 처리합니다.")

    try:
        while True:
            # 1. Output (진행률) 읽기 - (content_id, query) 쌍 단위로 추적
            processed_pairs = load_processed_pairs(args.output_file)

            # 2-1. KeyScene Summary 읽기
            summary_map = load_summary_map(args.keyscene_summary_file)

            # 2-2. Input 읽기 - 새 포맷: 각 줄 = {"content_id", "query", "answers"}
            #    content_id별로 queries 리스트로 재그룹핑
            content_answers_dict = {}  # content_id -> {"content_id": ..., "queries": [...]}
            content_query_order = {}   # content_id -> [query, ...] (순서 보존)
            for data in load_jsonl(args.answers_file):
                c_id = data.get("content_id")
                query = data.get("query")
                scene_idx = data.get("scene_idx")
                answers = data.get("answers")
                if c_id and query and answers:
                    if c_id not in content_answers_dict:
                        content_answers_dict[c_id] = {"content_id": c_id, "queries": []}
                        content_query_order[c_id] = []
                    if query not in content_query_order[c_id]:
                        content_query_order[c_id].append(query)
                        ref_text = summary_map.get((c_id, scene_idx), data.get("reference", ""))
                        content_answers_dict[c_id]["queries"].append({
                            "query": query,
                            "scene_idx": scene_idx,
                            "reference": ref_text,
                            "answers": answers
                        })
            content_answers_list = list(content_answers_dict.values())
            if not content_answers_list and not args.continuous:
                print(f"Error: {args.answers_file} 파일이 존재하지 않거나 비어 있습니다.")
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
                        for m in ["video", "raw", "img_desc", "mm_desc"]
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
                    
                    if not reference_answer or str(reference_answer).startswith("Error") or str(reference_answer).startswith("Warning"):
                        print(f"[{content_id}]  [Warning] KeyScene Summary가 없거나 오류입니다. 이 쿼리를 건너뜁니다.")
                        continue
                    
                    judge_results = {}  # mode -> score_dict
                    
                    def judge_for_mode(mode):
                        generated_answer = answers.get(mode)
                        if not generated_answer or not str(generated_answer).strip() or str(generated_answer).startswith("Error"):
                            print(f"[{content_id}]  Evaluating [{mode}] skipped (no valid answer).")
                            return mode, None

                        print(f"[{content_id}]  Evaluating [{mode}]...")
                        time.sleep(1)

                        score_dict = retry_parse_json(
                            lambda: evaluate_answer_session(
                                client=client,
                                model_name=args.uq_judge_model,
                                judge_config=judge_config,
                                user_prompt=user_prompt,
                                generated_answer=generated_answer,
                                keyscene_summary=reference_answer
                            ),
                            label=f"[{content_id}] [{mode}] Judge",
                        )
                        return mode, score_dict

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as mode_executor:
                        futures = [mode_executor.submit(judge_for_mode, m) for m in ["video", "raw", "img_desc", "mm_desc"]]
                        for future in concurrent.futures.as_completed(futures):
                            mode, score_dict = future.result()
                            if score_dict is not None:
                                judge_results[mode] = score_dict
                    
                    # 쿼리 한 개 평가가 끝나면 (content_id, query) 단위로 1줄 append
                    if judge_results:
                        # mode 순서 정렬 (video, raw, img_desc, mm_desc)
                        ordered_judge = {m: judge_results[m] for m in ["video", "raw", "img_desc", "mm_desc"] if m in judge_results}
                        score_record = {
                            "content_id": content_id,
                            "query": user_prompt,
                            "judge": ordered_judge
                        }
                        append_jsonl(args.output_file, score_record, lock=file_write_lock)
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
        os._exit(1)
        
    if not args.continuous:
        if not args.skip_aggregate:
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
