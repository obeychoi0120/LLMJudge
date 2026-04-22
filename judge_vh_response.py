import os
import json
import time
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    _retry_api_call, retry_parse_json,
    ensure_output_dir, load_processed_pairs,
    init_pipeline, load_jsonl, append_jsonl,
    load_summary_map,
    print_pipeline_banner, print_pipeline_done,
)

# ============================================================
# Judge Prompts
# ============================================================

_JUDGE_SYSTEM_PROMPT = """\
You are an objective, expert evaluator assessing the quality of AI-generated responses about video content.
The AI model generates answers based on various representations of the original video (visual frames, audio, text metadata, or multimodal descriptions).

Your goal is to evaluate how well the [Candidate Answer] responds to the [User Question], using the [Anchor (KeyScene Summary)] as the sole ground-truth reference.
The Anchor contains a summary of past events and a detailed description of the current scene.
Do NOT use external knowledge — evaluate strictly against the Anchor.

Evaluate across 3 criteria, each scored 1–5:

**1. Answer Relevance**
Does the Candidate directly and substantively answer the User Question using information from the Anchor?
- 5: Directly addresses the question with precise, relevant information that fully satisfies the query.
- 4: Addresses the question well with minor gaps in relevance or slight tangential content.
- 3: Partially answers the question but includes noticeable irrelevant content or misses the question's focus.
- 2: Only loosely related to the question; significant portions are off-topic.
- 1: Fails to address the question or provides a completely unrelated response.

**2. Factual Precision**
How accurately does the Candidate reflect verifiable facts from the Anchor: proper nouns, events, dialogue content, numbers, and specific claims?
- 5: All key facts, names, and events match the Anchor with high fidelity.
- 4: Most facts are correct with minor omissions, but NO fabricated information.
- 3: Gets core facts right but omits several important details, or contains 1–2 minor inaccuracies.
- 2: Contains multiple factual errors or significant omissions.
- 1: Fabricates facts not present in the Anchor or severely misidentifies key entities.
IMPORTANT: Fabrication (inventing facts not in the Anchor) must be penalized MORE harshly than omission (failing to mention facts). A response that omits a detail is better than one that invents a wrong detail.

**3. Response Quality**
Is the response well-structured, natural, and appropriate for a viewer?
- 5: Reads naturally, is well-organized with logical flow, and uses no system terminology.
- 4: Mostly natural with minor awkwardness in phrasing or structure.
- 3: Understandable but noticeably awkward, verbose, or poorly organized.
- 2: Difficult to follow, excessively verbose, or contains system terminology leakage (e.g., 'metadata', 'current scene', 'field name').
- 1: Incoherent, unreadable, or dominated by system/internal terminology.

Output ONLY the following JSON. No other text.
{
    "answer_relevance": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence from Anchor vs Candidate>",
        "score": <integer 1-5>
    },
    "factual_precision": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence from Anchor vs Candidate>",
        "score": <integer 1-5>
    },
    "response_quality": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence from Anchor vs Candidate>",
        "score": <integer 1-5>
    }
}"""


def make_judge_config(thinking_level=None):
    return make_generate_config(system_instruction=_JUDGE_SYSTEM_PROMPT, thinking_level=thinking_level)


def evaluate_answer(client, model_name, judge_config, user_prompt, generated_answer, keyscene_summary):
    """
    Judge 모델이 KeyScene Summary를 기준으로 generated_answer를 비교 평가합니다.
    """
    return _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name,
            contents=[
                "--- [Anchor] (Ground-truth KeyScene Summary, Korean) ---",
                keyscene_summary,
                "--- [Candidate Answer] ---",
                generated_answer,
                "--- Request ---",
                f"Evaluate the Candidate Answer to the following User Question against the Anchor "
                f"using the 3 criteria defined in the system prompt.\n"
                f"[User Question]: {user_prompt}\n\n"
                f"Output ONLY the JSON with answer_relevance, factual_precision, and response_quality scores.",
            ],
            config=judge_config
        ).text,
        label="VH Response Judge API",
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
                    
                    _SCORE_KEYS = ["answer_relevance", "factual_precision", "response_quality"]
                    _MODE_ORDER = ["video", "raw", "img_desc", "mm_desc"]
                    judge_results = {}  # mode -> score_dict

                    def judge_for_mode(mode):
                        generated_answer = answers.get(mode)
                        if not generated_answer or not str(generated_answer).strip() or str(generated_answer).startswith("Error"):
                            return mode, None

                        score_dict = retry_parse_json(
                            lambda: evaluate_answer(
                                client=client,
                                model_name=args.uq_judge_model,
                                judge_config=judge_config,
                                user_prompt=user_prompt,
                                generated_answer=generated_answer,
                                keyscene_summary=reference_answer
                            ),
                            label=f"VH Judge ({content_id}, {mode})",
                        )
                        return mode, score_dict

                    # 4모드 병렬 Judge
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                    try:
                        futures = {m: executor.submit(judge_for_mode, m) for m in _MODE_ORDER}

                        # 정규 순서로 결과 수집 & 출력
                        for mode in _MODE_ORDER:
                            mode_result, score_dict = futures[mode].result()
                            if score_dict is None:
                                print(f"[{mode}] Skip (no valid answer)")
                                continue

                            judge_results[mode] = score_dict
                            total = sum(
                                (score_dict.get(k, {}).get("score", 0) if isinstance(score_dict.get(k), dict) else 0)
                                for k in _SCORE_KEYS
                            )
                            scores_str = " | ".join(
                                f"{k}={score_dict.get(k, {}).get('score', '?')}" for k in _SCORE_KEYS
                            )
                            print(f"[{mode}] Total: {total}/15 | {scores_str}")
                            for k in _SCORE_KEYS:
                                item = score_dict.get(k, {})
                                rationale = item.get("rationale", "N/A") if isinstance(item, dict) else "N/A"
                                score = item.get("score", "?") if isinstance(item, dict) else "?"
                                print(f"- {k} ({score}/5): {rationale}")
                    finally:
                        executor.shutdown(wait=False, cancel_futures=True)

                    # 쿼리 한 개 평가가 끝나면 (content_id, query) 단위로 1줄 append
                    if judge_results:
                        ordered_judge = {m: judge_results[m] for m in _MODE_ORDER if m in judge_results}
                        score_record = {
                            "content_id": content_id,
                            "query": user_prompt,
                            "judge": ordered_judge
                        }
                        append_jsonl(args.output_file, score_record, lock=file_write_lock)
                        processed_pairs.add((content_id, user_prompt))
                    print(f"\n[{content_id}] -> Score 저장 완료")
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
