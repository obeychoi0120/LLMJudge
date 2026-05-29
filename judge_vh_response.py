import os
import time
import concurrent.futures
import threading
from collections import OrderedDict
from utils import (
    get_common_argparser,
    make_generate_config,
    _retry_api_call, retry_parse_json,
    ensure_output_dir, load_processed_pairs,
    init_pipeline, load_jsonl, append_jsonl,
    load_summary_map, load_content_indices,
    sort_jsonl_file,
    print_pipeline_banner, print_pipeline_done,
    print_scores_summary, ProgressTracker,
)

# ============================================================
# Judge Prompts
# ============================================================

_JUDGE_SYSTEM_PROMPT = """You are an objective, expert evaluator assessing the quality of AI-generated responses about video content.
The AI model generates answers based on various representations of the original video (visual frames, audio, text metadata, or multimodal descriptions).

Your goal is to evaluate how well the [Candidate Answer] responds to the [User Question].
The [Anchor (KeyScene Summary)] provides a summary of past events and a detailed description of the current scene. Use the Anchor as a factual reference, but NOT as the only acceptable source of information.

IMPORTANT EVALUATION PHILOSOPHY:
- The Candidate is expected to combine video-derived context with its own World Knowledge to provide the most helpful and satisfying answer to the viewer.
- Information from external knowledge that is factually accurate and relevant to the question should NOT be penalized, even if it is absent from the Anchor.
- However, information that CONTRADICTS the Anchor (i.e., conflicts with what is shown/said in the video) must be penalized.
- The primary evaluation question is: "Would a knowledgeable viewer find this answer helpful, accurate, and satisfying?"

Evaluate across 3 criteria, each scored 1–5:

**1. Answer Relevance**
Does the Candidate directly and substantively answer the User Question?
- 5: Directly addresses the question with precise, relevant information that fully satisfies the query. May enrich the answer with accurate supplementary knowledge.
- 4: Addresses the question well with minor gaps in relevance or slight tangential content.
- 3: Partially answers the question but includes noticeable irrelevant content or misses the question's focus.
- 2: Only loosely related to the question; significant portions are off-topic.
- 1: Fails to address the question or provides a completely unrelated response.

**2. Factual Precision**
How factually accurate is the Candidate's response, considering both the Anchor and general World Knowledge?
- 5: All stated facts are accurate — both video-derived facts (matching the Anchor) and any supplementary World Knowledge are correct. Accurate external knowledge that supplements the Anchor should be rewarded, not penalized.
- 4: Most facts are correct with minor omissions; any external knowledge used is accurate and relevant.
- 3: Gets core facts right but contains 1–2 minor inaccuracies, or external knowledge is partially incorrect.
- 2: Contains multiple factual errors, OR directly contradicts what the Anchor describes (e.g., misidentifying visible entities, misquoting dialogue). Anchor contradictions are weighted more heavily than inaccurate external knowledge.
- 1: Severely misidentifies key entities from the video, or provides fundamentally incorrect external knowledge.

**3. Informativeness**
Does the response provide specific, non-obvious information that genuinely extends the viewer's understanding?
NOTE: This criterion does NOT assess accuracy (covered by Factual Precision) or topicality (covered by Answer Relevance). Focus SOLELY on the depth and novelty of information provided.
- 5: Provides concrete, specific details the viewer could not easily infer from watching alone — e.g., behind-the-scenes context, historical background, expert-level insight, or quantitative data. Genuinely enriches the viewing experience.
- 4: Offers mostly specific information with good depth, but one or two points remain at surface level.
- 3: Mix of specific and generic information. Some useful details but also includes commonly known facts or vague statements that add little value.
- 2: Mostly generic or surface-level. Restates what is already obvious from the video or provides only broad generalizations.
- 1: Entirely generic, trivially obvious, or empty filler content with no informational value.

Output ONLY the following JSON. No other text.
{
    "answer_relevance": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    },
    "factual_precision": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    },
    "informativeness": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
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
                f"Output ONLY the JSON with answer_relevance, factual_precision, and informativeness scores.",
            ],
            config=judge_config
        ).text,
        label="VH Response Judge API",
    )


def is_content_id_fully_evaluated_vh_response(content_id, input_file, target_sources_set, summary_map, processed_pairs):
    """content_id에 해당하는 모든 target_source의 응답들이 processed_pairs에 존재하여 평가가 완료되었는지 확인합니다."""
    expected_scenes = {s_idx for (c_id, s_idx) in summary_map.keys() if c_id == content_id}
    if not expected_scenes:
        return False

    # 해당 content_id에 대해 입력 파일에 존재하는 실제 모드들을 추출
    existing_modes = set()
    for obj in load_jsonl(input_file):
        if obj.get("content_id") == content_id:
            m = obj.get("mode")
            if m:
                existing_modes.add(m)
                
    active_modes = target_sources_set & existing_modes
    if not active_modes:
        return False

    actual_answers = {}
    for obj in load_jsonl(input_file):
        if obj.get("pipeline_done"):
            continue
        c_id = obj.get("content_id")
        if c_id != content_id:
            continue
        s_idx = obj.get("scene_idx")
        mode = obj.get("mode")
        if mode not in active_modes:
            continue
        if s_idx not in expected_scenes:
            continue
            
        generated_answer = obj.get("answer")
        if not generated_answer or str(generated_answer).startswith("Error"):
            continue
            
        query = obj.get("query")
        if query:
            actual_answers.setdefault((s_idx, mode), []).append(query)

    for s_idx in expected_scenes:
        for mode in active_modes:
            key = (s_idx, mode)
            if key not in actual_answers:
                return False
            queries = actual_answers[key]
            for query in queries:
                if (content_id, s_idx, mode, query) not in processed_pairs:
                    return False
                    
    return True



def main():
    parser = get_common_argparser(description="Evaluate VH Responses using Judge model")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--output_file", default="assets/vh_response_scores.jsonl", help="평가 결과 저장 경로")
    parser.add_argument("--sources", nargs="+", default=["blank", "video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph"], 
                        choices=["blank", "video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph"], help="평가할 Source 직접 지정")
    parser.add_argument("--query_source", choices=["kss", "sourcewise"], default="kss",
                        help="평가할 Voice Hint 질문의 출처 (kss: KSS 기반 공통 질문, sourcewise: 각 모드별로 생성된 질문)")

    args, client = init_pipeline(parser.parse_args())
    content_indices = load_content_indices()

    query_source = args.query_source

    # query_source에 따라 input_file, output_file 경로 변경
    args.input_file = f"assets/vh_responses_{query_source}.jsonl"
    if args.output_file == "assets/vh_response_scores.jsonl":
        args.output_file = f"assets/vh_response_scores_{query_source}.jsonl"

    judge_config = make_judge_config(thinking_level=args.vh_response_judge_thinking_level)

    ensure_output_dir(args.output_file)

    if not os.path.exists(args.input_file):
        print(f"[Error] {args.input_file} 파일이 없습니다. generate_vh_response.py를 먼저 실행하세요.")
        return

    # KSS Anchor 로드
    summary_map = load_summary_map(args.keyscene_summary_file)
    if summary_map:
        print(f"[KSS Anchor] {len(summary_map)}개 Scene의 Summary 로드됨.")
    else:
        print(f"[Warning] KSS 파일을 찾을 수 없거나 비어있습니다: {args.keyscene_summary_file}")

    print_pipeline_banner("VH Response 품질 평가 파이프라인을 시작합니다.")

    file_write_lock = threading.Lock()
    target_sources_set = set(args.sources)
    if query_source == "sourcewise" and "blank" in target_sources_set:
        target_sources_set.remove("blank")
    _SCORE_KEYS = ["answer_relevance", "factual_precision", "informativeness"]
    _MODE_ORDER = ["blank", "video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_graph"]

    def judge_query_item(data):
        """단일 {content_id, scene_idx, mode, query, answer} 레코드를 평가합니다. (로깅 없이 결과만 반환)"""
        c_id = data["content_id"]
        s_idx = data.get("scene_idx")
        mode = data.get("mode")
        query = data["query"]
        query_type = data.get("query_type", "")
        generated_answer = data.get("answer")
        anchor = summary_map.get((c_id, s_idx), "")

        if not anchor:
            return None
        if not generated_answer or str(generated_answer).startswith("Error"):
            return None

        score_dict = retry_parse_json(
            lambda: evaluate_answer(
                client=client,
                model_name=args.vh_response_judge_model,
                judge_config=judge_config,
                user_prompt=query,
                generated_answer=generated_answer,
                keyscene_summary=anchor,
            ),
            label=f"VH Judge ({c_id}, {mode})",
        )

        if not score_dict:
            return None

        total = sum(
            (score_dict.get(k, {}).get("score", 0) if isinstance(score_dict.get(k), dict) else 0)
            for k in _SCORE_KEYS
        )

        record = OrderedDict([
            ("content_id", c_id),
            ("scene_idx",  s_idx),
            ("mode",       mode),
            ("query_type", query_type),
            ("query",      query),
            ("judge",      score_dict),
            ("total",      total),
        ])
        with file_write_lock:
            append_jsonl(args.output_file, record)
            processed_pairs.add((c_id, s_idx, mode, query))
        return record

    def _process_query_group(query_text, items):
        """같은 Query의 여러 mode를 병렬 평가 후, Query 한 번 출력 → mode별 점수를 정렬 출력합니다."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = {executor.submit(judge_query_item, obj): obj for obj in items}
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        if not results:
            return 0

        # Query 한 번 출력
        c_id = results[0]["content_id"]
        s_idx = results[0]["scene_idx"]
        print(f"\n[{c_id} | Scene {s_idx}] \nQuery: {query_text}")

        # mode 정렬 출력
        results.sort(key=lambda r: _MODE_ORDER.index(r["mode"]) if r["mode"] in _MODE_ORDER else 99)
        for r in results:
            scores_str = " | ".join(
                f"{k}={r['judge'].get(k, {}).get('score', '?')}" for k in _SCORE_KEYS
            )
            print(f"  -> [{r['mode']}] {r['total']}/15 ({scores_str})")

        return len(results)

    def _process_scene_group_sourcewise(c_id, s_idx, items):
        """sourcewise 모드: 한 Scene 내의 모든 Response(서로 다른 Query)를 병렬 평가 후, 점수를 정렬 출력합니다."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = {executor.submit(judge_query_item, obj): obj for obj in items}
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        if not results:
            return 0

        # mode 순서 및 query 기준으로 정렬하여 출력
        results.sort(key=lambda r: (_MODE_ORDER.index(r["mode"]) if r["mode"] in _MODE_ORDER else 99, r["query"]))
        
        print(f"\n[{c_id} | Scene {s_idx}]")
        for r in results:
            scores_str = " | ".join(
                f"{k}={r['judge'].get(k, {}).get('score', '?')}" for k in _SCORE_KEYS
            )
            print(f"  -> [{r['mode']}] {r['total']}/15 ({scores_str})")

        return len(results)


    printed_content_ids = set()

    def check_and_print_summaries(pairs_set):
        all_content_ids = set()
        for obj in load_jsonl(args.input_file):
            c_id = obj.get("content_id")
            if c_id:
                all_content_ids.add(c_id)
        for c_id in sorted(all_content_ids, key=lambda c: content_indices.get(c, 999)):
            if c_id not in printed_content_ids:
                if is_content_id_fully_evaluated_vh_response(c_id, args.input_file, target_sources_set, summary_map, pairs_set):
                    printed_content_ids.add(c_id)
                    print_scores_summary(args.output_file, c_id, _SCORE_KEYS, _MODE_ORDER, max_score=15)

    try:
        discovery_pass = 0
        while True:
            discovery_pass += 1

            # 매 pass마다 입력 파일을 다시 읽어 새로 추가된 Response를 감지
            processed_pairs = set()
            if os.path.exists(args.output_file):
                for rec in load_jsonl(args.output_file):
                    c_id  = rec.get("content_id")
                    s_idx = rec.get("scene_idx")
                    mode  = rec.get("mode")
                    query = rec.get("query")
                    if c_id and s_idx is not None and mode and query:
                        processed_pairs.add((c_id, s_idx, mode, query))
            
            check_and_print_summaries(processed_pairs)

            if discovery_pass == 1 and processed_pairs:
                print(f"[기처리] {len(processed_pairs)}개 항목이 이미 평가 완료됨.")

            # 전체 파일 로드 → 미처리 항목 그룹핑
            all_data = [r for r in load_jsonl(args.input_file) if not r.get("pipeline_done")]

            if not all_data:
                if discovery_pass == 1:
                    print("[Error] 평가할 Response 데이터가 없습니다. generate_vh_response.py를 먼저 실행하세요.")
                    return
                else:
                    print("[완료] 입력 파일에 처리할 Response가 없습니다.")
                    break

            unprocessed_groups = OrderedDict()
            for obj in all_data:
                c_id  = obj.get("content_id")
                s_idx = obj.get("scene_idx")
                mode  = obj.get("mode")
                query = obj.get("query")
                if not (c_id and s_idx is not None and mode and query):
                    continue
                if mode not in target_sources_set:
                    continue
                if (c_id, s_idx, mode, query) in processed_pairs:
                    continue
                if query_source == "kss":
                    q_key = (c_id, s_idx, query)
                else:  # sourcewise
                    q_key = (c_id, s_idx)
                unprocessed_groups.setdefault(q_key, []).append(obj)

            if not unprocessed_groups:
                if discovery_pass == 1:
                    print("[완료] 평가할 항목이 없거나 이미 모두 처리되었습니다.")
                    break
                else:
                    empty_streak = getattr(main, '_empty_streak', 0) + 1
                    main._empty_streak = empty_streak
                    if empty_streak >= 3:
                        print(f"[완료] {empty_streak}회 연속 추가 항목 없음. 모든 평가가 완료되었습니다.")
                        break
                    print(f"[대기] 추가 항목 없음 ({empty_streak}/3). 20초 후 재확인...")
                    time.sleep(20)
                    continue
            else:
                main._empty_streak = 0

            total_pending = sum(len(v) for v in unprocessed_groups.values())
            print(f"\n{'='*50}")
            if query_source == "kss":
                print(f"[Discovery {discovery_pass}] {len(unprocessed_groups)}개 Query, 미처리 {total_pending}개 항목 발견")
            else:
                print(f"[Discovery {discovery_pass}] {len(unprocessed_groups)}개 Scene, 미처리 {total_pending}개 항목 발견")
            print(f"{'='*50}")

            pass_evaluated = 0
            tracker = ProgressTracker(total_pending, unit="responses", action="evaluated")
            if query_source == "kss":
                for (c_id, s_idx, q_text), items in unprocessed_groups.items():
                    pass_evaluated += _process_query_group(q_text, items)
                    check_and_print_summaries(processed_pairs)
                    tracker.update(pass_evaluated)
            else:  # sourcewise
                for (c_id, s_idx), items in unprocessed_groups.items():
                    pass_evaluated += _process_scene_group_sourcewise(c_id, s_idx, items)
                    check_and_print_summaries(processed_pairs)
                    tracker.update(pass_evaluated)
            print(f"\n▶ [Discovery {discovery_pass}] {pass_evaluated}개 평가 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    sort_jsonl_file(args.output_file)
    print_pipeline_done(args.output_file)

if __name__ == "__main__":
    main()
