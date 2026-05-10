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
    sort_jsonl_file,
    print_pipeline_banner, print_pipeline_done,
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
- 5: All stated facts are accurate — both video-derived facts (matching the Anchor) and any supplementary World Knowledge are correct and well-integrated.
- 4: Most facts are correct with minor omissions; any external knowledge used is accurate and relevant.
- 3: Gets core facts right but contains 1–2 minor inaccuracies, or external knowledge is partially incorrect.
- 2: Contains multiple factual errors — either misrepresents video content or provides incorrect external information.
- 1: Severely misidentifies key entities from the video, or provides dangerously incorrect external knowledge.
IMPORTANT: Information that CONTRADICTS the Anchor (i.e., conflicts with what the video shows) must be penalized MORE harshly than inaccurate external knowledge. Accurate external knowledge that supplements the Anchor should be rewarded, not penalized.

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
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    },
    "factual_precision": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    },
    "response_quality": {
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
                f"Output ONLY the JSON with answer_relevance, factual_precision, and response_quality scores.",
            ],
            config=judge_config
        ).text,
        label="VH Response Judge API",
    )


def main():
    parser = get_common_argparser(description="Evaluate VH Responses using Judge model")
    parser.add_argument("--answers_file", default="assets/vh_responses.jsonl", help="VH Response JSONL 경로 (generate_vh_response.py 출력)")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--output_file", default="assets/vh_response_scores.jsonl", help="평가 결과 저장 경로")
    parser.add_argument("--watch", action="store_true", help="answers_file을 모니터링하며 새로운 Response를 실시간으로 평가합니다.")

    args, client = init_pipeline(parser.parse_args())
    judge_config = make_judge_config(thinking_level=args.vh_response_judge_thinking_level)

    ensure_output_dir(args.output_file)

    # 기처리분 로드: (content_id, scene_idx, mode, query) 단위
    processed_pairs = set()
    if os.path.exists(args.output_file):
        for rec in load_jsonl(args.output_file):
            c_id  = rec.get("content_id")
            s_idx = rec.get("scene_idx")
            mode  = rec.get("mode")
            query = rec.get("query")
            if c_id and s_idx is not None and mode and query:
                processed_pairs.add((c_id, s_idx, mode, query))
    if processed_pairs:
        print(f"[기처리] {len(processed_pairs)}개 항목이 이미 평가 완료됨.")

    # KSS Anchor 로드
    summary_map = load_summary_map(args.keyscene_summary_file)
    if summary_map:
        print(f"[KSS Anchor] {len(summary_map)}개 Scene의 Summary 로드됨.")
    else:
        print(f"[Warning] KSS 파일을 찾을 수 없거나 비어있습니다: {args.keyscene_summary_file}")

    print_pipeline_banner(f"VH Response 품질 평가 파이프라인을 시작합니다. (Watch 모드: {args.watch})")

    file_write_lock = threading.Lock()
    _SCORE_KEYS = ["answer_relevance", "factual_precision", "response_quality"]
    _MODE_ORDER = ["video", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk3", "imgvlm_graph"]

    if not os.path.exists(args.answers_file) and not args.watch:
        print(f"[Info] {args.answers_file} 파일이 존재하지 않습니다. 평가를 건너뜁니다.")
        print_pipeline_done(args.output_file)
        return

    last_position = 0
    pipeline_done = False
    total_evaluated = 0

    def judge_query_item(data):
        """단일 {content_id, scene_idx, mode, query, answer} 레코드를 평가합니다. (로깅 없이 결과만 반환)"""
        c_id = data["content_id"]
        s_idx = data.get("scene_idx")
        mode = data.get("mode")
        query = data["query"]
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

        record = {
            "content_id": c_id,
            "scene_idx":  s_idx,
            "mode":       mode,
            "query":      query,
            "judge":      score_dict,
            "total":      total,
        }
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

    try:
        while True:
            if not os.path.exists(args.answers_file):
                time.sleep(3)
                continue

            with open(args.answers_file, "r", encoding="utf-8") as f:
                f.seek(last_position)
                new_lines     = f.readlines()
                last_position = f.tell()

            pending = []
            for line in new_lines:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("pipeline_done"):
                    pipeline_done = True
                    continue

                c_id  = obj.get("content_id")
                s_idx = obj.get("scene_idx")
                mode  = obj.get("mode")
                query = obj.get("query")
                if not (c_id and s_idx is not None and mode and query):
                    continue
                if (c_id, s_idx, mode, query) in processed_pairs:
                    continue
                pending.append(obj)

            if pending:
                # Query별 그룹핑 → Query 단위로 순차, 모드는 병렬
                from collections import OrderedDict
                query_groups = OrderedDict()
                c_ids = set()
                for obj in pending:
                    c_ids.add(obj["content_id"])
                    q_key = (obj["content_id"], obj["scene_idx"], obj["query"])
                    query_groups.setdefault(q_key, []).append(obj)

                c_id_str = ", ".join(sorted(str(c) for c in c_ids))
                print(f"\n[Judge] [{c_id_str}] 새 항목 {len(pending)}개 ({len(query_groups)}개 Query) 평가 시작")
                for (c_id, s_idx, q_text), items in query_groups.items():
                    total_evaluated += _process_query_group(q_text, items)

                print(f"\n▶ [Judge] 누적 평가 완료: {total_evaluated}개")

            if args.watch:
                if pipeline_done:
                    print("\n[Watch] pipeline_done 시그널 감지. 종료합니다.")
                    break
                time.sleep(3)
            else:
                break

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    sort_jsonl_file(args.output_file)
    print_pipeline_done(args.output_file)

if __name__ == "__main__":
    main()
