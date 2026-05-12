import os
import concurrent.futures
import threading
from collections import OrderedDict
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

    args, client = init_pipeline(parser.parse_args())
    judge_config = make_judge_config(thinking_level=args.vh_response_judge_thinking_level)

    ensure_output_dir(args.output_file)

    if not os.path.exists(args.answers_file):
        print(f"[Error] {args.answers_file} 파일이 없습니다. generate_vh_response.py를 먼저 실행하세요.")
        return

    # KSS Anchor 로드
    summary_map = load_summary_map(args.keyscene_summary_file)
    if summary_map:
        print(f"[KSS Anchor] {len(summary_map)}개 Scene의 Summary 로드됨.")
    else:
        print(f"[Warning] KSS 파일을 찾을 수 없거나 비어있습니다: {args.keyscene_summary_file}")

    print_pipeline_banner("VH Response 품질 평가 파이프라인을 시작합니다.")

    file_write_lock = threading.Lock()
    _SCORE_KEYS = ["answer_relevance", "factual_precision", "response_quality"]
    _MODE_ORDER = ["video", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_graph"]

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
            if discovery_pass == 1 and processed_pairs:
                print(f"[기처리] {len(processed_pairs)}개 항목이 이미 평가 완료됨.")

            # 전체 파일 로드 → 미처리 항목 그룹핑
            all_data = [r for r in load_jsonl(args.answers_file) if not r.get("pipeline_done")]

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
                if (c_id, s_idx, mode, query) in processed_pairs:
                    continue
                q_key = (c_id, s_idx, query)
                unprocessed_groups.setdefault(q_key, []).append(obj)

            if not unprocessed_groups:
                if discovery_pass == 1:
                    print("[완료] 평가할 항목이 없거나 이미 모두 처리되었습니다.")
                else:
                    print("[완료] 추가 항목이 없습니다. 모든 평가가 완료되었습니다.")
                break

            total_pending = sum(len(v) for v in unprocessed_groups.values())
            print(f"\n{'='*50}")
            print(f"[Discovery {discovery_pass}] {len(unprocessed_groups)}개 Query, 미처리 {total_pending}개 항목 발견")
            print(f"{'='*50}")

            pass_evaluated = 0
            for (c_id, s_idx, q_text), items in unprocessed_groups.items():
                pass_evaluated += _process_query_group(q_text, items)

            print(f"\n▶ [Discovery {discovery_pass}] {pass_evaluated}개 평가 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    sort_jsonl_file(args.output_file)
    print_pipeline_done(args.output_file)

if __name__ == "__main__":
    main()
