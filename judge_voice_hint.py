import os
import argparse
import json
import time
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    parse_json_response,
    _retry_api_call, retry_parse_json,
    ensure_output_dir, load_processed_pairs,
    init_pipeline, load_jsonl, append_jsonl,
    load_summary_map, check_input_file,
    sort_jsonl_file,
    print_pipeline_banner, print_pipeline_done,
    print_scores_summary,
)

# ───────────────────────────────────────────────
# Voice Hint Judge 프롬프트 — query_type별 분리
# ───────────────────────────────────────────────

# 공통 도입부
_QUERY_JUDGE_COMMON_INTRO = """You are a top-tier Customer Experience (CX) expert and Product Manager designing a 'personalized interactive viewing experience' for a smart TV platform. Your goal is to evaluate the business value of AI-generated questions intended for viewers.

The ultimate business objectives of these generated questions are to stimulate passive viewers to:
1) Immediately interact with the TV using their remote control.
2) Prevent attention drift (e.g., looking at their smartphones to search for information).
3) Lead to further content exploration within the TV platform, increasing retention time.

You will be provided with a [Reference Scene Summary (Past summary and current scene description)], which gives the context of when the question was generated. Based on this text summary, evaluate the generated question on the following 2 criteria, scoring each from 1 to 5."""

# ── Content-Anchored 질문 전용 Judge ──
_QUERY_JUDGE_PROMPT_CONTENT_ANCHORED = _QUERY_JUDGE_COMMON_INTRO + """

[IMPORTANT EVALUATION PHILOSOPHY: Content-Anchored Questions]
This question is a "Content-Anchored" question — it is EXPECTED to directly engage with the CORE topic, event, or person of the current scene. Evaluate it based on how deeply and precisely it targets the scene's central narrative or subject matter. Do NOT require tangential/external knowledge for this type — instead, reward depth of insight into the scene's main content.

[Evaluation Criteria]

1. Temporal Immersion & Answerability (Platform Constraints)
Does the question logically fit the [Current Scene] without making future predictions, and can the TV system immediately answer it based on current information?
- 5: Perfectly timed. Excludes future speculations like "What will happen next?". Focuses purely on immediately identifiable objects or hidden background knowledge based on specific visual/audio/narrative clues in the [Current Scene].
- 4: Highly relevant to the current scene and answerable, but the timing or clarity of the clues might be very slightly off.
- 3: Relevant to the current scene, but the clues are too vague or the focus is too broad, making it less appealing for immediate curiosity resolution.
- 2: Answerable, but somewhat out of context or might unnecessarily distract the viewer from the current screen.
- 1: Unanswerable "future prediction" questions (e.g., "What will the result be?") that require watching more of the video to know, OR "late" questions asking about facts already revealed in the [Past Scene Summary].

2. Content Depth & Relevance (Core Topic Engagement)
How deeply and precisely does the question engage with the CORE topic, event, or person of the current scene?
- 5: Precisely targets the central event/topic of the current scene and pushes the viewer to think one level deeper — e.g., asking about the psychological motivation behind a character's action, the tactical significance of a play, or the policy implications of a news story. Goes beyond surface-level fact-checking.
- 4: Clearly connected to the scene's core topic with meaningful depth, but the angle or specificity could be slightly sharper.
- 3: Related to the scene but focuses on a secondary element rather than the core topic, or stays at a surface level without pushing for deeper insight.
- 2: Only loosely tied to the current scene's main subject; could apply to many generic scenes.
- 1: Completely disconnected from the scene's core narrative, or is a trivial factual question that anyone could answer without watching."""

# ── Tangential 질문 전용 Judge ──
_QUERY_JUDGE_PROMPT_TANGENTIAL = _QUERY_JUDGE_COMMON_INTRO + """

[IMPORTANT EVALUATION PHILOSOPHY: Tangential World Knowledge over Narrative]
This question is a "Tangential Expansion" question — it is EXPECTED to go BEYOND the main narrative to explore interesting background knowledge triggered by visual/audio cues. The Reference Scene Summary (KSS) is primarily focused on the 'narrative/story progression'. However, an excellent tangential question does NOT need to be tied to the main story. We highly value "Tangential World Knowledge" triggered by visual/audio cues.
For example:
- Dramas/Variety: Asking about the historical origin of a prop or filming location.
- Sports: Asking about a player's recent form, transfer history, or team history for newbies.
- Gaming: Asking about character meta, item updates, or strategic nuances.
- News/Docs: Asking about the historical context or economic ripple effects of the topic.
If a question successfully leverages such deep, tangential background knowledge to spark curiosity, it MUST be highly rewarded. Do NOT penalize a question simply because it asks about external knowledge not explicitly written in the KSS.

[Evaluation Criteria]

1. Temporal Immersion & Answerability (Platform Constraints)
Does the question logically fit the [Current Scene] without making future predictions, and can the TV system immediately answer it based on current information?
- 5: Perfectly timed. Excludes future speculations like "What will happen next?". Focuses purely on immediately identifiable objects or hidden background knowledge based on specific visual/audio/narrative clues in the [Current Scene].
- 4: Highly relevant to the current scene and answerable, but the timing or clarity of the clues might be very slightly off.
- 3: Relevant to the current scene, but the clues are too vague or the focus is too broad, making it less appealing for immediate curiosity resolution.
- 2: Answerable, but somewhat out of context or might unnecessarily distract the viewer from the current screen.
- 1: Unanswerable "future prediction" questions (e.g., "What will the result be?") that require watching more of the video to know, OR "late" questions asking about facts already revealed in the [Past Scene Summary].

2. Curiosity & Hook (Intrinsic Intrigue & Tone)
Evaluate the psychological hook, conversational tone, and naturalness of the question ITSELF, entirely independent of Criterion 1. Even if the question asks about the future or is poorly timed, how engaging is the wording?
- 5: An incredibly engaging question that leverages deep Tangential World Knowledge (historical, cultural, sports stats, gaming meta, etc.) based on screen elements. Politely and elegantly stimulates curiosity like an expert critic. Concise and natural.
- 4: Interesting enough to encourage interaction, but the wording is slightly generic, or the applied world knowledge is somewhat shallow.
- 3: A reasonable question related to the video content, but lacks a strong motivational hook to actually force interaction.
- 2: Has an informational purpose but is too stiff, unnatural, or requires the viewer to think too hard about the question's intent.
- 1: A completely obvious question, or one that is phrased very mechanically, generating zero curiosity."""

_QUERY_JUDGE_FORMAT_PROMPT_CONTENT_ANCHORED = """[Output Format]
Output ONLY the following JSON. Do NOT output any other text.
Write the rationale first, followed by the score for each criterion.
{
    "temporal_immersion": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    },
    "content_depth": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    }
}"""

_QUERY_JUDGE_FORMAT_PROMPT_TANGENTIAL = """[Output Format]
Output ONLY the following JSON. Do NOT output any other text.
Write the rationale first, followed by the score for each criterion.
{
    "temporal_immersion": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    },
    "curiosity_and_hook": {
        "rationale": "<Concise evaluation reasoning in English, citing specific evidence>",
        "score": <integer 1-5>
    }
}"""

def make_query_judge_config(thinking_level=None):
    """query_type별 Judge config를 반환합니다."""
    return {
        "content_anchored": make_generate_config(system_instruction=_QUERY_JUDGE_PROMPT_CONTENT_ANCHORED, thinking_level=thinking_level),
        "tangential": make_generate_config(system_instruction=_QUERY_JUDGE_PROMPT_TANGENTIAL, thinking_level=thinking_level),
    }


# query_type별 score key 매핑
_SCORE_KEYS_BY_TYPE = {
    "content_anchored": ["temporal_immersion", "content_depth"],
    "tangential": ["temporal_immersion", "curiosity_and_hook"],
}

def evaluate_query(client, model_name, judge_config, detailed_summary, query_text, query_type="tangential"):
    """생성된 질문을 KeyScene Summary 기반으로 평가합니다. query_type에 따라 다른 기준 적용."""
    format_prompt = _QUERY_JUDGE_FORMAT_PROMPT_CONTENT_ANCHORED if query_type == "content_anchored" else _QUERY_JUDGE_FORMAT_PROMPT_TANGENTIAL
    config = judge_config.get(query_type, judge_config.get("tangential"))

    contents = []
    if detailed_summary:
        contents += ["--- [Reference Scene Summary (Past summary and current scene description)] ---", detailed_summary]
    contents += [
        f"[Candidate Question]\n{query_text}\n\n",
        "Based on the [Reference Scene Summary] and the [Evaluation Criteria] defined above, evaluate the quality of the candidate question.\n\n" + format_prompt
    ]

    return _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=config
        ).text,
        label=f"Query Judge API ({query_type})",
    )

def judge_one(q_item, content_id, scene_idx, detailed_summary,
              client, args, judge_config, file_write_lock):
    query_text = q_item["query"]
    mode = q_item.get("mode", "unknown")
    query_type = q_item.get("query_type", "tangential")
    if not detailed_summary:
        return {"mode": mode, "query_type": query_type, "query": query_text, "success": False, "msg": f"[Warning] Scene {scene_idx}에 KeyScene Summary가 없습니다. 스킵합니다."}

    try:
        time.sleep(1)

        score_dict = retry_parse_json(
            lambda: evaluate_query(
                client, args.vh_judge_model, judge_config,
                detailed_summary, query_text, query_type=query_type
            ),
            label=f"VH Judge (Scene {scene_idx}, {mode}, {query_type})",
        )

        score_keys = _SCORE_KEYS_BY_TYPE.get(query_type, ["temporal_immersion", "curiosity_and_hook"])
        total = sum(
            (score_dict.get(k, {}).get("score", 0) if isinstance(score_dict.get(k), dict) else 0)
            for k in score_keys
        ) if score_dict else 0

        score_record = {
            "content_id": content_id,
            "scene_idx": scene_idx,
            "mode": mode,
            "query_type": query_type,
            "query": query_text,
            "judge": score_dict,
            "total_score": total,
        }

        _ITEM_LABELS_BY_TYPE = {
            "content_anchored": [
                ("temporal_immersion", "시점 몰입도"),
                ("content_depth", "콘텐츠 핵심 깊이"),
            ],
            "tangential": [
                ("temporal_immersion", "시점 몰입도"),
                ("curiosity_and_hook", "호기심 유도력"),
            ],
        }
        item_labels = _ITEM_LABELS_BY_TYPE.get(query_type, _ITEM_LABELS_BY_TYPE["tangential"])

        out_str = ""
        if score_dict:
            score_details = ", ".join([
                f"{label}: {score_dict.get(key, {}).get('score', 'N/A')}/5"
                for key, label in item_labels
            ])
            type_tag = "CA" if query_type == "content_anchored" else "TG"
            out_str = f"[{type_tag}] Query: {query_text}\nScore: {total}/10 | {score_details}"
            
        append_jsonl(args.scores_file, score_record, lock=file_write_lock)
        return {"mode": mode, "query_type": query_type, "query": query_text, "success": True, "out_str": out_str, "total": total}

    except Exception as e:
        return {"mode": mode, "query_type": query_type, "query": query_text, "success": False, "msg": f"[Error] Judge 최종 실패 ({mode}, {query_type}): {e}"}

def main():
    parser = get_common_argparser(description="Voice Hint 질문을 KeyScene Summary 기반으로 품질 평가")
    parser.add_argument("--input_file", default="assets/voice_hint.jsonl", help="Voice Hint 질문 목록 JSONL 경로")
    parser.add_argument("--kss_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--scores_file", default="assets/voice_hint_scores.jsonl", help="Voice Hint 질문별 Judge 점수 저장 경로")
    parser.add_argument("--modes", nargs="+", default=["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk2_meta", "imgvlm_graph", "imgvlm_graph_meta"], choices=["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk2_meta", "imgvlm_graph", "imgvlm_graph_meta"], help="평가할 모드 직접 지정")
    

    args, client = init_pipeline(parser.parse_args())
    judge_config = make_query_judge_config(thinking_level=args.vh_judge_thinking_level)

    ensure_output_dir(args.scores_file)

    if not check_input_file(args.input_file, hint="먼저 generate_voice_hint.py를 실행하세요."):
        return

    # Summary 맵 로드: (content_id, scene_idx) -> summary_text
    summary_map = load_summary_map(args.kss_file)
    if summary_map:
        print(f"[Summary] {len(summary_map)}개 Scene의 Summary 로드됨 ({args.kss_file})")
    elif not os.path.exists(args.kss_file):
        print(f"[Warning] Summary 파일을 찾을 수 없습니다: {args.kss_file}")

    print_pipeline_banner("Voice Hint 질문 품질 평가 프로세스 시작")

    file_write_lock = threading.Lock()
    target_modes_set = set(args.modes)
    target_mode_order = ["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk2_meta", "imgvlm_graph", "imgvlm_graph_meta"]
    try:
        discovery_pass = 0
        while True:
            discovery_pass += 1

            # 매 pass마다 입력 파일을 다시 읽어 새로 추가된 VH 레코드를 감지
            processed_pairs = load_processed_pairs(args.scores_file, key_fields=("content_id", "mode", "query"))
            if discovery_pass == 1 and processed_pairs:
                print(f"[{len(processed_pairs)}] 개의 (content_id, mode, query) 쌍이 이미 처리됨.")

            # 미처리 항목 그룹핑
            accumulated_groups = {}
            for scene_item in load_jsonl(args.input_file):
                if scene_item.get("pipeline_done"):
                    continue
                content_id   = scene_item.get("content_id")
                scene_idx    = scene_item.get("scene_idx")
                mode         = scene_item.get("mode", "")
                queries_list = scene_item.get("queries", [])
                # query_types 필드가 없으면 index 기반으로 추론
                query_types_list = scene_item.get("query_types", ["content_anchored", "tangential"][:len(queries_list)])

                if not content_id or scene_idx is None or not mode:
                    continue
                if mode not in target_modes_set:
                    continue

                detailed_summary = summary_map.get((content_id, scene_idx), "")
                group_key = (content_id, scene_idx)

                for q_idx, q_text in enumerate(queries_list):
                    q_type = query_types_list[q_idx] if q_idx < len(query_types_list) else "tangential"
                    if (content_id, mode, q_text) not in processed_pairs:
                        accumulated_groups.setdefault(group_key, []).append({
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "mode": mode,
                            "query_type": q_type,
                            "query": q_text,
                            "detailed_summary": detailed_summary,
                        })

            if not accumulated_groups:
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

            total_pending = sum(len(v) for v in accumulated_groups.values())
            print(f"\n{'='*50}")
            print(f"[Discovery {discovery_pass}] {len(accumulated_groups)}개 Scene, 미처리 {total_pending}개 질문 발견")
            print(f"{'='*50}")

            pass_evaluated = 0
            completed_content_ids = set()
            prev_c_id = None
            _VH_SCORE_KEYS = ["temporal_immersion", "content_depth", "curiosity_and_hook"]
            for (c_id, s_idx), items in accumulated_groups.items():
                if prev_c_id and prev_c_id != c_id and prev_c_id not in completed_content_ids:
                    completed_content_ids.add(prev_c_id)
                    print_scores_summary(args.scores_file, prev_c_id, _VH_SCORE_KEYS, target_mode_order, max_score=10)
                prev_c_id = c_id
                print(f"\n[{c_id} | Scene {s_idx}] {len(items)}개 질문(전체 모드) 병렬 평가 시작...")

                results_list = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(items))) as executor:
                    futures = [
                        executor.submit(
                            judge_one,
                            {"mode": item["mode"], "query_type": item.get("query_type", "tangential"), "query": item["query"]},
                            item["content_id"], item["scene_idx"], item["detailed_summary"],
                            client, args, judge_config, file_write_lock
                        )
                        for item in items
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res:
                            results_list.append(res)
                        pass_evaluated += 1

                def sort_key(r):
                    m = r.get("mode", "")
                    return target_mode_order.index(m) if m in target_mode_order else 999

                results_list.sort(key=sort_key)
                print(f"--- [{c_id} | Scene {s_idx} 평가 완료] ---")

                # Mode별 그룹핑 출력
                from collections import defaultdict
                mode_groups = defaultdict(list)
                for r in results_list:
                    if r.get("success") and r.get("out_str"):
                        mode_groups[r["mode"]].append(r)
                    elif not r.get("success") and r.get("msg"):
                        print(r["msg"])

                for mode in [m for m in target_mode_order if m in mode_groups]:
                    print(f"\nMode ({mode})")
                    for r in mode_groups[mode]:
                        print(r["out_str"])

            # 마지막 content_id 집계
            if prev_c_id and prev_c_id not in completed_content_ids:
                print_scores_summary(args.scores_file, prev_c_id, _VH_SCORE_KEYS, target_mode_order, max_score=10)

            print(f"\n▶ [Discovery {discovery_pass}] {pass_evaluated}개 평가 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    sort_jsonl_file(args.scores_file)
    print_pipeline_done(args.scores_file)

if __name__ == "__main__":
    main()
