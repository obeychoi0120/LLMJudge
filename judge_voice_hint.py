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
)

# ───────────────────────────────────────────────
# Voice Hint Judge 프롬프트 (Text-based)
# ───────────────────────────────────────────────

_QUERY_JUDGE_PROMPT = """You are a top-tier Customer Experience (CX) expert and Product Manager designing a 'personalized interactive viewing experience' for a smart TV platform. Your goal is to evaluate the business value of AI-generated questions intended for viewers.

The ultimate business objectives of these generated questions are to stimulate passive viewers to:
1) Immediately interact with the TV using their remote control.
2) Prevent attention drift (e.g., looking at their smartphones to search for information).
3) Lead to further content exploration within the TV platform, increasing retention time.

You will be provided with a [Reference Scene Summary (Past summary and current scene description)], which gives the context of when the question was generated. Based on this text summary, evaluate the generated question on the following 2 criteria, scoring each from 1 to 5.

[IMPORTANT EVALUATION PHILOSOPHY: Tangential World Knowledge over Narrative]
The Reference Scene Summary (KSS) is primarily focused on the 'narrative/story progression'. However, an excellent interactive question does NOT need to be tied to the main story. We highly value "Tangential World Knowledge" triggered by visual/audio cues.
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

_QUERY_JUDGE_FORMAT_PROMPT = """[Output Format]
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
    return make_generate_config(system_instruction=_QUERY_JUDGE_PROMPT, thinking_level=thinking_level)


def evaluate_query(client, model_name, judge_config, detailed_summary, query_text):
    """생성된 질문을 KeyScene Summary 기반으로 평가합니다."""
    contents = []
    if detailed_summary:
        contents += ["--- [Reference Scene Summary (Past summary and current scene description)] ---", detailed_summary]
    contents += [
        f"[Candidate Question]\n{query_text}\n\n",
        "Based on the [Reference Scene Summary] and the [Evaluation Criteria] defined above, evaluate the quality of the candidate question.\n\n" + _QUERY_JUDGE_FORMAT_PROMPT
    ]

    return _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=judge_config
        ).text,
        label="Query Judge API",
    )

def judge_one(q_item, content_id, scene_idx, detailed_summary,
              client, args, judge_config, file_write_lock):
    query_text = q_item["query"]
    mode = q_item.get("mode", "unknown")
    if not detailed_summary:
        return {"mode": mode, "query": query_text, "success": False, "msg": f"[Warning] Scene {scene_idx}에 KeyScene Summary가 없습니다. 스킵합니다."}

    try:
        time.sleep(1)

        score_dict = retry_parse_json(
            lambda: evaluate_query(
                client, args.vh_judge_model, judge_config,
                detailed_summary, query_text
            ),
            label=f"VH Judge (Scene {scene_idx}, {mode})",
        )

        _SCORE_KEYS = ["temporal_immersion", "curiosity_and_hook"]
        total = sum(
            (score_dict.get(k, {}).get("score", 0) if isinstance(score_dict.get(k), dict) else 0)
            for k in _SCORE_KEYS
        ) if score_dict else 0

        score_record = {
            "content_id": content_id,
            "scene_idx": scene_idx,
            "mode": mode,
            "query": query_text,
            "judge": score_dict,
            "total_score": total,
        }

        _ITEM_LABELS = [
            ("temporal_immersion", "시점 몰입도 및 답변 가능성"),
            ("curiosity_and_hook", "호기심 및 상호작용 유도력"),
        ]

        out_str = ""
        if score_dict:
            score_details = ", ".join([
                f"{label}: {score_dict.get(key, {}).get('score', 'N/A')}/5"
                for key, label in _ITEM_LABELS
            ])
            out_str = f"Mode ({mode}) | Score: {total}/10 | {score_details}\nQuery: {query_text}\n"
            
        append_jsonl(args.scores_file, score_record, lock=file_write_lock)
        return {"mode": mode, "query": query_text, "success": True, "out_str": out_str}

    except Exception as e:
        return {"mode": mode, "query": query_text, "success": False, "msg": f"[Error] Judge 최종 실패 ({mode}): {e}"}

def main():
    parser = get_common_argparser(description="Voice Hint 질문을 KeyScene Summary 기반으로 품질 평가")
    parser.add_argument("--input_file", default="assets/voice_hint.jsonl", help="Voice Hint 질문 목록 JSONL 경로")
    parser.add_argument("--kss_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--scores_file", default="assets/voice_hint_scores.jsonl", help="Voice Hint 질문별 Judge 점수 저장 경로")
    parser.add_argument("--watch", action="store_true", help="파일을 계속 모니터링하여 새 질문을 실시간으로 평가합니다.")
    parser.add_argument("--modes", nargs="+", default=["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk3", "imgvlm_graph"], choices=["kss", "video", "raw", "frag", "frag_with_vlm", "imgvlm_chunk2", "imgvlm_chunk3", "imgvlm_graph", "raw_with_mmvlm"], help="평가할 모드 직접 지정 (기본값: kss, video, raw_with_mmvlm, imgvlm_chunk2, imgvlm_chunk3, imgvlm_graph)")
    

    args, client = init_pipeline(parser.parse_args())
    judge_config = make_query_judge_config(thinking_level=args.vh_judge_thinking_level)

    ensure_output_dir(args.scores_file)

    processed_pairs = load_processed_pairs(args.scores_file, key_fields=("content_id", "mode", "query"))
    if processed_pairs:
        print(f"[{len(processed_pairs)}] 개의 (content_id, mode, query) 쌍이 이미 처리됨.")

    if not check_input_file(args.input_file, hint="먼저 generate_voice_hint.py를 실행하세요."):
        return

    # Summary 맵 로드: (content_id, scene_idx) -> summary_text
    summary_map = load_summary_map(args.kss_file)
    if summary_map:
        print(f"[Summary] {len(summary_map)}개 Scene의 Summary 로드됨 ({args.kss_file})")
    elif not os.path.exists(args.kss_file):
        print(f"[Warning] Summary 파일을 찾을 수 없습니다: {args.kss_file}")

    print_pipeline_banner(f"Voice Hint 질문 품질 평가 프로세스 시작 (Watch 모드: {args.watch})")

    file_write_lock = threading.Lock()
    last_position = 0
    pipeline_done = False
    
    total_generated = 0
    total_evaluated = len(processed_pairs)
    
    accumulated_groups = {}
    accumulated_modes = {}
    target_modes_set = set(args.modes)
    target_mode_order = ["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk3", "imgvlm_graph"]

    try:
        while True:
            if not os.path.exists(args.input_file):
                if not args.watch:
                    print(f"Error: {args.input_file} 파일이 존재하지 않습니다.")
                    return
                time.sleep(2)
                continue
                
            with open(args.input_file, "r", encoding="utf-8") as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
                
            new_items = []
            for line in new_lines:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if obj.get("pipeline_done"):
                            pipeline_done = True
                        else:
                            new_items.append(obj)
                    except json.JSONDecodeError:
                        pass
                        
            if new_items:
                for scene_item in new_items:
                    content_id = scene_item.get("content_id")
                    scene_idx  = scene_item.get("scene_idx")
                    mode        = scene_item.get("mode", "")
                    queries_list = scene_item.get("queries", [])

                    if not content_id or scene_idx is None or not mode:
                        continue
                    if mode not in target_modes_set:
                        continue

                    group_key = (content_id, scene_idx)
                    accumulated_modes.setdefault(group_key, set()).add(mode)

                    detailed_summary = summary_map.get((content_id, scene_idx), "")

                    for q_text in queries_list:
                        total_generated += 1
                        if (content_id, mode, q_text) not in processed_pairs:
                            accumulated_groups.setdefault(group_key, []).append({
                                "content_id": content_id,
                                "scene_idx": scene_idx,
                                "mode": mode,
                                "query": q_text,
                                "detailed_summary": detailed_summary,
                            })

            # 준비된 그룹(씬) 판별: 해당 씬의 모든 타겟 모드가 기록되었거나 파이프라인이 끝났을 때
            ready_groups = {}
            for group_key, modes_set in list(accumulated_modes.items()):
                if pipeline_done or modes_set.issuperset(target_modes_set):
                    items = accumulated_groups.pop(group_key, [])
                    if items:
                        ready_groups[group_key] = items
                    del accumulated_modes[group_key]

            if ready_groups or pipeline_done:
                # 그룹별 순차 처리, 같은 Scene 내의 수집된 모든 질문은 병렬 처리
                for (c_id, s_idx), items in ready_groups.items():
                    print(f"\n[{c_id} | Scene {s_idx}] {len(items)}개 질문(전체 모드) 병렬 평가 시작...")
                    
                    results_list = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
                        futures = [
                            executor.submit(
                                judge_one,
                                {"mode": item["mode"], "query": item["query"]},
                                item["content_id"], item["scene_idx"], item["detailed_summary"],
                                client, args, judge_config, file_write_lock
                            )
                            for item in items
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            res = future.result()
                            if res:
                                results_list.append(res)
                            total_evaluated += 1

                    processed_pairs.update(
                        (item["content_id"], item["mode"], item["query"]) for item in items
                    )
                    
                    # 지정된 mode 순서대로 결과 정렬
                    def sort_key(r):
                        m = r.get("mode", "")
                        return target_mode_order.index(m) if m in target_mode_order else 999

                    results_list.sort(key=sort_key)
                    
                    # 정렬된 순서대로 한꺼번에 출력
                    print(f"--- [{c_id} | Scene {s_idx} 평가 완료] ---")
                    for r in results_list:
                        if r.get("success"):
                            if r.get("out_str"):
                                print(r["out_str"])
                        else:
                            if r.get("msg"):
                                print(r["msg"])

                # TODO List 현황 출력
                pending_count = total_generated - total_evaluated
                print(f"\n▶ [Judging TODO List] Total Generated: {total_generated} | Evaluated: {total_evaluated} | Pending: {pending_count}")

            if args.watch:
                if pipeline_done:
                    print("\n[Watch] Generation 파이프라인의 종료 시그널(pipeline_done)을 감지했습니다. 모든 처리를 완료하고 종료합니다.")
                    break
                time.sleep(3)
            else:
                break

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    sort_jsonl_file(args.scores_file)
    print_pipeline_done(args.scores_file)

if __name__ == "__main__":
    main()
