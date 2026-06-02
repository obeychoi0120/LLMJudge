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
    load_summary_map, check_input_file, load_content_indices,
    sort_jsonl_file,
    print_pipeline_banner, print_pipeline_done,
    print_scores_summary, ProgressTracker,
)

# ───────────────────────────────────────────────
# Interactive Query Judge 프롬프트 — query_type별 분리
# ───────────────────────────────────────────────

# 공통 도입부
_QUERY_JUDGE_COMMON_INTRO = """You are a top-tier Customer Experience (CX) expert and Product Manager designing a 'personalized interactive viewing experience' for a smart TV platform. Your goal is to evaluate the business value of AI-generated questions intended for viewers.

The ultimate business objectives of these generated questions are to stimulate passive viewers to:
1) Immediately interact with the TV using their remote control.
2) Prevent attention drift (e.g., looking at their smartphones to search for information).
3) Lead to further content exploration within the TV platform, increasing retention time.

You will be provided with a [Reference Scene Summary (Past summary and current scene description)], which gives the context of when the question was generated. Based on this text summary, evaluate the generated question on the following 2 criteria, scoring each from 1 to 5."""

_SCENE_RELEVANCE_CRITERION = """1. Scene Relevance & Answerability (Platform Gate)
Evaluate whether the candidate question is well-timed for the [Current Scene] and can be answered at this exact moment. This criterion is NOT about how interesting, deep, or clickable the question is. It is only about whether the question belongs here, now.

A question should score high only when it is naturally triggered by the current viewing context and would feel misplaced if shown much earlier, much later, or in a generic part of the same video.

[Source-Aware Evidence Rules]
When identifying the current-scene trigger, distinguish evidence sources carefully:
- Spoken narration/dialogue is strong evidence for narrative, explanatory, interview, lecture, or documentary questions.
- On-screen captions, titles, labels, maps, menus, scoreboards, subtitles, product names, and visible UI text are valid triggers when they are salient in the scene.
- Visual objects, places, outfits, food, tools, animals, gestures, player actions, game items, vehicles, diagrams, charts, or environmental details are valid triggers, especially for tangential questions.
- Chat comments, scrolling comments, noisy OCR fragments, repeated viewer reactions, usernames, or incidental background text are weak evidence unless the current scene explicitly focuses on that text/comment.
- If ASR/OCR/VLM evidence conflicts, prefer the evidence most central to the current scene: spoken explanation for talk/lecture scenes, visible action for action scenes, and prominent labels/captions for information graphics.

[Talk / Explainer Content Rule]
For talk, lecture, documentary, interview, news, or explainer content, the current-scene trigger does NOT need to be a unique visual action. A currently discussed concept, claim, comparison, named entity, statistic, example, chart, or argument can be a strong scene trigger.
Do not penalize a question merely because it is triggered by spoken explanation rather than by a visual object.

[Past Context Rule]
Past context may be used when the [Current Scene] actively reintroduces, contrasts, applies, develops, or visually recalls that earlier information.
Penalize the question only when it merely repeats a fact that was already fully resolved in the [Past Scene Summary] without any fresh reason in the [Current Scene].

[Tangential Cue Distance Rule]
For tangential questions, judge the distance between the current-scene cue and the external knowledge being asked:
- Direct cue: visible/currently mentioned food -> origin, cooking science, ingredient, or local culture of that food.
- Direct cue: visible/currently mentioned animal -> behavior, biology, habitat, or conservation issue.
- Direct cue: visible/currently mentioned place/landmark -> geography, architecture, history, or local culture.
- Direct cue: visible/currently mentioned prop/tool/clothing/UI/chart -> design, function, origin, technology, or convention.
- Moderate cue: broad scene setting -> related background knowledge that still clearly helps understand the current scene.
- Weak cue: generic studio, generic background, generic topic, or video title only -> broad trivia not tied to the moment.
- Invalid cue: no identifiable current-scene cue, or only an inferred topic from metadata/title.

To evaluate this accurately, analyze the following checks in your rationale:
- Current Scene Trigger: Identify the concrete cue in the [Current Scene]. State whether it comes from speech, visible action, object, OCR text, caption, chart, place, person, or another source.
- Scene Specificity: Would this question feel specifically timed to this scene, or could it be asked almost anywhere in the same video?
- Immediate Answerability: Can the platform reasonably answer it now using available context or stable background knowledge, without needing future scenes?
- Future/Spoiler Risk: Does the question ask what will happen next, reveal a future event, or require information from later scenes?
- Past Redundancy: Was this already fully explained in the [Past Scene Summary]? If yes, does the [Current Scene] provide a fresh reason to ask it again?
- Reference Clarity: Are the entities clear from the current context, or are references like "this person," "that thing," or "the event" ambiguous?

Scoring Rubric:
- 5: Perfectly timed and scene-specific. The question is triggered by a concrete and salient current-scene cue. It would feel misplaced in most other scenes. It is immediately answerable and has no future/spoiler or stale-past issue.
- 4: Clearly relevant and answerable, but somewhat broader. The question is clearly connected to the current scene, but the trigger is a broader scene topic, recurring issue, or ongoing discussion rather than a highly unique detail. It would still feel reasonable here, though it might also work in a few nearby scenes.
- 3: Moderately timed but generic. The question is answerable and has some current-scene connection, but it is mostly derivable from the video's general topic, title, or recurring theme. It does not feel wrong here, but it is not strongly tied to this exact moment.
- 2: Poorly timed or weakly triggered. The current-scene trigger is missing, incidental, noisy, ambiguous, or too generic. The question could be asked almost anywhere in the video, depends heavily on stale past context, or relies on weak OCR/chat/background clues.
- 1: Critical violation. The question requires future scenes, asks what will happen next, leaks spoilers, is not answerable at the current moment, directly repeats already-resolved past information with no fresh current-scene trigger, or is unrelated to the current viewing context."""

# ── Content-Anchored 질문 전용 Judge ──
_QUERY_JUDGE_PROMPT_CONTENT_ANCHORED = _QUERY_JUDGE_COMMON_INTRO + """

[IMPORTANT EVALUATION PHILOSOPHY: Content-Anchored Questions]
This question is a "Content-Anchored" question — it is EXPECTED to directly engage with the CORE topic, event, or person of the current scene. Evaluate it based on how deeply and precisely it targets the scene's central narrative or subject matter. Do NOT require tangential/external knowledge for this type — instead, reward depth of insight into the scene's main content.

[Evaluation Criteria]

""" + _SCENE_RELEVANCE_CRITERION + """

2. Content Depth (Analytical Depth)
Evaluate the intellectual depth of the question, entirely independent of Criterion 1 (scene relevance). Assume the question IS relevant; how much deeper understanding would answering it provide to the viewer?
- 5: Multi-layered analysis. The question probes underlying mechanisms, motivations, or implications that are NOT obvious from simply watching (e.g., "Why did the character choose THIS specific approach?" reveals psychological depth; "What tactical shift caused the momentum change?" demands expert-level analysis). Answering it would genuinely enrich the viewer's understanding.
- 4: One level deeper than surface. The question pushes beyond what's directly shown or said, but the analytical angle is somewhat predictable for an informed viewer.
- 3: Surface-level factual. The question asks "what/who/when" rather than "why/how." Answerable with a simple fact lookup without requiring deeper analysis or critical thinking.
- 2: States the near-obvious. The question asks about something already largely apparent from watching, adding minimal new insight. A viewer would think "I can already see that."
- 1: Trivially obvious or nonsensical. The answer is self-evident from the screen, or the question is too vague/broad to generate any meaningful information."""

# ── Tangential 질문 전용 Judge ──
_QUERY_JUDGE_PROMPT_TANGENTIAL = _QUERY_JUDGE_COMMON_INTRO + """

[IMPORTANT EVALUATION PHILOSOPHY: Tangential World Knowledge over Narrative]
This question is a "Tangential Knowledge" question — it is EXPECTED to go BEYOND the main narrative to explore interesting background knowledge triggered by visual/audio cues. The Reference Scene Summary (KSS) is primarily focused on the 'narrative/story progression'. However, an excellent tangential question does NOT need to be tied to the main story. We highly value "Tangential World Knowledge" triggered by visual/audio cues.
For example:
- Dramas/Variety: Asking about the historical origin of a prop or filming location.
- Sports: Asking about a player's recent form, transfer history, or team history for newbies.
- Gaming: Asking about character meta, item updates, or strategic nuances.
- News/Docs: Asking about the historical context or economic ripple effects of the topic.
If a question successfully leverages such deep, tangential background knowledge to spark curiosity, it MUST be highly rewarded. Do NOT penalize a question simply because it asks about external knowledge not explicitly written in the KSS.

[Evaluation Criteria]

""" + _SCENE_RELEVANCE_CRITERION + """

2. Curiosity Hook (Viewer Action Drive)
Evaluate whether this question would make a viewer ACTUALLY PICK UP THE REMOTE and press a button to see the answer. This criterion is entirely independent of Criterion 1 (scene relevance). Focus purely on: does the question create an irresistible information gap?
- 5: Irresistible urge to know. The question reveals a surprising premise or counter-intuitive framing (e.g., "you thought X, but actually...") that creates a large information gap. The viewer would feel genuine discomfort NOT knowing the answer. Natural, conversational tone like an expert friend sharing a fascinating insight.
- 4: Genuinely interesting — the viewer would likely press the button, but "could also let it go." The premise is somewhat predictable, or the information gap is moderate rather than compelling.
- 3: Reasonable but forgettable. The viewer thinks "huh, that's a fair question" but feels no urgency to interact. No strong pull toward action.
- 2: The intent is visible but the question is too stiff, academic, or confusingly worded. The viewer has to spend effort just understanding what is being asked, killing any spontaneous curiosity.
- 1: Completely obvious (answer is trivially known) or so mechanically phrased that it generates zero curiosity. No viewer would bother interacting."""

_QUERY_JUDGE_FORMAT_PROMPT_CONTENT_ANCHORED = """[Output Format]
Output ONLY the following JSON. Do NOT output any other text.
Write the rationale first, followed by the score for each criterion.
{
    "scene_relevance": {
        "rationale": "Current Scene Trigger: <Identify the cue and source: speech / visible action / object / OCR / caption / chart / place / etc.>. Scene Specificity: <Explain exact-scene vs general-topic fit>. Immediate Answerability: <Explain whether it can be answered now>. Future/Spoiler Risk: <Explain if future knowledge/spoiler is required or leaked>. Past Redundancy: <Explain if it repeats resolved past information or has a fresh current trigger>. Reference Clarity: <Explain any ambiguity>. Final Judgment: <Concise justification for the score>.",
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
    "scene_relevance": {
        "rationale": "Current Scene Trigger: <Identify the cue and source: speech / visible action / object / OCR / caption / chart / place / etc.>. Scene Specificity: <Explain exact-scene vs general-topic fit>. Immediate Answerability: <Explain whether it can be answered now>. Future/Spoiler Risk: <Explain if future knowledge/spoiler is required or leaked>. Past Redundancy: <Explain if it repeats resolved past information or has a fresh current trigger>. Reference Clarity: <Explain any ambiguity>. Final Judgment: <Concise justification for the score>.",
        "score": <integer 1-5>
    },
    "curiosity_hook": {
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
    "content_anchored": ["scene_relevance", "content_depth"],
    "tangential": ["scene_relevance", "curiosity_hook"],
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
                client, args.interactive_query_judge_model, judge_config,
                detailed_summary, query_text, query_type=query_type
            ),
            label=f"Interactive Query Judge (Scene {scene_idx}, {mode}, {query_type})",
        )

        score_keys = _SCORE_KEYS_BY_TYPE.get(query_type, ["scene_relevance", "curiosity_hook"])
        raw_total = sum(
            (score_dict.get(k, {}).get("score", 0) if isinstance(score_dict.get(k), dict) else 0)
            for k in score_keys
        ) if score_dict else 0
        scene_relevance_score = (
            score_dict.get("scene_relevance", {}).get("score", 0)
            if score_dict and isinstance(score_dict.get("scene_relevance"), dict)
            else 0
        )
        gate_applied = scene_relevance_score <= 2
        total = scene_relevance_score if gate_applied else raw_total

        from collections import OrderedDict
        score_record = OrderedDict([
            ("content_id", content_id),
            ("scene_idx", scene_idx),
            ("mode", mode),
            ("query_type", query_type),
            ("query", query_text),
            ("judge", score_dict),
            ("gate_applied", gate_applied),
            ("total_score", total),
        ])

        _ITEM_LABELS_BY_TYPE = {
            "content_anchored": [
                ("scene_relevance", "씬 적절성"),
                ("content_depth", "콘텐츠 핵심 깊이"),
            ],
            "tangential": [
                ("scene_relevance", "씬 적절성"),
                ("curiosity_hook", "호기심 유도력"),
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
            gate_note = " | Gate: scene_relevance<=2 -> secondary score ignored" if gate_applied else ""
            out_str = f"[{type_tag}] Query: {query_text}\nScore: {total}/10 | {score_details}{gate_note}"
            
        append_jsonl(args.scores_file, score_record, lock=file_write_lock)
        return {"mode": mode, "query_type": query_type, "query": query_text, "success": True, "out_str": out_str, "total": total}

    except Exception as e:
        return {"mode": mode, "query_type": query_type, "query": query_text, "success": False, "msg": f"[Error] Judge 최종 실패 ({mode}, {query_type}): {e}"}

def is_content_id_fully_evaluated_interactive_query(content_id, input_file, target_modes_set, summary_map, scores_file):
    """content_id에 해당하는 모든 target_mode의 질문들이 scores_file에 정상적으로 평가 완료되었는지 확인합니다."""
    expected_scenes = {s_idx for (c_id, s_idx) in summary_map.keys() if c_id == content_id}
    if not expected_scenes:
        return False

    # 해당 content_id에 대해 입력 파일에 존재하는 실제 모드들을 추출
    existing_modes = set()
    for scene_item in load_jsonl(input_file):
        if scene_item.get("content_id") == content_id:
            m = scene_item.get("mode")
            if m:
                existing_modes.add(m)
                
    active_modes = target_modes_set & existing_modes
    if not active_modes:
        return False

    actual_records = {}
    for scene_item in load_jsonl(input_file):
        if scene_item.get("pipeline_done"):
            continue
        c_id = scene_item.get("content_id")
        if c_id != content_id:
            continue
        s_idx = scene_item.get("scene_idx")
        mode = scene_item.get("mode")
        if mode not in active_modes:
            continue
        if s_idx not in expected_scenes:
            continue
        queries_list = scene_item.get("queries", [])
        if queries_list:
            actual_records[(s_idx, mode)] = queries_list

    processed_pairs = load_processed_pairs(scores_file, key_fields=("content_id", "mode", "query"))
    
    for s_idx in expected_scenes:
        for mode in active_modes:
            key = (s_idx, mode)
            if key not in actual_records:
                return False
            queries = actual_records[key]
            for q_text in queries:
                if (content_id, mode, q_text) not in processed_pairs:
                    return False
                    
    return True



def main():
    parser = get_common_argparser(description="Interactive Query 질문을 KeyScene Summary 기반으로 품질 평가")
    parser.add_argument("--input_file", default="assets/interactive_queries.jsonl", help="Interactive Query 질문 목록 JSONL 경로")
    parser.add_argument("--kss_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--scores_file", default="assets/interactive_query_scores.jsonl", help="Interactive Query 질문별 Judge 점수 저장 경로")
    parser.add_argument("--modes", nargs="+", default=["video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph", "meta"], 
    choices=["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph", "meta"], help="평가할 모드 직접 지정")
    

    args, client = init_pipeline(parser.parse_args())
    content_indices = load_content_indices()
    judge_config = make_query_judge_config(thinking_level=args.interactive_query_judge_thinking_level)

    ensure_output_dir(args.scores_file)

    if not check_input_file(args.input_file, hint="먼저 generate_interactive_query.py를 실행하세요."):
        return

    # Summary 맵 로드: (content_id, scene_idx) -> summary_text
    summary_map = load_summary_map(args.kss_file)
    if summary_map:
        print(f"[Summary] {len(summary_map)}개 Scene의 Summary 로드됨 ({args.kss_file})")
    elif not os.path.exists(args.kss_file):
        print(f"[Warning] Summary 파일을 찾을 수 없습니다: {args.kss_file}")

    print_pipeline_banner("Interactive Query 질문 품질 평가 프로세스 시작")

    file_write_lock = threading.Lock()
    target_modes_set = set(args.modes)
    target_mode_order = ["video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_graph", "meta"]
    
    printed_content_ids = set()
    _INTERACTIVE_QUERY_SCORE_KEYS = ["scene_relevance", "content_depth", "curiosity_hook"]

    def check_and_print_summaries():
        all_content_ids = set()
        for scene_item in load_jsonl(args.input_file):
            c_id = scene_item.get("content_id")
            if c_id:
                all_content_ids.add(c_id)
        for c_id in sorted(all_content_ids, key=lambda c: content_indices.get(c, 999)):
            if c_id not in printed_content_ids:
                if is_content_id_fully_evaluated_interactive_query(c_id, args.input_file, target_modes_set, summary_map, args.scores_file):
                    printed_content_ids.add(c_id)
                    print_scores_summary(args.scores_file, c_id, _INTERACTIVE_QUERY_SCORE_KEYS, target_mode_order, max_score=10)

    try:
        discovery_pass = 0
        while True:
            discovery_pass += 1
            check_and_print_summaries()

            # 매 pass마다 입력 파일을 다시 읽어 새로 추가된 Interactive Query 레코드를 감지
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
            tracker = ProgressTracker(total_pending, unit="queries", action="evaluated")
            for (c_id, s_idx), items in accumulated_groups.items():
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

                check_and_print_summaries()
                tracker.update(pass_evaluated)
                
            check_and_print_summaries()
            print(f"\n▶ [Discovery {discovery_pass}] {pass_evaluated}개 평가 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    sort_jsonl_file(args.scores_file)
    print_pipeline_done(args.scores_file)

if __name__ == "__main__":
    main()
