import os
import time
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    check_gcs_files_exist,
    _retry_api_call,
    ensure_output_dir,
    preload_content_metadata,
    build_mode_parts,
    init_pipeline, load_jsonl, append_jsonl,
    sort_and_validate_jsonl,
    load_keypoints_by_content, check_input_file,
    print_pipeline_banner, print_pipeline_done,
)

# ───────────────────────────────────────────────
# Prompts
# ───────────────────────────────────────────────

_KSD_PROMPT_VIDEO = """You are an expert video content analyst. Your task is to generate a detailed, accurate English description of the provided video clip.

Your description must cover the following aspects, in this order:

1. **Scene & Visual Elements**
   - Setting, environment, lighting, camera angles, and scene composition.
   - Characters present: appearance, clothing, positioning, and spatial relationships.
   - Key actions and interactions between characters or with objects.
   - Any on-screen text: subtitles, captions, logos, lower-thirds, or text overlays.

2. **Dialogue & Factual Details**
   - Listen carefully to the audio. Transcribe or closely paraphrase all dialogue and narration.
   - Capture every proper noun: person names, place names, brand names, program titles, and any specific terms mentioned.
   - Record specific numbers, dates, or quantities if mentioned.
   - If a name is spoken but unclear, provide your best interpretation rather than omitting it.

3. **Narrative & Emotional Context**
   - Describe the narrative progression: what is happening and why.
   - Note cause-effect relationships between events in the scene.
   - Capture the emotional tone and atmosphere (e.g., tense, celebratory, melancholic).
   - Identify any thematic elements or recurring motifs.

Guidelines:
- Be precise and comprehensive. Vague summaries are unacceptable.
- Include specific details: colors, quantities, spatial relationships, identifiable objects.
- Actively use BOTH visual AND audio information from the video.
- Do NOT omit dialogue or proper nouns — these are critical.
- If you recognize a person, place, or brand from the video, use their correct known name even if not explicitly stated on screen.
- Focus on describing what is observable in the video. Do NOT add speculative interpretations or external context beyond proper identification."""

_KSD_PROMPT_RAW = """You are an expert video content analyst. Your task is to generate a detailed English scene description based on the provided audio and on-screen text metadata.

You will receive a JSON record containing:
- **speech**: ASR transcription of spoken dialogue or narration.
- **texts**: OCR detection of text appearing on screen.

Your description must cover the following aspects, in this order:

1. **Scene & Setting Inference**
   - Based on the dialogue content and on-screen text, infer what kind of scene this is (interview, cooking segment, outdoor exploration, etc.).
   - Describe the likely setting and context as implied by the audio and text clues.
   - Do NOT fabricate specific visual details (colors, camera angles, etc.) that cannot be inferred from the text.

2. **Dialogue & Factual Details**
   - Incorporate the dialogue or narration seamlessly into the description.
   - Capture every proper noun: person names, place names, brand names, and specific terms you can identify.
   - Record specific numbers, dates, or quantities if identifiable.
   - If a name appears garbled, provide your best corrected interpretation rather than omitting it.

3. **Narrative & Emotional Context**
   - Describe the narrative progression as inferred from the dialogue.
   - Note cause-effect relationships between events discussed.
   - Capture the emotional tone implied by the dialogue.
   - Identify any thematic elements or recurring topics.

Guidelines:
- Be precise and comprehensive. Vague summaries are unacceptable.
- Prioritize factual accuracy — proper nouns and dialogue content are critical.
- Write the description naturally, as if you had full knowledge of the scene.
- Do NOT mention that you are working from transcripts or metadata.
- If contextual clues allow you to identify a specific person, place, or brand, use their correct known name.
- Focus on describing the scene as conveyed through the provided data. Do NOT add speculative context beyond proper identification."""

_KSD_PROMPT_FRAG = """You are an expert video content analyst. Your task is to generate a detailed English scene description based on the provided fragmented speech and on-screen text metadata.

You will receive a JSON record containing:
- **speech_fragments**: Alphabetically sorted word fragments extracted from ASR transcription. The original sentence order has been intentionally destroyed for copyright protection.
- **text_fragments**: Alphabetically sorted word fragments extracted from OCR detection. The original order has been destroyed.

IMPORTANT: The fragments are intentionally shuffled into a Bag-of-Words format. You must reconstruct the likely meaning by analyzing the word set as a whole, using contextual clues and common sense to infer the original sentences.

Your description must cover the following aspects, in this order:

1. **Scene & Setting Inference**
   - Based on the reconstructed dialogue content and on-screen text, infer what kind of scene this is (interview, cooking segment, outdoor exploration, etc.).
   - Describe the likely setting and context as implied by the speech fragments and text clues.
   - Do NOT fabricate specific visual details (colors, camera angles, etc.) that cannot be inferred from the text.

2. **Dialogue & Factual Details**
   - Reconstruct the likely dialogue or narration from the shuffled speech fragments using context and common sense.
   - Capture every proper noun: person names, place names, brand names, and specific terms you can identify from the fragments.
   - Record specific numbers, dates, or quantities if identifiable.
   - If a name appears fragmented or garbled, provide your best corrected interpretation rather than omitting it.

3. **Narrative & Emotional Context**
   - Describe the narrative progression as inferred from the reconstructed dialogue.
   - Note cause-effect relationships between events discussed in the speech.
   - Capture the emotional tone implied by the dialogue (e.g., excitement, concern, humor).
   - Identify any thematic elements or recurring topics.

Guidelines:
- Be precise and comprehensive. Vague summaries are unacceptable.
- Prioritize factual accuracy — proper nouns and dialogue content are critical.
- Write the description naturally, as if you had full knowledge of the scene.
- Do NOT mention that you are working from fragments, transcripts, or metadata.
- If contextual clues allow you to identify a specific person, place, or brand, use their correct known name.
- Focus on describing the scene as conveyed through the provided data. Do NOT add speculative context beyond proper identification."""

_KSD_PROMPT_FRAG_WITH_VLM = """You are an expert video content analyst. Your task is to generate a detailed English scene description based on the provided VLM metadata and fragmented speech/on-screen text.

You will receive a JSON record containing:
- **vlm_mm_structure**: Visual metadata containing Subject, Environment, Actions, and alphabetically sorted Context keywords.
- **timeline**: A timeline of shots, where each shot includes:
  - **speech_fragments**: Alphabetically sorted word fragments extracted from ASR. Original order destroyed.
  - **text_fragments**: Alphabetically sorted word fragments extracted from OCR. Original order destroyed.

IMPORTANT: The speech/text fragments and Context keywords are intentionally shuffled. You must reconstruct the likely meaning by analyzing the word set as a whole, using contextual clues and common sense.

Your description must cover the following aspects, in this order:

1. **Scene & Visual Elements**
   - Describe the setting, environment, characters, and key actions using the provided vlm_mm_structure.
   - Expand on the basic VLM output to create a coherent narrative of what is happening visually.

2. **Dialogue & Factual Details**
   - Reconstruct the likely dialogue or narration from the shuffled speech fragments.
   - Capture every proper noun, specific numbers, and key terms you can identify.
   - Merge the visual context with the inferred dialogue to create a comprehensive picture.

3. **Narrative & Emotional Context**
   - Describe the narrative progression combining visual actions and reconstructed dialogue.
   - Capture the emotional tone implied by both the visual scene and the spoken words.

Guidelines:
- Be precise and comprehensive. Vague summaries are unacceptable.
- Write the description naturally, as if you had full knowledge of the scene.
- Do NOT mention that you are working from fragments, transcripts, or VLM structures.
- Focus on describing the scene as conveyed through the provided data. Do NOT add speculative context beyond proper identification."""


def make_ksd_gen_config(thinking_level=None):
    """KeyScene Description 생성용 GenerateContentConfig를 반환합니다."""
    return {
        "video":         make_generate_config(system_instruction=_KSD_PROMPT_VIDEO,         thinking_level=thinking_level),
        "raw":           make_generate_config(system_instruction=_KSD_PROMPT_RAW,           thinking_level=thinking_level),
        "frag":          make_generate_config(system_instruction=_KSD_PROMPT_FRAG,          thinking_level=thinking_level),
        "frag_with_vlm": make_generate_config(system_instruction=_KSD_PROMPT_FRAG_WITH_VLM, thinking_level=thinking_level),
    }


def generate_ksd_mode(client, model_name, config, mode, data_part, end_time):
    """지정된 모드와 데이터로 KeyScene Description을 생성합니다."""
    if mode == "video":
        contents = [
            "--- [Current Video Clip] ---",
            data_part,
            "--- Request ---",
            "Watch the provided video clip carefully and generate a detailed English description of the scene. "
            "Cover visual elements, characters' actions, on-screen text, and any inferable audio context.",
        ]
    elif mode == "raw":
        contents = [
            "--- [Current Scene Metadata (Intact ASR & OCR)] ---",
            data_part,
            "--- Request ---",
            "Based solely on the provided audio and on-screen text metadata above, "
            "infer the likely scene context and generate a detailed English description "
            "of what is happening in this scene. "
            "Do not fabricate visual details beyond what the text implies.",
        ]
    elif mode == "frag":
        contents = [
            "--- [Current Scene Metadata (speech_fragments & text_fragments)] ---",
            data_part,
            "--- Request ---",
            "Based solely on the provided speech fragments and text fragments metadata above, "
            "reconstruct the likely dialogue and scene context, then generate a detailed English description "
            "of what is happening in this scene. The fragments are alphabetically sorted and their original "
            "order has been destroyed — use context and common sense to reconstruct meaning. "
            "Do not fabricate visual details beyond what the text implies.",
        ]
    elif mode == "frag_with_vlm":
        contents = [
            "--- [Current Scene Metadata (VLM Structure & Fragments)] ---",
            data_part,
            "--- Request ---",
            "Based on the provided VLM metadata and the shuffled speech/text fragments, "
            "reconstruct the dialogue and integrate it with the visual description to generate "
            "a comprehensive English description of the scene.",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    t0 = time.time()
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=config
        ).text,
        label=f"KSD {mode} API (end={end_time:.1f}s)"
    )
    return text, time.time() - t0


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint Scene 목록을 입력받아 KeyScene Description을 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl",
                        help="Keypoint Scene 목록 JSONL 경로 (identify_keyscene.py 출력)")
    parser.add_argument("--output_file", default="assets/keyscene_description.jsonl",
                        help="KeyScene Description 저장 경로")
    parser.add_argument("--modes", nargs="+", default=["video", "raw", "frag", "frag_with_vlm"],
                        choices=["video", "raw", "frag", "frag_with_vlm"],
                        help="생성할 모드 직접 지정")

    args, client = init_pipeline(parser.parse_args())

    gen_configs = make_ksd_gen_config(thinking_level=args.ksd_gen_thinking_level)

    if not check_input_file(args.input_file, hint="먼저 identify_keyscene.py를 실행하세요."):
        return

    # Keypoint 목록 로드
    keypoints_by_content = load_keypoints_by_content(args.input_file)
    if not keypoints_by_content:
        print(f"Error: {args.input_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    # 기처리분 로드: (content_id, scene_idx, mode) 단위 추적
    done_modes_by_scene = set()
    for rec in load_jsonl(args.output_file):
        c_id  = rec.get("content_id")
        s_idx = rec.get("scene_idx")
        mode  = rec.get("mode")
        if c_id and s_idx is not None and mode:
            done_modes_by_scene.add((c_id, s_idx, mode))

    # 정규 모드 순서: JSONL에 쓰는 순서를 보장합니다.
    _MODE_ORDER = ["video", "raw", "frag", "frag_with_vlm"]
    target_modes = sorted(args.modes, key=lambda m: _MODE_ORDER.index(m) if m in _MODE_ORDER else 99)

    print_pipeline_banner("KeyScene Description 생성 파이프라인을 시작합니다.")

    file_lock = threading.Lock()

    try:
        for content_id, keypoints in keypoints_by_content.items():

            # 각 Keypoint별 누락 모드 추적
            remaining = []
            fully_done_count = 0

            for kp in keypoints:
                real_idx   = keypoints.index(kp)
                s_idx      = kp.get("scene_idx", real_idx)
                missing    = [m for m in target_modes if (content_id, s_idx, m) not in done_modes_by_scene]
                if missing:
                    remaining.append((kp, missing))
                else:
                    fully_done_count += 1

            if not remaining:
                print(f"\n[Skip] '{content_id}': 모든 Scene 완료 (누락 모드 없음)")
                continue

            if fully_done_count > 0:
                print(f"\n[Resume] '{content_id}': {fully_done_count}/{len(keypoints)}개 Scene 완료, {len(remaining)}개 재개")

            print(f"\n{'='*50}")
            print(f"Processing Content: '{content_id}' ({len(remaining)}/{len(keypoints)}개 Keypoint 처리)")
            print(f"{'='*50}")

            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue

            # JSONL 메타데이터 프리로드
            preload_content_metadata(args.gs_bucket_name, content_id)

            for kp, missing_modes in remaining:
                real_idx  = keypoints.index(kp)
                scene_idx = kp.get("scene_idx", real_idx)
                start_time = float(kp.get("start_time", 0.0))
                end_time   = float(kp.get("end_time",   0.0))

                print(f"[{real_idx}/{len(keypoints)}] Scene {scene_idx} | "
                      f"Range=[{start_time:.1f}s ~ {end_time:.1f}s] | Modes={missing_modes}")

                # 모드별 데이터 사전 로드
                _, data_parts = build_mode_parts(
                    args.gs_bucket_name, content_id, missing_modes,
                    scene_idx, scene_idx,
                )

                def _run_mode(mode):
                    return generate_ksd_mode(
                        client, args.ksd_gen_model, gen_configs[mode], mode, data_parts[mode], end_time
                    )

                try:
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(missing_modes))
                    try:
                        futures = {}
                        for mode in missing_modes:
                            futures[mode] = executor.submit(_run_mode, mode)

                        # 정규 순서(video → raw → frag → frag_with_vlm)로 결과를 수집하여 쓰기
                        for mode in missing_modes:  # 이미 _MODE_ORDER로 정렬된 상태
                            desc_text, elapsed = futures[mode].result()

                            record = {
                                "content_id": content_id,
                                "scene_idx":  scene_idx,
                                "mode":       mode,
                                "start_time": start_time,
                                "end_time":   end_time,
                                "description": desc_text,
                            }
                            with file_lock:
                                append_jsonl(args.output_file, record)
                                done_modes_by_scene.add((content_id, scene_idx, mode))

                            print(f"  -> [{mode}] 완료 (~{len(desc_text.split())}단어, {elapsed:.2f}초)")
                    finally:
                        executor.shutdown(wait=False, cancel_futures=True)

                    print("--------------------------------------------------------------------------------")

                except Exception as e:
                    print(f"  [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                    continue

            done_count = len({s for (c, s, _) in done_modes_by_scene if c == content_id})
            print(f"\n[OK] '{content_id}' - {done_count}개 Scene 처리 확인됨")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    # 결과 파일 정렬 및 누락 점검
    sort_and_validate_jsonl(
        args.output_file,
        keypoints_by_content,
        expected_modes=target_modes,
        mode_key="description"
    )

    # 완료 시그널 기록
    try:
        append_jsonl(args.output_file, {"pipeline_done": True})
        print(f"\n[Signal] Pipeline 종료 시그널을 기록했습니다.")
    except Exception as e:
        print(f"\n[Warning] Pipeline 완료 시그널 기록 실패: {e}")

    print_pipeline_done(args.output_file)


if __name__ == "__main__":
    main()
