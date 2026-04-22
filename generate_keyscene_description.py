import os
import time
import json
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_file_by_scene_idx, check_gcs_files_exist,
    _retry_api_call,
    ensure_output_dir, load_processed_pairs,
    preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
    sort_and_validate_jsonl,
    load_keypoints_by_content, check_input_file,
    print_pipeline_banner, print_pipeline_done,
)

# ───────────────────────────────────────────────
# Prompt: Video-based Description (영어)
# ───────────────────────────────────────────────

_VIDEO_DESC_SYSTEM_PROMPT = """You are an expert video content analyst. Your task is to generate a detailed, accurate description of the provided video clip.

Describe the following aspects in a single cohesive paragraph or structured text:
- **Visual Elements**: Scene composition, camera angles, environment/background, lighting
- **Characters & Actions**: Who appears, what they are doing, any notable behaviors or interactions
- **On-screen Text**: Any subtitles, captions, logos, or text overlays visible
- **Audio Cues** (if inferable from visuals): Any apparent dialogue, narration, or sound context

Guidelines:
- Be precise and objective. Avoid speculation beyond what is clearly visible.
- Include specific details: colors, quantities, spatial relationships, identifiable objects.
- Do NOT summarize vaguely. Capture as much detail as possible."""

# ───────────────────────────────────────────────
# Prompt: Ref JSON-based Description (영어)
# ───────────────────────────────────────────────

_REF_DESC_SYSTEM_PROMPT = """You are an expert video content analyst. Your task is to generate a detailed scene description based solely on the provided metadata.

You will receive a JSON record containing:
- **speech**: Dialogue or narration transcribed from the audio (may contain ASR errors)
- **texts**: On-screen text, subtitles, or OCR-extracted captions (may contain OCR errors)

Your task:
1. Correct obvious transcription/OCR errors using context clues.
2. Based on the speech and texts, describe the scene in as much detail as possible:
   - What is being said or narrated, and by whom (if identifiable)
   - What the on-screen text indicates about the scene context
   - What can be inferred about the visual setting from the dialogue/narration

Guidelines:
- Acknowledge that you are working from transcripts only (no video).
- Be as specific as possible about names, places, objects mentioned in the speech/texts.
- Do NOT fabricate visual details not implied by the text."""


def make_kd_gen_config(thinking_level=None):
    """KeyScene Description 생성용 GenerateContentConfig를 반환합니다."""
    return {
        "video": make_generate_config(system_instruction=_VIDEO_DESC_SYSTEM_PROMPT, thinking_level=thinking_level),
        "ref":   make_generate_config(system_instruction=_REF_DESC_SYSTEM_PROMPT,   thinking_level=thinking_level),
    }


def generate_video_desc(client, model_name, config, video_part, end_time):
    """비디오 클립만 보고 장면을 영어로 묘사합니다."""
    contents = [
        "--- [Current Video Clip] ---",
        video_part,
        "--- Request ---",
        "Watch the provided video clip carefully and generate a detailed English description of the scene. "
        "Cover visual elements, characters' actions, on-screen text, and any inferable audio context.",
    ]
    t0 = time.time()
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=config
        ).text,
        label=f"KD video_desc API (end={end_time:.1f}s)"
    )
    return text, time.time() - t0


def generate_ref_desc(client, model_name, config, ref_part, end_time):
    """Ref JSONL(speech + texts)만 보고 장면을 영어로 묘사합니다."""
    contents = [
        "--- [Current Scene Metadata (speech & texts)] ---",
        ref_part,
        "--- Request ---",
        "Based solely on the provided speech transcript and on-screen text metadata above, "
        "generate a detailed English description of what is happening in this scene. "
        "Correct any obvious ASR/OCR errors using context. Do not fabricate visual details beyond what the text implies.",
    ]
    t0 = time.time()
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=config
        ).text,
        label=f"KD ref_desc API (end={end_time:.1f}s)"
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
    parser.add_argument("--modes", nargs="+", default=["video_desc", "ref_desc"],
                        choices=["video_desc", "ref_desc"],
                        help="생성할 모드 직접 지정 (기본값: 모두 생성)")

    args, client = init_pipeline(parser.parse_args())

    gen_configs = make_kd_gen_config(thinking_level=args.kd_gen_thinking_level)

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

    target_modes = args.modes

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

                print(f"\n[{real_idx}/{len(keypoints)}] Scene {scene_idx} | "
                      f"Range=[{start_time:.1f}s ~ {end_time:.1f}s] | Modes={missing_modes}")

                # GCS Part 사전 로드
                video_part = None
                ref_part   = None
                if "video_desc" in missing_modes:
                    video_part = process_gcs_file_by_scene_idx(
                        args.gs_bucket_name, content_id, "video", scene_idx, scene_idx
                    )
                if "ref_desc" in missing_modes:
                    ref_part = process_gcs_file_by_scene_idx(
                        args.gs_bucket_name, content_id, "ref", scene_idx, scene_idx
                    )

                def _run_mode(mode):
                    if mode == "video_desc":
                        return generate_video_desc(
                            client, args.kd_gen_model, gen_configs["video"], video_part, end_time
                        )
                    else:  # ref_desc
                        return generate_ref_desc(
                            client, args.kd_gen_model, gen_configs["ref"], ref_part, end_time
                        )

                try:
                    # 2개 모드 병렬 생성
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        futures = {m: executor.submit(_run_mode, m) for m in missing_modes}

                        for mode in missing_modes:
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

                    print("--------------------------------------------------------------------------------")

                except Exception as e:
                    print(f"  [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                    continue

            done_count = len({s for (c, s, _) in done_modes_by_scene if c == content_id})
            print(f"\n[OK] '{content_id}' - {done_count}개 Scene 처리 확인됨")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        import os as _os
        _os.exit(1)

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
