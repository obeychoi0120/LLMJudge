import os
import time
import argparse
import json
from gemini_api_utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_file_range, check_gcs_files_exist,
    _retry_api_call,
    ensure_output_dir, load_processed_pairs,
    preload_content_metadata, get_gcs_text_range, download_gcs_text, truncate_jsonl_range,
    init_pipeline, load_jsonl, append_jsonl,
)

# ───────────────────────────────────────────────
# KeyScene Summary 생성 모델 프롬프트
# ───────────────────────────────────────────────

_SUMMARY_GEN_PROMPT = """\
당신은 영상 콘텐츠의 맥락을 완벽히 이해하고 대본 및 상황을 파악하는 전문가입니다.
당신에게는 이전 씬의 참조용 메타데이터(Past Reference Metadata) 및 이전 요약 텍스트(Past Summary)와, 
현재 씬의 참조용 메타데이터(Current Reference Metadata) 및 비디오(Current Video)가 제공됩니다.
당신의 목표는 이 정보들을 바탕으로 스토리가 어떻게 이어지고 있는지 파악한 후, 아래 출력 포맷에 맞추어 명확하게 구분하여 작성하는 것입니다.

[참조용 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- sounds: 환경음 및 효과음
- texts: 화면 속 자막, 간판 정보 등
- speech: 등장인물들의 대사 (영어 또는 한국어)

[참조용 메타데이터 사용 시 주의사항]
- speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다. 따라서 당신이 적절히 비디오와 교차 검증하여 교정해야 합니다.
- sounds: 효과음 분류 오류가 빈번합니다. 반드시 비디오에서 본 정보를 우선시하고, 비디오에 존재하지 않는 효과음이 있다면 무시하세요.
- texts: OCR 오류로 인해 화면 텍스트의 철자가 틀릴 수 있습니다. 비디오와 적절히 교차 검증하여 교정하세요.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다. 비디오와 적절히 교차 검증하여 교정하세요.

[작성 규칙]
- 아래 출력 포맷에 맞게 두 파트로 나누어 작성하세요.
- 과거 장면([Past Reference Metadata], [Past Summary])이 제공되지 않은 경우, "1. 과거 장면 요약" 항목은 "해당 없음"으로 표기하세요.
- 언어는 **한국어**로 작성하되, 인물 이름과 고유명사는 원어를 병기하세요 (예: 일각고래(narwhal), 셰즈 은데예(Chez Ndeye)).
- 각각의 파트는 텍스트의 길이 제한 없이 최대한 핵심을 상세하게 묘사하세요.

[출력 포맷]
**1. 과거 장면 요약**
(지금까지 발생한 주요 사건의 흐름, 등장인물 간의 대화와 관계, 인물의 목적이나 동기, 나오는 물체 등을 텍스트의 길이 제한 없이 시간순으로 상세하게 요약합니다.)

**2. 현재 장면**
(현재 장면에서 벌어지고 있는 구체적인 상황, 인물의 행동, 대화 내용, 감정 변화, 갈등 요소, 나오는 물체 등을 텍스트의 길이 제한 없이 상세하게 묘사합니다.)"""


def make_summary_config(thinking_budget=None):
    """Summary 생성용 GenerateContentConfig를 반환합니다."""
    return make_generate_config(system_instruction=_SUMMARY_GEN_PROMPT, thinking_budget=thinking_budget)


def process_summary(client, summary_model_name, summary_config, past_summary_text, past_ref_metadata, current_parts, end_time, use_ref=False):
    contents = []
    if past_summary_text is not None:
        if past_ref_metadata is not None:
            contents += ["--- [Past Reference Metadata] ---", past_ref_metadata]
        contents += ["--- [Past Summary] ---", past_summary_text]
    if use_ref:
        contents += ["--- [Current Reference Metadata] ---", current_parts["ref"]]
    contents += ["--- [Current Video] ---", current_parts["video"]]
    request_msg = (
        "[Past Reference Metadata]와 [Past Summary] 내용을 모두 종합하여 '1. 과거 장면 요약' 섹션을 작성하고, "
        "제공된 [Current Reference Metadata]의 힌트를 참고하여 시각적으로 묘사된 [Current Video] 영상을 종합하여 '2. 현재 장면' 섹션을 새롭게 작성하세요. "
        "반드시 1번과 2번 항목을 분리하여 포맷을 준수해 주세요."
        if past_summary_text is not None else
        "과거 정보가 없으므로 '1. 과거 장면 요약'은 '해당 없음'으로 기재하고, "
        "제공된 [Current Reference Metadata]와 [Current Video] 영상을 바탕으로 '2. 현재 장면' 섹션을 상세하게 작성해 주세요."
    )
    contents += ["--- 요청 사항 ---", request_msg]
    t0 = time.time()
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=summary_model_name, contents=contents, config=summary_config
        ).text,
        label=f"Summary 생성 (end={end_time:.1f}s)"
    )
    return text, time.time() - t0

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint Scene 목록을 입력받아 KeyScene Summary를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary 별도 저장 경로")

    args, client = init_pipeline(parser.parse_args())

    summary_config = make_summary_config(thinking_budget=args.keyscene_summary_thinking_budget)

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 identify_keypoint.py를 실행하세요.")
        return

    # Keypoint 목록 로드
    keypoints_by_content = {}
    for data in load_jsonl(args.input_file):
        c_id = data.get("content_id")
        kps = data.get("keypoints", [])
        if c_id and kps:
            keypoints_by_content[c_id] = kps

    if not keypoints_by_content:
        print(f"Error: {args.input_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

    # 출력 디렉토리 확인
    ensure_output_dir(args.keyscene_summary_file)

    summary_pairs = load_processed_pairs(args.keyscene_summary_file, key_fields=("content_id", "scene_idx"))

    # 생성된 Summary 텍스트를 캐싱하여 다음 Scene의 과거 요약 정보로 활용
    summary_texts_by_scene = {}
    if os.path.exists(args.keyscene_summary_file):
        for data in load_jsonl(args.keyscene_summary_file):
            c_id = data.get("content_id")
            s_idx = data.get("scene_idx")
            if c_id and s_idx is not None:
                summary_texts_by_scene[(c_id, s_idx)] = data

    print("\n" + "="*50)
    print("KeyScene Summary 생성 파이프라인을 시작합니다.")
    print("="*50)

    try:
        for content_id, keypoints in keypoints_by_content.items():
            done_scenes = {s_idx for (c_id, s_idx) in summary_pairs if c_id == content_id}
            remaining = [kp for kp in keypoints if kp.get("scene_idx") not in done_scenes]

            if not remaining:
                print(f"\n[Skip] '{content_id}': 모든 Scene 완료")
                continue
            if done_scenes:
                print(f"\n[Resume] '{content_id}': {len(done_scenes)}/{len(keypoints)}개 Scene 기완료, {len(remaining)}개 재개")

            print(f"\n{'='*50}")
            print(f"Processing Content: '{content_id}' ({len(remaining)}/{len(keypoints)}개 Keypoint)")
            print(f"{'='*50}")

            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue

            # JSONL 메타데이터 프리로드 (캐시 워밍업)
            preload_content_metadata(args.gs_bucket_name, content_id)

            for kp in remaining:
                real_idx = keypoints.index(kp)
                scene_idx = kp.get("scene_idx", real_idx)
                start_time = float(kp.get("start_time", 0.0))
                end_time = float(kp.get("end_time", 0.0))

                print(f"[{real_idx+1}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s]")
                
                def _run_keypoint():
                    # 직전 완료된 scene 탐색 (scene_idx보다 작은 것 중에서 최댓값)
                    past_scene_indices = [s for (c, s) in summary_pairs if c == content_id and s < scene_idx]
                    past_summary_text = None
                    past_ref_metadata = None
                    if past_scene_indices:
                        last_scene_idx = max(past_scene_indices)
                        past_record = summary_texts_by_scene.get((content_id, last_scene_idx))
                        if past_record:
                            past_summary_text = past_record.get("summary")
                            if args.use_ref_for_keyscene_summary:
                                p_start = float(past_record.get("start_time", 0.0))
                                p_end = float(past_record.get("end_time", 0.0))
                                past_ref_metadata = process_gcs_file_range(args.gs_bucket_name, content_id, "ref", p_start, p_end)

                    current_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", start_time, end_time),
                        "ref":   process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   start_time, end_time),
                    }
                    return process_summary(
                        client, args.keyscene_summary_model, summary_config,
                        past_summary_text, past_ref_metadata, current_parts, end_time, use_ref=args.use_ref_for_keyscene_summary
                    )

                try:
                    summary_text, summary_elapsed = _retry_api_call(
                        _run_keypoint,
                        label=f"KeyScene Summary (Scene {scene_idx})"
                    )

                    scene_key = (content_id, scene_idx)
                    if scene_key not in summary_pairs:
                        summary_record = {
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "summary": summary_text,
                        }
                        append_jsonl(args.keyscene_summary_file, summary_record)
                        summary_pairs.add(scene_key)
                        summary_texts_by_scene[scene_key] = summary_record

                    print(f"\n{summary_text}")
                    print(f"\n-> [Summary] 생성 완료 ({len(summary_text)}자, {summary_elapsed:.2f}초)")
                    print(f"------------------------------------------------------\n")

                except Exception as e:
                    print(f"    [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                    continue

            done_count = len({s_idx for (c_id, s_idx) in summary_pairs if c_id == content_id})
            print(f"\n[OK] '{content_id}' - {done_count}개 Scene 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    print("\n" + "="*50)
    print(f"모든 작업이 완료되었습니다. 저장 위치: {args.keyscene_summary_file}")
    print("="*50)


if __name__ == "__main__":
    main()
