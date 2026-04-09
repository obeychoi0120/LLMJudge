import os
import time
import argparse
import json
from gemini_api_utils import (
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
당신은 영상 콘텐츠의 맥락을 완벽히 이해하고 이를 상세하게 요약하는 전문가입니다.
사용자는 원본 비디오 프레임과 Reference 메타데이터(JSONL)를 함께 제공합니다.
시청자는 **과거 정보 (Past Information)** 와 **현재 정보 (Current Information)** 를 모두 시청했습니다.
이를 종합하여, 지금까지 발생한 중요한 사건, 대화, 인물의 목적, 그리고 현재 장면에서 벌어지고 있는 갈등이나 구체적인 상황 등을 포괄적으로 묘사하는 하나의 상세한 요약(Detailed Summary)을 작성하세요.
이 요약 문단은 파이프라인의 후속 단계에서 질문의 적절성 여부를 평가하기 위한 강력한 Reference로 사용됩니다.
단 하나의 문자열(일반 텍스트)로 출력하되, 텍스트의 길이나 형식 제한 없이 최대한 핵심을 상세하게 묘사하세요.

[Reference 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- sounds: 환경음 및 효과음
- texts: 화면 속 자막, 간판 정보 등
- speech: 등장인물들의 대사 (영어 또는 한국어)

[메타데이터 사용 시 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다. 반드시 비디오 프레임의 시각 정보를 우선 참고하세요.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식될 수 있습니다.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다.


[작성 규칙]
- 아래 출력 포맷에 맞게 두 파트로 나누어 작성하세요.
- 과거 장면(Past Information)이 제공되지 않은 경우, "1. 과거 장면 요약" 항목은 "해당 없음"으로 표기하세요.
- 언어는 반드시 **한국어**로 작성하세요. 영어 콘텐츠의 경우 고유명사는 원어 병기(예: 일각고래(narwhal))를 허용합니다.
- 각각의 파트는 텍스트의 길이 제한 없이 최대한 핵심을 상세하게 묘사하세요.

[출력 포맷]
**1. 과거 장면 요약**
(지금까지 발생한 주요 사건의 흐름, 등장인물 간의 대화와 관계, 인물의 목적이나 동기 등을 시간순으로 요약합니다.)

**2. 현재 장면**
(현재 장면에서 벌어지고 있는 구체적인 상황, 인물의 행동, 대화 내용, 감정 변화, 갈등 요소 등을 상세하게 묘사합니다.)"""


def make_summary_config(thinking_budget=None):
    """Summary 생성용 GenerateContentConfig를 반환합니다."""
    return make_generate_config(system_instruction=_SUMMARY_GEN_PROMPT, thinking_budget=thinking_budget)


def process_summary(client, summary_model_name, summary_config, past_parts, current_parts, end_time, use_ref=False):
    contents = []
    if past_parts is not None:
        contents += ["--- Past Information (Context) ---", past_parts["video"]]
        if use_ref:
            contents.append(past_parts["ref"])
    contents += ["--- Current Information (Focus Zone) ---", current_parts["video"]]
    if use_ref:
        contents.append(current_parts["ref"])
    request_msg = ("제공된 과거(Past Information)와 현재(Current Information) 영상 전체를 바탕으로 상세 요약을 작성하세요."
                   if past_parts is not None else
                   "제공된 현재 정보(Current Information) 영상을 바탕으로 상세 요약을 작성하세요.")
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
    parser = argparse.ArgumentParser(description="Keypoint Scene 목록을 입력받아 KeyScene Summary를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary 별도 저장 경로")

    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")

    parser.add_argument("--keyscene_summary_model", default="gemini-2.5-pro", help="Summary 생성에 사용할 Premium 모델명")
    parser.add_argument("--keyscene_summary_thinking_budget", type=int, default=1024,
                        help="KeyScene Summary 생성 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")
    parser.add_argument("--use_ref_for_keyscene_summary", type=lambda x: str(x).lower() == 'true', default=False, help="Summary 생성 시 Ref JSONL 참조 여부")

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
                    past_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", 0.0, start_time),
                        "ref":   process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   0.0, start_time),
                    } if start_time > 0.0 else None
                    current_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", start_time, end_time),
                        "ref":   process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   start_time, end_time),
                    }
                    return process_summary(
                        client, args.keyscene_summary_model, summary_config,
                        past_parts, current_parts, end_time, use_ref=args.use_ref_for_keyscene_summary
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

                    print(f"\n-> [Summary] 생성 완료 ({len(summary_text)}자, {summary_elapsed:.2f}초)")
                    print(f"\n{summary_text}")
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
