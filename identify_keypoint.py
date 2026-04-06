import os
import time
import argparse
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from gemini_api_utils import (
    process_gcs_file, check_gcs_files_exist, SAFETY_SETTINGS,
    load_config, parse_json_response, _retry_api_call, download_gcs_text
)

# ============================================================
# Keypoint 식별 모델 프롬프트
# ============================================================

_KEYPOINT_SYSTEM_PROMPT = """\
당신은 영상 콘텐츠를 분석하여 시청자가 보는 도중 자연스럽게 궁금해할 만한
'핵심 씬(Keypoint Scene)'을 식별하는 전문가입니다.
사용자는 원본 비디오 프레임과 Reference 메타데이터(JSONL)를 함께 제공합니다.

[Reference 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- speech: 등장인물들의 대사
- texts: 화면 속 자막, 간판 정보 등
- sounds: 환경음 및 효과음

[메타데이터 정확도 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 잘못 전사될 수 있습니다.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식되거나, 의미 없는 워터마크/로고가 포함될 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다 (예: 괴물 소리를 고양이 골골송으로 인식하는 등). 반드시 비디오 프레임의 시각 정보를 우선적으로 참고하고, 메타데이터는 보조 자료로만 활용하세요.

[Keypoint Scene 선별 기준]
시청자가 해당 시점까지 영상을 보다가 멈추고 궁금해할 만한 순간을 고르세요:
1. 극적인 전환점 또는 갑작스러운 상황 변화
2. 인물이 새로운 행동을 시작하거나, 중요한 결정을 내리는 순간
3. 시각적으로 인상적이거나 화면에 새로운 정보가 등장하는 순간
4. 대화 중 호기심을 자극하는 발언이나 사건이 발생하는 순간

[주의사항]
- 제공된 **Scene List** 중에서 가장 적합한 Scene을 선택하세요.
- 각 Keypoint는 반드시 특정 Scene의 종료 시점(end_time)을 기준으로 합니다.
- 최대 10개, 전체 Scene 수보다 많이 선택할 수 없음
- 반드시 아래 JSON 배열 형식으로만 출력하세요 (다른 설명 추가 금지)

[출력 형식 예시]
[
    {"scene_idx": 3, "reason": "미팅 장소에 도착하는 시점"},
    {"scene_idx": 7, "reason": "주인공의 인상적인 발언"},
    {"scene_idx": 16, "reason": "사건의 전환을 암시하는 행동"}
]\
"""


def init_keypoint_model(model_name):
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[_KEYPOINT_SYSTEM_PROMPT],
        safety_settings=SAFETY_SETTINGS
    )


def identify_keypoints(model, video_part, ref_part, scene_list_text):
    prompt = (
        "제공된 비디오, Reference 메타데이터와 아래 Scene List를 분석하여, "
        "시청자가 영상을 보는 도중 자연스럽게 궁금해할 만한 핵심 전환점/사건 Scene을 "
        "최대 10개 (전체 Scene 수 이하) 골라내세요.\n\n"
        f"[Scene List]\n{scene_list_text}\n\n"
        "반드시 지정된 JSON 배열 형식으로만 출력하세요 (scene_idx와 reason 필수)."
    )
    return _retry_api_call(
        lambda: model.generate_content([video_part, ref_part, prompt]).text,
        label="Keypoint 식별",
    )


def main():
    parser = argparse.ArgumentParser(description="영상 콘텐츠에서 Keypoint Scene을 식별하고 저장합니다.")
    parser.add_argument("--input_file", default="content_list.json", help="입력 JSON 파일 경로 (content_id 리스트)")
    parser.add_argument("--output_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 저장 경로")

    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")

    parser.add_argument("--keypoint_model", default="gemini-2.5-flash", help="Keypoint 식별에 사용할 모델명")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (config.json을 생성하세요)")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}...")
    vertexai.init(project=args.gcp_project_id, location=args.location)

    model = init_keypoint_model(args.keypoint_model)

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다.")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        input_list = json.load(f)

    # 출력 디렉토리 확인
    odir = os.path.dirname(args.output_file)
    if odir and not os.path.exists(odir):
        os.makedirs(odir, exist_ok=True)

    # 기처리분 건너뛰기
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["content_id"])
                    except: pass

    if processed_ids:
        print(f"[{len(processed_ids)}] 개의 콘텐츠가 이미 처리되어 건너뜁니다.")

    print("\n" + "="*50)
    print("Keypoint Scene 식별 파이프라인을 시작합니다.")
    print("="*50)

    try:
        for item in input_list:
            content_id = item if isinstance(item, str) else item.get("content_id")
            if not content_id:
                continue
            if content_id in processed_ids:
                print(f"\n[Skip] '{content_id}': 이미 처리됨")
                continue

            print(f"\n{'='*50}")
            print(f"Processing Content: '{content_id}'")
            print(f"{'='*50}")

            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue

            print(f"\n[Step 1] Keypoint Scene 식별 중... ({args.keypoint_model})")

            ref_jsonl_content = download_gcs_text(args.gs_bucket_name, f"jsonl/{content_id}_Ref.jsonl")
            ref_scenes = [json.loads(l) for l in ref_jsonl_content.strip().split("\n")]

            scene_list_text = ""
            for s in ref_scenes:
                idx = s.get("scene_idx", "?")
                st = s.get("start_time", 0.0)
                et = s.get("end_time", 0.0)
                sp = s.get("speech", "")
                scene_list_text += f"- Scene {idx}: {st:.1f}s ~ {et:.1f}s | {sp}...\n"

            video_meta = process_gcs_file(args.gs_bucket_name, content_id, mode="video")
            ref_meta = process_gcs_file(args.gs_bucket_name, content_id, mode="ref")

            try:
                time.sleep(2)
                keypoint_text = identify_keypoints(model, video_meta, ref_meta, scene_list_text)
                raw_keypoints = parse_json_response(keypoint_text)[:20]

                keypoints = []
                for rk in raw_keypoints:
                    s_idx = rk.get("scene_idx")
                    target = next((s for s in ref_scenes if s.get("scene_idx") == s_idx), None)
                    if target:
                        keypoints.append({
                            "scene_idx": s_idx,
                            "start_time": target["start_time"],
                            "end_time": target["end_time"],
                            "reason": rk.get("reason", "")
                        })
            except Exception as e:
                print(f"  [ERROR] Keypoint 식별 실패: {e}")
                continue

            print(f"\n총 {len(keypoints)}개의 Keypoint가 식별되었습니다:")
            for i, kp in enumerate(keypoints, 1):
                print(f"  {i:2d}. [Scene {kp['scene_idx']:2d}] {kp['start_time']:.1f}s ~ {kp['end_time']:.1f}s | {kp['reason']}")

            while True:
                user_input = input(f"\n위 {len(keypoints)}개 Keypoint를 저장하시겠습니까? (Y/N): ").strip().upper()
                if user_input in ("Y", "N"):
                    break

            if user_input == "N":
                print("저장을 취소합니다.")
                continue

            kp_record = {
                "content_id": content_id,
                "keypoints": keypoints
            }
            with open(args.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(kp_record, ensure_ascii=False) + "\n")

            processed_ids.add(content_id)
            print(f"[OK] '{content_id}' - {len(keypoints)}개 Keypoint 저장 완료: {args.output_file}")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    print("\n" + "="*50)
    print(f"Keypoint 식별 완료. 저장 위치: {args.output_file}")
    print("="*50)


if __name__ == "__main__":
    main()
