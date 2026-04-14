import os
import time
import argparse
import json
from gemini_api_utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_file_range,
    check_gcs_files_exist,
    parse_json_response, _retry_api_call,
    ensure_output_dir,
    preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
)

# ───────────────────────────────────────────────
# Step 2: User Query 생성 모델 프롬프트
# ───────────────────────────────────────────────

_USER_QUERY_GENERATION_PROMPT = """\
당신은 영상 콘텐츠의 전체적인 흐름을 바탕으로 질문을 생성하는 전문가입니다.
당신에게 참조용 메타데이터(JSONL)와 원본 비디오 프레임을 차례로 제공합니다.
시청자는 **과거 정보 (Past Information)** 와 **현재 정보 (Current Information)** 를 모두 시청했습니다.
두 정보를 모두 고려하여, 지금까지 누적해서 본 내용이나 전체 맥락 속에서 자연스럽게 가질 만한 종합적인 질문 3개를 생성하세요. 미래 내용은 절대 유추하지 마세요.

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
- 어투: 인터넷 커뮤니티나 친구에게 물어보는 매우 캐주얼한 구어체 (반말 위주)
- 현재 장면 뿐만 아니라 누적된 이야기 흐름이나 앞선 사건과의 연관성에 관한 질문도 좋습니다.

[출력 형식 예시]
반드시 아래와 같은 형태의 JSON 배열 안에 3개의 질문을 작성해야 합니다.
[
    "저번 사건 때문에 지금 등장인물들이 저러는 건가?",
    "앞에 나왔던 아이템이 지금 또 등장했는데, 이게 복선이야?",
    "지금까지의 등장인물 관계를 보면 얘가 왜 이런 선택을 한 거지?"
]\
"""

def generate_user_query(client, model_name, query_config, past_parts, current_parts, end_time):
    contents = []
    if past_parts is not None:
        contents += [
            "--- Past Information (Context) ---",
            past_parts["meta"], past_parts["video"],
        ]
    contents += [
        "--- Current Information (Focus Zone) ---",
        current_parts["meta"], current_parts["video"],
        "--- 요청 사항 ---",
    ]
    if past_parts is not None:
        contents.append("과거와 현재 정보를 바탕으로 누적 맥락에서 질문 3개를 생성하세요.")
    else:
        contents.append("현재 정보를 바탕으로 시청자가 자연스럽게 가질 만한 질문 3개를 생성하세요.")
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=query_config
        ).text,
        label=f"User Query 생성 (end={end_time:.1f}s)"
    )
    return parse_json_response(text)

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint 기반 User 파이프라인 (User Query) 자동 생성")
    parser.add_argument("--input_file", default="assets/voice_hint.jsonl", help="Voice Hint 생성된 JSONL 파일 (Keypoint 추출용, keypoints_file 없을 시 폴백)")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL (우선 사용, 없으면 input_file에서 추출)")
    parser.add_argument("--output_file", default="assets/user_query.jsonl", help="User Query 목록 저장 경로")

    args, client = init_pipeline(parser.parse_args())
    query_config = make_generate_config(system_instruction=_USER_QUERY_GENERATION_PROMPT,
                                        thinking_level=args.uq_gen_thinking_level)

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_voice_hint.py를 실행하세요.")
        return

    # Keypoint 정보 로드: keypoints_file 우선, 없으면 voice_hint.jsonl에서 추출
    content_keypoints_map = {}

    if os.path.exists(args.keypoints_file):
        print(f"️  keypoint_scenes.jsonl 보존 파일을 사용: {args.keypoints_file}")
        for data in load_jsonl(args.keypoints_file):
            c_id = data.get("content_id")
            keypoints = data.get("keypoints", [])
            if c_id and keypoints:
                content_keypoints_map[c_id] = [
                    {
                        "scene_idx": kp.get("scene_idx"),
                        "start_time": kp.get("start_time", 0.0),
                        "end_time": kp.get("end_time", 0.0)
                    }
                    for kp in keypoints
                ]
    
    if not content_keypoints_map:
        print(f"keypoint_scenes.jsonl 없음 → {args.input_file} 에서 Keypoint 추출")
        if not os.path.exists(args.input_file):
            print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_voice_hint.py를 실행하세요.")
            return
        for data in load_jsonl(args.input_file):
            c_id = data.get("content_id")
            queries = data.get("queries", [])
            kp_dict = {}
            for q in queries:
                s_idx = q.get("scene_idx")
                if s_idx not in kp_dict:
                    kp_dict[s_idx] = {
                        "scene_idx": s_idx,
                        "start_time": q.get("start_time", 0.0),
                        "end_time": q.get("end_time", 0.0)
                    }
            if c_id and kp_dict:
                content_keypoints_map[c_id] = list(kp_dict.values())

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    # 기처리분 건너뛰기 로직
    processed_scenes = set()
    for data in load_jsonl(args.output_file):
        c_id = data.get("content_id")
        if c_id:
            for q in data.get("queries", []):
                s_idx = q.get("scene_idx")
                if s_idx is not None:
                    processed_scenes.add((c_id, s_idx))
                    
    if processed_scenes:
        processed_contents = set([c_id for c_id, _ in processed_scenes])
        print(f"[{len(processed_contents)}] 개의 콘텐츠(씬 부분 완료 포함)가 이미 {args.output_file}에 존재하여 건너뜁니다.")

    print("\n" + "="*50)
    print("User Query 생성 파이프라인을 시작합니다.")
    print("="*50)

    try:
        for content_id, keypoints in content_keypoints_map.items():
            all_scenes_done = all((content_id, kp["scene_idx"]) in processed_scenes for kp in keypoints)
            if all_scenes_done and len(keypoints) > 0:
                print(f"\n[Skip] '{content_id}': 모든 Scene 이미 처리됨")
                continue

            print(f"\n{'='*50}")
            print(f"Processing Content: '{content_id}'")
            print(f"총 {len(keypoints)}개의 Keypoint에서 User Query를 생성합니다.")
            print(f"{'='*50}")

            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue

            # JSONL 메타데이터 프리로드 (캐시 워밍업)
            preload_content_metadata(args.gs_bucket_name, content_id)

            for idx, kp in enumerate(keypoints):
                scene_idx = kp["scene_idx"]
                start_time = float(kp["start_time"])
                end_time = float(kp["end_time"])

                if (content_id, scene_idx) in processed_scenes:
                    print(f"\n[{idx+1}/{len(keypoints)}] Range=[{start_time:.1f}s ~ {end_time:.1f}s] - [Skip] 이미 처리됨")
                    continue

                print(f"\n[{idx+1}/{len(keypoints)}] Range=[{start_time:.1f}s ~ {end_time:.1f}s]")

                try:
                    time.sleep(2)
                    past_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", 0.0, start_time),
                        "meta":  process_gcs_file_range(args.gs_bucket_name, content_id, "ref",  0.0, start_time)
                    } if start_time > 0.0 else None
                    current_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", start_time, end_time),
                        "meta":  process_gcs_file_range(args.gs_bucket_name, content_id, "ref",  start_time, end_time)
                    }

                    user_query_list = generate_user_query(
                        client, args.uq_gen_model, query_config,
                        past_parts, current_parts, end_time
                    )
                    
                    scene_queries = user_query_list[:3]
                    for q in scene_queries:
                        print(f"Q: {q}")

                    with open(args.output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "queries": scene_queries
                        }, ensure_ascii=False) + "\n")

                    processed_scenes.add((content_id, scene_idx))

                except Exception as e:
                    print(f"[ERROR] 질문 생성 실패 (end={end_time:.1f}s): {e}")
                    continue

            print(f"\n[OK] '{content_id}' - User Query 생성 세션 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    print("\n" + "="*50)
    print(f"모든 작업이 완료되었습니다. 저장 위치: {args.output_file}")
    print("="*50)


if __name__ == "__main__":
    main()
