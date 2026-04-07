import os
import time
import argparse
import json
from gemini_api_utils import (
    create_client, make_generate_config,
    process_gcs_file_range,
    check_gcs_files_exist,
    load_config, parse_json_response, _retry_api_call,
    ensure_output_dir, load_processed_content_ids,
    preload_content_metadata,
)

# ───────────────────────────────────────────────
# Step 2: User Query 생성 모델 프롬프트
# ───────────────────────────────────────────────

_USER_QUERY_GENERATION_PROMPT = """\
당신은 영상 콘텐츠의 전체적인 흐름을 바탕으로 질문을 생성하는 전문가입니다.
사용자는 원본 비디오 프레임과 Reference 메타데이터(JSONL)를 함께 제공합니다.
시청자는 **과거 정보 (Past Information)** 와 **현재 정보 (Current Information)** 를 모두 시청했습니다.
두 정보를 모두 고려하여, 지금까지 누적해서 본 내용이나 전체 맥락 속에서 자연스럽게 가질 만한 종합적인 질문 3개를 생성하세요. 미래 내용은 절대 유추하지 마세요.

[Reference 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- speech: 등장인물들의 대사
- texts: 화면 속 자막, 간판 정보 등
- sounds: 환경음 및 효과음

[메타데이터 사용 시 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 잘못 전사될 수 있습니다.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식되거나, 의미 없는 워터마크/로고가 포함될 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다 (예: 괴물 소리를 고양이 골골송으로 인식하는 등). 반드시 비디오 프레임의 시각 정보를 우선적으로 참고하고, 메타데이터는 보조 자료로만 활용하세요.

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
    contents = [
        "--- Past Information (Context) ---",
        past_parts["video"], past_parts["meta"],
        "--- Current Information (Focus Zone) ---",
        current_parts["video"], current_parts["meta"],
        "--- 요청 사항 ---",
        "과거와 현재 정보를 바탕으로 누적 맥락에서 질문 3개를 생성하세요."
    ]
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
    parser = argparse.ArgumentParser(description="Keypoint 기반 User 파이프라인 (User Query) 자동 생성")
    parser.add_argument("--input_file", default="assets/bubble_query.jsonl", help="Bubble Query 생성된 JSONL 파일 (Keypoint 추출용, keypoints_file 없을 시 폴백)")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL (우선 사용, 없으면 input_file에서 추출)")
    parser.add_argument("--output_file", default="assets/user_query.jsonl", help="User Query 목록 저장 경로")
    
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")
    
    parser.add_argument("--uq_gen_model", default="gemini-2.5-pro", help="User Query 생성에 사용할 Premium 모델명")
    parser.add_argument("--uq_gen_thinking_budget", type=int, default=2048,
                        help="UQ 생성 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (config.json을 생성하세요)")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}...")
    client = create_client(args.gcp_project_id, args.location)
    query_config = make_generate_config(system_instruction=_USER_QUERY_GENERATION_PROMPT,
                                        thinking_budget=args.uq_gen_thinking_budget)

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_bubble_query.py를 실행하세요.")
        return

    # Keypoint 정보 로드: keypoints_file 우선, 없으면 bubble_query.jsonl에서 추출
    content_keypoints_map = {}

    if os.path.exists(args.keypoints_file):
        print(f"️  keypoint_scenes.jsonl 보존 파일을 사용: {args.keypoints_file}")
        with open(args.keypoints_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
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
                except json.JSONDecodeError:
                    pass
    
    if not content_keypoints_map:
        print(f"keypoint_scenes.jsonl 없음 → {args.input_file} 에서 Keypoint 추출")
        if not os.path.exists(args.input_file):
            print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_bubble_query.py를 실행하세요.")
            return
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
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
                except json.JSONDecodeError:
                    pass

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    # 기처리분 건너뛰기 로직
    processed_ids = load_processed_content_ids(args.output_file)
    if processed_ids:
        print(f"[{len(processed_ids)}] 개의 콘텐츠가 이미 {args.output_file}에 존재하여 건너뜁니다.")

    print("\n" + "="*50)
    print("User Query 생성 파이프라인을 시작합니다.")
    print("="*50)

    try:
        for content_id, keypoints in content_keypoints_map.items():
            if content_id in processed_ids:
                print(f"\n[Skip] '{content_id}': 이미 처리됨")
                continue

            print(f"\n{'='*50}")
            print(f"Processing Content: '{content_id}'")
            print(f"총 {len(keypoints)}개의 Keypoint에서 User Query를 생성합니다.")
            print(f"{'='*50}")

            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue

            # JSONL 메타데이터 프리로드 (캐시 워밍업)
            preload_content_metadata(args.gs_bucket_name, content_id)

            all_user_queries = []

            for idx, kp in enumerate(keypoints):
                scene_idx = kp["scene_idx"]
                start_time = float(kp["start_time"])
                end_time = float(kp["end_time"])

                print(f"\n  [{idx+1}/{len(keypoints)}] Range=[{start_time:.1f}s ~ {end_time:.1f}s]")

                try:
                    time.sleep(2)
                    past_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", 0.0, start_time),
                        "meta":  process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   0.0, start_time)
                    }
                    current_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", start_time, end_time),
                        "meta":  process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   start_time, end_time)
                    }

                    user_list = generate_user_query(
                        client, args.uq_gen_model, query_config,
                        past_parts, current_parts, end_time
                    )
                    
                    for q in user_list[:3]:
                        all_user_queries.append({
                            "scene_idx": scene_idx,
                            "query": q,
                            "start_time": start_time,
                            "end_time": end_time
                        })

                    print(f"    -> [User Query] 생성 완료: {len(user_list[:3])}개")

                except Exception as e:
                    print(f"    [ERROR] 질문 생성 실패 (end={end_time:.1f}s): {e}")
                    continue

            if all_user_queries:
                with open(args.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"content_id": content_id, "queries": all_user_queries}, ensure_ascii=False) + "\n")

            processed_ids.add(content_id)
            print(f"\n[OK] '{content_id}' - User Query({len(all_user_queries)}개) 저장 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    print("\n" + "="*50)
    print(f"모든 작업이 완료되었습니다. 저장 위치: {args.output_file}")
    print("="*50)


if __name__ == "__main__":
    main()
