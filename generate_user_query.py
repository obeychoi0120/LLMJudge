import os
import time
import argparse
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from gemini_api_utils import (
    process_gcs_file_range,
    check_gcs_files_exist, SAFETY_SETTINGS,
    load_config, parse_json_response, _retry_api_call,
)

# ───────────────────────────────────────────────
# Step 2: User Query 생성 모델 프롬프트
# ───────────────────────────────────────────────

_USER_QUERY_PROMPT = """\
당신은 영상 콘텐츠의 전체적인 흐름을 바탕으로 질문을 생성하는 전문가입니다.
시청자는 **과거 정보 (Past Information)** 와 **현재 정보 (Current Information)** 를 모두 시청했습니다.
두 정보를 모두 고려하여, 지금까지 누적해서 본 내용이나 전체 맥락 속에서 자연스럽게 가질 만한 종합적인 질문 3개를 생성하세요. 미래 내용은 절대 유추하지 마세요.
반드시 아래와 같은 형태의 JSON 배열 안에 3개의 질문을 작성해야 합니다.

[작성 규칙]
- 어투: 인터넷 커뮤니티나 친구에게 물어보는 매우 캐주얼한 구어체 (반말 위주)
- 현재 장면 뿐만 아니라 누적된 이야기 흐름이나 앞선 사건과의 연관성에 관한 질문도 좋습니다.

[출력 형식 예시]
[
    "저번 사건 때문에 지금 등장인물들이 저러는 건가?",
    "앞에 나왔던 아이템이 지금 또 등장했는데, 이게 복선이야?",
    "지금까지의 등장인물 관계를 보면 얘가 왜 이런 선택을 한 거지?"
]\
"""

def init_user_query_model(model_name):
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[_USER_QUERY_PROMPT],
        safety_settings=SAFETY_SETTINGS
    )

def generate_user_query(query_model, past_parts, current_parts, end_time):
    contents = [
        "--- Past Information (Context) ---",
        past_parts["video"], past_parts["meta"],
        "--- Current Information (Focus Zone) ---",
        current_parts["video"], current_parts["meta"],
        "--- 요청 사항 ---",
        "과거와 현재 정보를 바탕으로 누적 맥락에서 질문 3개를 생성하세요."
    ]
    text = _retry_api_call(lambda: query_model.generate_content(contents).text, label=f"User Query 생성 (end={end_time:.1f}s)")
    return parse_json_response(text)

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keypoint 기반 User 파이프라인 (User Query) 자동 생성")
    parser.add_argument("--input_file", default="assets/bubble_query_generated.jsonl", help="Bubble Query 생성된 JSONL 파일 (Keypoint 추출용)")
    parser.add_argument("--output_file", default="assets/user_query_generated.jsonl", help="User Query 목록 저장 경로")
    
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")
    
    parser.add_argument("--query_gen_model", default="gemini-2.5-flash", help="질문 생성에 사용할 Budget 모델명")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (config.json을 생성하세요)")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}...")
    vertexai.init(project=args.gcp_project_id, location=args.location)

    user_model = init_user_query_model(args.query_gen_model)

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_bubble_query.py를 실행하세요.")
        return

    # Bubble Query 파일에서 Keypoint 정보 추출
    content_keypoints_map = {}
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

    if not content_keypoints_map:
        print(f"Error: {args.input_file} 에서 Keypoint 정보를 추출할 수 없습니다.")
        return

    # 출력 디렉토리 확인
    odir = os.path.dirname(args.output_file)
    if odir and not os.path.exists(odir):
        os.makedirs(odir)

    # 기처리분 건너뛰기 로직
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["content_id"])
                    except: pass
                    
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
                        user_model, past_parts, current_parts, end_time
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
