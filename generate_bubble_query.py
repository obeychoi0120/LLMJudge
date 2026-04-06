import os
import time
import argparse
import json
import concurrent.futures
import vertexai
from vertexai.generative_models import GenerativeModel
from gemini_api_utils import (
    process_gcs_file_range, check_gcs_files_exist, SAFETY_SETTINGS,
    load_config, parse_json_response, _retry_api_call,
)

# ───────────────────────────────────────────────
# Bubble Query 생성 모델 프롬프트
# ───────────────────────────────────────────────

_BUBBLE_QUERY_PROMPT = """\
당신은 현재 시청 중인 방금 본 장면에서 자연스러운 궁금증을 유도하는 데이터 생성 전문가입니다.
사용자는 원본 비디오 프레임과 Reference 메타데이터(JSONL)를 함께 제공합니다.
시청자에게는 오직 **현재 정보 (Current Information)** (방금 본 Scene)만 제공됩니다.
현재 장면의 구체적인 상황, 인물의 행동, 화면 속 디테일 등에 집중하여 시청자가 가질 수 있는 질문 3개를 생성하세요.
과거의 맥락은 제공되지 않으므로, 철저하게 '이 장면에 보이는 것만'으로 만들어질 수 있는 질문이어야 합니다. 미래에 일어날 일을 짐작하지 마세요.
반드시 아래와 같은 형태의 JSON 배열 안에 3개의 질문을 작성해야 합니다.

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
- 단순히 구글링으로 해결되거나 너무 명백한 질문은 피하세요.

[출력 형식 예시]
[
    "방금 저 사람이 입고 있는 패딩이 어떤 거야?",
    "여기서 주인공의 표정이 어두워진 이유가 뭘까?",
    "화면에 잠시 보였던 저 파일에 뭐라고 적혀 있었지?"
]\
"""

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
- speech: 등장인물들의 대사
- texts: 화면 속 자막, 간판 정보 등
- sounds: 환경음 및 효과음

[메타데이터 사용 시 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 잘못 전사될 수 있습니다.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식되거나, 의미 없는 워터마크/로고가 포함될 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다 (예: 괴물 소리를 고양이 골골송으로 인식하는 등). 반드시 비디오 프레임의 시각 정보를 우선적으로 참고하고, 메타데이터는 보조 자료로만 활용하세요.\
"""


def init_bubble_query_model(model_name):
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[_BUBBLE_QUERY_PROMPT],
        safety_settings=SAFETY_SETTINGS
    )


def init_summary_gen_model(model_name):
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[_SUMMARY_GEN_PROMPT],
        safety_settings=SAFETY_SETTINGS
    )


def process_bubble_parallel(bubble_model, summary_model, past_parts, current_parts, end_time):
    """하나의 Keypoint에 대해 Bubble Query 및 Summary를 병렬로 수행합니다."""
    def generate_bubble_queries():
        contents = [
            "--- Current Information (Focus Zone) ---",
            current_parts["video"], current_parts["meta"],
            "--- 요청 사항 ---",
            "제공된 현재 장면(Current Information)만을 기반으로 시청자가 자연스럽게 가질 수 있는 질문 3개를 생성하세요."
        ]
        text = _retry_api_call(lambda: bubble_model.generate_content(contents).text, label=f"Bubble Query 생성 (end={end_time:.1f}s)")
        return parse_json_response(text)

    def generate_summary():
        contents = [
            "--- Past Information (Context) ---",
            past_parts["video"], past_parts["meta"],
            "--- Current Information (Focus Zone) ---",
            current_parts["video"], current_parts["meta"],
            "--- 요청 사항 ---",
            "제공된 과거(Past Information)와 현재(Current Information) 영상 전체를 바탕으로 상세 요약을 작성하세요."
        ]
        return _retry_api_call(lambda: summary_model.generate_content(contents).text, label=f"Summary 생성 (end={end_time:.1f}s)")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(generate_bubble_queries)
        f_sum = executor.submit(generate_summary)
        bubble_queries = f1.result()[:3]
        summary_text = f_sum.result()

    return bubble_queries, summary_text


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keypoint Scene 목록을 입력받아 Bubble Query & Detailed Summary를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--output_file", default="assets/bubble_query.jsonl", help="Bubble Query 목록 저장 경로")

    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")

    parser.add_argument("--query_gen_model", default="gemini-2.5-flash", help="질문 생성에 사용할 Budget 모델명")
    parser.add_argument("--summary_gen_model", default="gemini-2.5-pro", help="Summary 생성에 사용할 Premium 모델명")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (config.json을 생성하세요)")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}...")
    vertexai.init(project=args.gcp_project_id, location=args.location)

    bubble_model = init_bubble_query_model(args.query_gen_model)
    summary_model = init_summary_gen_model(args.summary_gen_model)

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 identify_keypoint.py를 실행하세요.")
        return

    # Keypoint 목록 로드
    keypoints_by_content = {}
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                c_id = data.get("content_id")
                kps = data.get("keypoints", [])
                if c_id and kps:
                    keypoints_by_content[c_id] = kps
            except json.JSONDecodeError:
                pass

    if not keypoints_by_content:
        print(f"Error: {args.input_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

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
    print("Bubble Query & Detailed Summary 생성 파이프라인을 시작합니다.")
    print("="*50)

    try:
        for content_id, keypoints in keypoints_by_content.items():
            if content_id in processed_ids:
                print(f"\n[Skip] '{content_id}': 이미 처리됨")
                continue

            print(f"\n{'='*50}")
            print(f"Processing Content: '{content_id}' ({len(keypoints)}개 Keypoint)")
            print(f"{'='*50}")

            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue

            all_bubble_queries = []

            for idx, kp in enumerate(keypoints):
                scene_idx = kp.get("scene_idx", idx)
                start_time = float(kp.get("start_time", 0.0))
                end_time = float(kp.get("end_time", 0.0))
                reason = kp.get("reason", "")

                print(f"\n  [{idx+1}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s] | {reason}")

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

                    bubble_list, summary_text = process_bubble_parallel(
                        bubble_model, summary_model,
                        past_parts, current_parts, end_time
                    )

                    for q in bubble_list:
                        all_bubble_queries.append({
                            "scene_idx": scene_idx,
                            "query": q,
                            "start_time": start_time,
                            "end_time": end_time,
                            "detailed_summary": summary_text
                        })

                    print(f"    -> [Bubble Query] 생성 완료: {len(bubble_list)}개")
                    print(f"    -> [Summary] 생성 완료 ({len(summary_text)}자)")

                except Exception as e:
                    print(f"    [ERROR] 생성 실패 (Scene {scene_idx}): {e}")
                    continue

            if all_bubble_queries:
                with open(args.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"content_id": content_id, "queries": all_bubble_queries}, ensure_ascii=False) + "\n")

            processed_ids.add(content_id)
            print(f"\n[OK] '{content_id}' - Bubble Query({len(all_bubble_queries)}개) 저장 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    print("\n" + "="*50)
    print(f"모든 작업이 완료되었습니다. 저장 위치: {args.output_file}")
    print("="*50)


if __name__ == "__main__":
    main()
