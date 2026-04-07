import os
import time
import argparse
import json
import concurrent.futures
from gemini_api_utils import (
    create_client, make_generate_config,
    process_gcs_file_range, check_gcs_files_exist,
    load_config, parse_json_response, _retry_api_call,
    ensure_output_dir, load_processed_pairs,
    preload_content_metadata,
)

# ───────────────────────────────────────────────
# Bubble Query 생성 모델 프롬프트
# ───────────────────────────────────────────────

_BUBBLE_QUERY_BASE = """\
당신은 현재 시청 중인 방금 본 장면에서 자연스러운 궁금증을 유도하는 데이터 생성 전문가입니다.
시청자에게는 오직 **현재 정보 (Current Information)** (방금 본 Scene)만 제공됩니다.
당신에게는 장면 설명이 담긴 Reference 메타데이터를 제공합니다.
현재 장면의 구체적인 상황, 인물의 행동, 화면 속 디테일 등에 집중하여 시청자가 가질 수 있는 질문 3개를 생성하세요.

[Reference 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- speech: 등장인물들의 대사 (영어 또는 한국어)
- texts: 화면 속 자막, 간판 정보 등
- sounds: 환경음 및 효과음
{description_field}

[메타데이터 사용 시 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다. **따라서, 당신이 자연스럽게 교정하여 답변을 생성해야 합니다.**
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식될 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다 (예: 괴물 소리를 사자의 울음소리로 인식하는 등). 반드시 비디오 프레임의 시각 정보를 우선적으로 참고하고, 메타데이터는 보조 자료로만 활용하세요.

[작성 규칙]
- (중요) 메타데이터를 통해 이미 명확하게 알 수 있는 내용은 질문하지 마세요.
- 과거의 맥락은 제공되지 않으므로, 철저하게 '이 장면에 보이는 것만' 으로 만들어질 수 있는 질문이어야 합니다. 
- 당신은 다음에 어떤 컨텐츠가 제공될 지 알 수 없습니다. 미래 시점에 대해서 질문하지 마세요.

[출력 형식]
- 어투: 반말 위주, 인터넷 커뮤니티나 친구에게 물어보는 매우 캐주얼한 구어체
- 언어: 반드시 **한국어**로 작성하세요. 영어 콘텐츠의 경우 고유명사는 원어 병기(예: 일각고래(narwhal))를 허용합니다.
- 형식: 아래의 [예시] 처럼 3개의 질문만 생성하세요. 다른 설명은 덧붙이지 마십시오.

[출력 예시]
[
    "지금 저 여자가 입고 있는 옷을 찾아줘.",
    "여기서 저 남자가 왜 갑자기 저렇게 행동하는 거야?",
    "아까 주인공이 먹었던 빵 은담베(Pain Ndambe)는 어떻게 만드는 거야?"
]"""

_DESCRIPTION_LINE = "- description: 해당 장면 속 인물의 행동과 배경 장면 묘사\n"

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
- speech: 등장인물들의 대사 (영어 또는 한국어)
- texts: 화면 속 자막, 간판 정보 등
- sounds: 환경음 및 효과음

[메타데이터 사용 시 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다. **따라서, 당신이 자연스럽게 교정하여 답변을 생성해야 합니다.**
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식될 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다 (예: 괴물 소리를 사자의 울음소리로 인식하는 등). 반드시 비디오 프레임의 시각 정보를 우선적으로 참고하고, 메타데이터는 보조 자료로만 활용하세요.

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


def make_bubble_query_config(mode="part", thinking_budget=0):
    """Bubble Query 생성용 GenerateContentConfig를 반환합니다."""
    if mode == "full":
        prompt = _BUBBLE_QUERY_BASE.format(description_field=_DESCRIPTION_LINE)
    else:  # part
        prompt = _BUBBLE_QUERY_BASE.format(description_field="")
    return make_generate_config(system_instruction=prompt, thinking_budget=thinking_budget)


def make_summary_config(thinking_budget=None):
    """Summary 생성용 GenerateContentConfig를 반환합니다."""
    return make_generate_config(system_instruction=_SUMMARY_GEN_PROMPT, thinking_budget=thinking_budget)


def process_bubble_parallel(client, bubble_model_name, summary_model_name,
                            bubble_config_full, bubble_config_part, summary_config,
                            past_parts, current_parts, end_time):
    """하나의 Keypoint에 대해 Bubble Query(Full/Part) 및 Summary를 병렬로 수행합니다."""
    def generate_bubble_queries(mode):
        if mode == "full":
            data = current_parts["full"]
            config = bubble_config_full
        else:
            data = current_parts["part"]
            config = bubble_config_part
        contents = [
            "--- Current Information (Focus Zone) ---",
            data,
            "--- 요청 사항 ---",
            "제공된 현재 장면(Current Information)만을 기반으로 시청자가 자연스럽게 가질 수 있는 질문 3개를 생성하세요."
        ]
        t0 = time.time()
        text = _retry_api_call(
            lambda: client.models.generate_content(
                model=bubble_model_name, contents=contents, config=config
            ).text,
            label=f"Bubble Query({mode}) 생성 (end={end_time:.1f}s)"
        )
        return parse_json_response(text)[:3], time.time() - t0

    def generate_summary():
        if past_parts is not None:
            contents = [
                "--- Past Information (Context) ---",
                past_parts["video"], past_parts["ref"],
                "--- Current Information (Focus Zone) ---",
                current_parts["video"], current_parts["ref"],
                "--- 요청 사항 ---",
                "제공된 과거(Past Information)와 현재(Current Information) 영상 전체를 바탕으로 상세 요약을 작성하세요."
            ]
        else:
            contents = [
                "--- Current Information (Focus Zone) ---",
                current_parts["video"], current_parts["ref"],
                "--- 요청 사항 ---",
                "제공된 현재(Current Information) 영상을 바탕으로 상세 요약을 작성하세요."
            ]
        t0 = time.time()
        text = _retry_api_call(
            lambda: client.models.generate_content(
                model=summary_model_name, contents=contents, config=summary_config
            ).text,
            label=f"Summary 생성 (end={end_time:.1f}s)"
        )
        return text, time.time() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_full = executor.submit(generate_bubble_queries, "full")
        f_part = executor.submit(generate_bubble_queries, "part")
        f_sum  = executor.submit(generate_summary)
        bubble_full_list, bubble_full_elapsed = f_full.result()
        bubble_part_list, bubble_part_elapsed = f_part.result()
        summary_text, summary_elapsed = f_sum.result()

    return bubble_full_list[:3], bubble_part_list[:3], summary_text, bubble_full_elapsed, bubble_part_elapsed, summary_elapsed


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keypoint Scene 목록을 입력받아 Bubble Query & Detailed Summary를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--output_file", default="assets/bubble_query.jsonl", help="Bubble Query 목록 저장 경로")
    parser.add_argument("--summary_file", default="assets/bubble_summary.jsonl", help="Detailed Summary 별도 저장 경로")

    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")

    parser.add_argument("--bq_gen_model", default="gemini-2.5-flash", help="질문 생성에 사용할 Budget 모델명")
    parser.add_argument("--bq_summary_model", default="gemini-2.5-pro", help="Summary 생성에 사용할 Premium 모델명")
    parser.add_argument("--bq_thinking_budget", type=int, default=0, help="Bubble Query 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")
    parser.add_argument("--bq_summary_thinking_budget", type=int, default=1024,
                        help="BQ Summary 생성 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (config.json을 생성하세요)")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}...")
    client = create_client(args.gcp_project_id, args.location)

    bubble_config_full = make_bubble_query_config(mode="full", thinking_budget=args.bq_thinking_budget)
    bubble_config_part = make_bubble_query_config(mode="part", thinking_budget=args.bq_thinking_budget)
    summary_config = make_summary_config(thinking_budget=args.bq_summary_thinking_budget)

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
    ensure_output_dir(args.output_file)
    ensure_output_dir(args.summary_file)

    # 각 파일에서 독립적으로 기처리분 로드
    bq_pairs = load_processed_pairs(args.output_file,  key_fields=("content_id", "scene_idx"))
    summary_pairs = load_processed_pairs(args.summary_file, key_fields=("content_id", "scene_idx"))
    # API 호출 스킵 기준: 두 파일 모두 완료된 scene
    fully_done_pairs = bq_pairs & summary_pairs
    if fully_done_pairs:
        print(f"[{len(fully_done_pairs)}]개의 Scene이 이미 처리되어 건너뜁니다.")

    print("\n" + "="*50)
    print("Bubble Query & Detailed Summary 생성 파이프라인을 시작합니다.")
    print("="*50)

    try:
        for content_id, keypoints in keypoints_by_content.items():
            done_scenes = {s_idx for (c_id, s_idx) in fully_done_pairs if c_id == content_id}
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

            for idx, kp in enumerate(remaining):
                scene_idx = kp.get("scene_idx", idx)
                start_time = float(kp.get("start_time", 0.0))
                end_time = float(kp.get("end_time", 0.0))
                reason = kp.get("reason", "")

                print(f"\n  [{idx+1}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s]")
                kp_start = time.time()

                def _run_keypoint():
                    past_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", 0.0, start_time),
                        "ref":  process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   0.0, start_time)
                    } if start_time > 0.0 else None
                    current_parts = {
                        "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", start_time, end_time),
                        "ref":  process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   start_time, end_time),
                        "full":  process_gcs_file_range(args.gs_bucket_name, content_id, "full",  start_time, end_time),
                        "part":  process_gcs_file_range(args.gs_bucket_name, content_id, "part",  start_time, end_time),
                    }
                    return process_bubble_parallel(
                        client,
                        args.bq_gen_model, args.bq_summary_model,
                        bubble_config_full, bubble_config_part, summary_config,
                        past_parts, current_parts, end_time
                    )

                try:
                    bubble_full_list, bubble_part_list, summary_text, bubble_full_elapsed, bubble_part_elapsed, summary_elapsed = _retry_api_call(
                        _run_keypoint,
                        label=f"Bubble+Summary (Scene {scene_idx})"
                    )

                    scene_key = (content_id, scene_idx)

                    # bubble_query.jsonl: 해당 파일에 없는 경우에만 기록
                    if scene_key not in bq_pairs:
                        scene_queries = [
                            {"mode": "full", "queries": bubble_full_list[:3]},
                            {"mode": "part", "queries": bubble_part_list[:3]},
                        ]
                        scene_record = {
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "queries": scene_queries,
                        }
                        with open(args.output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(scene_record, ensure_ascii=False) + "\n")
                        bq_pairs.add(scene_key)
                    else:
                        print(f"-> [BQ] Scene {scene_idx} 이미 존재, 스킵")

                    # bubble_summary.jsonl: 해당 파일에 없는 경우에만 기록
                    if scene_key not in summary_pairs:
                        summary_record = {
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "summary": summary_text,
                        }
                        with open(args.summary_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
                        summary_pairs.add(scene_key)
                    else:
                        print(f"-> [Summary] Scene {scene_idx} 이미 존재, 스킵")

                    print(f"-> [BQ - Full] {len(bubble_full_list)}개 ({bubble_full_elapsed:.2f}초)")
                    for qi, q in enumerate(bubble_full_list, 1):
                        print(f"    {qi}. {q}")
                    print(f"-> [BQ - Part] {len(bubble_part_list)}개 ({bubble_part_elapsed:.2f}초)")
                    for qi, q in enumerate(bubble_part_list, 1):
                        print(f"    {qi}. {q}")
                    print(f"-> [Summary] 생성 완료 ({len(summary_text)}자, {summary_elapsed:.2f}초)")
                    print(f"\n{summary_text}")
                    print(f"------------------------------------------------------")

                except Exception as e:
                    print(f"    [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                    continue

            done_count = len({s_idx for (c_id, s_idx) in (bq_pairs & summary_pairs) if c_id == content_id})
            print(f"\n[OK] '{content_id}' - {done_count}개 Scene 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    print("\n" + "="*50)
    print(f"모든 작업이 완료되었습니다. 저장 위치: {args.output_file}")
    print("="*50)



if __name__ == "__main__":
    main()
