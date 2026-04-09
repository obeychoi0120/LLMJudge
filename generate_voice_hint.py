import os
import time
import argparse
import json
import concurrent.futures
from gemini_api_utils import (
    make_generate_config,
    process_gcs_file_range, check_gcs_files_exist,
    parse_json_response, _retry_api_call,
    ensure_output_dir, load_processed_pairs,
    preload_content_metadata, get_gcs_text_range, download_gcs_text, truncate_jsonl_range,
    init_pipeline, load_jsonl, append_jsonl,
)

# ───────────────────────────────────────────────
# Voice Hint 생성 모델 프롬프트
# ───────────────────────────────────────────────

_VOICE_HINT_BASE = """\
당신은 현재 시청 중인 방금 본 장면에서 자연스러운 궁금증을 유도하는 데이터 생성 전문가입니다.
시청자에게는 오직 **현재 정보 (Current Information)** (방금 본 Scene)만 제공됩니다.
당신에게는 장면 설명이 담긴 Description 메타데이터를 제공합니다.
현재 장면의 구체적인 상황, 인물의 행동, 화면 속 디테일 등에 집중하여 시청자가 가질 수 있는 질문 3개를 생성하세요.

[Description 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- description: 해당 Scene의 시각적 상황, 인물 행동, 대사, 화면 자막, 환경음 등을 종합한 자세한 묘사

[작성 규칙]
- (중요) description을 통해 이미 명확하게 알 수 있는 내용은 질문하지 마세요.
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

def make_voice_hint_config(thinking_budget=0):
    """Voice Hint 생성용 GenerateContentConfig를 반환합니다."""
    return make_generate_config(system_instruction=_VOICE_HINT_BASE, thinking_budget=thinking_budget)

def process_vh_parallel(client, vh_model_name, vh_config, current_parts, end_time):
    """하나의 Keypoint에 대해 Voice Hint(img_desc, mm_desc 2개 모드)를 병렬로 수행합니다."""
    def generate_voice_hints(mode):
        contents = [
            "--- Current Information (Focus Zone) ---",
            current_parts[mode],
            "--- 요청 사항 ---",
            "제공된 현재 장면(Current Information)만을 기반으로 시청자가 자연스럽게 가질 수 있는 질문 3개를 생성하세요."
        ]
        t0 = time.time()
        text = _retry_api_call(
            lambda: client.models.generate_content(
                model=vh_model_name, contents=contents, config=vh_config
            ).text,
            label=f"Voice Hint({mode}) 생성 (end={end_time:.1f}s)"
        )
        return parse_json_response(text)[:3], time.time() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_vh_img  = executor.submit(generate_voice_hints, "img_desc")
        f_vh_mm  = executor.submit(generate_voice_hints, "mm_desc")
        
        vh_list_img, elapsed_img = f_vh_img.result()
        vh_list_mm, elapsed_mm = f_vh_mm.result()

    return {"img_desc": vh_list_img[:3], "mm_desc": vh_list_mm[:3]}, {"img_desc": elapsed_img, "mm_desc": elapsed_mm}

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Keypoint Scene 목록을 입력받아 Voice Hint를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--output_file", default="assets/voice_hint.jsonl", help="Voice Hint 목록 저장 경로")

    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")

    parser.add_argument("--vh_gen_model", default="gemini-2.5-flash", help="질문 생성에 사용할 Budget 모델명")
    parser.add_argument("--vh_thinking_budget", type=int, default=0, help="Voice Hint 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args, client = init_pipeline(parser.parse_args())

    vh_config = make_voice_hint_config(thinking_budget=args.vh_thinking_budget)

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
    ensure_output_dir(args.output_file)

    # 기처리분 로드
    vh_pairs = load_processed_pairs(args.output_file,  key_fields=("content_id", "scene_idx"))

    print("\n" + "="*50)
    print("Voice Hint 생성 파이프라인을 시작합니다.")
    print("="*50)

    try:
        for content_id, keypoints in keypoints_by_content.items():
            done_scenes = {s_idx for (c_id, s_idx) in vh_pairs if c_id == content_id}
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
                reason = kp.get("reason", "")

                print(f"[{real_idx+1}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s]")
                
                # 로깅용 Desc 텍스트 추출 및 즉시 출력
                img_desc_text = get_gcs_text_range(args.gs_bucket_name, content_id, "img_desc", start_time, end_time)
                print(f"\n[Desc (img_desc)]\n{img_desc_text}\n")

                mm_desc_text = get_gcs_text_range(args.gs_bucket_name, content_id, "mm_desc", start_time, end_time)
                print(f"\n[Desc (mm_desc)]\n{mm_desc_text}\n")

                def _run_keypoint():
                    current_parts = {
                        "img_desc":  process_gcs_file_range(args.gs_bucket_name, content_id, "img_desc",  start_time, end_time),
                        "mm_desc":  process_gcs_file_range(args.gs_bucket_name, content_id, "mm_desc",  start_time, end_time),
                    }
                    return process_vh_parallel(client, args.vh_gen_model, vh_config, current_parts, end_time)

                try:
                    vh_dict, vh_elapsed_dict = _retry_api_call(
                        _run_keypoint,
                        label=f"Voice Hint (Scene {scene_idx})"
                    )

                    scene_key = (content_id, scene_idx)

                    # voice_hint.jsonl: 해당 파일에 없는 경우에만 기록
                    if scene_key not in vh_pairs:
                        scene_record = {
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "queries": [
                                {"mode": "img_desc", "queries": vh_dict["img_desc"]},
                                {"mode": "mm_desc", "queries": vh_dict["mm_desc"]}
                            ],
                        }
                        append_jsonl(args.output_file, scene_record)
                        vh_pairs.add(scene_key)

                    print(f"-> [VH - img_desc] ({vh_elapsed_dict['img_desc']:.2f}초)")
                    for qi, q in enumerate(vh_dict["img_desc"], 1):
                        print(f"    {qi}. {q}")

                    print(f"\n-> [VH - mm_desc] ({vh_elapsed_dict['mm_desc']:.2f}초)")
                    for qi, q in enumerate(vh_dict["mm_desc"], 1):
                        print(f"    {qi}. {q}")
                    print(f"------------------------------------------------------\n")

                except Exception as e:
                    print(f"    [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                    continue

            done_count = len({s_idx for (c_id, s_idx) in vh_pairs if c_id == content_id})
            print(f"\n[OK] '{content_id}' - {done_count}개 Scene 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    print("\n" + "="*50)
    print(f"모든 작업이 완료되었습니다. 저장 위치: {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
