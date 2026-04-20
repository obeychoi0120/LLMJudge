import os
import time
import argparse
import json
from utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_file_by_scene_idx,
    check_gcs_files_exist,
    parse_json_response, _retry_api_call,
    ensure_output_dir,
    preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
    load_keypoints_by_content, load_summary_map, check_input_file,
    print_pipeline_banner, print_pipeline_done,
)

# ───────────────────────────────────────────────
# Step 2: User Query 생성 모델 프롬프트
# ───────────────────────────────────────────────

_USER_QUERY_GENERATION_PROMPT = """당신은 영상 콘텐츠의 전체적인 흐름을 바탕으로 질문을 생성하는 전문가입니다.
당신에게 그동안 논리적으로 정리된 누적 기억인 **[지금까지의 핵심 요약 및 현재 상황 (KeyScene Summary)]** 텍스트와 현재 장면의 메타데이터 / 원본 비디오를 함께 제공합니다.
시청자는 화면을 보면서 당신이 읽는 요약본 내용만큼의 과거 스토리를 이미 알고 있다고 가정합니다.
이 요약된 전체 맥락 속에서 자연스럽게 가질 만한 종합적인 질문 3개를 생성하세요. 미래 내용은 절대 유추하지 마세요.

[메타데이터 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- sounds: 환경음 및 효과음
- texts: 화면 속 자막, 간판 정보 등
- speech: 등장인물들의 대사 (영어 또는 한국어)

[메타데이터 해석 시 주의사항]
- speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다. 따라서 당신이 적절히 비디오와 교차 검증하여 교정해야 합니다.
- sounds: 효과음 분류 오류가 빈번합니다. 반드시 비디오에서 본 정보를 우선시하고, 비디오에 존재하지 않는 효과음이 있다면 무시하세요.
- texts: OCR 오류로 인해 화면 텍스트의 철자가 틀릴 수 있습니다. 비디오와 적절히 교차 검증하여 교정하세요.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다. 비디오와 적절히 교차 검증하여 교정하세요.

[작성 규칙]
- 어투: 인터넷 커뮤니티나 친구에게 물어보는 매우 캐주얼한 구어체 (반말 위주)
- 현재 장면에만 국한되지 않고, 누적된 이야기 흐름이나 앞선 사건과의 연관성에 관한 거시적인 질문이어야 합니다.

[출력 형식 예시]
반드시 아래와 같은 형태의 JSON 배열 안에 3개의 질문을 작성해야 합니다.
[
    "저번 사건 때문에 지금 등장인물들이 저러는 건가?",
    "앞에 나왔던 아이템이 지금 또 등장했는데, 이게 복선이야?",
    "지금까지의 등장인물 관계를 보면 얘가 왜 이런 선택을 한 거지?"
]"""

def generate_user_query(client, model_name, query_config, summary_text, current_parts, scene_idx):
    contents = []
    if summary_text:
        contents += [
            "--- [지금까지의 핵심 요약 및 현재 상황 (KeyScene Summary)] ---",
            summary_text,
        ]
    contents += [
        "--- [Current Metadata] ---",
        current_parts["meta"], 
        "--- [Current Video] ---",
        current_parts["video"],
        "--- 요청 사항 ---",
    ]
    if summary_text:
        contents.append("제공된 텍스트 요약(과거+현재)과 현재 데이터를 복합적으로 분석하여 시청자가 가질 만한 종합적인 질문 3개를 생성하세요.")
    else:
        contents.append("현재 데이터를 바탕으로 시청자가 자연스럽게 가질 만한 질문 3개를 생성하세요.")
        
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=query_config
        ).text,
        label=f"User Query 생성 (Scene {scene_idx})"
    )
    return parse_json_response(text)

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint 기반 User 파이프라인 (User Query) 자동 생성")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL (과거 문맥용)")
    parser.add_argument("--output_file", default="assets/user_query.jsonl", help="User Query 목록 저장 경로")

    args, client = init_pipeline(parser.parse_args())
    query_config = make_generate_config(system_instruction=_USER_QUERY_GENERATION_PROMPT,
                                        thinking_level=args.uq_gen_thinking_level)

    # Keypoint 정보 로드
    if not check_input_file(args.keypoints_file, hint="먼저 identify_keypoint.py를 실행하세요."):
        return

    print(f"️  keypoint_scenes.jsonl을 사용: {args.keypoints_file}")
    content_keypoints_map = load_keypoints_by_content(args.keypoints_file)

    # KeyScene Summary 맵 로드
    summary_map = load_summary_map(args.keyscene_summary_file)
    if summary_map:
        print(f"[Summary] {len(summary_map)}개 Scene의 Summary 로드됨 ({args.keyscene_summary_file})")
    else:
        print(f"[Warning] Summary 파일을 찾을 수 없습니다: {args.keyscene_summary_file}")

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

    print_pipeline_banner("User Query 생성 파이프라인을 시작합니다.")

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

                if (content_id, scene_idx) in processed_scenes:
                    print(f"\n[{idx}/{len(keypoints)}] Scene {scene_idx} - [Skip] 이미 처리됨")
                    continue

                print(f"\n[{idx}/{len(keypoints)}] Scene {scene_idx}")

                try:
                    time.sleep(2)
                    summary_text = summary_map.get((content_id, scene_idx), "")
                    current_parts = {
                        "video": process_gcs_file_by_scene_idx(args.gs_bucket_name, content_id, "video", scene_idx, scene_idx),
                        "meta":  process_gcs_file_by_scene_idx(args.gs_bucket_name, content_id, "ref",   scene_idx, scene_idx)
                    }

                    user_query_list = generate_user_query(
                        client, args.uq_gen_model, query_config,
                        summary_text, current_parts, scene_idx
                    )
                    
                    scene_queries = user_query_list[:3]
                    for i, q in enumerate(scene_queries):
                        print(f"  {i+1}: {q}")

                    append_jsonl(args.output_file, {
                        "content_id": content_id,
                        "scene_idx": scene_idx,
                        "queries": scene_queries
                    })

                    processed_scenes.add((content_id, scene_idx))

                except Exception as e:
                    print(f"[ERROR] 질문 생성 실패 (Scene {scene_idx}): {e}")
                    continue

            print(f"\n[OK] '{content_id}' - User Query 생성 세션 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    print_pipeline_done(args.output_file)


if __name__ == "__main__":
    main()
