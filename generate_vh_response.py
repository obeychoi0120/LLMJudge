import time
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_video_part,
    check_gcs_files_exist,
    _retry_api_call,
    get_gcs_descriptions_by_scene_idx,
    get_gcs_raw_fields_by_scene_idx,
    ensure_output_dir,
    preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
    load_keypoints_by_content, check_input_file,
    load_scenes,
    print_pipeline_banner, print_pipeline_done,
)

# ============================================================
# System Prompts
# ============================================================

_VH_RESPONSE_PROMPT_TEXT = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

당신에게는 영상의 처음부터 시청자가 현재 보고 있는 장면까지, 소형 AI가 자동으로 분석·요약한 **누적 시청 기억(Accumulated Memory)**이 제공됩니다.
이 기억에는 Scene 단위로 분절된 영상 정보가 시간순으로 나열되어 있으며, **마지막 Scene이 시청자가 지금 집중하고 있는 장면**입니다.

이 정보를 바탕으로 시청자의 질문에 자연스럽고 정확하게 답변하여 '개인화된 인터랙티브 시청 경험'을 극대화해 주세요.

[시청 기억의 구조]
시청 기억은 다음 두 가지 형식 중 하나로 제공됩니다:

(A) Description 형식 (Image-based / Multimodal Description):
  - scene_idx: Scene 인덱스
  - start_time / end_time: Scene 시작·종료 시간 (초)
  - description: 시각적 상황, 인물 행동, 대사, 자막, 환경 등을 종합한 자세한 영문 묘사

(B) Raw Metadata 형식:
  - scene_idx: Scene 인덱스
  - start_time / end_time: Scene 시작·종료 시간 (초)
  - speech: 해당 Scene에서 인식된 음성(ASR) 텍스트
  - texts: 해당 Scene에서 검출된 화면 텍스트(OCR) 목록

[분석 및 대화 지시사항]

1. **현재 장면 우선 (Recency Bias)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.
   - 과거 Scene은 질문의 맥락을 이해하는 데 참고하되, 답변의 생동감은 현재 장면에서 끌어오세요.

2. **VLM 노이즈 자체 교정 (매우 중요)**
   - 시청 기억은 소형 AI가 자동 생성한 텍스트이므로 오탈자(예: '셰프'→'세프', '인정받은'→'정받은'), 띄어쓰기 오류, 상충 정보, 환각(Hallucination)이 포함될 수 있습니다.
   - 표면적 텍스트를 맹신하지 말고, 앞뒤 맥락과 풍부한 일반 상식(World Knowledge)을 결합하여 명백한 오류를 자연스럽게 교정·필터링하세요.

3. **유연한 추론과 상식의 결합**
   - 화면에 직접 명시되지 않은 질문이라도 "알 수 없습니다"로 대화를 끊지 마세요.
   - 외부 웹 검색은 하지 않되, 시각적 단서와 내재된 사전 지식을 결합하여 "화면에 직접 나오진 않지만~", "보통 이런 경우~"처럼 논리적이고 유익하게 추론하세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "JSON", "타임스탬프", "시청 기록에 따르면" 등 시스템 용어는 절대 금지합니다.
   - "지금 화면을 보면~", "방금 자막에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""

_VH_RESPONSE_PROMPT_VIDEO = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

당신에게는 영상의 처음부터 시청자가 현재 보고 있는 장면까지의 **비디오 클립**이 직접 제공됩니다.
클립의 **마지막 부분이 시청자가 지금 집중하고 있는 장면**입니다.

이 영상을 직접 시청·분석하여 시청자의 질문에 자연스럽고 정확하게 답변해 주세요.

[분석 및 대화 지시사항]

1. **현재 장면 우선 (Recency Bias)**
   - 영상 후반부(= 현재 장면)에 가장 높은 우선순위를 두세요.
   - 영상 초반은 질문의 맥락을 이해하는 데 참고하되, 답변의 생동감은 현재 장면에서 끌어오세요.

2. **정밀한 시각·청각 분석**
   - 화면에 보이는 인물, 행동, 자막, 로고, 배경, 소리 등 모든 정보를 종합적으로 활용하세요.
   - 외부 웹 검색은 절대 하지 말고, 영상에서 직접 관찰한 시각·청각 정보와 내재된 사전 지식만을 결합하세요.

3. **유연한 추론과 상식의 결합**
   - 화면에 직접 명시되지 않은 질문이라도 "알 수 없습니다"로 대화를 끊지 마세요.
   - 시각적 단서와 사전 지식을 결합하여 "화면에 직접 나오진 않지만~", "보통 이런 경우~"처럼 논리적이고 유익하게 추론하세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "프레임", "타임스탬프", "비디오 분석 결과" 등 시스템 용어는 절대 금지합니다.
   - "지금 화면을 보면~", "방금 나온 장면에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""


def make_vh_gen_config(thinking_level=None):
    """VH Response 생성용 GenerateContentConfig 딕셔너리를 반환합니다."""
    return {
        "text": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_TEXT,
            thinking_level=thinking_level,
        ),
        "video": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_VIDEO,
            thinking_level=thinking_level,
        ),
    }


# ============================================================
# Source Builders
# ============================================================

def _build_source(gs_bucket_name, content_id, scene_idx, keypoints, mode, max_past_scenes=None):
    """지정 모드의 Source Part / 텍스트를 빌드합니다.

    Args:
        gs_bucket_name: GCS 버킷명
        content_id: 콘텐츠 ID
        scene_idx: 현재 KeyScene의 scene_idx
        keypoints: 해당 content_id의 전체 keypoint 목록 (list of dict)
        mode: 'img_desc' | 'mm_desc' | 'video' | 'raw'
        max_past_scenes: None이면 Scene 0부터 전체, 정수이면 현재 KeyScene 기준 최근 N개 Scene만 사용

    Returns:
        (source, label) — source는 API contents 리스트에 넣을 수 있는 Part 또는 str
    """
    # 현재 KeyScene의 end_time을 ref JSONL에서 조회
    ref_scenes = load_scenes(gs_bucket_name, content_id, mode="ref")

    if mode == "video":
        # video: start_offset ~ end_offset 클리핑
        if max_past_scenes is not None:
            # 현재 KeyScene이 keypoints의 몇 번째인지 찾기
            kp_idx = next((i for i, kp in enumerate(keypoints) if kp.get("scene_idx") == scene_idx), None)
            if kp_idx is not None:
                past_start_kp_idx = max(0, kp_idx - max_past_scenes)
                past_start_scene_idx = keypoints[past_start_kp_idx].get("scene_idx", 0)
            else:
                past_start_scene_idx = 0
            start_scene = next((s for s in ref_scenes if s.get("scene_idx") == past_start_scene_idx), None)
            video_start = float(start_scene.get("start_time", 0.0)) if start_scene else 0.0
        else:
            video_start = 0.0

        end_scene = next((s for s in ref_scenes if s.get("scene_idx") == scene_idx), None)
        video_end = float(end_scene.get("end_time", 0.0)) if end_scene else 0.0

        return process_gcs_video_part(gs_bucket_name, content_id, video_start, video_end)

    else:
        # 텍스트 모드: start_idx 결정
        if max_past_scenes is not None:
            kp_idx = next((i for i, kp in enumerate(keypoints) if kp.get("scene_idx") == scene_idx), None)
            if kp_idx is not None:
                past_start_kp_idx = max(0, kp_idx - max_past_scenes)
                start_idx = keypoints[past_start_kp_idx].get("scene_idx", 0)
            else:
                start_idx = 0
        else:
            start_idx = 0

        if mode in ("img_desc", "mm_desc"):
            return get_gcs_descriptions_by_scene_idx(
                gs_bucket_name, content_id, mode, start_idx, scene_idx
            )
        else:  # raw
            return get_gcs_raw_fields_by_scene_idx(
                gs_bucket_name, content_id, start_idx, scene_idx
            )


def _generate_for_mode(client, model_name, gen_configs, source, mode, query, scene_idx):
    """단일 모드에 대해 Response를 생성합니다."""
    try:
        time.sleep(1)

        if mode == "video":
            contents = [
                "--- [Video Clip (처음부터 현재 장면까지)] ---",
                source,
                "--- 질문 ---",
                query,
            ]
            config = gen_configs["video"]
        else:
            label_map = {
                "img_desc": "Image-based Description (처음부터 현재 장면까지)",
                "mm_desc":  "Multimodal Description (처음부터 현재 장면까지)",
                "raw":      "Raw Metadata (speech & texts, 처음부터 현재 장면까지)",
            }
            contents = [
                f"--- [{label_map.get(mode, mode)}] ---",
                source,
                "--- 질문 ---",
                query,
            ]
            config = gen_configs["text"]

        answer = _retry_api_call(
            lambda: client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            ).text,
            label=f"VH Response [{mode}] (Scene {scene_idx})",
        )
        return mode, answer

    except Exception as e:
        print(f"  [ERROR] [{mode}] 생성 실패 (Scene {scene_idx}): {e}")
        return mode, f"Error: {str(e)}"


# ============================================================
# Progress Helpers
# ============================================================

def _load_completed_pairs(output_path):
    """완료된 (content_id, query) 쌍을 반환합니다.
    모든 4개 모드(img_desc, mm_desc, video, raw)가 에러 없이 완료된 경우만 완료로 간주합니다."""
    completed = set()
    _MODES = ("img_desc", "mm_desc", "video", "raw")
    for rec in load_jsonl(output_path):
        c_id = rec.get("content_id")
        query = rec.get("query")
        if not c_id or not query:
            continue
        answers = rec.get("answers", {})
        if all(
            answers.get(m) and not str(answers.get(m, "")).startswith("Error")
            for m in _MODES
        ):
            completed.add((c_id, query))
    return completed


# ============================================================
# Main
# ============================================================

def main():
    parser = get_common_argparser(description="Voice Hint (KSS 모드) 질문에 대해 4가지 Source 모드로 Response를 생성합니다.")
    parser.add_argument("--input_file",     default="assets/voice_hint.jsonl",   help="Voice Hint JSONL 경로 (KSS 모드만 사용)")
    parser.add_argument("--output_file",    default="assets/vh_responses.jsonl", help="VH Response 저장 경로")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로")
    parser.add_argument("--max_past_scenes", type=int, default=None,
                        help="현재 KeyScene 기준 최근 N개 KeyPoint 내의 Scene만 Source로 사용 (기본값: 제한 없음)")
    parser.add_argument("--continuous", action="store_true",
                        help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")

    args, client = init_pipeline(parser.parse_args())

    # 입력 파일 확인
    if not check_input_file(args.input_file, hint="먼저 generate_voice_hint.py를 실행하세요."):
        return

    # Keypoint 로드 (scene_idx → start/end_time 매핑용)
    keypoints_by_content = load_keypoints_by_content(args.keypoints_file)
    if not keypoints_by_content:
        print(f"Error: {args.keypoints_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

    # Gen configs
    gen_configs = make_vh_gen_config(thinking_level=args.vh_response_thinking_level)

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    print_pipeline_banner("VH Response 생성 파이프라인을 시작합니다.")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다.")
    if args.max_past_scenes:
        print(f"[Window] max_past_scenes={args.max_past_scenes} 설정: 현재 KeyScene 기준 최근 {args.max_past_scenes}개 KeyPoint 내 Scene만 Source로 사용합니다.")

    file_write_lock = threading.Lock()

    try:
        while True:
            # 진행 현황 로드
            completed_pairs = _load_completed_pairs(args.output_file)

            # voice_hint.jsonl에서 kss 모드 레코드만 수집
            # 포맷: {content_id, scene_idx, mode, queries: [...], start_time, end_time, ...}
            kss_records = []
            for rec in load_jsonl(args.input_file):
                if rec.get("pipeline_done"):
                    continue
                if rec.get("mode") != "kss":
                    continue
                c_id    = rec.get("content_id")
                s_idx   = rec.get("scene_idx")
                queries = rec.get("queries", [])
                if c_id and s_idx is not None and queries:
                    kss_records.append(rec)

            if not kss_records and not args.continuous:
                print(f"Error: {args.input_file} 에 KSS 모드 데이터가 없습니다.")
                return

            # pending 작업 계산
            pending = []
            for rec in kss_records:
                c_id  = rec["content_id"]
                s_idx = rec["scene_idx"]
                for q_text in rec.get("queries", []):
                    if (c_id, q_text) not in completed_pairs:
                        pending.append({
                            "content_id": c_id,
                            "scene_idx":  s_idx,
                            "query":      q_text,
                        })

            if pending:
                print(f"\n[TODO] 처리할 Query: {len(pending)}개")
                for item in pending:
                    print(f"  - [{item['content_id']}] Scene {item['scene_idx']} | \"{item['query'][:50]}\"")
                print("-" * 60)

            new_data_processed = False

            # content_id별로 그룹핑하여 순차 처리
            pending_by_content = {}
            for item in pending:
                pending_by_content.setdefault(item["content_id"], []).append(item)

            for content_id, items in pending_by_content.items():
                print(f"\nProcessing Content: '{content_id}' ({len(items)}개 Query)")

                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    continue

                preload_content_metadata(args.gs_bucket_name, content_id)
                keypoints = keypoints_by_content.get(content_id, [])

                for item in items:
                    scene_idx = item["scene_idx"]
                    query     = item["query"]

                    # 이미 완료된 경우 skip (루프 내에서 추가될 수 있으므로 재확인)
                    if (content_id, query) in completed_pairs:
                        continue

                    print(f"\n  [Scene {scene_idx}] Query: \"{query[:60]}\"")

                    # Source 빌드 (4개 모드 모두)
                    _MODES = ("img_desc", "mm_desc", "video", "raw")
                    sources = {}
                    try:
                        for mode in _MODES:
                            sources[mode] = _build_source(
                                args.gs_bucket_name, content_id,
                                scene_idx, keypoints, mode,
                                max_past_scenes=args.max_past_scenes,
                            )
                    except Exception as e:
                        print(f"  [ERROR] Source 빌드 실패 (Scene {scene_idx}): {e}")
                        continue

                    # 4개 모드 병렬 Response 생성
                    answers = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {
                            executor.submit(
                                _generate_for_mode,
                                client, args.vh_response_model, gen_configs,
                                sources[mode], mode, query, scene_idx,
                            ): mode
                            for mode in _MODES
                        }
                        for future in concurrent.futures.as_completed(futures):
                            mode, answer = future.result()
                            answers[mode] = answer
                            status = "OK" if not answer.startswith("Error") else "ERROR"
                            print(f"    [{mode}] {status}")

                    # 저장
                    record = {
                        "content_id": content_id,
                        "scene_idx":  scene_idx,
                        "query":      query,
                        "answers":    {m: answers[m] for m in _MODES if m in answers},
                    }
                    append_jsonl(args.output_file, record, lock=file_write_lock)
                    completed_pairs.add((content_id, query))
                    print(f"  -> Scene {scene_idx} 저장 완료")

                new_data_processed = True
                print(f"\n[OK] '{content_id}' 완료")

            if not args.continuous:
                break
            if not new_data_processed:
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        import os as _os
        _os.exit(1)

    print_pipeline_done(args.output_file)


if __name__ == "__main__":
    main()
