import os
import time
import argparse
import json
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_file_by_scene_idx, check_gcs_files_exist,
    _retry_api_call,
    ensure_output_dir, load_processed_pairs,
    preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
    sort_and_validate_jsonl,
    load_keypoints_by_content, check_input_file,
    print_pipeline_banner, print_pipeline_done,
)

# ───────────────────────────────────────────────
# [Session 1] 과거 장면 요약 전용 프롬프트 (텍스트 only)
# ───────────────────────────────────────────────

_PAST_SUMMARY_PROMPT = """당신은 영상 콘텐츠의 맥락을 완벽히 이해하고 대본 및 상황을 파악하는 전문가입니다.
당신에게는 지금까지의 흐름을 시간 순서대로 나열한 과거 연대기인 **'과거 기록(Past History)'**이 제공됩니다.
과거 기록에는 이전 주요 장면(KeyScene)의 상세 묘사와, 주요 장면 사이 구간의 참조용 메타데이터가 시간순으로 포함되어 있습니다.
당신의 목표는 이 과거 기록을 종합하여, 지금까지 발생한 사건의 흐름을 하나의 상세한 과거 장면 요약으로 작성하는 것입니다.

[메타데이터 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- speech: 등장인물들의 대사 (영어 또는 한국어)
- texts: 화면 속 자막, 간판 정보 등

[메타데이터 해석 시 주의사항]
- 이전 주요 장면의 상세 묘사(비디오 기반으로 이미 교정된 텍스트)가 함께 제공되므로, 메타데이터와 상세 묘사 사이에 불일치가 있을 경우 **상세 묘사를 우선시** 하세요.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다. 적절히 교차 검증하여 교정하세요.
- texts: OCR 오류로 인해 화면 텍스트의 철자가 틀릴 수 있습니다. 적절히 교차 검증하여 교정하세요.

[강조 사항 및 작성 규칙]
- **대화 상세 묘사:** 인물 간의 대화는 요약에 그치지 말고, 중요한 대사일 경우 직접 인용("...")하거나 화자와 청자의 관계, 의도 및 감정 상태를 포함하여 매우 구체적으로 서술하세요.
- **키워드 및 핵심 소재:** 등장하는 중요한 물체(Object), 텍스트(Text), 환경적 특징(Background) 등 사건 전개의 실마리가 되는 키워드들은 누락하지 말고 반드시 명시하여 묘사하세요.
- 언어는 **한국어**로 작성하되, 중요한 인물 이름, 고유명사, 등장하는 중요한 텍스트 등은 원어를 병기하여 정확성을 높이세요 (예: 일각고래(narwhal), 셰즈 은데예(Chez Ndeye)).
- 텍스트의 길이 제한 없이 최대한 핵심을 상세하게 묘사하세요."""

# ───────────────────────────────────────────────
# [Session 2] 현재 장면 묘사 전용 프롬프트 (멀티모달)
# ───────────────────────────────────────────────

_CURRENT_SCENE_PROMPT = """당신은 영상 콘텐츠의 맥락을 완벽히 이해하고 대본 및 상황을 파악하는 전문가입니다.
당신에게는 지금까지의 흐름을 종합한 **'과거 장면 요약(Past Summary)'**과,
현재 장면의 **'현재 참조용 메타데이터(Current Reference Metadata)'** 및 **'현재 장면의 비디오(Current Video)'**가 순서대로 제공됩니다.
당신의 목표는 과거 장면 요약을 통해 이전 맥락을 파악한 후, 현재 장면에서 벌어지고 있는 상황을 비디오와 메타데이터를 교차 검증하여 상세하게 묘사하는 것입니다.

[메타데이터 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- speech: 등장인물들의 대사 (영어 또는 한국어)
- texts: 화면 속 자막, 간판 정보 등

[메타데이터 해석 시 주의사항]
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다. 비디오와 적절히 교차 검증하여 교정하세요.
- texts: OCR 오류로 인해 화면 텍스트의 철자가 틀릴 수 있습니다. 비디오와 적절히 교차 검증하여 교정하세요.

[강조 사항 및 작성 규칙]
- **과거 맥락 활용:** 과거 장면 요약에서 파악한 인물 관계, 갈등 구조, 목적 등을 현재 장면의 행동과 연결지어 서술하세요.
- **대화 상세 묘사:** 인물 간의 대화는 요약에 그치지 말고, 중요한 대사일 경우 직접 인용("...")하거나 화자와 청자의 관계, 의도 및 감정 상태를 포함하여 매우 구체적으로 서술하세요.
- **키워드 및 핵심 소재:** 화면에 등장하는 중요한 물체(Object), 텍스트(Text), 환경적 특징(Background) 등 사건 전개의 실마리가 되는 키워드들은 누락하지 말고 반드시 명시하여 묘사하세요.
- 언어는 **한국어**로 작성하되, 중요한 인물 이름, 고유명사, 등장하는 중요한 텍스트 등은 원어를 병기하여 정확성을 높이세요 (예: 일각고래(narwhal), 셰즈 은데예(Chez Ndeye)).
- 텍스트의 길이 제한 없이 최대한 핵심을 상세하게 묘사하세요."""


def make_past_summary_config(thinking_level=None):
    """[Session 1] 과거 장면 요약 전용 GenerateContentConfig를 반환합니다."""
    return make_generate_config(system_instruction=_PAST_SUMMARY_PROMPT, thinking_level=thinking_level)

def make_current_scene_config(thinking_level=None):
    """[Session 2] 현재 장면 묘사 전용 GenerateContentConfig를 반환합니다."""
    return make_generate_config(system_instruction=_CURRENT_SCENE_PROMPT, thinking_level=thinking_level)

def extract_current_scene_desc(text):
    """요약본 텍스트에서 '2. 현재 장면 묘사' 섹션의 내용만 파싱하여 반환합니다."""
    tag = "[2. 현재 장면 묘사]"
    if tag in text:
        return text.split(tag)[-1].strip()
    return text.strip()

def process_past_summary(client, model_name, config, past_history, end_time):
    """[Session 1] 과거 연대기 텍스트를 종합하여 '1. 과거 장면 요약'을 생성합니다."""
    contents = ["--- [Past History] ---"]
    if isinstance(past_history, list):
        contents.extend(past_history)
    else:
        contents.append(past_history)

    request_msg = (
        "시간 순서대로 제공된 [Past History] 연대기 내용을 종합하여 과거 장면 요약을 작성하세요. "
        "이전 주요 장면의 상세 묘사와 중간 구간 메타데이터를 모두 반영하여, 핵심 사건·대화·키워드를 누락 없이 상세하게 담아주세요."
    )
    contents += ["--- 요청 사항 ---", request_msg]

    t0 = time.time()
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=config
        ).text,
        label=f"[Session 1] 과거 요약 생성 (end={end_time:.1f}s)"
    )
    return text, time.time() - t0


def process_current_scene(client, model_name, config, past_summary_text, current_parts, end_time, use_ref=False):
    """[Session 2] 과거 요약 + 현재 메타/비디오로 '2. 현재 장면'을 생성합니다."""
    contents = []

    if past_summary_text is not None:
        contents += ["--- [Past Summary] ---", past_summary_text]

    if use_ref:
        contents += ["--- [Current Reference Metadata] ---", current_parts["ref"]]
    contents += ["--- [Current Video] ---", current_parts["video"]]

    if past_summary_text is not None:
        request_msg = (
            "제공된 [Past Summary]로 이전 맥락을 파악한 후, "
            "[Current Reference Metadata]의 힌트를 참고하고 [Current Video] 영상을 교차 검증하여 "
            "현재 장면 묘사를 상세하게 작성해 주세요."
        )
    else:
        request_msg = (
            "과거 정보가 없으므로, 제공된 [Current Reference Metadata]와 [Current Video] 영상을 바탕으로 "
            "현재 장면 묘사를 상세하게 작성해 주세요."
        )

    contents += ["--- 요청 사항 ---", request_msg]

    t0 = time.time()
    text = _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=config
        ).text,
        label=f"[Session 2] 현재 장면 생성 (end={end_time:.1f}s)"
    )
    return text, time.time() - t0

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint Scene 목록을 입력받아 KeyScene Summary를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary 별도 저장 경로")
    parser.add_argument("--watch", action="store_true", help="입력 파일에 새로운 데이터가 추가되는지 주기적으로 감지하고 계속 처리합니다.")
    parser.add_argument("--parallel", type=int, default=4, help="동시에 병렬 처리할 비디오(Content) 수 (기본값: 4)")

    args, client = init_pipeline(parser.parse_args())

    past_summary_config = make_past_summary_config(thinking_level=args.kss_past_summary_thinking_level)
    current_scene_config = make_current_scene_config(thinking_level=args.kss_current_scene_thinking_level)

    if not args.watch:
        if not check_input_file(args.input_file, hint="먼저 identify_keypoint.py를 실행하세요."):
            return

    # 출력 디렉토리 확인
    ensure_output_dir(args.keyscene_summary_file)

    # 기존 파일 정리: 빈 요약이나 중복 제거 후 다시 쓰기
    summary_pairs = set()
    summary_texts_by_scene = {}
    if os.path.exists(args.keyscene_summary_file):
        existing_records = load_jsonl(args.keyscene_summary_file)
        cleaned_records = {}
        for r in existing_records:
            c_id = r.get("content_id")
            s_idx = r.get("scene_idx")
            summary = r.get("summary", "")
            if c_id and s_idx is not None and summary.strip():
                # 중복이 있을 경우 가장 마지막(최신) 레코드로 덮어씀
                cleaned_records[(c_id, s_idx)] = r
        
        with open(args.keyscene_summary_file, "w", encoding="utf-8") as f:
            for r in cleaned_records.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        summary_pairs = set(cleaned_records.keys())
        summary_texts_by_scene = cleaned_records

    print_pipeline_banner("KeyScene Summary 생성 파이프라인을 시작합니다.")

    file_lock = threading.Lock()
    stop_event = threading.Event()

    def process_content(content_id, keypoints):
        if stop_event.is_set():
            return

        with file_lock:
            done_scenes = {s_idx for (c_id, s_idx) in summary_pairs if c_id == content_id}
        remaining = [kp for kp in keypoints if kp.get("scene_idx") not in done_scenes]

        if not remaining:
            print(f"\n[Skip] '{content_id}': 모든 Scene 완료")
            return
        if done_scenes:
            print(f"\n[Resume] '{content_id}': {len(done_scenes)}/{len(keypoints)}개 Scene 기완료, {len(remaining)}개 재개")

        print(f"\n{'='*50}")
        print(f"Processing Content: '{content_id}' ({len(remaining)}/{len(keypoints)}개 Keypoint)")
        print(f"{'='*50}")

        if not check_gcs_files_exist(args.gs_bucket_name, content_id):
            return

        # JSONL 메타데이터 프리로드 (캐시 워밍업)
        preload_content_metadata(args.gs_bucket_name, content_id)

        for kp in remaining:
            if stop_event.is_set():
                break

            real_idx = keypoints.index(kp)
            scene_idx = kp.get("scene_idx", real_idx)

            scene_key = (content_id, scene_idx)
            with file_lock:
                already_done = scene_key in summary_pairs
            
            if already_done:
                # 이미 동일한 Scene에 속하는 중복 Keypoint라면 무의미한 중복 API 호출을 방지하기 위해 건너뜁니다.
                continue

            start_time = float(kp.get("start_time", 0.0))
            end_time = float(kp.get("end_time", 0.0))

            print(f"[{content_id}] [{real_idx}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s]")

            def _run_keypoint():
                # ── 과거 연대기 구축 (KSS desc → Gap 순서) ──
                with file_lock:
                    past_scene_indices = sorted([s for (c, s) in summary_pairs if c == content_id and s < scene_idx])
                
                accumulated_past = []

                if past_scene_indices:
                    for i, s_idx in enumerate(past_scene_indices):
                        with file_lock:
                            record = summary_texts_by_scene.get((content_id, s_idx))
                        if not record:
                            continue

                        # 1) 첫 KSS 이전의 Gap (Scene 0 ~ s_idx-1): 완전성을 위해 수집
                        if i == 0 and s_idx > 0 and args.use_ref_for_keyscene_summary:
                            gap_ref = process_gcs_file_by_scene_idx(args.gs_bucket_name, content_id, "ref", 0, s_idx - 1)
                            if gap_ref:
                                accumulated_past.append(f"[초반 구간의 메타데이터: Scene 0 ~ Scene {s_idx - 1}]")
                                accumulated_past.append(gap_ref)

                        # 2) KSS Description (방향 지시 역할)
                        desc = extract_current_scene_desc(record.get("summary", ""))
                        accumulated_past.append(f"[이전 Scene {s_idx}의 장면 묘사]\n{desc}")

                        # 3) Gap: 이 KSS end → 다음 KSS start (또는 현재 Scene) 사이
                        if i + 1 < len(past_scene_indices):
                            next_s_idx = past_scene_indices[i + 1]
                        else:
                            next_s_idx = scene_idx  # 마지막 과거 KSS → 현재 Scene

                        gap_start_scene = s_idx + 1
                        gap_end_scene = next_s_idx - 1

                        if gap_end_scene >= gap_start_scene and args.use_ref_for_keyscene_summary:
                            gap_ref = process_gcs_file_by_scene_idx(args.gs_bucket_name, content_id, "ref", gap_start_scene, gap_end_scene)
                            if gap_ref:
                                accumulated_past.append(f"[중간 구간의 메타데이터: Scene {gap_start_scene} ~ Scene {gap_end_scene}]")
                                accumulated_past.append(gap_ref)

                else:
                    # 과거 Scene이 없는 경우: Scene 0 ~ scene_idx-1까지의 Gap 수집
                    if scene_idx > 0 and args.use_ref_for_keyscene_summary:
                        gap_ref = process_gcs_file_by_scene_idx(args.gs_bucket_name, content_id, "ref", 0, scene_idx - 1)
                        if gap_ref:
                            accumulated_past.append(f"[초반 구간의 메타데이터: Scene 0 ~ Scene {scene_idx - 1}]")
                            accumulated_past.append(gap_ref)

                # ── Phase 1: 과거 장면 요약 (텍스트 only) ──
                past_summary_text = None
                past_elapsed = 0.0
                if accumulated_past:
                    past_summary_text, past_elapsed = process_past_summary(
                        client, args.kss_past_summary_model, past_summary_config,
                        accumulated_past, start_time
                    )
                    print(f"[{content_id}] -> [Session 1] 과거 요약 완료 ({len(past_summary_text)}자, {past_elapsed:.2f}초)")

                # ── Phase 2: 현재 장면 묘사 (멀티모달) ──
                current_parts = {
                    "video": process_gcs_file_by_scene_idx(args.gs_bucket_name, content_id, "video", scene_idx, scene_idx),
                    "ref":   process_gcs_file_by_scene_idx(args.gs_bucket_name, content_id, "ref",   scene_idx, scene_idx),
                }
                current_scene_text, scene_elapsed = process_current_scene(
                    client, args.kss_current_scene_model, current_scene_config,
                    past_summary_text, current_parts, end_time, use_ref=args.use_ref_for_keyscene_summary
                )
                print(f"[{content_id}] -> [Session 2] 현재 장면 완료 ({len(current_scene_text)}자, {scene_elapsed:.2f}초)")

                # ── 최종 조합 (extract_current_scene_desc 호환성 유지) ──
                final_summary = f"[1. 과거 장면 요약]\n\n{past_summary_text or '해당 없음'}\n\n[2. 현재 장면 묘사]\n\n{current_scene_text}"
                total_elapsed = past_elapsed + scene_elapsed
                return final_summary, total_elapsed

            try:
                summary_text, summary_elapsed = _retry_api_call(
                    _run_keypoint,
                    label=f"KeyScene Summary (Scene {scene_idx})"
                )

                scene_key = (content_id, scene_idx)
                with file_lock:
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
                        summary_texts_by_scene[scene_key] = summary_record

                # print(f"\n{summary_text}")
                print(f"[{content_id}] -> [Summary] 생성 완료 ({len(summary_text)}자, {summary_elapsed:.2f}초)")

            except Exception as e:
                print(f"    [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                continue

        with file_lock:
            done_count = len({s_idx for (c_id, s_idx) in summary_pairs if c_id == content_id})
        print(f"[OK] '{content_id}' - {done_count}개 Scene 완료")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel)
    last_processed_ids = set()

    try:
        while not stop_event.is_set():
            if os.path.exists(args.input_file):
                keypoints_by_content = load_keypoints_by_content(args.input_file)
            else:
                keypoints_by_content = {}
            
            if not args.watch and not keypoints_by_content:
                print(f"Error: {args.input_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
                break

            new_contents = {c_id: k_pts for c_id, k_pts in keypoints_by_content.items() if c_id not in last_processed_ids}

            if new_contents:
                if args.watch and last_processed_ids:
                    print(f"\n[Watch] 새로운 데이터 {len(new_contents)}건이 발견되었습니다. 처리를 시작합니다...")
                
                futures = []
                for content_id, keypoints in new_contents.items():
                    futures.append(executor.submit(process_content, content_id, keypoints))
                    last_processed_ids.add(content_id)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] Content 처리 중 예외 발생: {e}")

            if not args.watch:
                # 결과 파일 정렬 및 누락 점검 (utils.py 공통 함수 사용)
                sort_and_validate_jsonl(args.keyscene_summary_file, keypoints_by_content, expected_modes=[])
                break
            
            # Watch 모드인 경우 주기적으로 확인
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        stop_event.set()
        # 아직 대기 중인 future들을 취소하고 즉시 shutdown
        executor.shutdown(wait=False, cancel_futures=True)
        os._exit(1)
    finally:
        executor.shutdown(wait=True)

    print_pipeline_done(args.keyscene_summary_file)


if __name__ == "__main__":
    main()
