import os
import time
import concurrent.futures
import argparse
import json
from utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_file, process_gcs_file_by_scene_idx, check_gcs_files_exist,
    parse_json_response, parse_duration_to_times,
    _retry_api_call, load_scenes,
    ensure_output_dir, load_processed_content_ids,
    preload_content_metadata, init_pipeline, append_jsonl,
    check_input_file, print_pipeline_banner, print_pipeline_done,
)

# ============================================================
# 프롬프트 정의
# ============================================================

# ---- 공통 메타데이터 설명 블록 (재사용) ----
_METADATA_FIELD_DESC = """\
[Reference 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- speech: 등장인물들의 대사
- texts: 화면 속 자막, 간판 정보 등
- sounds: 환경음 및 효과음"""

_KEYPOINT_CRITERIA = """\
[Keypoint Scene 선별 기준]
시청자가 해당 시점까지 영상을 보다가 멈추고 궁금해할 만한 순간을 고르세요:
1. 극적인 전환점 또는 갑작스러운 상황 변화
2. 인물이 새로운 행동을 시작하거나, 중요한 결정을 내리는 순간
3. 시각적으로 인상적이거나 화면에 새로운 정보가 등장하는 순간
4. 대화 중 호기심을 자극하는 발언이나 사건이 발생하는 순간"""

# ---- (B) Stage 1: 세그먼트별 Candidate 생성용 ----
_CANDIDATE_SYSTEM_PROMPT = f"""\
당신은 영상 콘텐츠의 **일부 구간**을 분석하여 시청자가 보는 도중 자연스럽게 
궁금해할 만한 '핵심 씬 후보(Candidate)'를 식별하는 전문가입니다.

사용자는 전체 영상 중 **해당 구간의 Reference 메타데이터(JSONL)**와 **특정 구간의 비디오 프레임**을 차례로 제공합니다.
※ 전체 영상이 아닌 일부 구간만 제공되므로, 제공된 구간 내에서만 판단하세요.

{_METADATA_FIELD_DESC}

{_KEYPOINT_CRITERIA}

[주의사항]
- 제공된 **Scene List** 중에서 가장 적합한 Scene을 선택하세요.
- 사용자 요청 개수에 맞춰서, 기준에 부합하는 가장 중요한 후보를 선택하세요. (만약 개수 제한이 명시되지 않았다면 기준에 부합하는 모든 후보를 선택하세요)
- 각 후보에 대해 아래 순서대로 필드를 포함하세요:
  - rationale: 해당 Scene을 선택한 구체적 이유 (무엇이 일어났는지, 왜 시청자가 궁금해할지 2~3문장으로 상세히 기술)
  - category: 선별 기준 카테고리 ("전환점", "새로운행동", "시각적임팩트", "호기심발언" 중 택 1)
  - impact: 시청자 호기심 유발 강도 (1~5, 5가 가장 강렬)
  - scene_idx: 최종 판단한 Scene 인덱스
- 반드시 아래 JSON 배열 형식으로만 출력하세요 (다른 설명 추가 금지)

[출력 형식 예시]
[
    {{"rationale": "주인공이 처음 시장에 도착하여 놀라운 표정을 짓는다. 익숙한 장소지만 새로운 발견이 있음을 암시하며 시청자의 기대감을 높인다.", "category": "전환점", "impact": 4, "scene_idx": 3}},
    {{"rationale": "나물에 대한 독특한 비유를 사용하며 전문 지식을 드러낸다. 일반적이지 않은 관점이 시청자의 호기심을 자극한다.", "category": "호기심발언", "impact": 3, "scene_idx": 7}}
]"""


# ============================================================
# 분할 유틸리티
# ============================================================

# 세그먼트 분할 임계값
_SPLIT_THRESHOLD_2SEG = 16   # 이 값 이상이면 2등분
_SPLIT_THRESHOLD_3SEG = 33   # 이 값 이상이면 3등분
_SPLIT_THRESHOLD_4SEG = 64   # 이 값 이상이면 4등분


def split_scenes_into_segments(ref_scenes, num_segments=3):
    """Scene 리스트를 num_segments개의 균등한 세그먼트로 분할합니다.

    Returns:
        list of list: 각 세그먼트에 해당하는 Scene 딕셔너리 리스트
    """
    n = len(ref_scenes)
    seg_size = n // num_segments
    remainder = n % num_segments
    segments = []
    start = 0
    for i in range(num_segments):
        end = start + seg_size + (1 if i < remainder else 0)
        segments.append(ref_scenes[start:end])
        start = end
    return segments


def _get_scene_times(scene):
    """Scene 딕셔너리에서 (start_time, end_time)을 추출합니다.
    start_time/end_time 필드가 있으면 직접 사용하고, 없으면 duration을 파싱합니다."""
    if "start_time" in scene and "end_time" in scene:
        return float(scene["start_time"]), float(scene["end_time"])
    duration = scene.get("duration")
    if duration:
        return parse_duration_to_times(duration)
    return 0.0, 0.0


def build_scene_list_text(scenes):
    """Scene 리스트를 프롬프트용 텍스트로 변환합니다."""
    lines = []
    for s in scenes:
        idx = s.get("scene_idx", "?")
        st, et = _get_scene_times(s)
        sp = s.get("speech", "")
        lines.append(f"- Scene {idx}: {st:.1f}s ~ {et:.1f}s | {sp}")
    return "\n".join(lines)


# ============================================================
# 모델 초기화
# ============================================================

def make_candidate_config(thinking_level=None):
    """Stage 1: 세그먼트별 Candidate 생성용 config"""
    return make_generate_config(system_instruction=_CANDIDATE_SYSTEM_PROMPT, thinking_level=thinking_level)


def generate_candidates_for_segment(client, model_name, candidate_config, video_part, ref_part, scene_list_text, seg_label, num_pick=None):
    """하나의 세그먼트에서 Candidate를 생성합니다."""
    
    if num_pick is not None:
        pick_instruction = f"최대 {num_pick}개 골라내세요."
    else:
        pick_instruction = "개수 제한 없이 모두 골라내세요."
        
    prompt = (
        f"이 영상 구간({seg_label})의 Reference 메타데이터 및 비디오와 "
        "아래 Scene List를 분석하여, "
        "시청자가 영상을 보는 도중 자연스럽게 궁금해할 만한 핵심 전환점/사건 Scene 후보를 "
        f"{pick_instruction}\n\n"
        f"[Scene List]\n{scene_list_text}\n\n"
        "반드시 지정된 JSON 배열 형식으로만 출력하세요 (rationale과 scene_idx 필수)."
    )
    return _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=[ref_part, video_part, prompt], config=candidate_config
        ).text,
        label=f"Candidate 생성 ({seg_label})",
    )


# ============================================================
# Candidate → Keypoint 매핑 헬퍼
# ============================================================

def resolve_keypoints(raw_list, ref_scenes):
    """LLM 응답의 raw scene_idx 리스트를 start_time/end_time이 포함된 keypoint로 변환하고 중복을 제거합니다."""
    merged_map = {}
    for rk in raw_list:
        s_idx = rk.get("scene_idx")
        if s_idx in merged_map:
            existing_rationale = merged_map[s_idx].get("rationale", "").strip()
            new_rationale = rk.get("rationale", "").strip()
            # 이유가 비어있지 않고, 기존 내용에 완벽히 포함되지 않은 경우에만 덧붙임
            if new_rationale and new_rationale not in existing_rationale:
                if existing_rationale:
                    merged_map[s_idx]["rationale"] = existing_rationale + " / " + new_rationale
                else:
                    merged_map[s_idx]["rationale"] = new_rationale
        else:
            merged_map[s_idx] = rk.copy()
            
    keypoints = []
    for s_idx, rk in merged_map.items():
        target = next((s for s in ref_scenes if s.get("scene_idx") == s_idx), None)
        if target:
            st, et = _get_scene_times(target)
            keypoints.append({
                "scene_idx": s_idx,
                "start_time": st,
                "end_time": et,
                "rationale": rk.get("rationale", ""),
                "category": rk.get("category", ""),
                "impact": rk.get("impact", "")
            })
            
    # 시간 순(scene_idx)으로 정렬
    keypoints.sort(key=lambda x: x["scene_idx"])
    return keypoints


# ============================================================
# main
# ============================================================

def main():
    parser = get_common_argparser(description="영상 콘텐츠에서 Keypoint Scene을 식별하고 저장합니다.")
    parser.add_argument("--input_file", default="content_list.json", help="입력 JSON 파일 경로 (content_id 리스트)")
    parser.add_argument("--output_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 저장 경로")
    
    args, client = init_pipeline(parser.parse_args())
    candidate_config = make_candidate_config(thinking_level=args.keypoint_thinking_level)

    if not check_input_file(args.input_file):
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        input_list = json.load(f)

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    # 기처리분 건너뛰기
    processed_ids = load_processed_content_ids(args.output_file)
    if processed_ids:
        print(f"[{len(processed_ids)}] 개의 콘텐츠가 이미 처리되어 건너뜁니다.")

    print_pipeline_banner("Keypoint Scene 식별 파이프라인을 시작합니다.")

    # ---- 사전 스캔: content_id별 Scene 수 출력 ----
    print("\n[Pre-scan] content_id별 Scene 수 조회 중...")
    pending_items = []
    for item in input_list:
        content_id = item if isinstance(item, str) else item.get("content_id")
        if not content_id or content_id in processed_ids:
            continue
        pending_items.append(content_id)

    if pending_items:
        print(f"{'─'*55}")
        print(f"  {'content_id':<35} {'Scenes':>6}  경로")
        print(f"{'─'*55}")
        for cid in pending_items:
            try:
                preload_content_metadata(args.gs_bucket_name, cid)
                scenes = load_scenes(args.gs_bucket_name, cid, mode="ref")
                n = len(scenes)
                if n <= 8:
                    route = "전체 사용"
                elif n < _SPLIT_THRESHOLD_2SEG:
                    route = "분할없이 8개"
                elif n < _SPLIT_THRESHOLD_3SEG:
                    route = "2등분 (4×2=8)"
                elif n < _SPLIT_THRESHOLD_4SEG:
                    route = "3등분 (4×3=12)"
                else:
                    route = "4등분 (4×4=16)"
                print(f"  {cid:<35} {n:>6}  {route}")
            except Exception as e:
                print(f"  {cid:<35}  {'ERROR':>6}  {e}")
        print(f"{'─'*55}")
        print(f"  처리 대상: {len(pending_items)}개 콘텐츠\n")
    else:
        print("  처리할 콘텐츠가 없습니다.\n")

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

            # ---- JSONL 로드 & Scene List 구성 ----
            preload_content_metadata(args.gs_bucket_name, content_id)
            ref_scenes = load_scenes(args.gs_bucket_name, content_id, mode="ref")
            total_scenes = len(ref_scenes)
            full_scene_list_text = build_scene_list_text(ref_scenes)

            print(f"\n[Info] 전체 Scene 수: {total_scenes}")

            # ======================================================
            # 경로 A: Scene ≤ 8 → LLM 건너뛰고 전체 사용
            # ======================================================
            if total_scenes <= 8:
                print(f"  -> Scene 수가 8개 이하이므로 전체 Scene을 Keypoint로 자동 사용합니다.")
                keypoints = []
                for s in ref_scenes:
                    if s.get("scene_idx") is None:
                        continue
                    st, et = _get_scene_times(s)
                    keypoints.append({
                        "scene_idx": s.get("scene_idx"),
                        "start_time": st,
                        "end_time": et,
                        "rationale": "",
                        "category": "",
                        "impact": ""
                    })

            # ======================================================
            # 경로 B: 9~15 Scene → 분할 없이 전체에서 8개 판별
            # ======================================================
            elif total_scenes < _SPLIT_THRESHOLD_2SEG:
                num_pick = 8
                print(f"  -> Scene 수가 9~{_SPLIT_THRESHOLD_2SEG-1}개이므로 분할 없이 {num_pick}개의 Keypoint를 판별합니다.")

                video_part = process_gcs_file(args.gs_bucket_name, content_id, "video")
                ref_part = process_gcs_file(args.gs_bucket_name, content_id, "ref")

                print(f"\n[Candidate 생성] 전체 영상에서 {num_pick}개 추출 중... ({args.keypoint_model})")
                cand_text = generate_candidates_for_segment(
                    client, args.keypoint_model, candidate_config,
                    video_part, ref_part,
                    full_scene_list_text, "전체", num_pick=num_pick
                )
                raw_keypoints = parse_json_response(cand_text)[:num_pick]
                keypoints = resolve_keypoints(raw_keypoints, ref_scenes)

            # ======================================================
            # 경로 C: 16+ Scene → 세그먼트 분할 후 각 4개씩 판별
            # ======================================================
            else:
                picks_per_seg = 4
                if total_scenes < _SPLIT_THRESHOLD_3SEG:
                    num_segments = 2
                    print(f"  -> Scene 수가 {_SPLIT_THRESHOLD_2SEG}~{_SPLIT_THRESHOLD_3SEG-1}개이므로 {num_segments}등분하여 각각 {picks_per_seg}개의 후보를 추출, 총 {num_segments*picks_per_seg}개의 Keypoint를 결합합니다.")
                elif total_scenes < _SPLIT_THRESHOLD_4SEG:
                    num_segments = 3
                    print(f"  -> Scene 수가 {_SPLIT_THRESHOLD_3SEG}~{_SPLIT_THRESHOLD_4SEG-1}개이므로 {num_segments}등분하여 각각 {picks_per_seg}개의 후보를 추출, 총 {num_segments*picks_per_seg}개의 Keypoint를 결합합니다.")
                else:
                    num_segments = 4
                    print(f"  -> Scene 수가 {_SPLIT_THRESHOLD_4SEG}개 이상이므로 {num_segments}등분하여 각각 {picks_per_seg}개의 후보를 추출, 총 {num_segments*picks_per_seg}개의 Keypoint를 결합합니다.")

                max_keypoints = num_segments * picks_per_seg

                segments = split_scenes_into_segments(ref_scenes, num_segments)
                for si, seg in enumerate(segments):
                    first_idx = seg[0].get("scene_idx", "?")
                    last_idx = seg[-1].get("scene_idx", "?")
                    seg_start, _ = _get_scene_times(seg[0])
                    _, seg_end = _get_scene_times(seg[-1])
                    print(f"     세그먼트 {si+1}: Scene {first_idx}~{last_idx} "
                          f"({seg_start:.1f}s ~ {seg_end:.1f}s, {len(seg)}개)")

                # 세그먼트별 Candidate 생성 (병렬)
                print(f"\n[Candidate 생성] 세그먼트별 병렬 생성 중... ({args.keypoint_model})")

                def _run_segment_candidate(si_seg):
                    """단일 세그먼트에 대한 Candidate 생성 작업 (스레드에서 실행)."""
                    si, seg = si_seg
                    seg_label = f"세그먼트 {si+1}/{num_segments}"
                    seg_start_time, _ = _get_scene_times(seg[0])
                    _, seg_end_time = _get_scene_times(seg[-1])
                    seg_scene_text = build_scene_list_text(seg)

                    print(f"  [{seg_label}] 시작 - "
                          f"Scene {seg[0].get('scene_idx')}~{seg[-1].get('scene_idx')} "
                          f"({seg_start_time:.1f}s ~ {seg_end_time:.1f}s)")

                    # 각 스레드마다 독립 config 사용 (thread-safe)
                    seg_cand_config = make_candidate_config(thinking_level=args.keypoint_thinking_level)

                    seg_start_idx = seg[0].get("scene_idx")
                    seg_end_idx = seg[-1].get("scene_idx")

                    video_part = process_gcs_file_by_scene_idx(
                        args.gs_bucket_name, content_id, "video",
                        seg_start_idx, seg_end_idx
                    )
                    ref_part = process_gcs_file_by_scene_idx(
                        args.gs_bucket_name, content_id, "ref",
                        seg_start_idx, seg_end_idx
                    )

                    cand_text = generate_candidates_for_segment(
                        client, args.keypoint_model, seg_cand_config,
                        video_part, ref_part,
                        seg_scene_text, seg_label, num_pick=picks_per_seg
                    )
                    seg_candidates = parse_json_response(cand_text)
                    for c in seg_candidates:
                        c["segment"] = si + 1

                    print(f"  [{seg_label}] 완료 - {len(seg_candidates)}개 Candidate 생성")
                    return si, seg_candidates

                all_candidates = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_segments) as executor:
                    futures = {executor.submit(_run_segment_candidate, (si, seg)): si
                               for si, seg in enumerate(segments)}

                    results = {}  # si -> seg_candidates
                    for future in concurrent.futures.as_completed(futures):
                        si = futures[future]
                        seg_label = f"세그먼트 {si+1}/{num_segments}"
                        try:
                            seg_idx, seg_candidates = future.result()
                            results[seg_idx] = seg_candidates
                        except Exception as e:
                            print(f"    [ERROR] {seg_label} Candidate 생성 실패: {e}")

                # segment 순서대로 정렬하여 취합
                for si in sorted(results):
                    all_candidates.extend(results[si])

                if not all_candidates:
                    print(f"  [ERROR] 모든 세그먼트에서 Candidate 생성 실패")
                    continue

                print(f"  -> 총 {len(all_candidates)}개 Candidate 수집 완료")

                # 단순 결합: 세그먼트별 추출 결과를 최대 max_keypoints개까지 병합
                raw_keypoints = all_candidates[:max_keypoints]
                keypoints = resolve_keypoints(raw_keypoints, ref_scenes)

            # ---- 결과 출력 ----
            print(f"\n총 {len(keypoints)}개의 Keypoint가 식별되었습니다:")
            for i, kp in enumerate(keypoints, 0):
                rationale_str = f" | {kp['rationale']}" if kp.get("rationale") else ""
                print(f"  {i:2d}. [Scene {kp['scene_idx']:2d}] {kp['start_time']:.1f}s ~ {kp['end_time']:.1f}s{rationale_str}")

            # ---- 저장 ----
            kp_record = {
                "content_id": content_id,
                "keypoints": keypoints
            }
            append_jsonl(args.output_file, kp_record)

            processed_ids.add(content_id)
            print(f"[OK] '{content_id}' - {len(keypoints)}개 Keypoint 저장 완료: {args.output_file}")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    print_pipeline_done(args.output_file)


if __name__ == "__main__":
    main()
