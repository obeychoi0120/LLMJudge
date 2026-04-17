import os
import time
import argparse
import json
import concurrent.futures
from gemini_api_utils import (
    get_common_argparser,
    make_generate_config,
    check_gcs_files_exist,
    parse_json_response, _retry_api_call,
    ensure_output_dir, load_processed_pairs,
    preload_content_metadata, get_gcs_descriptions_by_scene_idx,
    init_pipeline, load_jsonl, append_jsonl,
)

# ───────────────────────────────────────────────
# Voice Hint 생성 모델 프롬프트
# ───────────────────────────────────────────────

_VOICE_HINT_BASE = """당신은 제공되는 메타데이터를 기반으로 스마트 TV 플랫폼에서 시청자의 리모컨 상호작용과 플랫폼 체류 시간을 극대화하는 '개인화된 예상 질문' 생성 전문가입니다.

시청자에게는 오직 현재 정보만 주어지는 것이 아닙니다. 시청자는 지금까지 시청해 온 **[이전까지의 과거 시청 맥락]**을 인지한 상태로 방금 **[현재 시청 중인 장면]**을 보았습니다.
이 두 정보를 바탕으로, TV 화면의 버튼을 눌러 답을 확인하고 싶게 만드는 매력적인 질문 3개를 생성하세요.

[입력 형식 설명]
당신에게는 영상 Scene 묘사가 다음과 같은 형식으로 제공됩니다.
- 각 Scene은 `[Scene N] 묘사` 형태로 제공됩니다. N은 영상 내 Scene 인덱스(순서)입니다.
- 묘사(description)는 소형 AI가 해당 Scene의 시각적 상황, 인물 행동, 대사, 자막 등을 종합해 자동 생성한 원시(Raw) 텍스트로, 어색한 단어나 오탈자가 포함될 수 있습니다.

[질문 생성 핵심 전략]
1. 오류 교정 및 노이즈 필터링 (최우선): 제공된 묘사는 소형 AI의 결과물이므로 음성 인식 오류나 자막 OCR 오탈자(예: '세프'->'셰프', '봄고레'->'범고래') 등 환각이 포함될 수 있습니다. 텍스트를 기계적으로 맹신하지 말고, 상식과 문맥을 바탕으로 오류를 교정한 뒤 기획해야 합니다. 최종 노출될 질문에 오탈자가 포함되어서는 절대 안 됩니다.
2. 정보 공백 파악을 통한 호기심 유도 (Hook & Curiosity): 질문의 핵심 소재(Trigger)는 반드시 [현재 시청 중인 장면]에서 '새롭게' 등장한 단서로만 한정합니다. **[이전까지의 과거 시청 맥락]에서 이미 설명되거나 유추할 수 있는 사실(예: 인물의 정체, 과거의 사건 등)을 묻는 '뒷북 질문'은 무조건 0점 처리됩니다.** 오직 방금 새롭게 발생한 의문점과 정보의 공백(미지수)만을 짚어내세요.
3. 현재 시점 몰입 (톤앤매너 준수 및 뒷북/스포일러 금지): 답변을 알려면 반드시 '이후의 영상'을 봐야만 알 수 있도록 질문을 기획하세요. 특히 질문의 어조(Tone)는 상황에 따라 묘사된 씬의 분위기(긴장감, 슬픔, 경쾌함 등)를 절대 깨지 않도록 일치시켜야 합니다. (기본적으로는 옆사람에게 툭 던지는 자연스러운 구어체를 쓰되, 상황에 맞게 톤을 조절하세요.)

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 3단계를 순서대로 작성하여 논리적으로 사고하세요.
- 1단계 (오류 교정): 제공된 묘사 텍스트에 오탈자나 명백한 AI 환각이 있다면 올바른 단어/상황으로 교정하여 명시합니다. (오류가 없으면 '특이사항 없음' 표기)
- 2단계 (과거 정보 영구 차단): 이전 과거 맥락에서 이미 밝혀진 사실(시청자가 이미 아는 내용)을 요약한 뒤, "이 정보들은 시청자가 이미 알고 있으므로 질문 소재에서 영구 제외한다"라고 명시적으로 선언하세요.
- 3단계 (신규 단서 및 질문 기획): 현재 장면에서 새롭게 포착된 단서를 바탕으로, 시청자의 어떤 호기심을 자극할 것인지 기획 의도를 설명합니다.

[출력 형식]
- 언어: 한국어 (단, 영어 콘텐츠의 고유명사는 원어 병기 허용. 예: 일각고래(Narwhal))
- 형식: 반드시 아래 JSON 형식으로 출력하세요. 다른 설명은 덧붙이지 마십시오.

[JSON 형식 예시]
{
    "rationale": "1. 오류 교정: 묘사 중 '봄고레 때'는 문맥상 '범고래 떼'의 오탈자이므로 교정함. / 2. 과거 정보 영구 차단: 혹등고래가 굶주리고 있다는 사실은 이전 장면에서 파악됨. 이 정보는 시청자가 이미 알고 있으므로 질문 소재에서 영구 제외함. / 3. 신규 단서 및 질문 기획: 현재 장면에서 갑자기 범고래 떼가 나타나는 새로운 단서 포착. 범고래의 등장 이유에 대한 궁금증을 타겟팅함.",
    "queries": [
        "방금 나타난 범고래 떼, 설마 혹등고래를 노리고 온 걸까?",
        "지금 범고래들이 내는 저 소리, 사냥 시작한다는 신호 아니야?",
        "이 지역에 범고래가 원래 이렇게 자주 나타나?"
    ]
}"""

def make_voice_hint_config(thinking_level=0):
    """Voice Hint 생성용 GenerateContentConfig를 반환합니다."""
    return make_generate_config(system_instruction=_VOICE_HINT_BASE, thinking_level=thinking_level)

def process_vh_parallel(client, vh_model_name, vh_config, past_parts, current_parts, end_time):
    """하나의 Keypoint에 대해 Voice Hint(img_desc, mm_desc 2개 모드)를 병렬로 수행합니다."""
    def generate_voice_hints(mode):
        contents = []
        past_text = past_parts[mode]
        has_past = bool(past_text.strip())

        # 샌드위치 구조 (소형 모델 앵커링용) : [현재] -> [과거 (참고)] -> [현재 (재확인)]
        contents += ["--- [현재 시청 중인 장면] 시작 ---", current_parts[mode], "--- [현재 시청 중인 장면] 끝 ---"]

        if has_past:
            contents += ["--- [이전까지의 과거 시청 맥락 (참조용)] 시작 ---", past_text, "--- [이전까지의 과거 시청 맥락 (참조용)] 끝 ---"]
            contents += ["--- [현재 시청 중인 장면 (재확인)] 시작 ---", current_parts[mode], "--- [현재 시청 중인 장면 (재확인)] 끝 ---"]
        
        contents += [
            "--- 요청 사항 ---",
            "위의 [현재 시청 중인 장면]과 [이전까지의 과거 시청 맥락]을 인지한 상태에서, 시스템 프롬프트의 [사고 과정 가이드] 3단계를 철저히 준수하여 질문 3개를 JSON 형식으로 생성하세요. 과거 맥락에서 이미 아는 내용으로 질문하는 것은 피하고, 오직 방금 본 [현재 시청 중인 장면] 에 '새롭게' 등장한 단서에 집중하세요." 
            if has_past else "제공된 [현재 시청 중인 장면] 을 바탕으로 시스템 프롬프트의 [사고 과정 가이드] 지침에 따라 매력적인 질문 3개를 JSON 형식으로 생성하세요."
        ]
        
        t0 = time.time()
        text = _retry_api_call(
            lambda: client.models.generate_content(
                model=vh_model_name, contents=contents, config=vh_config
            ).text,
            label=f"Voice Hint({mode}) 생성 (end={end_time:.1f}s)"
        )
        
        parsed = parse_json_response(text)
        rationale = ""
        
        if isinstance(parsed, dict):
            queries = parsed.get("queries", [])
            rationale = parsed.get("rationale", "")
        elif isinstance(parsed, list):
            queries = parsed
        else:
            queries = []
            
        return {"queries": queries[:3], "rationale": rationale}, time.time() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_vh_img  = executor.submit(generate_voice_hints, "img_desc")
        f_vh_mm  = executor.submit(generate_voice_hints, "mm_desc")
        
        vh_res_img, elapsed_img = f_vh_img.result()
        vh_res_mm, elapsed_mm = f_vh_mm.result()

    return {
        "img_desc": vh_res_img["queries"], 
        "mm_desc": vh_res_mm["queries"],
        "rationales": {
            "img_desc": vh_res_img["rationale"],
            "mm_desc": vh_res_mm["rationale"]
        }
    }, {"img_desc": elapsed_img, "mm_desc": elapsed_mm}

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint Scene 목록을 입력받아 Voice Hint를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--output_file", default="assets/voice_hint.jsonl", help="Voice Hint 목록 저장 경로")

    args, client = init_pipeline(parser.parse_args())

    vh_config = make_voice_hint_config(thinking_level=args.vh_thinking_level)

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
                rationale = kp.get("rationale", "")

                print(f"[{real_idx}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s]")
                
                # 로깅용 Desc 텍스트 추출 및 즉시 출력 (description-only 형식)
                img_desc_text = get_gcs_descriptions_by_scene_idx(args.gs_bucket_name, content_id, "img_desc", scene_idx, scene_idx)
                print(f"\n[Desc (img_desc)]\n{img_desc_text}\n")

                mm_desc_text = get_gcs_descriptions_by_scene_idx(args.gs_bucket_name, content_id, "mm_desc", scene_idx, scene_idx)
                print(f"\n[Desc (mm_desc)]\n{mm_desc_text}\n")

                def _run_keypoint():
                    # 과거 N개 구역(Scene)을 위한 scene_idx 계산 (Sliding Window)
                    past_start_kp_idx = max(0, real_idx - args.vh_gen_past_scenes_size)
                    past_start_scene_idx = keypoints[past_start_kp_idx].get("scene_idx", 0)
                    past_end_scene_idx = scene_idx - 1  # 현재 Scene 직전까지

                    if past_end_scene_idx >= past_start_scene_idx:
                        past_parts = {
                            "img_desc": get_gcs_descriptions_by_scene_idx(args.gs_bucket_name, content_id, "img_desc", past_start_scene_idx, past_end_scene_idx),
                            "mm_desc":  get_gcs_descriptions_by_scene_idx(args.gs_bucket_name, content_id, "mm_desc",  past_start_scene_idx, past_end_scene_idx),
                        }
                    else:
                        past_parts = {"img_desc": "", "mm_desc": ""}

                    current_parts = {
                        "img_desc": img_desc_text,
                        "mm_desc": mm_desc_text,
                    }
                    return process_vh_parallel(client, args.vh_gen_model, vh_config, past_parts, current_parts, end_time)

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
                                {"mode": "img_desc", "queries": vh_dict["img_desc"], "rationale": vh_dict.get("rationales", {}).get("img_desc", "")},
                                {"mode": "mm_desc", "queries": vh_dict["mm_desc"], "rationale": vh_dict.get("rationales", {}).get("mm_desc", "")}
                            ],
                        }
                        append_jsonl(args.output_file, scene_record)
                        vh_pairs.add(scene_key)

                    print(f"-> [VH - img_desc] ({vh_elapsed_dict['img_desc']:.2f}초)")
                    if vh_dict.get("rationales", {}).get("img_desc"):
                        print(f"[Rationale]: {vh_dict['rationales']['img_desc']}")
                    for qi, q in enumerate(vh_dict["img_desc"], 1):
                        print(f"    {qi}. {q}")

                    print(f"\n-> [VH - mm_desc] ({vh_elapsed_dict['mm_desc']:.2f}초)")
                    if vh_dict.get("rationales", {}).get("mm_desc"):
                        print(f"[Rationale]: {vh_dict['rationales']['mm_desc']}")
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
