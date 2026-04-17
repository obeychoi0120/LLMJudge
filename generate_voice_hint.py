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

_VOICE_HINT_BASE_DESC = """당신은 제공되는 메타데이터를 기반으로 스마트 TV 플랫폼에서 시청자의 리모컨 상호작용과 플랫폼 체류 시간을 극대화하는 '개인화된 예상 질문' 생성 전문가입니다.

시청자에게는 오직 현재 정보만 주어지는 것이 아닙니다. 시청자는 지금까지 시청해 온 **[이전까지의 과거 시청 맥락]**을 인지한 상태로 방금 **[현재 시청 중인 장면]**을 보았습니다.
이 두 정보를 바탕으로, TV 화면의 버튼을 눌러 답을 확인하고 싶게 만드는 매력적인 질문 3개를 생성하세요.

[입력 형식 설명]
당신에게는 영상 Scene 묘사가 다음과 같은 형식으로 제공됩니다.
- 각 Scene은 `[Scene N] 묘사` 형태로 제공됩니다. N은 영상 내 Scene 인덱스(순서)입니다.
- 묘사(description)는 소형 AI가 해당 Scene의 시각적 상황, 인물 행동, 대사, 자막 등을 종합해 자동 생성한 원시(Raw) 텍스트로, 어색한 단어나 오탈자가 포함될 수 있습니다.

[질문 생성 핵심 전략]
1. 오류 교정 및 노이즈 필터링 (최우선): 제공된 묘사는 소형 AI의 결과물이므로 음성 인식 오류나 자막 OCR 오탈자(예: '세프'->'셰프', '봄고레'->'범고래') 등 환각이 포함될 수 있습니다. 텍스트를 기계적으로 맹신하지 말고, 상식과 문맥을 바탕으로 오류를 교정한 뒤 기획해야 합니다. 최종 노출될 질문에 오탈자가 포함되어서는 절대 안 됩니다.
2. 정보 공백 파악을 통한 호기심 유도 (Hook & Curiosity): 질문의 핵심 소재(Trigger)는 반드시 [현재 시청 중인 장면]에서 '새롭게' 등장한 단서로만 한정합니다. **[이전까지의 과거 시청 맥락]에서 이미 설명되거나 유추할 수 있는 사실(예: 인물의 정체, 과거의 사건 등)을 묻는 '뒷북 질문'은 무조건 0점 처리됩니다.** 오직 방금 새롭게 발생한 의문점과 정보의 공백(미지수)만을 짚어내세요.
3. 현재 시점 몰입 (미래 추측 금지 및 톤앤매너 준수): 질문은 오직 현재 즉시 정보를 제공할 수 있는 [현재 시청 중인 장면] 속 사물, 인물의 정체, 배경지식, 상황의 숨은 의미로 한정해야 합니다. **앞으로 전개될 스토리나 미래의 결과(예: "과연 어떻게 될까요?", "어떤 평가를 받을까요?")를 묻는 질문은 현재 시스템이 미리 답변해 줄 수 없으므로 철저히 0점 처리됩니다.** 어조(Tone)는 묘사된 씬의 분위기를 절대 깨지 않도록 '뛰어난 AI 비서'의 정중하고 흥미로운 존댓말로 작성하세요.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하여 논리적으로 사고하세요.
- 1단계 (오류 교정): 제공된 묘사 텍스트에 오탈자나 명백한 AI 환각이 있다면 올바른 단어/상황으로 교정하여 명시합니다. (오류가 없으면 '특이사항 없음' 표기)
- 2단계 (과거 정보 영구 차단): 이전 과거 맥락에서 이미 밝혀진 사실(시청자가 이미 아는 내용)을 요약한 뒤, "이 정보들은 시청자가 이미 알고 있으므로 질문 소재에서 영구 제외한다"라고 명시적으로 선언하세요.
- 3단계 (미래 추측 차단): "이후 벌어질 결과나 스토리의 전개를 추측하는 미래 지향적 질문은 시스템이 당장 답할 수 없어 오답 처리되므로 절대 생성하지 않겠다"라고 단호하게 선언하세요.
- 4단계 (신규 단서 및 질문 기획): 현재 장면에서 새롭게 포착된 단서에 집중하여, 현재 시점에서 당장 답변할 수 있는 흥미로운 정보(인물/사물의 정체, 배경지식, 숨은 의미 등)를 묻는 기획을 설명합니다.

[출력 형식]
- 언어: 한국어 (단, 영어 콘텐츠의 고유명사는 원어 병기 허용. 예: 일각고래(Narwhal))
- 형식: 반드시 아래 JSON 형식으로 출력하세요. 다른 설명은 덧붙이지 마십시오.

[JSON 형식 예시]
{
    "rationale": "1. 오류 교정: 묘사 중 '봄고레 때'는 문맥상 '범고래 떼'의 오탈자이므로 교정함. / 2. 과거 정보 영구 차단: 혹등고래가 굶주리고 있다는 사실은 이전 장면에서 파악됨. 질문 소재에서 영구 제외함. / 3. 미래 추측 차단: '범고래 사냥 결과' 등 미래 서사는 답할 수 없으므로 철저히 배제함. / 4. 질문 기획: 대신 시청자가 현재 접한 '범고래 떼의 소리'나 '생태계 특성' 등 당장 답변 가능한 배경지식으로 유도함.",
    "queries": [
        "방금 나타난 범고래 떼는 혹등고래를 노리고 온 것일까요?",
        "지금 범고래들이 내는 저 소리는 사냥을 시작한다는 신호일까요?",
        "이 지역에 범고래가 원래 자주 나타나는 편일까요?"
    ]
}"""

_VOICE_HINT_BASE_KSS = """당신은 제공되는 영상 요약을 기반으로 스마트 TV 플랫폼에서 시청자의 리모컨 상호작용과 플랫폼 체류 시간을 극대화하는 '개인화된 예상 질문' 생성 전문가입니다.

시청자에게는 오직 현재 정보만 주어지는 것이 아닙니다. 시청자는 지금까지 시청해 온 과거 맥락을 인지한 상태로 방금 현재 장면을 보았습니다.
이 두 정보의 흐름을 파악하여, TV 화면의 버튼을 눌러 답을 확인하고 싶게 만드는 매력적인 질문 3개를 생성하세요.

[입력 형식 설명]
당신에게는 해당 장면의 맥락을 완벽하게 요약한 **[KeyScene Summary]** 형태의 텍스트가 제공됩니다.
- 이 요약본에는 [1. 과거 장면 요약]과 [2. 현재 장면 묘사]가 포함되어 있습니다.

[질문 생성 핵심 전략]
1. 정보 공백 파악을 통한 호기심 유도 (Hook & Curiosity): 질문의 핵심 소재(Trigger)는 반드시 [2. 현재 장면 묘사]에서 '새롭게' 등장한 단서로만 한정합니다. **[1. 과거 장면 요약]에서 이미 설명되거나 유추할 수 있는 사실(예: 인물의 정체, 과거의 사건 등)을 묻는 '뒷북 질문'은 무조건 0점 처리됩니다.** 오직 방금 새롭게 발생한 의문점과 정보의 공백(미지수)만을 짚어내세요.
2. 현재 시점 몰입 (미래 추측 금지 및 톤앤매너 준수): 질문은 오직 현재 즉시 정보를 제공할 수 있는 [현재 장면] 속 사물, 인물의 정체, 배경지식, 상황의 숨은 의미로 한정해야 합니다. **앞으로 전개될 스토리나 미래의 결과(예: "과연 어떻게 될까요?", "어떤 평가를 받을까요?")를 묻는 질문은 현재 시스템이 미리 답변해 줄 수 없으므로 철저히 0점 처리됩니다.** 어조(Tone)는 KSS에 묘사된 씬의 분위기를 절대 깨지 않도록 '뛰어난 AI 비서'의 정중하고 흥미로운 존댓말로 작성하세요.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 3단계를 순서대로 작성하여 논리적으로 사고하세요.
- 1단계 (과거 정보 영구 차단): [1. 과거 장면 요약]에서 이미 밝혀지거나 설명된 사실(시청자가 이미 아는 내용)을 요약한 뒤, "이 정보들은 시청자가 이미 알고 있으므로 질문 소재에서 영구 제외한다"라고 명시적으로 선언하세요.
- 2단계 (미래 추측 차단): "이후 벌어질 결과나 스토리의 전개를 추측하는 미래 지향적 질문은 시스템이 당장 답할 수 없어 오답 처리되므로 절대 생성하지 않겠다"라고 선언하세요.
- 3단계 (신규 단서 및 질문 기획): [2. 현재 장면 묘사]에서 포착된 새로운 단서에 집중하여, 현재 시점에서 당장 답변할 수 있는 흥미로운 정보(인물/사물의 정체, 배경지식, 숨은 의미 등)를 묻는 방향으로 기획을 설명합니다.

[출력 형식]
- 언어: 한국어 (단, 영어 콘텐츠의 고유명사는 원어 병기 허용. 예: 일각돌고래(Narwhal), 셰즈 은데예(Chez Ndeye))
- 형식: 반드시 아래 JSON 형식으로 출력하세요. 다른 설명은 덧붙이지 마십시오.

[JSON 형식 예시]
{
    "rationale": "1. 과거 정보 영구 차단: 혹등고래가 굶주리고 있다는 사실은 이전 장면에서 파악됨. 이 정보는 질문 소재에서 영구 제외함. / 2. 미래 추측 차단: '범고래의 사냥 결과' 같은 미래 서사는 답을 알 수 없으므로 묻지 않음. / 3. 질문 기획: 대신 시청자가 현재 접한 '범고래 떼의 소리'나 '생태계 특성' 등 당장 답변할 수 있는 배경지식에 대한 호기심으로 질문을 기획함.",
    "queries": [
        "방금 등장한 범고래 떼는 보통 어떤 먹이 사냥 방식을 가지고 있을까요?",
        "지금 범고래들이 내는 저 독특한 소리는 무리 안에서 어떤 의미를 가질까요?",
        "이 지역의 바다 생태계에서 범고래 무리가 지금처럼 나타나는 것은 흔한 일일까요?"
    ]
}"""

def make_voice_hint_configs(thinking_level=0):
    """Voice Hint 생성용 GenerateContentConfig 정보를 반환합니다."""
    return {
        "desc": make_generate_config(system_instruction=_VOICE_HINT_BASE_DESC, thinking_level=thinking_level),
        "kss": make_generate_config(system_instruction=_VOICE_HINT_BASE_KSS, thinking_level=thinking_level)
    }

def process_vh_modes(client, vh_model_name, vh_configs, past_parts, current_parts, kss_summary_text, end_time):
    """하나의 Keypoint에 대해 Voice Hint(img_desc, mm_desc, kss)를 병렬로 수행합니다."""
    def generate_voice_hints(mode):
        contents = []
        if mode == "kss":
            if not kss_summary_text:
                return {"queries": [], "rationale": "No KSS Summary available"}, 0.0
            contents = [
                "--- [KeyScene Summary] ---", 
                kss_summary_text, 
                "--- 요청 사항 ---",
                "위의 [KeyScene Summary]를 읽고, 시스템 프롬프트의 [사고 과정 가이드] 지침을 철저히 준수하여 질문 3개를 JSON 형식으로 생성하세요. 오직 방금 본 [2. 현재 장면 묘사]에 등장한 '새로운' 단서에 호기심을 집중하세요."
            ]
            vh_config = vh_configs["kss"]
        else:
            past_text = past_parts.get(mode, "")
            has_past = bool(past_text.strip())

            # 샌드위치 구조 (소형 모델 앵커링용)
            contents += ["--- [현재 시청 중인 장면] 시작 ---", current_parts.get(mode, ""), "--- [현재 시청 중인 장면] 끝 ---"]

            if has_past:
                contents += ["--- [이전까지의 과거 시청 맥락 (참조용)] 시작 ---", past_text, "--- [이전까지의 과거 시청 맥락 (참조용)] 끝 ---"]
                contents += ["--- [현재 시청 중인 장면 (재확인)] 시작 ---", current_parts.get(mode, ""), "--- [현재 시청 중인 장면 (재확인)] 끝 ---"]
            
            contents += [
                "--- 요청 사항 ---",
                "위의 [현재 시청 중인 장면]과 [이전까지의 과거 시청 맥락]을 인지한 상태에서, 시스템 프롬프트의 [사고 과정 가이드] 4단계를 철저히 준수하여 질문 3개를 JSON 형식으로 생성하세요. 과거 맥락에서 이미 아는 내용으로 질문하는 것은 피하고, 오직 방금 본 [현재 시청 중인 장면] 에 '새롭게' 등장한 단서에 집중하세요." 
                if has_past else "제공된 [현재 시청 중인 장면] 을 바탕으로 시스템 프롬프트의 [사고 과정 가이드] 지침에 따라 매력적인 질문 3개를 JSON 형식으로 생성하세요."
            ]
            vh_config = vh_configs["desc"]
        
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_vh_img  = executor.submit(generate_voice_hints, "img_desc")
        f_vh_mm   = executor.submit(generate_voice_hints, "mm_desc")
        f_vh_kss  = executor.submit(generate_voice_hints, "kss")
        
        vh_res_img, elapsed_img = f_vh_img.result()
        vh_res_mm, elapsed_mm = f_vh_mm.result()
        vh_res_kss, elapsed_kss = f_vh_kss.result()

    return {
        "img_desc": vh_res_img["queries"], 
        "mm_desc": vh_res_mm["queries"],
        "kss": vh_res_kss["queries"],
        "rationales": {
            "img_desc": vh_res_img["rationale"],
            "mm_desc": vh_res_mm["rationale"],
            "kss": vh_res_kss["rationale"]
        }
    }, {"img_desc": elapsed_img, "mm_desc": elapsed_mm, "kss": elapsed_kss}

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint Scene 목록을 입력받아 Voice Hint를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--kss_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--output_file", default="assets/voice_hint.jsonl", help="Voice Hint 목록 저장 경로")

    args, client = init_pipeline(parser.parse_args())

    vh_configs = make_voice_hint_configs(thinking_level=args.vh_thinking_level)

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

    # KSS 맵 로드: (content_id, scene_idx) -> summary_text
    kss_map = {}
    if os.path.exists(args.kss_file):
        for rec in load_jsonl(args.kss_file):
            c_id = rec.get("content_id")
            s_idx = rec.get("scene_idx")
            if c_id and s_idx is not None:
                kss_map[(c_id, s_idx)] = rec.get("summary", "")
        print(f"[Summary] {len(kss_map)}개 Scene의 KSS 로드됨 ({args.kss_file})")
    else:
        print(f"[Warning] KSS 파일을 찾을 수 없습니다: {args.kss_file}")

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
                kss_summary_text = kss_map.get((content_id, scene_idx), "")

                print(f"[{real_idx}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s]")
                if kss_summary_text:
                    current_scene_only = kss_summary_text.split("[2. 현재 장면 묘사]")[-1].strip() if "[2. 현재 장면 묘사]" in kss_summary_text else kss_summary_text.strip()
                    print(f"\n[ --- KSS 현재 장면 --- ]\n\n{current_scene_only}\n")
                
                # 로깅용 Desc 텍스트 추출 및 즉시 출력 (description-only 형식)
                img_desc_text = get_gcs_descriptions_by_scene_idx(args.gs_bucket_name, content_id, "img_desc", scene_idx, scene_idx)
                mm_desc_text = get_gcs_descriptions_by_scene_idx(args.gs_bucket_name, content_id, "mm_desc", scene_idx, scene_idx)

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
                    return process_vh_modes(client, args.vh_gen_model, vh_configs, past_parts, current_parts, kss_summary_text, end_time)

                try:
                    vh_dict, vh_elapsed_dict = _retry_api_call(
                        _run_keypoint,
                        label=f"Voice Hint (Scene {scene_idx})"
                    )

                    scene_key = (content_id, scene_idx)

                    # voice_hint.jsonl: 해당 파일에 없는 경우에만 기록
                    if scene_key not in vh_pairs:
                        query_groups = [
                            {"mode": "img_desc", "queries": vh_dict["img_desc"], "rationale": vh_dict.get("rationales", {}).get("img_desc", "")},
                            {"mode": "mm_desc", "queries": vh_dict["mm_desc"], "rationale": vh_dict.get("rationales", {}).get("mm_desc", "")}
                        ]
                        if vh_dict.get("kss"):
                            query_groups.append({"mode": "kss", "queries": vh_dict["kss"], "rationale": vh_dict.get("rationales", {}).get("kss", "")})
                            
                        scene_record = {
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "start_time": start_time,
                            "end_time": end_time,
                            "queries": query_groups,
                        }
                        append_jsonl(args.output_file, scene_record)
                        vh_pairs.add(scene_key)

                    for mod in ["kss", "img_desc", "mm_desc"]:
                        if vh_dict.get(mod):
                            print(f"-> [VH - {mod}] ({vh_elapsed_dict.get(mod, 0.0):.2f}초)")
                            if vh_dict.get("rationales", {}).get(mod):
                                print(f"[Rationale]: {vh_dict['rationales'][mod]}")
                            for qi, q in enumerate(vh_dict[mod], 1):
                                print(f"    {qi}. {q}")
                            print("")
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
