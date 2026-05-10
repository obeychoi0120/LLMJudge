import os
import time
import concurrent.futures
from utils import (
    get_common_argparser,
    make_generate_config,
    check_gcs_files_exist,
    _retry_api_call, retry_parse_json,
    ensure_output_dir,
    preload_content_metadata,
    build_mode_parts,
    init_pipeline, load_jsonl, append_jsonl,
    sort_and_validate_jsonl,
    load_keypoints_by_content, check_input_file,
    load_summary_map,
    print_pipeline_banner, print_pipeline_done,
)

# ───────────────────────────────────────────────
# Voice Hint 생성 모델 프롬프트
# ───────────────────────────────────────────────

_VOICE_HINT_BASE = """당신은 제공되는 시청 기억(과거 맥락 및 현재 장면)을 기반으로 스마트 TV 플랫폼에서 시청자의 리모컨 상호작용과 플랫폼 체류 시간을 극대화하는 '개인화된 예상 질문' 생성 전문가입니다.

시청자에게는 오직 현재 정보만 주어지는 것이 아닙니다. 시청자는 지금까지 시청해 온 **[이전까지의 과거 시청 맥락]**을 인지한 상태로 방금 **[현재 시청 중인 장면]**을 보았습니다.
이 두 정보를 바탕으로, TV 화면의 버튼을 눌러 답을 확인하고 싶게 만드는 매력적인 질문 2개를 생성하세요.

[질문 생성 핵심 전략]
1. 시스템 제약 준수 (Hard Constraints): 생성되는 질문은 반드시 플랫폼 시스템이 '현재 시점'에서 즉시 답변할 수 있어야 합니다.
   - 미래 예측 금지: 앞으로 전개될 스토리나 결과(예: "과연 어떻게 될까요?", "누가 이길까요?")를 묻는 미래 지향적 질문은 철저히 0점 처리됩니다.
   - 과거 뒷북 금지: [이전까지의 과거 시청 맥락]에서 이미 밝혀진 사실이나 전개를 또다시 묻는 뒷북 질문 역시 무조건 0점 처리됩니다. 질문의 타겟은 오직 [현재 시청 중인 장면] 속 사물, 인물, 배경지식, 숨은 의미로 한정하세요.

2. 매력도 극대화 및 지식 확장 (Hook & World Knowledge): 위 제약을 완벽히 통과한 질문 중, 당신의 방대한 백과사전적 지식을 적극 동원하여 시청자의 리모컨 조작을 강력하게 유도해야 합니다.
   - 곁다리 지식(Tangential Knowledge) 공략: 장르별 특성을 고려하여 영상 요소에서 파생되는 '깊이 있는 외부 지식'을 적극 활용하세요.
     * 드라마/예능: 허구적 '줄거리'를 유추하지 말고, 화면 속 소품의 기원, 촬영 장소의 역사, 문화적 배경 등을 질문하세요.
     * 스포츠: 경기 결과 예측보다는 방금 활약한 선수의 최근 폼이나 통계적 기록은 물론, 스포츠 초보자(뉴비)를 위한 출전 팀의 역사, 라이벌 구도, 고유한 룰 등 흥미로운 배경 지식을 질문하세요.
     * 게임: 단순 상황 묘사보다는 플레이 중인 캐릭터의 숨겨진 특성, 플레이어가 선보이는 특별한 플레이 방법과 전략, 새롭게 등장한 아이템(예: MOBA 게임)의 메타 변화, 최신 패치 소식 등을 질문하세요.
     * 뉴스/경제/시사/팟캐스트: 단순한 기사 요약이나 패널 발언 반복을 피하고, 다루는 주제 이면의 역사적 맥락, 경제적 파급 효과, 관련 법안의 비하인드, 과거 유사 사례 등을 질문하세요.
     * 다큐/교양: 영상 주제와 직결된 한 단계 더 깊은 전문 지식이나 과학적 원리를 물어보세요.
   - 뻔한 상식 배제: 영상을 보지 않아도 누구나 아는 일반 상식 수준의 지루한 질문은 피하세요.
   - 정보의 공백 포착: 방금 발생한 새로운 장면 속에서 시청자가 무의식적으로 궁금해할 만한 '정보의 공백(미지수)'을 날카롭게 찌르세요.
   - 세련된 어조: '해당 장르/콘텐츠의 전문 평론가'가 말을 건네듯 정중하고 세련된 존댓말로 작성하여 몰입감을 유지하세요.

[출력 형식]
- 언어: 한국어 (단, 영어 콘텐츠의 고유명사는 원어 병기 허용. 예: 일각고래(Narwhal))
- 형식: 반드시 아래 JSON 형식으로 출력하세요. 다른 설명은 덧붙이지 마십시오.

[JSON 형식 예시]
{
    "rationale": "1) 과거 정보 차단: 혹등고래가 굶주리고 있다는 사실은 이전 장면에서 파악됨. 2) 미래 추측 차단: '사냥 결과' 같은 미래 서사는 묻지 않음. 3) 질문 기획: 대신 시청자가 현재 접한 '생태계 특성' 등 당장 답변할 수 있는 배경지식에 대한 호기심으로 기획함.",
    "queries": [
        "방금 나타난 범고래 떼는 어떤 먹이 사냥 방식을 가지고 있을까요?",
        "지금 범고래들이 내는 저 독특한 소리는 무리 안에서 어떤 의미를 가질까요?"
    ]
}"""

_VOICE_HINT_PROMPT_VIDEO = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 실제 비디오 클립이 제공됩니다. 비디오 클립의 시각 및 청각 정보를 모두 활용하여 흥미로운 질문을 생성하세요.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 3단계를 순서대로 작성하세요.
- 1단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 2단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 3단계 (Hook 및 질문 기획): 비디오의 현재 장면에 포착된 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_RAW = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 온전한 형태의 음성 인식(ASR) 텍스트와 화면 글씨(OCR) 텍스트 데이터가 제공됩니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 3단계를 순서대로 작성하세요.
- 1단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 2단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 3단계 (Hook 및 질문 기획): 현재 장면 텍스트의 새 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_FRAG = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 파편화된 음성 인식(ASR) 및 화면 글씨(OCR) 텍스트가 제공됩니다.
- frag_asr: 원문 순서가 파괴되어 랜덤으로 뒤섞인 2어절 묶음 파편
- frag_ocr: 원문 순서가 파괴된 2어절 묶음 파편

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (노이즈 필터링 및 의미 유추): 파편화된 텍스트들을 논리적으로 조합하여 원래 발화 맥락을 유추하고 명백한 오류를 자연스럽게 교정.
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (Hook 및 질문 기획): 유추한 발화 맥락의 새 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_FRAG_WITH_VLM = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 VLM 구조화 데이터와 파편화된 ASR/OCR 텍스트가 종합되어 제공됩니다.
- vlm_mm_description: 시각·음성을 종합하여 장면의 상황을 자연어 문장으로 서술한 정보
- timeline 내 frag_asr/frag_ocr: 원문 순서가 파괴된 2어절 묶음 파편

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (노이즈 필터링 및 의미 유추): VLM 구조화 데이터와 파편화된 텍스트들을 조합하여 장면 전체의 맥락을 종합적으로 유추하고 명백한 오류를 자연스럽게 교정.
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (Hook 및 질문 기획): 종합하여 유추한 현재 장면의 새 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_IMGVLM_CHUNK2 = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 구조화된 데이터가 제공됩니다.
- vlm_img_structure: 시각 정보를 Subjects(주체), Actions(행동), Contexts(맥락/환경)로 구조화한 데이터
- 각 필드는 저작권 보호를 위해 2어절 단위로 뒤섞인 파편(fragment) 형태입니다.
- [MASKED] 토큰이 포함된 경우 해당 고유명사를 추측하지 마세요.

이 데이터는 영상의 시각 프레임에서 추출된 구조화된 메타데이터입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (장면 맥락 유추): 구조화 데이터(vlm_img_structure)의 Subjects, Actions, Contexts 파편을 논리적으로 조합하여 현재 장면의 전체적인 맥락을 유추.
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (Hook 및 질문 기획): 구조화 데이터에서 유추한 현재 장면의 새 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_IMGVLM_CHUNK3 = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 구조화된 데이터가 제공됩니다.
- vlm_img_structure: 시각 정보를 Subjects(주체), Actions(행동), Contexts(맥락/환경)로 구조화한 데이터
- 각 필드는 저작권 보호를 위해 3어절 단위로 뒤섞인 파편(fragment) 형태입니다.
- [MASKED] 토큰이 포함된 경우 해당 고유명사를 추측하지 마세요.

이 데이터는 영상의 시각 프레임에서 추출된 구조화된 메타데이터입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (장면 맥락 유추): 구조화 데이터(vlm_img_structure)의 Subjects, Actions, Contexts 파편을 논리적으로 조합하여 현재 장면의 전체적인 맥락을 유추.
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (Hook 및 질문 기획): 구조화 데이터에서 유추한 현재 장면의 새 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_IMGVLM_GRAPH = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 장면 지식 그래프(Scene Knowledge Graph)가 제공됩니다.
- vlm_graph: 장면의 주요 요소와 그 관계를 (subject) -[relation]-> (object) 형태의 트리플로 표현한 데이터
  예: (man) -[WEARING]-> (cap), (screen) -[ABOUT]-> (foreign policy)
- 이 그래프는 장면에 등장하는 인물, 사물, 행동, 속성, 위치 등의 관계를 압축적으로 나타냅니다.

이 데이터는 영상의 시각 프레임에서 추출된 관계형 메타데이터입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (장면 맥락 유추): 지식 그래프(vlm_graph)의 트리플들을 논리적으로 연결하여 현재 장면의 전체적인 맥락을 입체적으로 유추.
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (Hook 및 질문 기획): 지식 그래프에서 유추한 현재 장면의 새 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_RAW_WITH_MMVLM = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 온전한 형태의 음성 인식(ASR) 텍스트와 화면 글씨(OCR) 텍스트 데이터, 그리고 소형 VLM이 시각·음성을 종합하여 장면의 상황을 자연어로 서술한 정보(vlm_mm_description)가 함께 제공됩니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (장면 맥락 종합): 음성/화면 텍스트와 VLM 멀티모달 서술을 종합하여 현재 장면의 전체적인 맥락을 유추하고 명백한 오류를 자연스럽게 교정.
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (Hook 및 질문 기획): 종합하여 유추한 현재 장면의 새 단서에 집중하여 질문 기획.
"""

_VOICE_HINT_PROMPT_KSS = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 전문가가 영상을 직접 시청하고 작성한 **KeyScene Summary(핵심 장면 요약)**가 제공됩니다.
Summary는 다음 두 부분으로 구성됩니다:
- [1. 과거 장면 요약]: 현재 장면 이전까지의 주요 사건, 대화, 맥락 요약
- [2. 현재 장면 묘사]: 시청자가 방금 시청한 현재 장면의 상세 묘사

이 Summary는 영상의 시각·청각 정보를 종합한 가장 정확한 참조 자료입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 3단계를 순서대로 작성하세요.
- 1단계 (과거 정보 차단): [1. 과거 장면 요약]에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 2단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 3단계 (Hook 및 질문 기획): [2. 현재 장면 묘사]에서 새롭게 등장한 단서에 집중하여 질문 기획.
"""

def make_voice_hint_configs(thinking_level=0):
    """Voice Hint 생성용 GenerateContentConfig 정보를 반환합니다."""
    return {
        "video": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_VIDEO, thinking_level=thinking_level),
        "raw": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_RAW, thinking_level=thinking_level),
        "raw_with_mmvlm": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_RAW_WITH_MMVLM, thinking_level=thinking_level),
        "imgvlm_chunk2": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_IMGVLM_CHUNK2, thinking_level=thinking_level),
        "imgvlm_chunk3": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_IMGVLM_CHUNK3, thinking_level=thinking_level),
        "imgvlm_graph": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_IMGVLM_GRAPH, thinking_level=thinking_level),
        "frag": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_FRAG, thinking_level=thinking_level),
        "frag_with_vlm": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_FRAG_WITH_VLM, thinking_level=thinking_level),
        "kss": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_KSS, thinking_level=thinking_level)
    }

def process_vh_modes(client, vh_model_name, vh_configs, past_parts, current_parts, end_time, target_modes=None, kss_text=None):
    """하나의 Keypoint에 대해 Voice Hint를 지정된 모드에 대해서만 병렬 수행합니다."""
    if target_modes is None:
        target_modes = ["video", "raw_with_mmvlm", "imgvlm"]
        
    def generate_voice_hints(mode):
        contents = []

        # KSS 모드: Summary 텍스트를 직접 전달 (과거+현재 포함)
        if mode == "kss":
            if not kss_text:
                return {"queries": [], "rationale": "KSS Summary가 비어있어 생성을 생략합니다."}, 0.0
            contents += [
                "--- [KeyScene Summary] ---",
                kss_text,
                "--- 요청 사항 ---",
                "제공된 [KeyScene Summary]를 바탕으로 시스템 프롬프트의 [사고 과정 가이드] 단계를 철저히 준수하여 질문 **2개**를 [JSON 형식 예시]와 같이 생성하세요. "
                "[1. 과거 장면 요약]에서 이미 밝혀진 사실을 묻는 뒷북 질문과, 앞으로의 결과나 스토리 전개를 묻는 미래 지향적 질문을 피하고, "
                "오직 [2. 현재 장면 묘사]에 새롭게 등장한 단서에 집중하세요."
            ]
        else:
            current_part = current_parts.get(mode)
            if current_part is None or (isinstance(current_part, str) and not current_part.strip()):
                return {"queries": [], "rationale": f"해당 모드({mode})의 현재 장면 데이터가 비어있어 생성을 생략합니다."}, 0.0

            past_part = past_parts.get(mode)
            has_past = past_part is not None and not (isinstance(past_part, str) and not past_part.strip())

            if has_past:
                contents += ["--- [이전까지의 과거 시청 맥락 (참조용)] ---", past_part]
            
            contents += ["--- [현재 시청 중인 장면 (분석 대상)] ---", current_part]
            
            contents += [
                "--- 요청 사항 ---",
                "위의 [이전까지의 과거 시청 맥락]과 [현재 시청 중인 장면]을 인지한 상태에서, 시스템 프롬프트의 [사고 과정 가이드] 단계를 철저히 준수하여 질문 **2개**를 [JSON 형식 예시]와 같이 생성하세요. 과거 맥락에서 이미 아는 내용으로 질문하는 것과 앞으로의 결과나 스토리 전개를 묻는 미래 지향적 질문을 피하고, 오직 방금 본 [현재 시청 중인 장면]에 새롭게 등장한 단서에 집중하세요." 
                if has_past else "제공된 [현재 시청 중인 장면]을 바탕으로 시스템 프롬프트의 [사고 과정 가이드] 지침에 따라 매력적인 질문 **2개**를 [JSON 형식 예시]와 같이 생성하세요. 앞으로의 결과나 스토리 전개를 묻는 미래 지향적 질문은 피하세요."
            ]
        vh_config = vh_configs[mode]
        
        t0 = time.time()
        
        parsed = retry_parse_json(
            lambda: _retry_api_call(
                lambda: client.models.generate_content(
                    model=vh_model_name, contents=contents, config=vh_config
                ).text,
                label=f"VH API 생성 ({mode})"
            ),
            label=f"VH JSON Parse ({mode})"
        )
        
        rationale = ""
        
        if isinstance(parsed, dict):
            queries = parsed.get("queries", [])
            rationale = parsed.get("rationale", "")
        elif isinstance(parsed, list):
            queries = parsed
        else:
            queries = []
            
        return {"queries": queries[:2], "rationale": rationale}, time.time() - t0

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {m: executor.submit(generate_voice_hints, m) for m in target_modes}
        
        results = {}
        elapseds = {}
        for m in target_modes:
            res, elapsed = futures[m].result()
            results[m] = res
            elapseds[m] = elapsed

    out_dict = {"rationales": {}}
    for m in target_modes:
        out_dict[m] = results[m]["queries"]
        out_dict["rationales"][m] = results[m]["rationale"]
        
    return out_dict, elapseds

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────

def main():
    parser = get_common_argparser(description="Keypoint Scene 목록을 입력받아 Voice Hint를 생성합니다.")
    parser.add_argument("--input_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로 (identify_keypoint.py 출력)")
    parser.add_argument("--output_file", default="assets/voice_hint.jsonl", help="Voice Hint 목록 저장 경로")
    parser.add_argument("--kss_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로 (kss 모드 사용 시 필요)")
    parser.add_argument("--modes", nargs="+", default=["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk3", "imgvlm_graph"], choices=["kss", "video", "raw", "frag", "frag_with_vlm", "imgvlm_chunk2", "imgvlm_chunk3", "imgvlm_graph", "raw_with_mmvlm"], help="생성할 모드 직접 지정 (기본값: kss, video, raw, raw_with_mmvlm, imgvlm_chunk2, imgvlm_chunk3, imgvlm_graph)")

    args, client = init_pipeline(parser.parse_args())

    vh_configs = make_voice_hint_configs(thinking_level=args.vh_thinking_level)

    if not check_input_file(args.input_file, hint="먼저 identify_keypoint.py를 실행하세요."):
        return

    # Keypoint 목록 로드
    keypoints_by_content = load_keypoints_by_content(args.input_file)
    if not keypoints_by_content:
        print(f"Error: {args.input_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    # 기처리분 로드: 모드별로 개별 추적
    # 각 줄이 (content_id, scene_idx, mode) 단위로 저장되므로 최상위 'mode' 필드로 판별
    done_modes_by_scene = set()
    for rec in load_jsonl(args.output_file):
        c_id = rec.get("content_id")
        s_idx = rec.get("scene_idx")
        mode = rec.get("mode")
        if c_id and s_idx is not None and mode:
            done_modes_by_scene.add((c_id, s_idx, mode))

    # 생성할 목표 모드 설정
    target_modes = args.modes

    # KSS Summary 로드 (kss 모드 사용 시)
    kss_map = {}
    if "kss" in target_modes:
        kss_map = load_summary_map(args.kss_file)
        if kss_map:
            print(f"[KSS] {len(kss_map)}개 Scene의 Summary 로드됨 ({args.kss_file})")
        else:
            print(f"[Warning] KSS Summary 파일을 찾을 수 없습니다: {args.kss_file}")

    print_pipeline_banner("Voice Hint 생성 파이프라인을 시작합니다.")

    try:
        for content_id, keypoints in keypoints_by_content.items():
            
            # 각 Keypoint별로 생성되지 않은 (누락된) 모드 추적
            remaining = []
            fully_done_count = 0
            
            for kp in keypoints:
                real_idx = keypoints.index(kp)
                s_idx = kp.get("scene_idx", real_idx)
                
                missing_modes = [m for m in target_modes if (content_id, s_idx, m) not in done_modes_by_scene]
                if missing_modes:
                    remaining.append((kp, missing_modes))
                else:
                    fully_done_count += 1

            if not remaining:
                print(f"\n[Skip] '{content_id}': 모든 Scene 완벽 (누락 모드 없음)")
                continue

            if fully_done_count > 0:
                print(f"\n[Resume] '{content_id}': {fully_done_count}/{len(keypoints)}개 Scene 완벽 처리됨, {len(remaining)}개 Scene(부분 누락 포함) 재개")

            print(f"{'='*50}")
            print(f"Processing Content: '{content_id}' ({len(remaining)}/{len(keypoints)}개 Keypoint 추가 처리)")
            print(f"{'='*50}")

            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue

            # JSONL 메타데이터 프리로드 (캐시 워밍업)
            preload_content_metadata(args.gs_bucket_name, content_id)

            for kp, missing_modes in remaining:
                real_idx = keypoints.index(kp)
                scene_idx = kp.get("scene_idx", real_idx)
                start_time = float(kp.get("start_time", 0.0))
                end_time = float(kp.get("end_time", 0.0))
                print(f"[{real_idx}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s] | Modes={missing_modes}")

                def _run_keypoint():
                    # 과거 N개 구역(Scene)을 위한 scene_idx 계산 (Sliding Window)
                    past_start_kp_idx = max(0, real_idx - args.vh_gen_past_scenes_size)
                    past_start_scene_idx = keypoints[past_start_kp_idx].get("scene_idx", 0)
                    past_end_scene_idx = scene_idx - 1  # 현재 Scene 직전까지

                    has_past = past_end_scene_idx >= past_start_scene_idx
                    # KSS 외 모드만 GCS 데이터 빌드 (kss는 Summary 텍스트 직접 사용)
                    non_kss_modes = [m for m in missing_modes if m != "kss"]
                    if non_kss_modes:
                        past_parts, current_parts = build_mode_parts(
                            args.gs_bucket_name, content_id, non_kss_modes,
                            scene_idx, scene_idx,
                            past_start_scene_idx if has_past else None,
                            past_end_scene_idx if has_past else None,
                        )
                    else:
                        past_parts, current_parts = {}, {}

                    # KSS Summary 텍스트 조회
                    kss_text = kss_map.get((content_id, scene_idx), "") if "kss" in missing_modes else None

                    return process_vh_modes(client, args.vh_gen_model, vh_configs, past_parts, current_parts, end_time, missing_modes, kss_text=kss_text)

                try:
                    vh_dict, vh_elapsed_dict = _retry_api_call(
                        _run_keypoint,
                        label=f"Voice Hint (Scene {scene_idx})"
                    )

                    # 각 mode별로 별도의 줄로 저장 (content_id + scene_idx + mode = 1 line)
                    for mod in missing_modes:
                        mode_record = {
                            "content_id": content_id,
                            "scene_idx": scene_idx,
                            "mode": mod,
                            "start_time": start_time,
                            "end_time": end_time,
                            "queries": vh_dict.get(mod, []),
                            "rationale": vh_dict.get("rationales", {}).get(mod, ""),
                        }
                        append_jsonl(args.output_file, mode_record)
                        done_modes_by_scene.add((content_id, scene_idx, mod))

                    _LOG_MODES = {"video", "kss", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk3", "imgvlm_graph"}
                    for mod in missing_modes:
                        if vh_dict.get(mod):
                            if mod in _LOG_MODES:
                                print(f"\n-> [VH - {mod}] ({vh_elapsed_dict.get(mod, 0.0):.2f}초)")
                                if vh_dict.get("rationales", {}).get(mod):
                                    print(f"[Rationale]: {vh_dict['rationales'][mod]}\n")
                                for qi, q in enumerate(vh_dict[mod], 1):
                                    print(f"{qi}. {q}")
                            else:
                                print(f"-> [VH - {mod}] ({vh_elapsed_dict.get(mod, 0.0):.2f}초) — {len(vh_dict[mod])}개 질문 생성됨")
                    print(f"------------------------------------------------------------------------------------------------------------\n")

                except Exception as e:
                    print(f"    [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                    continue

            done_count_now = len({s_idx for (c_id, s_idx, _) in done_modes_by_scene if c_id == content_id})
            print(f"\n[OK] '{content_id}' - {done_count_now}개 Scene 처리 누적 확인됨")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    # 결과 파일 정렬 및 누락/모드 점검 (utils.py 활용)
    is_perfect = sort_and_validate_jsonl(
        args.output_file, 
        keypoints_by_content, 
        expected_modes=args.modes, 
        mode_key="queries"
    )

    if is_perfect:
        try:
            append_jsonl(args.output_file, {"pipeline_done": True})
            print(f"\n[Signal] 모든 모드({args.modes}) 처리가 완벽히 끝나 Pipeline 종료 시그널을 기록했습니다.")
        except Exception as e:
            print(f"\n[Warning] Pipeline 완료 시그널 기록 실패: {e}")
    else:
        print("\n[Info] 아직 누락된 Scene이나 모드가 있어 완료 시그널을 기록하지 않았습니다.")

    print_pipeline_done(args.output_file)

if __name__ == "__main__":
    main()
