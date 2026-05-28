import os
import time
import json
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
    load_video_metadata, format_video_context,
    print_pipeline_banner, print_pipeline_done, ProgressTracker,
)

# ───────────────────────────────────────────────
# Voice Hint 생성 모델 프롬프트
# ───────────────────────────────────────────────

_VOICE_HINT_BASE = """당신은 제공되는 시청 기억(과거 맥락 및 현재 장면)을 기반으로 스마트 TV 플랫폼에서 시청자의 리모컨 상호작용과 플랫폼 체류 시간을 극대화하는 '개인화된 예상 질문' 생성 전문가입니다.

당신에게는 [Video Context]로 영상의 채널명과 제목이 제공될 수 있습니다. 이를 통해 콘텐츠의 장르(드라마/스포츠/게임/뉴스/다큐 등)와 도메인을 먼저 파악하고, 아래 장르별 전략에 정확히 매칭하세요.

시청자에게는 오직 현재 정보만 주어지는 것이 아닙니다. 시청자는 지금까지 시청해 온 **[이전까지의 과거 시청 맥락]**을 인지한 상태로 방금 **[현재 시청 중인 장면]**을 보았습니다.
이 두 정보를 바탕으로, 아래 역할 분담에 따라 TV 화면의 버튼을 눌러 답을 확인하고 싶게 만드는 매력적인 **콘텐츠 특화 질문(Content-Anchored) 2개**를 생성하세요.

[질문 유형 — 콘텐츠 특화 질문 2개 생성]
생성하는 질문 2개(queries[0], queries[1])는 모두 아래 지침을 따르는 '콘텐츠 특화 질문'이어야 합니다.

■ Content-Anchored (콘텐츠 특화 질문)
  현재 장면의 **핵심 사건·인물·주제에 직접 연결된** 깊이 있는 질문입니다.
  단순 사실 확인("이 사람은 누구?")이 아닌, 핵심 주제를 한 단계 깊이 파고들어 시청자가 "더 알고 싶다"고 느낄 만한 해설·분석·맥락 질문을 지향하세요.
  - 드라마/예능: 방금 인물이 보인 행동·대사의 심리적 배경, 작중 갈등 구도의 맥락, 연출 의도
  - 스포츠: 방금 활약한 선수의 최근 폼·이적 경위·팀 내 역할 변화, 현재 경기의 전술적 포인트
  - 게임: 지금 사용 중인 전략/빌드의 메타 배경, 플레이어의 의사결정 분석, 패치 영향
  - 뉴스/시사/팟캐스트: 다루고 있는 이슈의 핵심 쟁점, 이해관계 구도, 정책적 파급 효과
  - 다큐/교양: 영상이 집중 조명하는 현상·원리에 대한 전문적 깊이의 후속 질문

[질문 생성 핵심 전략]
1. 시스템 제약 준수 (Hard Constraints): 생성되는 질문은 반드시 플랫폼 시스템이 '현재 시점'에서 즉시 답변할 수 있어야 합니다.
   - 미래 예측 금지: 앞으로 전개될 스토리나 결과(예: "과연 어떻게 될까요?", "누가 이길까요?")를 묻는 미래 지향적 질문은 철저히 0점 처리됩니다.
   - 과거 뒷북 금지: [이전까지의 과거 시청 맥락]에서 이미 밝혀진 사실이나 전개를 또다시 묻는 뒷북 질문 역시 무조건 0점 처리됩니다. 질문의 타겟은 오직 [현재 시청 중인 장면] 속 사물, 인물, 배경지식, 숨은 의미로 한정하세요.

2. 두 질문 모두에 대한 공통 원칙:
   - 뻔한 상식 배제: 영상을 보지 않아도 누구나 아는 일반 상식 수준의 지루한 질문은 피하세요.
   - 정보의 공백 포착: 방금 발생한 새로운 장면 속에서 시청자가 무의식적으로 궁금해할 만한 '정보의 공백(미지수)'을 날카롭게 찌르세요.
   - 세련된 어조: '해당 장르/콘텐츠의 전문 평론가'가 말을 건네듯 정중하고 세련된 존댓말로 작성하여 몰입감을 유지하세요.
   - 백과사전적 지식 적극 활용: 당신의 방대한 외부 지식을 동원하여 시청자의 리모컨 조작을 강력하게 유도하세요.

[출력 형식]
- 언어: 한국어 (단, 영어 콘텐츠의 고유명사는 원어 병기 허용. 예: 일각고래(Narwhal))
- 형식: 반드시 아래 JSON 형식으로 출력하세요. 다른 설명은 덧붙이지 마십시오.
- **queries 배열의 2개 요소 모두 반드시 서로 다른 주제의 매력적인 Content-Anchored 질문이어야 합니다.**

[JSON 형식 예시]
{
    "rationale": "1) 과거 정보 차단: 혹등고래가 굶주리고 있다는 사실은 이전 장면에서 파악됨. 2) 미래 추측 차단: '사냥 결과' 같은 미래 서사는 묻지 않음. 3) Content-Anchored 1 기획: 현재 장면의 핵심인 범고래 떼의 협동 사냥 전략에 대한 깊이 있는 질문. 4) Content-Anchored 2 기획: 범고래의 사냥 부위나 이동 행동 패턴의 과학적 배경지식을 묻는 질문.",
    "queries": [
        "방금 화면에서 범고래 떼가 보여준 협동 사냥은 어떤 전략적 메커니즘으로 이루어지나요?",
        "범고래들이 사냥할 때 먹이의 특정 부위(예: 간 등)만 골라 먹는 생태학적 이유는 무엇인가요?"
    ]
}"""

_VOICE_HINT_BASE_LOW_CONTEXT = """당신은 제공되는 시청 기억(과거 맥락 및 현재 장면)을 기반으로 스마트 TV 플랫폼에서 시청자의 리모컨 상호작용과 플랫폼 체류 시간을 극대화하는 '개인화된 예상 질문' 생성 전문가입니다.

당신에게는 [Video Context]로 영상의 채널명과 제목이 제공될 수 있습니다. 이를 통해 콘텐츠의 장르(드라마/스포츠/게임/뉴스/다큐 등)와 도메인을 참고하고, 아래 전략에 맞추세요. (단, 저작권 제약으로 인해 과거 맥락과 현재 장면 모두 비식별화된 VLM 추출 데이터 형식으로 제공됩니다.)

시청자에게는 오직 현재 정보만 주어지는 것이 아닙니다. 시청자는 지금까지 시청해 온 **[이전까지의 과거 시청 맥락]**을 인지한 상태로 방금 **[현재 시청 중인 장면]**을 보았습니다.
이 두 정보를 바탕으로, TV 화면의 버튼을 눌러 답을 확인하고 싶게 만드는 매력적인 **곁다리 지식 질문(Tangential Knowledge) 2개**를 생성하세요.

[질문 유형 — 곁다리 지식 질문 2개 생성]
생성하는 질문 2개(queries[0], queries[1])는 모두 아래 지침을 따르는 '곁다리 지식 질문'이어야 합니다.

■ Tangential Knowledge (곁다리 지식 질문)
  현재 장면에서 포착된 시각적 단서(인물, 사물, 배경 등)에서 파생되는 곁다리 지식 질문입니다.
  영상의 주된 서사와 직접 관련되지 않더라도, 화면 속 소품·장소·문화·기술 등에서 호기심을 자극하는 '깊이 있는 외부 지식'을 질문하세요.
  - 화면 속 소품의 기원, 촬영 장소의 역사, 의상·음식 등의 문화적 배경
  - 경기장 건축 양식, 유니폼·로고 디자인의 유래, 해당 도시의 스포츠 문화
  - 게임 세계관의 신화적·문학적 모티프, UI/사운드 디자인 철학, 개발사 비하인드
  - 배경 화면의 건물·지도·기호의 의미, 관련 역사적 사건, 과거 유사 사례
  - 화면에 스쳐 지나간 생물·지형·도구 등에 대한 흥미로운 부가 지식

[질문 생성 핵심 전략]
1. 시스템 제약 준수 (Hard Constraints): 생성되는 질문은 반드시 플랫폼 시스템이 '현재 시점'에서 즉시 답변할 수 있어야 합니다.
   - 미래 예측 금지: 앞으로 전개될 스토리나 결과(예: "과연 어떻게 될까요?", "누가 이길까요?")를 묻는 미래 지향적 질문은 철저히 0점 처리됩니다.
   - 과거 뒷북 금지: [이전까지의 과거 시청 맥락]에서 이미 밝혀진 사실이나 상식을 또다시 묻는 뒷북 질문 역시 무조건 0점 처리됩니다. 질문의 단서는 오직 [현재 시청 중인 장면] 속 사물, 인물, 배경지식으로 한정하세요.

2. 두 질문 모두에 대한 공통 원칙:
   - 뻔한 상식 배제: 영상을 보지 않아도 누구나 아는 일반 상식 수준의 지루한 질문은 피하세요.
   - 정보의 공백 포착: 방금 발생한 새로운 장면 속에서 시청자가 무의식적으로 궁금해할 만한 '정보의 공백(미지수)'을 날카롭게 찌르세요.
   - 세련된 어조: '해당 장르/콘텐츠의 전문 평론가'가 말을 건네듯 정중하고 세련된 존댓말로 작성하여 몰입감을 유지하세요.
   - 백과사전적 지식 적극 활용: 당신의 방대한 외부 지식을 동원하여 시청자의 리모컨 조작을 강력하게 유도하세요.

[출력 형식]
- 언어: 한국어 (단, 영어 콘텐츠의 고유명사는 원어 병기 허용. 예: 일각고래(Narwhal))
- 형식: 반드시 아래 JSON 형식으로 출력하세요. 다른 설명은 덧붙이지 마십시오.
- **queries 배열의 2개 요소 모두 반드시 서로 다른 주제의 매력적인 Tangential 질문이어야 합니다.**

[JSON 형식 예시]
{
    "rationale": "1) 과거 정보 차단: 일각고래가 물속에서 울음소리를 낸다는 사실은 이전 장면에서 파악됨. 2) 미래 추측 차단: 앞으로 어떻게 사냥을 시작할지 등 미래 서사는 묻지 않음. 3) Tangential 1 기획: 일각고래의 독특한 엄니(Tusk) 생물학적 기원에 대한 질문. 4) Tangential 2 기획: 북극 바다 얼음(빙하)의 물리적 특징이나 극지 지형과 관련된 질문.",
    "queries": [
        "일각고래의 머리 앞에 길게 뻗은 엄니는 실제 이빨이 발달한 것일까요, 아니면 뿔일까요?",
        "북극해의 얼어붙은 빙하 한가운데에 얼지 않고 동그랗게 뚫려 있는 구멍인 '폴리냐'는 어떤 원리로 형성되나요?"
    ]
}"""

_VOICE_HINT_PROMPT_VIDEO = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 실제 비디오 클립이 제공됩니다. 비디오 클립의 시각 및 청각 정보를 모두 활용하여 흥미로운 질문을 생성하세요.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 3단계를 순서대로 작성하세요.
- 1단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 2단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 3단계 (질문 기획): 비디오의 현재 장면에 포착된 단서에 집중하여 queries[0]과 queries[1] 모두 서로 다른 핵심 주제에 직결된 Content-Anchored 질문으로 기획.
"""

_VOICE_HINT_PROMPT_RAW = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 음성 인식(ASR) 텍스트와 화면 글씨(OCR) 텍스트 데이터가 Scene 단위로 제공됩니다.
각 Scene의 구조:
  - scene_idx: Scene 인덱스
  - duration: Scene 시작~종료 시간
  - speech: Scene 내 모든 음성을 시간 순서대로 이어 붙인 통합 텍스트
  - on_screen_text: Scene 내 화면 텍스트를 중복 제거하여 나열한 문자열

[분석 시 주의사항]
- speech는 연속된 내러티브로 읽고, 문맥과 상식을 동원하여 ASR 오탈자를 자연스럽게 교정하세요.
- ASR과 OCR은 서로 보정 단서가 됩니다. 교차 검증하여 오탈자를 교정하세요.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 3단계를 순서대로 작성하세요.
- 1단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 2단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 3단계 (질문 기획): 현재 장면 텍스트의 새 단서에 집중하여 queries[0]과 queries[1] 모두 서로 다른 핵심 주제에 직결된 Content-Anchored 질문으로 기획.
"""


_VOICE_HINT_PROMPT_IMGVLM_CHUNK2 = _VOICE_HINT_BASE_LOW_CONTEXT + """
[주의] [Video Context]는 저작권 제약으로 이 모드에서는 제공되지 않습니다. 오직 아래 데이터만으로 장르를 유추하세요.

[입력 형식 설명]
당신에게는 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 구조화된 데이터가 제공됩니다.
각 Scene은 시간 범위와 함께 <vlm_img_structure> 태그로 감싸진 형태입니다:
  - Subjects: 장면의 핵심 주체 (등장인물, 주요 피사체) — 2어절 단위 파편
  - Contexts: 장면의 행동, 배경, 환경 및 맥락 정보 — 2어절 단위 파편
  - 파편은 저작권 보호를 위해 원문을 2어절 단위로 분할하고 순서를 뒤섞은 것입니다.
  - 파편 구분자는 ' | '이며, [MASKED]는 저작권 보호를 위한 마스킹이므로 무시하세요.

이 데이터는 영상의 시각 프레임에서 추출된 구조화된 시각 정보입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (Scene Abstraction - 핵심 상황 유추):
  a) Subjects 파편에서 핵심 주체(인물, 동물, 사물 등)를 먼저 식별하세요.
  b) Contexts 파편에서 해당 주체의 행동, 위치, 상태를 교차 매칭하여 장면을 구체적으로 재구성하세요.
  c) 과거 맥락과 현재 파편들을 종합하여 현재 씬의 '핵심 상황/주제'를 1문장으로 유추하세요. (예: [이전 장면: 이탈리아 입국 -> 현재 장면: 피자, 빠니보틀] -> 유추: 이탈리아 정통 음식 문화 체험)
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (질문 기획): 1단계에서 유추된 상황을 바탕으로, 화면 내 사물에 대한 단순 묘사 질문을 엄격히 금지하고, queries[0]과 queries[1] 모두 시각 단서에서 파생된 서로 다른 주제의 Tangential 질문으로 기획.
"""


_VOICE_HINT_PROMPT_IMGVLM_SENTENCE = _VOICE_HINT_BASE_LOW_CONTEXT + """
[주의] [Video Context]는 저작권 제약으로 이 모드에서는 제공되지 않습니다. 오직 아래 데이터만으로 장르를 유추하세요.

[입력 형식 설명]
당신에게는 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 구조화된 데이터가 제공됩니다.
각 Scene은 시간 범위와 함께 <vlm_img_structure> 태그로 감싸진 형태입니다:
  - Subjects: 장면의 핵심 주체 (등장인물, 주요 피사체)
  - Contexts: 장면의 행동, 배경, 환경 및 맥락 정보

이 데이터는 영상의 시각 프레임에서 추출된 구조화된 시각 정보입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (Scene Abstraction - 핵심 상황 유추):
  a) Subjects에서 핵심 주체(인물, 동물, 사물 등)를 먼저 식별하세요.
  b) Contexts에서 해당 주체의 행동, 위치, 상태를 교차 매칭하여 장면을 구체적으로 재구성하세요.
  c) 과거 맥락과 현재 텍스트들을 종합하여 현재 씬의 '핵심 상황/주제'를 1문장으로 유추하세요. (예: [이전 장면: 이탈리아 입국 -> 현재 장면: 피자, 빠니보틀] -> 유추: 이탈리아 정통 음식 문화 체험)
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (질문 기획): 1단계에서 유추된 상황을 바탕으로, 화면 내 사물에 대한 단순 묘사 질문을 엄격히 금지하고, queries[0]과 queries[1] 모두 시각 단서에서 파생된 서로 다른 주제의 Tangential 질문으로 기획.
"""


_VOICE_HINT_PROMPT_IMGVLM_GRAPH = _VOICE_HINT_BASE_LOW_CONTEXT + """
[주의] [Video Context]는 저작권 제약으로 이 모드에서는 제공되지 않습니다. 오직 아래 데이터만으로 장르를 유추하세요.

[입력 형식 설명]
당신에게는 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 장면 지식 그래프(Scene Knowledge Graph)가 제공됩니다.
- vlm_graph: 장면의 주요 요소와 그 관계를 (subject) -[relation]-> (object) 형태의 트리플로 표현한 데이터
  예: (man) -[WEARING]-> (cap), (screen) -[ABOUT]-> (foreign policy)
- 이 그래프는 장면에 등장하는 인물, 사물, 행동, 속성, 위치 등의 관계를 압축적으로 나타냅니다.

이 데이터는 영상의 시각 프레임에서 추출된 관계형 메타데이터입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (Scene Abstraction - 핵심 상황 유추): 과거 맥락과 현재 지식 그래프의 트리플들을 논리적으로 연결하여 현재 씬의 '핵심 상황/주제'를 1문장으로 유추하세요. (예: [이전 장면: 이탈리아 입국 -> 현재 장면: 피자, 빠니보틀] -> 유추: 이탈리아 정통 음식 문화 체험)
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (질문 기획): 1단계에서 유추된 상황을 바탕으로, 화면 내 사물에 대한 단순 묘사 질문을 엄격히 금지하고, queries[0]과 queries[1] 모두 시각 단서에서 파생된 서로 다른 주제의 Tangential 질문으로 기획.
"""
_VOICE_HINT_PROMPT_RAW_WITH_MMVLM = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 음성 인식(ASR) 텍스트와 화면 글씨(OCR) 텍스트 데이터가 Scene 단위로 제공되며,
추가로 소형 VLM이 시각·음성을 종합하여 장면의 상황을 서술한 보조 참고 정보(vlm_mm_description)가 함께 제공됩니다.

각 Scene의 구조:
  - speech: Scene 내 모든 음성을 시간 순서대로 이어 붙인 통합 텍스트 **(1차 사실 소스)**
  - on_screen_text: Scene 내 화면 텍스트를 중복 제거하여 나열한 문자열 **(1차 사실 소스)**
  - vlm_mm_description: 시각·음성 종합 서술 **(보조 참고용 — 부정확할 수 있음)**

[분석 시 주의사항]
- **speech(음성)와 on_screen_text(화면 텍스트)를 먼저 읽고** 대화 흐름과 주제를 파악하세요.
- ASR과 OCR을 교차 검증하여 오탈자를 자연스럽게 교정하세요.
- vlm_mm_description은 시각적 맥락 보조로만 참고하되, **speech/on_screen_text와 충돌할 경우 speech/on_screen_text를 우선**하세요.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (장면 맥락 종합): speech와 on_screen_text를 먼저 분석하여 사실 관계를 파악한 뒤, vlm_mm_description으로 시각적 맥락을 보조적으로 보충. 충돌 시 speech/on_screen_text 우선.
- 2단계 (과거 정보 차단): 이전 과거 맥락에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 3단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 4단계 (질문 기획): 종합하여 유추한 현재 장면의 새 단서에 집중하여 queries[0]과 queries[1] 모두 서로 다른 핵심 주제에 직결된 Content-Anchored 질문으로 기획.
"""

_VOICE_HINT_PROMPT_KSS = _VOICE_HINT_BASE + """
[입력 형식 설명]
당신에게는 전문가가 영상을 직접 시청하고 작성한 **KeyScene Summary(핵심 장면 요약)**가 제공됩니다.
Summary는 다음 두 부분으로 구성됩니다:
- [1. 과거 장면 요약]: 현재 장면 이전까지의 주요 사건, 대화, 맥락 요약
- [2. 현재 장면 묘사]: 시청자가 방금 시청한 현재 장면의 상세 묘사

이 Summary는 영상의 시각·청각 정보를 종합한 가장 정확한 참조 자료입니다.

[사고 과정 (Chain-of-Thought) 가이드]
질문을 생성하기 전에 `rationale` 필드에 반드시 다음 4단계를 순서대로 작성하세요.
- 1단계 (과거 정보 차단): [1. 과거 장면 요약]에서 이미 밝혀진 사실이나 상식을 요약한 뒤 차단 선언.
- 2단계 (미래 추측 차단): 미래 지향적 질문을 차단하겠다고 선언.
- 3단계 (핵심 사건 유추): [2. 현재 장면 묘사]를 분석하여 지금 이 순간 일어나고 있는 핵심 사건(인물들의 중심 행동, 발언, 상황 등) 2가지를 명확히 식별하세요.
- 4단계 (질문 기획): 3단계에서 식별된 핵심 사건 2가지에 각각 매칭하여, queries[0]과 queries[1] 모두 서로 다른 핵심 주제에 직결된 Content-Anchored 질문으로 기획.
"""

_VOICE_HINT_PROMPT_META = """당신은 제공되는 비디오의 메타데이터를 기반으로 스마트 TV 플랫폼에서 시청자의 리모컨 상호작용과 플랫폼 체류 시간을 극대화하는 '개인화된 예상 질문' 생성 전문가입니다.

[주의] 이 모드는 저작권 및 플랫폼 제약으로 인해 영상 내용(비디오 및 상세 장면 맥락)에 전혀 접근할 수 없는 **메타데이터 전용 모드**입니다.
오직 제공되는 [Video Context] (채널명, 제목, 업로드 날짜, 설명 등)만을 분석하여 영상의 장르와 흐름을 추론하고, 시청자가 흥미로워할 만한 **곁다리 지식 질문(Tangential Knowledge) 2개**를 생성하세요.

[입력 형식 설명]
당신에게는 오직 [Video Context]만 제공되며, [이전까지의 과거 시청 맥락]이나 [현재 시청 중인 장면] 등 구체적인 영상의 정보나 대화/자막/시각 데이터는 일절 제공되지 않습니다.

■ Tangential Knowledge (곁다리 지식 질문)
  현재 장면에서 직접적인 영상 프레임 묘사 없이, 영상 제목/채널/설명에서 파생되는 흥미로운 곁다리 지식 질문입니다.
  영상의 주된 서사와 직접 관련되지 않더라도, 관련 분야 소품·장소·문화·기술 등에서 호기심을 자극하는 '깊이 있는 외부 지식'을 질문하세요.

[질문 생성 핵심 지침]
- 구체적인 장면이나 프레임 묘사 금지: 화면을 볼 수 없으므로 "방금 장면에서", "이 장면에 등장하는" 등 특정 시각적 프레임에 대한 언급은 피하세요. 대신 영상 제목이나 설명에 기반한 외부 상식 및 관련 지식 질문으로 구성하세요.
- 미래 예측 금지: 앞으로 전개될 스토리나 결과를 묻는 질문은 금지합니다.
- 서로 다른 주제의 질문 2개 생성: `queries` 배열에는 반드시 서로 다른 주제의 매력적인 곁다리 지식 질문 **2개**가 포함되어야 합니다.

[출력 형식]
- 언어: 한국어 (단, 영어 콘텐츠의 고유명사는 원어 병기 허용)
- 형식: 반드시 아래 JSON 형식으로 출력하세요. 다른 설명은 덧붙이지 마십시오.

[JSON 형식 예시]
{
    "rationale": "1) 메타데이터 분석: 채널명이 내셔널지오그래픽이고 제목이 일각고래 다큐멘터리임을 파악함. 2) 장면 언급 배제 선언: 구체적인 씬 묘사를 피하고 상식 질문으로 기획함. 3) Tangential 1 기획: 일각고래의 독특한 엄니(Tusk) 생물학적 기원에 대한 곁다리 질문을 기획함. 4) Tangential 2 기획: 북극 바다 얼음의 물리적 특징이나 극지 지형(폴리냐 등)에 대한 상식 질문을 기획함.",
    "queries": [
        "일각고래의 머리 앞에 길게 뻗은 엄니는 실제 이빨이 발달한 것일까요, 아니면 뿔일까요?",
        "북극해의 얼어붙은 빙하 한가운데에 얼지 않고 동그랗게 뚫려 있는 구멍인 '폴리냐'는 어떤 원리로 형성되나요?"
    ]
}"""

def make_voice_hint_configs(thinking_level=0):
    """Voice Hint 생성용 GenerateContentConfig 정보를 반환합니다."""
    return {
        "video": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_VIDEO, thinking_level=thinking_level),
        "raw": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_RAW, thinking_level=thinking_level),
        "raw_with_mmvlm": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_RAW_WITH_MMVLM, thinking_level=thinking_level),
        "imgvlm_chunk2": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_IMGVLM_CHUNK2, thinking_level=thinking_level),
        "imgvlm_sentence": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_IMGVLM_SENTENCE, thinking_level=thinking_level),
        "imgvlm_graph": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_IMGVLM_GRAPH, thinking_level=thinking_level),
        "kss": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_KSS, thinking_level=thinking_level),
        "meta": make_generate_config(system_instruction=_VOICE_HINT_PROMPT_META, thinking_level=thinking_level)
    }

def process_vh_modes(client, vh_model_name, vh_configs, past_parts, current_parts, end_time, target_modes=None, kss_text=None, video_context="", scene_idx=None, start_time=0.0):
    """하나의 Keypoint에 대해 Voice Hint를 지정된 모드에 대해서만 병렬 수행합니다."""
    if target_modes is None:
        target_modes = ["video", "raw_with_mmvlm", "imgvlm"]
        
    def generate_voice_hints(mode):
        contents = []

        # Video Context: 채널명/제목으로 도메인 컨텍스트 제공
        # (imgvlm 모드는 저작권 미계약 데이터이므로 메타데이터 주입 금지)
        _COPYRIGHT_SAFE_MODES = {"imgvlm_chunk2", "imgvlm_sentence", "imgvlm_graph"}
        if video_context and mode not in _COPYRIGHT_SAFE_MODES:
            contents += ["--- [Video Context (영상 기본 정보)] ---", video_context]

        # KSS 모드: Summary 텍스트를 직접 전달 (과거+현재 포함)
        if mode == "kss":
            if not kss_text:
                return {"queries": [], "rationale": "KSS Summary가 비어있어 생성을 생략합니다."}, 0.0
            contents += [
                "--- [KeyScene Summary] ---",
                kss_text,
                "--- 요청 사항 ---",
                "제공된 [KeyScene Summary]를 바탕으로 시스템 프롬프트의 [사고 과정 가이드] 4단계를 철저히 준수하여 질문 **2개**를 [JSON 형식 예시]와 같이 생성하세요. "
                "특히 3단계에서 [2. 현재 장면 묘사]의 핵심 사건들(인물들의 행동, 대화, 상황 등)에 집중하여, "
                "queries[0]과 queries[1] 모두 서로 다른 핵심 주제에 직결된 콘텐츠 특화 질문(Content-Anchored)으로 작성하세요. "
                "[1. 과거 장면 요약]에서 이미 밝혀진 사실을 묻는 뒷북 질문과 미래 지향적 질문은 금지입니다."
            ]
        elif mode == "meta":
            # meta 모드: 씬 식별 컨텍스트 없이 순수하게 메타데이터(Video Context)만을 사용하여 곁다리 질문 생성
            contents += [
                "--- 요청 사항 ---",
                "제공된 [Video Context]의 메타데이터만을 바탕으로, 시스템 프롬프트의 지침을 준수하여 서로 다른 주제의 매력적인 곁다리 지식 질문(Tangential Knowledge) **2개**를 [JSON 형식 예시]와 같이 생성하세요."
            ]
        else:
            current_part = current_parts.get(mode)
            if current_part is None or (isinstance(current_part, str) and not current_part.strip()):
                return {"queries": [], "rationale": f"해당 모드({mode})의 현재 장면 데이터가 비어있어 생성을 생략합니다."}, 0.0

            past_part = past_parts.get(mode)
            is_low_context = mode.startswith("imgvlm")
            has_past = past_part is not None and not (isinstance(past_part, str) and not past_part.strip())

            if has_past:
                contents += ["--- [이전까지의 과거 시청 맥락 (참조용)] ---", past_part]
            
            contents += ["--- [현재 시청 중인 장면 (분석 대상)] ---", current_part]
            
            if is_low_context:
                req_text = (
                    "위의 [이전까지의 과거 시청 맥락]과 [현재 시청 중인 장면]을 인지한 상태에서, 시스템 프롬프트의 [사고 과정 가이드] 지침에 따라 곁다리 지식 질문(Tangential Knowledge) **2개**를 [JSON 형식 예시]와 같이 생성하세요. 과거 맥락에서 이미 아는 내용으로 질문하는 것과 앞으로의 결과나 스토리 전개를 묻는 미래 지향적 질문을 피하고, 오직 방금 본 [현재 시청 중인 장면]에서 포착된 새로운 단서에 집중하세요."
                    if has_past else
                    "제공된 [현재 시청 중인 장면]을 바탕으로 시스템 프롬프트의 [사고 과정 가이드] 지침에 따라 곁다리 지식 질문(Tangential Knowledge) **2개**를 [JSON 형식 예시]와 같이 생성하세요. 앞으로의 결과나 스토리 전개를 묻는 미래 지향적 질문은 피하세요."
                )
            else:
                req_text = (
                    "위의 [이전까지의 과거 시청 맥락]과 [현재 시청 중인 장면]을 인지한 상태에서, 시스템 프롬프트의 [사고 과정 가이드] 단계를 철저히 준수하여 콘텐츠 특화 질문(Content-Anchored) **2개**를 [JSON 형식 예시]와 같이 생성하세요. 과거 맥락에서 이미 아는 내용으로 질문하는 것과 앞으로의 결과나 스토리 전개를 묻는 미래 지향적 질문을 피하고, 오직 방금 본 [현재 시청 중인 장면]에 새롭게 등장한 단서에 집중하세요."
                    if has_past else
                    "제공된 [현재 시청 중인 장면]을 바탕으로 시스템 프롬프트의 [사고 과정 가이드] 지침에 따라 콘텐츠 특화 질문(Content-Anchored) **2개**를 [JSON 형식 예시]와 같이 생성하세요. 앞으로의 결과나 스토리 전개를 묻는 미래 지향적 질문은 피하세요."
                )
            contents += ["--- 요청 사항 ---", req_text]
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
    parser.add_argument("--modes", nargs="+", default=["video", "kss", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph", "meta"], 
    choices=["kss", "video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph", "meta"], help="생성할 모드 직접 지정")

    args, client = init_pipeline(parser.parse_args())

    vh_configs = make_voice_hint_configs(thinking_level=args.vh_thinking_level)

    if not check_input_file(args.input_file, hint="먼저 identify_keypoint.py를 실행하세요."):
        return

    # Keypoint 목록 로드
    keypoints_by_content = load_keypoints_by_content(args.input_file)
    if not keypoints_by_content:
        print(f"Error: {args.input_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

    # content_id -> index 매핑 로드
    content_indices = {}
    if os.path.exists(args.input_file):
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        cid = data.get("content_id")
                        idx = data.get("index")
                        if cid and idx is not None:
                            content_indices[cid] = idx
                    except Exception:
                        pass
    metadata_path = "video_metadata.jsonl"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        cid = data.get("content_id")
                        idx = data.get("index")
                        if cid and idx is not None:
                            content_indices[cid] = idx
                    except Exception:
                        pass

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    # 기존 파일 정리: 빈 질문이나 중복 제거 후 다시 쓰기
    done_modes_by_scene = set()
    if os.path.exists(args.output_file):
        existing_records = load_jsonl(args.output_file)
        cleaned_records = {}
        for r in existing_records:
            c_id = r.get("content_id")
            s_idx = r.get("scene_idx")
            mode = r.get("mode")
            queries = r.get("queries", [])
            if c_id and s_idx is not None and mode and queries:
                # 중복이 있을 경우 가장 마지막(최신) 레코드로 덮어씀
                cleaned_records[(c_id, s_idx, mode)] = r
        
        with open(args.output_file, "w", encoding="utf-8") as f:
            for r in cleaned_records.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        done_modes_by_scene = set(cleaned_records.keys())

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

    # 1. 미처리 Keypoint 총 개수 및 작업 목록 집계
    total_keypoints_to_process = 0
    all_content_remaining_jobs = {}
    for content_id, keypoints in keypoints_by_content.items():
        remaining = []
        for kp in keypoints:
            real_idx = keypoints.index(kp)
            s_idx = kp.get("scene_idx", real_idx)
            missing_modes = [m for m in target_modes if (content_id, s_idx, m) not in done_modes_by_scene]
            if missing_modes:
                remaining.append((kp, missing_modes))
        if remaining:
            all_content_remaining_jobs[content_id] = remaining
            total_keypoints_to_process += len(remaining)

    print_pipeline_banner("Voice Hint 생성 파이프라인을 시작합니다.")
    print(f"Modes={target_modes}")
    print(f"[Info] 총 {len(all_content_remaining_jobs)}개 콘텐츠, {total_keypoints_to_process}개 Keypoint 생성 작업 대기 중")

    processed_keypoints = 0
    tracker = ProgressTracker(total_keypoints_to_process, unit="Keypoints", action="processed")

    try:
        for content_id, keypoints in keypoints_by_content.items():
            remaining = all_content_remaining_jobs.get(content_id, [])
            if not remaining:
                print(f"\n[Skip] '{content_id}': 모든 Scene 완벽 (누락 모드 없음)")
                continue

            fully_done_count = len(keypoints) - len(remaining)
            if fully_done_count > 0:
                print(f"\n[Resume] '{content_id}': {fully_done_count}/{len(keypoints)}개 Scene 완벽 처리됨, {len(remaining)}개 Scene(부분 누락 포함) 재개")

            print(f"{'='*50}")
            print(f"Processing Content: '{content_id}' ({len(remaining)}/{len(keypoints)}개 Keypoint 추가 처리)")
            print(f"{'='*50}")

            # GCS check and preload only if we need GCS data (i.e. modes other than kss and meta)
            requires_gcs = any(
                any(m not in ("kss", "meta") for m in original_missing_modes)
                for kp, original_missing_modes in remaining
            )
            if requires_gcs:
                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    continue
                # JSONL 메타데이터 프리로드 (캐시 워밍업)
                preload_content_metadata(args.gs_bucket_name, content_id)

            # Video Metadata 로드 (채널명, 제목 등)
            try:
                video_metadata = load_video_metadata(args.gs_bucket_name, content_id)
            except Exception as e:
                print(f"[ERROR] '{content_id}'의 메타데이터 로드 실패: {e}")
                continue

            video_context = format_video_context(video_metadata)
            if video_context:
                print(f"[Video Context] {video_metadata.get('channel', '')} — {video_metadata.get('title', '')}")
                print(f"\n[Description] {video_metadata.get('description', '')}")

            for kp, original_missing_modes in remaining:
                missing_modes = list(original_missing_modes)
                if not missing_modes:
                    processed_keypoints += 1
                    tracker.update(processed_keypoints)
                    continue

                real_idx = keypoints.index(kp)
                scene_idx = kp.get("scene_idx", real_idx)
                start_time = float(kp.get("start_time", 0.0))
                end_time = float(kp.get("end_time", 0.0))
                current_dur = end_time - start_time
                past_n = min(args.vh_gen_past_scenes_size, scene_idx)
                past_approx_sec = past_n * (start_time / scene_idx) if scene_idx > 0 else 0
                print(f"[{real_idx}/{len(keypoints)}] Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s] | Current: {current_dur:.1f}s, Past: {past_approx_sec:.0f}s ({past_n} scenes)")

                def _run_keypoint():
                    # 과거 N개 Scene을 위한 scene_idx 계산 (현재 Scene 직전 N개)
                    past_start_scene_idx = max(0, scene_idx - args.vh_gen_past_scenes_size)
                    past_end_scene_idx = scene_idx - 1  # 현재 Scene 직전까지

                    has_past = past_end_scene_idx >= past_start_scene_idx
                    # KSS 및 meta 외 모드만 GCS 데이터 빌드
                    non_kss_modes = [m for m in missing_modes if m not in ("kss", "meta")]
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

                    return process_vh_modes(
                        client, args.vh_gen_model, vh_configs, past_parts, current_parts, end_time, missing_modes, 
                        kss_text=kss_text, video_context=video_context, scene_idx=scene_idx, start_time=start_time
                    )

                try:
                    vh_dict, vh_elapsed_dict = _retry_api_call(
                        _run_keypoint,
                        label=f"Voice Hint (Scene {scene_idx})"
                    )

                    # 각 mode별로 별도의 줄로 저장 (content_id + scene_idx + mode = 1 line)
                    for mod in missing_modes:
                        queries = vh_dict.get(mod, [])
                        # High-context 모드는 content_anchored, Low-context 모드(imgvlm, meta)는 tangential
                        if mod.startswith("imgvlm") or mod == "meta":
                            query_types = ["tangential"] * len(queries)
                        else:
                            query_types = ["content_anchored"] * len(queries)
                        from collections import OrderedDict
                        mode_record = OrderedDict()
                        mode_record["index"] = content_indices.get(content_id, 0)
                        mode_record["content_id"] = content_id
                        mode_record["scene_idx"] = scene_idx
                        mode_record["mode"] = mod
                        mode_record["start_time"] = start_time
                        mode_record["end_time"] = end_time
                        mode_record["queries"] = queries
                        mode_record["query_types"] = query_types
                        mode_record["rationale"] = vh_dict.get("rationales", {}).get(mod, "")
                        append_jsonl(args.output_file, mode_record)
                        done_modes_by_scene.add((content_id, scene_idx, mod))

                    _LOG_MODES = {"video", "kss", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph", "meta"}
                    for mod in missing_modes:
                        if vh_dict.get(mod):
                            if mod in _LOG_MODES:
                                print(f"\n-> [VH - {mod}] ({vh_elapsed_dict.get(mod, 0.0):.2f}초)")
                                for qi, q in enumerate(vh_dict[mod], 1):
                                    print(f"{qi}. {q}")
                            else:
                                print(f"-> [VH - {mod}] ({vh_elapsed_dict.get(mod, 0.0):.2f}초) — {len(vh_dict[mod])}개 질문 생성됨")
                    print(f"------------------------------------------------------------------------------------------------------------\n")

                except Exception as e:
                    print(f"    [ERROR] 치명적 오류로 Scene {scene_idx} 건너뜁니다: {e}")
                    continue
                finally:
                    processed_keypoints += 1
                    tracker.update(processed_keypoints)

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

    if not is_perfect:
        print("\n[Info] 아직 누락된 Scene이나 모드가 있습니다. 스크립트를 다시 실행하면 누락분만 재처리됩니다.")

    print_pipeline_done(args.output_file)

if __name__ == "__main__":
    main()
