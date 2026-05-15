import os
import json
import time
import concurrent.futures
import threading
from utils import (
    get_common_argparser,
    make_generate_config,
    process_gcs_video_part,
    check_gcs_files_exist,
    _retry_api_call,
    get_gcs_raw_fields_by_scene_idx,
    get_processed_vlm_descriptions_by_scene_idx,
    get_gcs_raw_with_mmvlm_by_scene_idx,
    parse_duration_to_times,
    ensure_output_dir,
    preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
    sort_jsonl_file,
    load_keypoints_by_content, check_input_file,
    load_scenes,
    load_video_metadata, format_video_context,
    print_pipeline_banner, print_pipeline_done,
)

# ============================================================
# System Prompts
# ============================================================

_VH_RESPONSE_PROMPT_BASE = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

당신에게는 [Video Context]로 영상의 채널명과 제목이 제공될 수 있습니다. 이를 통해 콘텐츠의 장르와 도메인을 먼저 파악하여 답변의 방향성을 설정하세요.

당신에게는 영상의 처음부터 시청자가 현재 보고 있는 장면까지, 소형 AI가 자동으로 분석·요약한 **누적 시청 기억(Accumulated Memory)**이 제공됩니다.
이 기억에는 Scene 단위로 분절된 영상 정보가 시간순으로 나열되어 있으며, **마지막 Scene이 시청자가 지금 집중하고 있는 장면**입니다.

이 정보를 바탕으로 시청자의 질문에 자연스럽고 정확하게 답변하여 '개인화된 인터랙티브 시청 경험'을 극대화해 주세요."""

_VH_RESPONSE_PROMPT_RAW = _VH_RESPONSE_PROMPT_BASE + """

[시청 기억의 구조]
제공되는 시청 기억은 '음성 기록(ASR)'과 '화면 텍스트(OCR)'로 구성되며, Scene 단위로 제공됩니다.
각 Scene의 구조:
  - scene_idx: Scene 인덱스
  - duration: Scene 시작~종료 시간
  - speech: Scene 내 모든 음성을 시간 순서대로 이어 붙인 통합 텍스트
  - on_screen_text: Scene 내 화면 텍스트를 중복 제거하여 나열한 문자열

[분석 및 대화 지시사항]
1. **텍스트 분석 및 노이즈 교정 (매우 중요)**
   - speech는 연속된 내러티브로 읽고, 문맥과 상식을 동원하여 ASR 오탈자를 자연스럽게 교정하세요.
   - ASR과 OCR을 교차 검증하여 오탈자를 교정하고, 단어들을 논리적으로 조합하여 상황을 유추해야 합니다.

2. **적극적 지식 활용과 만족스러운 답변 (최우선 원칙)**
   - 시청자에게 만족스럽고 유익한 답변을 제공하는 것이 최우선 목표입니다.
   - 영상 내용만으로 답변이 불충분할 경우, 당신이 가진 사전 지식(World Knowledge)을 적극적으로 활용하세요.
   - 유추한 발화 맥락과 사전 지식을 결합하여, 시청자가 "이 AI는 정말 똑똑하다!"라고 느끼도록 풍부하고 정확한 답변을 제공하세요.
   - 단, 데이터에서 유추한 내용과 명백히 모순되는 정보는 제공하지 마세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.
   - 과거 Scene은 질문의 맥락을 이해하는 데 참고하되, 답변의 생동감은 현재 장면에서 끌어오세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "텍스트 조각을 조합해보면" 등 데이터 구조를 암시하는 말은 절대 금지합니다.
   - "방금 들린 대화에 따르면~", "화면에 나온 정보에서~"처럼 실제 영상의 소리를 듣거나 텍스트를 본 것처럼 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""


_VH_RESPONSE_PROMPT_RAW_WITH_MMVLM = _VH_RESPONSE_PROMPT_BASE + """

[시청 기억의 구조]
제공되는 시청 기억은 '음성 기록(ASR)'과 '화면 텍스트(OCR)'가 Scene 단위로 구성되며,
추가로 소형 VLM이 시각·음성을 종합하여 장면을 서술한 보조 참고 정보(vlm_mm_description)가 함께 제공됩니다.

각 Scene의 구조:
  - scene_idx: Scene 인덱스
  - duration: Scene 시작~종료 시간
  - speech: Scene 내 모든 음성을 시간 순서대로 이어 붙인 통합 텍스트 **(1차 사실 소스)**
  - on_screen_text: Scene 내 화면 텍스트를 중복 제거하여 나열한 문자열 **(1차 사실 소스)**
  - vlm_mm_description: 시각·음성 종합 서술 **(보조 참고용 — 부정확할 수 있음)**

[분석 및 대화 지시사항]
1. **텍스트 우선 분석 (매우 중요)**
   - **speech(음성)와 on_screen_text(화면 텍스트)를 먼저 읽고** 대화 흐름, 등장인물, 주제를 파악하세요.
   - ASR과 OCR을 교차 검증하여 오탈자를 교정하세요.
   - vlm_mm_description은 시각적 맥락 보조로만 참고하되, **speech/on_screen_text와 충돌할 경우 speech/on_screen_text를 우선**하세요.

2. **적극적 지식 활용과 만족스러운 답변 (최우선 원칙)**
   - 시청자에게 만족스럽고 유익한 답변을 제공하는 것이 최우선 목표입니다.
   - 영상 내용만으로 답변이 불충분할 경우, 당신이 가진 사전 지식(World Knowledge)을 적극적으로 활용하세요.
   - 단, 데이터에서 유추한 내용과 명백히 모순되는 정보는 제공하지 마세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "텍스트 조각을 조합해보면" 등 데이터 구조를 암시하는 말은 절대 금지합니다.
   - "방금 들린 대화에 따르면~", "화면에 나온 정보에서~"처럼 실제 영상의 소리를 듣거나 텍스트를 본 것처럼 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""

_VH_RESPONSE_PROMPT_VIDEO = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

당신에게는 [Video Context]로 영상의 채널명과 제목이 제공될 수 있습니다. 이를 통해 콘텐츠의 장르와 도메인을 먼저 파악하여 답변의 방향성을 설정하세요.

당신에게는 영상의 처음부터 시청자가 현재 보고 있는 장면까지의 **비디오 클립**이 직접 제공됩니다.
클립의 **마지막 부분이 시청자가 지금 집중하고 있는 장면**입니다.

이 영상을 직접 시청·분석하여 시청자의 질문에 자연스럽고 정확하게 답변해 주세요.

[분석 및 대화 지시사항]

1. **정밀한 시각·청각 분석과 지식 보강**
   - 화면에 보이는 인물, 행동, 자막, 로고, 배경, 소리 등 모든 정보를 종합적으로 활용하세요.
   - 영상 내용만으로 답변이 불충분할 경우, 당신이 가진 사전 지식을 적극적으로 활용하여 풍부하고 유익한 답변을 제공하세요.

2. **적극적 지식 활용과 만족스러운 답변 (최우선 원칙)**
   - 시청자에게 만족스럽고 유익한 답변을 제공하는 것이 최우선 목표입니다.
   - 화면에 직접 명시되지 않은 질문이라도 "알 수 없습니다"로 대화를 끊지 마세요.
   - 영상에서 얻은 맥락과 사전 지식을 자연스럽게 결합하여 풍부하고 정확한 답변을 제공하세요.
   - 단, 영상 내용과 명백히 모순되는 정보는 제공하지 마세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 영상 후반부(= 현재 장면)에 가장 높은 우선순위를 두세요.
   - 영상 초반은 질문의 맥락을 이해하는 데 참고하되, 답변의 생동감은 현재 장면에서 끌어오세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "프레임", "타임스탬프", "비디오 분석 결과" 등 시스템 용어는 절대 금지합니다.
   - "지금 화면을 보면~", "방금 나온 장면에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""

_VH_RESPONSE_PROMPT_IMGVLM_CHUNK2 = _VH_RESPONSE_PROMPT_BASE + """
[주의] [Video Context]는 저작권 제약으로 이 모드에서는 제공되지 않습니다. 오직 아래 데이터만으로 장르를 유추하세요.

[시청 기억의 구조]
제공되는 시청 기억은 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 **구조화된 시각 정보**만으로 이루어져 있습니다.
각 Scene은 시간 범위와 함께 <vlm_img_structure> 태그로 감싸진 형태입니다:
  - Subjects: 장면의 핵심 주체 (등장인물, 주요 피사체) — 2어절 단위 파편
  - Contexts: 장면의 행동, 배경, 환경 및 맥락 정보 — 2어절 단위 파편
  - 파편은 저작권 보호를 위해 원문을 2어절 단위로 분할하고 순서를 뒤섞은 것입니다.
  - 파편 구분자는 ' | '이며, [MASKED]는 저작권 보호 마스킹이므로 무시하세요.

[분석 및 대화 지시사항]
1. **시각 정보의 역할 제한 (Ice-breaking 전용)**
   - 제공된 구조화 시각 데이터는 절대 정답의 논리적 출처나 본문의 논거로 억지로 끼워 넣지 마세요.
   - 오직 답변의 첫 문장에서 시청자가 방금 본 상황에 가볍게 공감하는 인사말(Ice-breaking 브릿지)을 만드는 용도로만 사용하세요.
   - 예: "아, 방금 이탈리아 골목에서 화덕 피자가 나오는 모습을 보셨군요!"

2. **방대한 외부 지식의 전면적 활용 (답변 본문)**
   - 답변의 본문은 전적으로 당신의 방대한 외부 지식(World Knowledge)을 최우선으로 활용하여 작성하세요.
   - 빈약한 파편화 정보에 억지로 논리를 맞추려다 문맥이 꼬이지 않도록 주의하세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.
   - 과거 Scene은 질문의 맥락을 이해하는 데 참고하되, 답변의 생동감은 현재 장면에서 끌어오세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "메타데이터", "구조화 데이터", "Subjects 필드", "파편", "vlm_img_structure" 등 시스템 용어는 절대 금지입니다.
   - "지금 화면을 보면~", "방금 나온 장면에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""

_VH_RESPONSE_PROMPT_IMGVLM_GRAPH = _VH_RESPONSE_PROMPT_BASE + """
[주의] [Video Context]는 저작권 제약으로 이 모드에서는 제공되지 않습니다. 오직 아래 데이터만으로 장르를 유추하세요.

[시청 기억의 구조]
제공되는 시청 기억은 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 **장면 지식 그래프(Scene Knowledge Graph)**만으로 이루어져 있습니다.
각 Scene별로 다음의 관계형 정보가 제공됩니다:
  - vlm_graph: 장면의 주요 요소와 그 관계를 (subject) -[relation]-> (object) 형태의 트리플로 표현
  예: (man) -[WEARING]-> (cap), (screen) -[ABOUT]-> (foreign policy)

이 그래프는 장면에 등장하는 인물, 사물, 행동, 속성, 위치 등의 관계를 압축적으로 나타냅니다.

[분석 및 대화 지시사항]
1. **시각 정보의 역할 제한 (Ice-breaking 전용)**
   - 제공된 지식 그래프(vlm_graph) 데이터는 절대 정답의 논리적 출처나 본문의 논거로 억지로 끼워 넣지 마세요.
   - 오직 답변의 첫 문장에서 시청자가 방금 본 상황에 가볍게 공감하는 인사말(Ice-breaking 브릿지)을 만드는 용도로만 사용하세요.
   - 예: "아, 방금 이탈리아 골목에서 화덕 피자가 나오는 모습을 보셨군요!"

2. **방대한 외부 지식의 전면적 활용 (답변 본문)**
   - 답변의 본문은 전적으로 당신의 방대한 외부 지식(World Knowledge)을 최우선으로 활용하여 작성하세요.
   - 빈약한 트리플 데이터에 억지로 논리를 맞추려다 문맥이 꼬이지 않도록 주의하세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "그래프", "트리플", "관계 데이터", "메타데이터" 등 시스템 용어는 절대 금지입니다.
   - "지금 화면을 보면~", "방금 나온 장면에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""
   
_VH_RESPONSE_PROMPT_IMGVLM_CHUNK2_META = _VH_RESPONSE_PROMPT_BASE + """

[시청 기억의 구조]
제공되는 시청 기억은 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 **구조화된 시각 정보**만으로 이루어져 있습니다.
각 Scene은 시간 범위와 함께 <vlm_img_structure> 태그로 감싸진 형태입니다:
  - Subjects: 장면의 핵심 주체 (등장인물, 주요 피사체) — 2어절 단위 파편
  - Contexts: 장면의 행동, 배경, 환경 및 맥락 정보 — 2어절 단위 파편
  - 파편은 저작권 보호를 위해 원문을 2어절 단위로 분할하고 순서를 뒤섞은 것입니다.
  - 파편 구분자는 ' | '이며, [MASKED]는 저작권 보호 마스킹이므로 무시하세요.

[분석 및 대화 지시사항]
1. **시각 정보의 역할 제한 (Ice-breaking 전용)**
   - 제공된 구조화 시각 데이터는 절대 정답의 논리적 출처나 본문의 논거로 억지로 끼워 넣지 마세요.
   - 오직 답변의 첫 문장에서 시청자가 방금 본 상황에 가벽게 공감하는 인사말(Ice-breaking 브릿지)을 만드는 용도로만 사용하세요.

2. **방대한 외부 지식의 전면적 활용 (답변 본문)**
   - 답변의 본문은 전적으로 당신의 방대한 외부 지식(World Knowledge)을 최우선으로 활용하여 작성하세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "메타데이터", "구조화 데이터", "Subjects 필드", "파편", "vlm_img_structure" 등 시스템 용어는 절대 금지입니다.
   - "지금 화면을 보면~", "방금 나온 장면에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑팡을 유도하세요."""

_VH_RESPONSE_PROMPT_IMGVLM_GRAPH_META = _VH_RESPONSE_PROMPT_BASE + """

[시청 기억의 구조]
제공되는 시청 기억은 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 **장면 지식 그래프(Scene Knowledge Graph)**만으로 이루어져 있습니다.
각 Scene별로 다음의 관계형 정보가 제공됩니다:
  - vlm_graph: 장면의 주요 요소와 그 관계를 (subject) -[relation]-> (object) 형태의 트리플로 표현
  예: (man) -[WEARING]-> (cap), (screen) -[ABOUT]-> (foreign policy)

이 그래프는 장면에 등장하는 인물, 사물, 행동, 속성, 위치 등의 관계를 압축적으로 나타냅니다.

[분석 및 대화 지시사항]
1. **시각 정보의 역할 제한 (Ice-breaking 전용)**
   - 제공된 지식 그래프(vlm_graph) 데이터는 절대 정답의 논리적 출처나 본문의 논거로 억지로 끼워 넣지 마세요.
   - 오직 답변의 첫 문장에서 시청자가 방금 본 상황에 가벽게 공감하는 인사말(Ice-breaking 브릿지)을 만드는 용도로만 사용하세요.

2. **방대한 외부 지식의 전면적 활용 (답변 본문)**
   - 답변의 본문은 전적으로 당신의 방대한 외부 지식(World Knowledge)을 최우선으로 활용하여 작성하세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "그래프", "트리플", "관계 데이터", "메타데이터" 등 시스템 용어는 절대 금지입니다.
   - "지금 화면을 보면~", "방금 나온 장면에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 다음 장면에 대한 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑팡을 유도하세요."""

_VH_RESPONSE_PROMPT_BLANK = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

당신에게는 [Video Context]로 영상의 채널명과 제목이 제공될 수 있습니다. 이를 참고하여 콘텐츠의 장르와 도메인을 파악하세요.

당신에게는 현재 영상에 대한 시청 정보는 제공되지 않습니다.
[Video Context]와 당신이 가진 사전 지식(World Knowledge)을 활용하여 시청자의 질문에 답변해 주세요.

[대화 지시사항]
1. **적극적 지식 활용과 만족스러운 답변 (최우선 원칙)**
   - 시청자에게 만족스럽고 유익한 답변을 제공하는 것이 최우선 목표입니다.
   - 질문의 맥락과 키워드를 단서로 삼아, 당신이 가진 사전 지식(World Knowledge)을 최대한 활용하세요.
   - 시청자가 "이 AI는 정말 똑똑하다!"라고 느끼도록 풍부하고 정확한 답변을 제공하세요.

2. **완벽한 TV 파트너 톤앤매너**
   - "데이터가 없어서" 등 시스템 한계를 암시하는 말은 절대 금지합니다.
   - "아, 그거요!" 처럼 자연스럽고 친숙한 구어체를 사용하세요.

3. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가벼운 공감이나 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""


def make_vh_gen_config(thinking_level=None):
    """VH Response 생성용 GenerateContentConfig 딕셔너리를 반환합니다."""
    return {
        "raw": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_RAW,
            thinking_level=thinking_level,
        ),
        "imgvlm_chunk2": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_IMGVLM_CHUNK2,
            thinking_level=thinking_level,
        ),
        "imgvlm_chunk2_meta": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_IMGVLM_CHUNK2_META,
            thinking_level=thinking_level,
        ),
        "imgvlm_graph": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_IMGVLM_GRAPH,
            thinking_level=thinking_level,
        ),
        "imgvlm_graph_meta": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_IMGVLM_GRAPH_META,
            thinking_level=thinking_level,
        ),
        "raw_with_mmvlm": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_RAW_WITH_MMVLM,
            thinking_level=thinking_level,
        ),
        "video": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_VIDEO,
            thinking_level=thinking_level,
        ),
        "blank": make_generate_config(
            system_instruction=_VH_RESPONSE_PROMPT_BLANK,
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
        mode: 'video' | 'raw' | 'raw_with_mmvlm' | 'imgvlm_chunk2' | 'imgvlm_graph' | 'blank'
        max_past_scenes: None이면 Scene 0부터 전체, 정수이면 현재 KeyScene 기준 최근 N개 Scene만 사용

    Returns:
        (source, label) — source는 API contents 리스트에 넣을 수 있는 Part 또는 str
    """
    # 현재 KeyScene의 end_time을 ref JSONL에서 조회
    ref_scenes = load_scenes(gs_bucket_name, content_id, mode="ref")

    if mode == "blank":
        return None

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
            video_start = parse_duration_to_times(start_scene["duration"])[0] if start_scene and start_scene.get("duration") else 0.0
        else:
            video_start = 0.0

        end_scene = next((s for s in ref_scenes if s.get("scene_idx") == scene_idx), None)
        video_end = parse_duration_to_times(end_scene["duration"])[1] if end_scene and end_scene.get("duration") else 0.0

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

        if mode == "raw":
            return get_gcs_raw_fields_by_scene_idx(
                gs_bucket_name, content_id, start_idx, scene_idx
            )
        elif mode in ("imgvlm_chunk2", "imgvlm_chunk2_meta"):
            return get_processed_vlm_descriptions_by_scene_idx(
                gs_bucket_name, content_id, "vlm_img_structure_chunk2", start_idx, scene_idx
            )
        elif mode in ("imgvlm_graph", "imgvlm_graph_meta"):
            return get_processed_vlm_descriptions_by_scene_idx(
                gs_bucket_name, content_id, "vlm_graph", start_idx, scene_idx
            )
        elif mode == "raw_with_mmvlm":
            return get_gcs_raw_with_mmvlm_by_scene_idx(
                gs_bucket_name, content_id, start_idx, scene_idx
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")


def _generate_for_mode(client, model_name, gen_configs, source, mode, query, scene_idx, video_context=""):
    """단일 모드에 대해 Response를 생성합니다."""
    start_time = time.time()
    try:
        time.sleep(1)

        # Video Context: 채널명/제목으로 도메인 컨텍스트 제공
        # (imgvlm 모드는 저작권 미계약 데이터이므로 메타데이터 주입 금지)
        _COPYRIGHT_SAFE_MODES = {"imgvlm_chunk2", "imgvlm_graph"}
        ctx_prefix = []
        if video_context and mode not in _COPYRIGHT_SAFE_MODES:
            ctx_prefix = ["--- [Video Context (영상 기본 정보)] ---", video_context]

        if mode == "blank":
            contents = ctx_prefix + [
                "--- 질문 ---",
                query,
            ]
        elif mode == "video":
            contents = ctx_prefix + [
                "--- [Video Clip (처음부터 현재 장면까지)] ---",
                source,
                "--- 질문 ---",
                query,
            ]
        else:
            label_map = {
                "raw": "Raw Metadata (speech & text, 처음부터 현재 장면까지)",
                "imgvlm_chunk2": "VLM Image Structure — 2-word Chunks (처음부터 현재 장면까지)",
                "imgvlm_chunk2_meta": "VLM Image Structure — 2-word Chunks + Meta (처음부터 현재 장면까지)",
                "imgvlm_graph": "VLM Scene Knowledge Graph (처음부터 현재 장면까지)",
                "imgvlm_graph_meta": "VLM Scene Knowledge Graph + Meta (처음부터 현재 장면까지)",
                "raw_with_mmvlm": "Raw ASR/OCR + VLM Multimodal Description (처음부터 현재 장면까지)",
            }
            contents = ctx_prefix + [
                f"--- [{label_map.get(mode, mode)}] ---",
                source,
                "--- 질문 ---",
                query,
            ]
            
        config = gen_configs[mode]

        answer = _retry_api_call(
            lambda: client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            ).text,
            label=f"VH Response [{mode}] (Scene {scene_idx})",
        )
        elapsed = time.time() - start_time
        return mode, answer, elapsed

    except Exception as e:
        print(f"  [ERROR] [{mode}] 생성 실패 (Scene {scene_idx}): {e}")
        elapsed = time.time() - start_time
        return mode, f"Error: {str(e)}", elapsed


# ============================================================
# Progress Helpers
# ============================================================

def _load_completed_pairs(output_path):
    """완료된 (content_id, scene_idx, mode, query) 쌍을 반환합니다."""
    completed = set()
    for rec in load_jsonl(output_path):
        c_id = rec.get("content_id")
        s_idx = rec.get("scene_idx")
        mode = rec.get("mode")
        query = rec.get("query")
        answer = rec.get("answer")
        
        if not (c_id and s_idx is not None and mode and query):
            continue
            
        if answer and not str(answer).startswith("Error"):
            completed.add((c_id, s_idx, mode, query))
    return completed


# ============================================================
# Validation
# ============================================================

def _validate_vh_responses(output_file, vh_input_file, target_modes):
    """VH Response 파일을 정렬하고, KSS Query × target mode 기준으로 누락을 점검합니다."""
    if not os.path.exists(output_file):
        print(f"[Warning] 파일을 찾을 수 없습니다: {output_file}")
        return

    # 1) 정렬
    sort_jsonl_file(output_file)

    print(f"\n{'='*50}")
    print(f"결과 파일 검증: {output_file}")

    data_records = load_jsonl(output_file)

    # 2) KSS Query × target mode 기준 누락 점검
    print("\n[최종 누락분 점검]")

    # KSS VH에서 기대되는 (content_id, scene_idx, query) 집합 추출
    expected = set()
    if os.path.exists(vh_input_file):
        for rec in load_jsonl(vh_input_file):
            if rec.get("mode") != "kss":
                continue
            c_id = rec.get("content_id")
            s_idx = rec.get("scene_idx")
            for q in rec.get("queries", []):
                for m in target_modes:
                    expected.add((c_id, s_idx, m, q))

    # 실제 존재하는 (content_id, scene_idx, mode, query) 집합
    done_set = {
        (x.get("content_id"), x.get("scene_idx"), x.get("mode"), x.get("query"))
        for x in data_records
    }

    missing = sorted(expected - done_set)
    if missing:
        # Scene 단위로 그룹핑하여 출력
        missing_by_scene = {}
        for c, s, m, q in missing:
            missing_by_scene.setdefault((c, s), []).append(m)

        print(f"-> 총 {len(missing)}개의 (Scene, Mode, Query) 처리가 누락되었습니다. ({len(missing_by_scene)}개 Scene)")
        for (c, s), modes in sorted(missing_by_scene.items())[:20]:
            mode_counts = {}
            for m in modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            summary = ", ".join(f"{m}({n})" for m, n in mode_counts.items())
            print(f"    - ({c}, scene={s}): {summary}")
        if len(missing_by_scene) > 20:
            print(f"    ... 외 {len(missing_by_scene) - 20}개 Scene")
    else:
        print("-> 모든 KSS Query × Mode 조합이 누락 없이 정상적으로 생성되었습니다.")
    print("=" * 50 + "\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = get_common_argparser(description="KSS 모드 Voice Hint의 질문을 공통 Query로 삼아, 각 모드의 Source로 Response를 생성합니다.")
    parser.add_argument("--input_file", default="assets/voice_hint.jsonl", help="Voice Hint JSONL 경로")
    parser.add_argument("--output_file", default="assets/vh_responses.jsonl", help="VH Response 저장 경로")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로")
    parser.add_argument("--modes", nargs="+",
                        default=["video", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk2_meta", "imgvlm_graph", "imgvlm_graph_meta", "blank"],
                        choices=["video", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_chunk2_meta", "imgvlm_graph", "imgvlm_graph_meta", "blank"],
                        help="Response를 생성할 대상 모드 (KSS Query를 이 모드들의 Source로 답변). blank=컨텍스트 없이 World Knowledge만 사용")

    args, client = init_pipeline(parser.parse_args())

    # Keypoint 로드 (scene_idx → start/end_time 매핑용)
    keypoints_by_content = load_keypoints_by_content(args.keypoints_file)
    if not keypoints_by_content:
        print(f"Error: {args.keypoints_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

    target_modes = args.modes

    # Gen configs
    gen_configs = make_vh_gen_config(thinking_level=args.vh_response_thinking_level)

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    if not check_input_file(args.input_file, hint="먼저 generate_voice_hint.py를 실행하세요."):
        return

    print_pipeline_banner("VH Response 생성 파이프라인을 시작합니다.")
    print(f"[Mode] KSS Query → Target Response Modes: {target_modes}")
    if args.vh_response_past_scenes_size:
        print(f"[Window] vh_response_past_scenes_size={args.vh_response_past_scenes_size} 설정: 현재 KeyScene 기준 최근 {args.vh_response_past_scenes_size}개 KeyPoint 내 Scene만 Source로 사용합니다.")

    file_write_lock = threading.Lock()
    _checked_contents = set()

    def _process_kss_record(rec):
        """KSS VH 레코드의 queries를 공통 Query로 삼아 모든 target 모드에 대해 Response를 병렬 생성합니다."""
        c_id = rec.get("content_id")
        s_idx = rec.get("scene_idx")
        queries = rec.get("queries", [])

        if not (c_id and s_idx is not None and queries):
            return 0

        # 각 target 모드 × 각 query 조합에서 미처리분만 추출
        pending_items = []
        for q_text in queries:
            for mode in target_modes:
                if (c_id, s_idx, mode, q_text) not in completed_pairs:
                    pending_items.append({
                        "content_id": c_id,
                        "scene_idx":  s_idx,
                        "mode":       mode,
                        "query":      q_text,
                    })

        if not pending_items:
            return 0

        keypoints = keypoints_by_content.get(c_id, [])
        if c_id not in _checked_contents:
            if not check_gcs_files_exist(args.gs_bucket_name, c_id):
                return 0
            preload_content_metadata(args.gs_bucket_name, c_id)
            _checked_contents.add(c_id)

        # Video Metadata 로드 (채널명, 제목 등)
        video_metadata = load_video_metadata(args.gs_bucket_name, c_id)
        video_context = format_video_context(video_metadata)
        start_time = float(rec.get("start_time", 0.0))
        end_time = float(rec.get("end_time", 0.0))
        print(f"\n[VH Response] '{c_id}' Scene {s_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s] | KSS Queries → {len(pending_items)}개 (mode×query) 조합 처리")

        # Source 캐시: 같은 Scene의 같은 mode는 Source를 1번만 빌드
        source_cache = {}
        count = 0

        # Query 단위로 그룹핑
        from collections import OrderedDict
        query_groups = OrderedDict()
        for item in pending_items:
            query_groups.setdefault(item["query"], []).append(item)

        for q_text, items in query_groups.items():
            print(f"\n[{c_id} | Scene {s_idx}] \nQuery: {q_text}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_modes)) as executor:
                futures = {}
                for item in items:
                    def process_item(item_data):
                        _s_idx = item_data["scene_idx"]
                        _m = item_data["mode"]
                        _q = item_data["query"]
                        try:
                            # Source 캐시 활용 (같은 Query 내에서는 _m이 중복되지 않으므로 Thread-safe)
                            if _m not in source_cache:
                                source_cache[_m] = _build_source(
                                    args.gs_bucket_name, c_id,
                                    _s_idx, keypoints, _m,
                                    max_past_scenes=args.vh_response_past_scenes_size,
                                )
                            source = source_cache[_m]
                            _, answer, elapsed = _generate_for_mode(
                                client, args.vh_response_model, gen_configs,
                                source, _m, _q, _s_idx, video_context=video_context
                            )
                            return item_data, answer, elapsed, None
                        except Exception as e:
                            return item_data, None, 0.0, str(e)

                    futures[executor.submit(process_item, item)] = item

                for future in concurrent.futures.as_completed(futures):
                    item, answer, elapsed, error = future.result()
                    scene_idx = item["scene_idx"]
                    mode      = item["mode"]
                    query     = item["query"]

                    if error:
                        print(f"  -> [{mode}] ERROR: {error}")
                        answer = f"Error: {error}"
                    else:
                        length_info = len(answer) if not answer.startswith("Error") else 0
                        print(f"  -> [{mode}] OK ({elapsed:.1f}s, {length_info}자)")

                    record = {
                        "content_id": c_id,
                        "scene_idx":  scene_idx,
                        "mode":       mode,
                        "query":      query,
                        "answer":     answer,
                    }
                    append_jsonl(args.output_file, record, lock=file_write_lock)
                    completed_pairs.add((c_id, scene_idx, mode, query))
                    count += 1

        return count

    MAX_RETRY = 3
    try:
        discovery_pass = 0
        while True:
            discovery_pass += 1

            # 매 pass마다 입력 파일을 다시 읽어 새로 추가된 KSS 레코드를 감지
            completed_pairs = _load_completed_pairs(args.output_file)
            if discovery_pass == 1 and completed_pairs:
                print(f"[기처리] {len(completed_pairs)}개 항목이 이미 처리 완료됨.")

            all_kss_records = [
                r for r in load_jsonl(args.input_file)
                if r.get("mode") == "kss" and not r.get("pipeline_done")
            ]

            if not all_kss_records:
                if discovery_pass == 1:
                    print("[Error] KSS 모드 레코드가 없습니다. generate_voice_hint.py를 먼저 실행하세요.")
                    return
                else:
                    print("[완료] 입력 파일에 처리할 KSS 레코드가 없습니다.")
                    break

            # 미처리 항목 개수 확인
            pending_count = sum(
                1 for r in all_kss_records
                for q in r.get("queries", [])
                for m in target_modes
                if (r.get("content_id"), r.get("scene_idx"), m, q) not in completed_pairs
            )

            if pending_count == 0:
                print("[완료] 모든 항목이 처리되었습니다.")
                break

            print(f"\n{'='*50}")
            print(f"[Discovery {discovery_pass}] KSS 레코드 {len(all_kss_records)}개, 미처리 {pending_count}개 조합 발견")
            print(f"{'='*50}")

            for rec in all_kss_records:
                _process_kss_record(rec)

            print(f"\n▶ [Discovery {discovery_pass}] 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    # 결과 파일 정렬 및 누락 점검
    _validate_vh_responses(args.output_file, args.input_file, target_modes)

    print_pipeline_done(args.output_file)


if __name__ == "__main__":
    main()
