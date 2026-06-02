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
    load_keypoints_by_content, check_input_file, load_content_indices,
    load_scenes,
    load_video_metadata, format_video_context,
    print_pipeline_banner, print_pipeline_done, ProgressTracker,
)

# ============================================================
# System Prompts
# ============================================================

_INTERACTIVE_QUERY_RESPONSE_PROMPT_BASE = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

당신에게는 [Video Context]로 영상의 채널명과 제목이 제공될 수 있습니다. 이를 통해 콘텐츠의 장르와 도메인을 먼저 파악하여 답변의 방향성을 설정하세요.

당신에게는 영상의 처음부터 시청자가 현재 보고 있는 장면까지, 소형 AI가 자동으로 분석·요약한 **누적 시청 기억(Accumulated Memory)**이 제공됩니다.
이 기억에는 Scene 단위로 분절된 영상 정보가 시간순으로 나열되어 있으며, **마지막 Scene이 시청자가 지금 집중하고 있는 장면**입니다.

이 정보를 바탕으로 시청자의 질문에 자연스럽고 정확하게 답변하여 '개인화된 인터랙티브 시청 경험'을 극대화해 주세요."""

_INTERACTIVE_QUERY_RESPONSE_PROMPT_RAW = _INTERACTIVE_QUERY_RESPONSE_PROMPT_BASE + """

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


_INTERACTIVE_QUERY_RESPONSE_PROMPT_RAW_WITH_MMVLM = _INTERACTIVE_QUERY_RESPONSE_PROMPT_BASE + """

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

_INTERACTIVE_QUERY_RESPONSE_PROMPT_VIDEO = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

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

_INTERACTIVE_QUERY_RESPONSE_PROMPT_IMGVLM_CHUNK2 = _INTERACTIVE_QUERY_RESPONSE_PROMPT_BASE + """
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

_INTERACTIVE_QUERY_RESPONSE_PROMPT_IMGVLM_SENTENCE = _INTERACTIVE_QUERY_RESPONSE_PROMPT_BASE + """
[주의] [Video Context]는 저작권 제약으로 이 모드에서는 제공되지 않습니다. 오직 아래 데이터만으로 장르를 유추하세요.

[시청 기억의 구조]
제공되는 시청 기억은 소형 VLM이 영상의 시각 프레임만을 분석하여 추출한 **구조화된 시각 정보**만으로 이루어져 있습니다.
각 Scene은 시간 범위와 함께 <vlm_img_structure> 태그로 감싸진 형태입니다:
  - Subjects: 장면의 핵심 주체 (등장인물, 주요 피사체)
  - Contexts: 장면의 행동, 배경, 환경 및 맥락 정보

이 데이터는 영상의 시각 프레임에서 추출된 구조화된 시각 정보입니다.

[분석 및 대화 지시사항]
1. **시각 정보의 역할 제한 (Ice-breaking 전용)**
   - 제공된 구조화 시각 데이터는 절대 정답의 논리적 출처나 본문의 논거로 억지로 끼워 넣지 마세요.
   - 오직 답변의 첫 문장에서 시청자가 방금 본 상황에 가볍게 공감하는 인사말(Ice-breaking 브릿지)을 만드는 용도로만 사용하세요.
   - 예: "아, 방금 이탈리아 골목에서 화덕 피자가 나오는 모습을 보셨군요!"

2. **방대한 외부 지식의 전면적 활용 (답변 본문)**
   - 답변의 본문은 전적으로 당신의 방대한 외부 지식(World Knowledge)을 최우선으로 활용하여 작성하세요.
   - 빈약한 텍스트 데이터에 억지로 논리를 맞추려다 문맥이 꼬이지 않도록 주의하세요.

3. **현재 장면 우선 (답변을 이끌어내는 시점)**
   - 누적 기억 중 **마지막 Scene(= 현재 장면)**에 가장 높은 우선순위를 두세요.
   - 과거 Scene은 질문의 맥락을 이해하는 데 참고하되, 답변의 생동감은 현재 장면에서 끌어오세요.

4. **완벽한 TV 파트너 톤앤매너**
   - "메타데이터", "구조화 데이터", "Subjects 필드", "vlm_img_structure" 등 시스템 용어는 절대 금지입니다.
   - "지금 화면을 보면~", "방금 나온 장면에서~"처럼 실제 시청자와 대화하듯 친숙한 구어체를 사용하세요.

5. **대화 이어가기**
   - 정보만 전달하고 끝내지 마세요.
   - 답변 마지막에 가볍게 공감하거나 호기심을 자극하는 '부드러운 꼬리 질문'을 던져 대화의 핑퐁을 유도하세요."""

_INTERACTIVE_QUERY_RESPONSE_PROMPT_IMGVLM_GRAPH = _INTERACTIVE_QUERY_RESPONSE_PROMPT_BASE + """
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
   


_INTERACTIVE_QUERY_RESPONSE_PROMPT_BLANK = """당신은 시청자와 나란히 소파에 앉아 TV를 함께 보며 즐겁게 대화를 나누는 '친절하고 똑똑한 비디오 전문 AI 시청 파트너'입니다.

당신에게는 현재 영상에 대한 시청 정보나 영상 메타데이터(채널명, 제목 등)는 제공되지 않습니다.
오직 질문 내용과 당신이 가진 사전 지식(World Knowledge)만을 활용하여 시청자의 질문에 답변해 주세요.

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


def make_interactive_query_gen_config(thinking_level=None):
    """Interactive Query Response 생성용 GenerateContentConfig 딕셔너리를 반환합니다."""
    return {
        "raw": make_generate_config(
            system_instruction=_INTERACTIVE_QUERY_RESPONSE_PROMPT_RAW,
            thinking_level=thinking_level,
        ),
        "imgvlm_chunk2": make_generate_config(
            system_instruction=_INTERACTIVE_QUERY_RESPONSE_PROMPT_IMGVLM_CHUNK2,
            thinking_level=thinking_level,
        ),
        "imgvlm_sentence": make_generate_config(
            system_instruction=_INTERACTIVE_QUERY_RESPONSE_PROMPT_IMGVLM_SENTENCE,
            thinking_level=thinking_level,
        ),
        "imgvlm_graph": make_generate_config(
            system_instruction=_INTERACTIVE_QUERY_RESPONSE_PROMPT_IMGVLM_GRAPH,
            thinking_level=thinking_level,
        ),
        "raw_with_mmvlm": make_generate_config(
            system_instruction=_INTERACTIVE_QUERY_RESPONSE_PROMPT_RAW_WITH_MMVLM,
            thinking_level=thinking_level,
        ),
        "video": make_generate_config(
            system_instruction=_INTERACTIVE_QUERY_RESPONSE_PROMPT_VIDEO,
            thinking_level=thinking_level,
        ),
        "blank": make_generate_config(
            system_instruction=_INTERACTIVE_QUERY_RESPONSE_PROMPT_BLANK,
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
        mode: 'video' | 'raw' | 'raw_with_mmvlm' | 'imgvlm_chunk2' | 'imgvlm_sentence' | 'imgvlm_graph' | 'blank'
        max_past_scenes: None이면 Scene 0부터 전체, 정수이면 현재 Scene 기준 직전 N개 Scene만 사용

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
            past_start_scene_idx = max(0, scene_idx - max_past_scenes)
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
            start_idx = max(0, scene_idx - max_past_scenes)
        else:
            start_idx = 0

        if mode == "raw":
            return get_gcs_raw_fields_by_scene_idx(
                gs_bucket_name, content_id, start_idx, scene_idx
            )
        elif mode == "imgvlm_chunk2":
            return get_processed_vlm_descriptions_by_scene_idx(
                gs_bucket_name, content_id, "vlm_img_structure_chunk2", start_idx, scene_idx
            )
        elif mode == "imgvlm_sentence":
            return get_processed_vlm_descriptions_by_scene_idx(
                gs_bucket_name, content_id, "vlm_img_structure_sentence", start_idx, scene_idx
            )
        elif mode == "imgvlm_graph":
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
    """단일 모드에 대해 Response를 스트리밍 생성하고 TTFT를 측정합니다.

    Returns:
        (mode, answer, elapsed, ttft) — ttft는 Time To First Token (초)
    """
    start_time = time.perf_counter()
    try:
        time.sleep(1)

        # Video Context: 채널명/제목으로 도메인 컨텍스트 제공
        # (imgvlm 모드 및 blank 모드는 메타데이터 주입 금지)
        _NO_METADATA_MODES = {"imgvlm_chunk2", "imgvlm_sentence", "imgvlm_graph", "blank"}
        ctx_prefix = []
        if video_context and mode not in _NO_METADATA_MODES:
            ctx_prefix = ["--- [Video Context (영상 기본 정보)] ---", video_context]

        if mode == "blank":
            contents = [
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
                "imgvlm_sentence": "VLM Image Structure — Sentences (처음부터 현재 장면까지)",
                "imgvlm_graph": "VLM Scene Knowledge Graph (처음부터 현재 장면까지)",
                "raw_with_mmvlm": "Raw ASR/OCR + VLM Multimodal Description (처음부터 현재 장면까지)",
            }
            contents = ctx_prefix + [
                f"--- [{label_map.get(mode, mode)}] ---",
                source,
                "--- 질문 ---",
                query,
            ]
            
        config = gen_configs[mode]

        def _stream_generate():
            """스트리밍 호출 + TTFT 측정"""
            req_time = time.perf_counter()
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
            chunks = []
            ttft = None
            for chunk in stream:
                if ttft is None:
                    ttft = time.perf_counter() - req_time
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks), ttft

        answer, ttft = _retry_api_call(
            _stream_generate,
            label=f"Interactive Query Response [{mode}] (Scene {scene_idx})",
        )
        elapsed = time.perf_counter() - start_time
        return mode, answer, elapsed, ttft

    except Exception as e:
        print(f"  [ERROR] [{mode}] 생성 실패 (Scene {scene_idx}): {e}")
        elapsed = time.perf_counter() - start_time
        return mode, f"Error: {str(e)}", elapsed, None


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

def _validate_interactive_query_responses(output_file, interactive_query_input_file, target_sources, query_source="kss"):
    """Interactive Query Response 파일을 정렬하고, 지정된 Query Source와 target source 기준으로 누락을 점검합니다."""
    if not os.path.exists(output_file):
        print(f"[Warning] 파일을 찾을 수 없습니다: {output_file}")
        return

    # 1) 정렬
    sort_jsonl_file(output_file)

    print(f"\n{'='*50}")
    print(f"결과 파일 검증: {output_file}")

    data_records = load_jsonl(output_file)

    # 2) Query × target source 기준 누락 점검
    print("\n[최종 누락분 점검]")

    expected = set()
    if os.path.exists(interactive_query_input_file):
        interactive_query_records = load_jsonl(interactive_query_input_file)
        interactive_query_lookup = {}
        for r in interactive_query_records:
            c_id = r.get("content_id")
            s_idx = r.get("scene_idx")
            mode = r.get("mode")
            if c_id and s_idx is not None and mode:
                interactive_query_lookup[(c_id, s_idx, mode)] = r.get("queries", [])

        # KSS Interactive Query 레코드를 기준으로 각 scene 식별
        for r in interactive_query_records:
            if r.get("mode") != "kss":
                continue
            c_id = r.get("content_id")
            s_idx = r.get("scene_idx")
            for m in target_sources:
                if query_source == "kss":
                    queries = r.get("queries", [])
                else:  # sourcewise
                    if m == "blank":
                        continue
                    if (c_id, s_idx, m) not in interactive_query_lookup:
                        queries = r.get("queries", [])
                    else:
                        queries = interactive_query_lookup[(c_id, s_idx, m)]
                for q in queries:
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

        print(f"-> 총 {len(missing)}개의 (Scene, Source, Query) 처리가 누락되었습니다. ({len(missing_by_scene)}개 Scene)")
        for (c, s), modes in sorted(missing_by_scene.items())[:20]:
            mode_counts = {}
            for m in modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            summary = ", ".join(f"{m}({n})" for m, n in mode_counts.items())
            print(f"    - ({c}, scene={s}): {summary}")
        if len(missing_by_scene) > 20:
            print(f"    ... 외 {len(missing_by_scene) - 20}개 Scene")
    else:
        print(f"-> 모든 {query_source.upper()} Query × Source 조합이 누락 없이 정상적으로 생성되었습니다.")
    print("=" * 50 + "\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = get_common_argparser(description="Interactive Query의 질문을 Query로 삼아, 각 Source 컨텍스트 정보를 활용해 Response를 생성합니다.")
    parser.add_argument("--input_file", default="assets/interactive_queries.jsonl", help="Interactive Query JSONL 경로")
    parser.add_argument("--output_file", default="assets/interactive_query_responses.jsonl", help="Interactive Query Response 저장 경로")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 JSONL 경로")
    parser.add_argument("--sources", nargs="+",
                        default=["blank", "video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph"],
                        choices=["blank", "video", "raw", "raw_with_mmvlm", "imgvlm_sentence", "imgvlm_chunk2", "imgvlm_graph"],
                        help="Response를 생성할 대상 Source (blank=컨텍스트 없이 World Knowledge만 사용)")
    parser.add_argument("--query_source", choices=["kss", "sourcewise"], default="kss",
                        help="Response 생성에 사용할 Interactive Query 질문의 출처 (kss: KSS 기반 공통 질문, sourcewise: 각 모드별로 생성된 질문)")

    args, client = init_pipeline(parser.parse_args())
    content_indices = load_content_indices()

    # Keypoint 로드 (scene_idx → start/end_time 매핑용)
    keypoints_by_content = load_keypoints_by_content(args.keypoints_file)
    if not keypoints_by_content:
        print(f"Error: {args.keypoints_file} 에서 Keypoint 데이터를 읽을 수 없습니다.")
        return

    target_sources = args.sources
    query_source = args.query_source

    # sourcewise source에서는 blank 모드의 Interactive Query Resp 생성을 제외
    if query_source == "sourcewise" and "blank" in target_sources:
        target_sources = [s for s in target_sources if s != "blank"]

    # query_source에 따라 output_file 경로 변경
    if args.output_file == "assets/interactive_query_responses.jsonl":
        args.output_file = f"assets/interactive_query_responses_{query_source}.jsonl"

    # Gen configs
    gen_configs = make_interactive_query_gen_config(thinking_level=args.interactive_query_response_thinking_level)

    # 출력 디렉토리 확인
    ensure_output_dir(args.output_file)

    if not check_input_file(args.input_file, hint="먼저 generate_interactive_query.py를 실행하세요."):
        return

    print_pipeline_banner("Interactive Query Response 생성 파이프라인을 시작합니다.")
    print(f"[Query Source] {query_source} | Target Response Sources: {target_sources}")
    if args.interactive_query_response_past_scenes_size:
        print(f"[Window] interactive_query_response_past_scenes_size={args.interactive_query_response_past_scenes_size} 설정: 현재 Scene 기준 직전 {args.interactive_query_response_past_scenes_size}개 Scene만 Source로 사용합니다.")

    file_write_lock = threading.Lock()
    _checked_contents = set()

    def _process_scene_record(rec):
        """KSS Interactive Query 레코드(scene 기준)에 대해 지정된 query_source에 기반한 질문들을 바탕으로 각 target source Response를 생성합니다."""
        c_id = rec.get("content_id")
        s_idx = rec.get("scene_idx")

        if not (c_id and s_idx is not None):
            return 0

        # 각 target source × 각 query 조합에서 미처리분만 추출
        pending_items = []
        for mode in target_sources:
            if query_source == "kss":
                mode_queries = rec.get("queries", [])
                mode_query_types = rec.get("query_types", ["content_anchored", "tangential"][:len(mode_queries)])
            else:  # sourcewise
                if mode == "blank":
                    continue
                if (c_id, s_idx, mode) not in interactive_query_lookup:
                    kss_info = interactive_query_lookup.get((c_id, s_idx, "kss"), {})
                    mode_queries = kss_info.get("queries", [])
                    mode_query_types = kss_info.get("query_types", [])
                else:
                    mode_info = interactive_query_lookup.get((c_id, s_idx, mode), {})
                    mode_queries = mode_info.get("queries", [])
                    mode_query_types = mode_info.get("query_types", [])

            for q_idx, q_text in enumerate(mode_queries):
                q_type = mode_query_types[q_idx] if q_idx < len(mode_query_types) else "tangential"
                if (c_id, s_idx, mode, q_text) not in completed_pairs:
                    pending_items.append({
                        "content_id": c_id,
                        "scene_idx":  s_idx,
                        "mode":       mode,
                        "query_type": q_type,
                        "query":      q_text,
                    })

        if not pending_items:
            return 0

        keypoints = keypoints_by_content.get(c_id, [])
        requires_gcs = any(item["mode"] != "blank" for item in pending_items)
        if requires_gcs and c_id not in _checked_contents:
            if not check_gcs_files_exist(args.gs_bucket_name, c_id):
                return 0
            preload_content_metadata(args.gs_bucket_name, c_id)
            _checked_contents.add(c_id)

        # Video Metadata 로드 (채널명, 제목 등)
        video_metadata = load_video_metadata(args.gs_bucket_name, c_id)
        video_context = format_video_context(video_metadata)
        start_time = float(rec.get("start_time", 0.0))
        end_time = float(rec.get("end_time", 0.0))
        current_dur = end_time - start_time
        past_n = min(args.interactive_query_response_past_scenes_size, s_idx) if args.interactive_query_response_past_scenes_size else s_idx
        past_approx_sec = past_n * (start_time / s_idx) if s_idx > 0 else 0
        print(f"\n[Interactive Query Response] '{c_id}' Scene {s_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s] | Current: {current_dur:.1f}s, Past: {past_approx_sec:.0f}s ({past_n} scenes) | Queries → {len(pending_items)}개 조합 처리")

        # Source 캐시: 같은 Scene의 같은 mode는 Source를 1번만 빌드
        source_cache = {}
        count = 0

        if query_source == "kss":
            # Query 단위로 그룹핑하여 출력 (동일 Query가 여러 mode에 적용되므로)
            from collections import OrderedDict
            query_groups = OrderedDict()
            for item in pending_items:
                query_groups.setdefault(item["query"], []).append(item)

            for q_text, items in query_groups.items():
                print(f"\nQuery: {q_text}")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
                    futures = {}
                    for item in items:
                        def process_item(item_data):
                            _s_idx = item_data["scene_idx"]
                            _m = item_data["mode"]
                            _q = item_data["query"]
                            try:
                                if _m not in source_cache:
                                    source_cache[_m] = _build_source(
                                        args.gs_bucket_name, c_id,
                                        _s_idx, keypoints, _m,
                                        max_past_scenes=args.interactive_query_response_past_scenes_size,
                                    )
                                source = source_cache[_m]
                                _, answer, elapsed, ttft = _generate_for_mode(
                                    client, args.interactive_query_response_model, gen_configs,
                                    source, _m, _q, _s_idx, video_context=video_context
                                )
                                return item_data, answer, elapsed, ttft, None
                            except Exception as e:
                                return item_data, None, 0.0, None, str(e)

                        futures[executor.submit(process_item, item)] = item

                    for future in concurrent.futures.as_completed(futures):
                        item, answer, elapsed, ttft, error = future.result()
                        scene_idx = item["scene_idx"]
                        mode      = item["mode"]
                        query     = item["query"]

                        if error:
                            print(f"  -> [{mode}] ERROR: {error}")
                            answer = f"Error: {error}"
                        else:
                            length_info = len(answer) if not answer.startswith("Error") else 0
                            ttft_str = f"TTFT={ttft:.2f}s, " if ttft is not None else ""
                            print(f"  -> [{mode}] OK ({ttft_str}total={elapsed:.1f}s, {length_info}자)")

                        record = OrderedDict([
                            ("content_id", c_id),
                            ("scene_idx",  scene_idx),
                            ("mode",       mode),
                            ("query_type", item.get("query_type", "tangential")),
                            ("query",      query),
                            ("answer",     answer),
                        ])
                        append_jsonl(args.output_file, record, lock=file_write_lock)
                        completed_pairs.add((c_id, scene_idx, mode, query))
                        count += 1
        else:
            # sourcewise: 각 mode마다 쿼리가 다르므로 동시에 모든 pending_items를 병렬로 생성 수행
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending_items)) as executor:
                futures = {}
                for item in pending_items:
                    def process_item(item_data):
                        _s_idx = item_data["scene_idx"]
                        _m = item_data["mode"]
                        _q = item_data["query"]
                        try:
                            if _m not in source_cache:
                                source_cache[_m] = _build_source(
                                    args.gs_bucket_name, c_id,
                                    _s_idx, keypoints, _m,
                                    max_past_scenes=args.interactive_query_response_past_scenes_size,
                                )
                            source = source_cache[_m]
                            _, answer, elapsed, ttft = _generate_for_mode(
                                client, args.interactive_query_response_model, gen_configs,
                                source, _m, _q, _s_idx, video_context=video_context
                            )
                            return item_data, answer, elapsed, ttft, None
                        except Exception as e:
                            return item_data, None, 0.0, None, str(e)

                    futures[executor.submit(process_item, item)] = item

                for future in concurrent.futures.as_completed(futures):
                    item, answer, elapsed, ttft, error = future.result()
                    scene_idx = item["scene_idx"]
                    mode      = item["mode"]
                    query     = item["query"]

                    if error:
                        print(f"  -> [{mode}] ERROR: {error}")
                        answer = f"Error: {error}"
                    else:
                        length_info = len(answer) if not answer.startswith("Error") else 0
                        ttft_str = f"TTFT={ttft:.2f}s, " if ttft is not None else ""
                        print(f"  -> [{mode}] OK ({ttft_str}total={elapsed:.1f}s, {length_info}자)")

                    record = OrderedDict([
                        ("content_id", c_id),
                        ("scene_idx",  scene_idx),
                        ("mode",       mode),
                        ("query_type", item.get("query_type", "tangential")),
                        ("query",      query),
                        ("answer",     answer),
                    ])
                    append_jsonl(args.output_file, record, lock=file_write_lock)
                    completed_pairs.add((c_id, scene_idx, mode, query))
                    count += 1

        return count

    MAX_RETRY = 3
    try:
        discovery_pass = 0
        while True:
            discovery_pass += 1

            # 매 pass마다 입력 파일을 다시 읽어 새로 추가된 레코드를 감지
            completed_pairs = _load_completed_pairs(args.output_file)
            if discovery_pass == 1 and completed_pairs:
                print(f"[기처리] {len(completed_pairs)}개 항목이 이미 처리 완료됨.")

            # interactive_query 파일 로드 및 lookup 테이블 구축
            interactive_query_records = load_jsonl(args.input_file)
            interactive_query_lookup = {}
            for r in interactive_query_records:
                c_id = r.get("content_id")
                s_idx = r.get("scene_idx")
                mode = r.get("mode")
                if c_id and s_idx is not None and mode:
                    interactive_query_lookup[(c_id, s_idx, mode)] = {
                        "queries": r.get("queries", []),
                        "query_types": r.get("query_types", [])
                    }

            all_kss_records = [
                r for r in interactive_query_records
                if r.get("mode") == "kss" and not r.get("pipeline_done")
            ]

            if not all_kss_records:
                if discovery_pass == 1:
                    print("[Error] KSS 모드 레코드가 없습니다. generate_interactive_query.py를 먼저 실행하세요.")
                    return
                else:
                    print("[완료] 입력 파일에 처리할 KSS 레코드가 없습니다.")
                    break

            # 미처리 항목 개수 확인
            pending_count = 0
            for r in all_kss_records:
                c_id = r.get("content_id")
                s_idx = r.get("scene_idx")
                for m in target_sources:
                    if query_source == "kss":
                        queries = r.get("queries", [])
                    else:  # sourcewise
                        if m == "blank":
                            continue
                        if (c_id, s_idx, m) not in interactive_query_lookup:
                            queries = r.get("queries", [])
                        else:
                            queries = interactive_query_lookup[(c_id, s_idx, m)].get("queries", [])
                    for q in queries:
                        if (c_id, s_idx, m, q) not in completed_pairs:
                            pending_count += 1

            if pending_count == 0:
                print("[완료] 모든 항목이 처리되었습니다.")
                break

            print(f"\n{'='*50}")
            print(f"[Discovery {discovery_pass}] KSS 레코드 {len(all_kss_records)}개, 미처리 {pending_count}개 조합 발견")
            print(f"{'='*50}")

            processed_count = 0
            tracker = ProgressTracker(pending_count, unit="items", action="processed")
            for rec_idx, rec in enumerate(all_kss_records):
                processed_in_rec = _process_scene_record(rec)
                if processed_in_rec > 0:
                    processed_count += processed_in_rec
                if processed_count > 0 and ((rec_idx + 1) % 10 == 0 or (rec_idx + 1) == len(all_kss_records)):
                    tracker.update(processed_count)

            print(f"\n▶ [Discovery {discovery_pass}] 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    # 결과 파일 정렬 및 누락 점검
    _validate_interactive_query_responses(args.output_file, args.input_file, target_sources, query_source=query_source)

    print_pipeline_done(args.output_file)


if __name__ == "__main__":
    main()
