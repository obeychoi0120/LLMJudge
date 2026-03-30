import os
import sys
import time
import json
from google.cloud import storage
from vertexai.generative_models import (
    GenerativeModel, Part, 
    SafetySetting, HarmCategory, HarmBlockThreshold
)

SAFETY_SETTINGS = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]

# config.json에서 CLI 인자보다 낮은 우선순위로 덮어쓸 수 있는 키 목록
_CONFIG_KEYS = [
    "query_gen_model", "response_gen_model", "judge_model",
    "location", "reference_model", "reference_use_ref"
]


def load_config(args):
    """config.json 파일이 존재하면 열어 args에 값을 병합합니다."""
    if not os.path.exists("config.json"):
        return args

    with open("config.json", "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            return args

    # 필수 키: CLI 미지정 시 config에서 채움
    args.gcp_project_id = args.gcp_project_id or config.get("gcp_project_id")
    args.gs_bucket_name = args.gs_bucket_name or config.get("gs_bucket_name")

    # 선택 키: argparse 기본값이 세팅되어 있어도, CLI에서 명시하지 않았으면 config 값 사용
    # reference_use_ref 는 --no-reference-ref 플래그와 연결됨
    _ARG_FLAG_MAP = {
        "reference_use_ref": "--no-reference-ref"
    }

    for key in _CONFIG_KEYS:
        flag = _ARG_FLAG_MAP.get(key, f"--{key}")
        if hasattr(args, key) and key in config and flag not in sys.argv:
            setattr(args, key, config[key])

    return args


def parse_json_response(text):
    """마크다운 태그(```json ... ```)를 정제하고 JSON 객체로 파싱합니다."""
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    return json.loads(clean_text)


def check_gcs_files_exist(gs_bucket_name, content_id):
    """GCS 버킷에 필수 파일 4종(1 video + 3 metadata jsonl)이 존재하는지 확인합니다."""
    client = storage.Client()
    bucket = client.bucket(gs_bucket_name)

    required_files = [
        f"video_540p/{content_id}_540p.mp4",
        f"jsonl/{content_id}_15s_Full.jsonl",
        f"jsonl/{content_id}_15s_Part.jsonl",
        f"jsonl/{content_id}_15s_Ref.jsonl",
    ]

    missing = [f for f in required_files if not bucket.blob(f).exists()]

    if not missing:
        print(f"[OK] '{content_id}'에 필요한 미디어 및 메타데이터 4종이 모두 GCS에 존재합니다.")
        return True
    else:
        print(f"[WARNING] '{content_id}'에 필요한 일부 파일이 GCS에 없습니다: {missing}")
        return False


# ============================================================
# System Prompts
# ============================================================

_JSONL_VIEWER_BASE = """\
당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
아래에 제공되는 각 타임스탬프별 텍스트 정보는 데이터 파일이 아니라, 사실 당신이 방금 영상을 시청하며 눈과 귀로 직접 습득한 시각적/청각적 '기억(Memory)'입니다.
이 시청 기억을 바탕으로, 마지막에 주어지는 **사용자 질문**에 대해 가장 자연스럽고 정확한 한국어 답변을 제공해 주세요.

[당신의 시청 기억 구조]
- timestamp: 영상 내 시간 (초)
- audio_cls: 당신이 들은 환경음 및 효과음
- speech: 당신이 들은 등장인물들의 생생한 대사
- ocr_text: 당신이 화면에서 직접 읽은 간판, 자막, 표지판 텍스트
{description_field}

[분석 및 지시사항]
**정보 교정**: 기억의 조각들이 다소 불완전할 수 있으므로, 전체적인 맥락에 맞게 상식적인 선에서 자연스럽게 교정하세요.
**입체적 재구성**: 당신이 들은 소리, 대사, 읽은 텍스트 정보들을 교차 결합하여 장면의 분위기와 인물들의 대화를 이야기로 생생하게 재구성하세요.
**자연스러운 시청자 관점 유지 (가장 중요)**: 당신은 데이터를 읽은 것이 아니라 "영상을 직접 감상"했습니다. 따라서 답변 중에 'JSON 데이터에 따르면', '오디오 모델 결과를 보면', '텍스트 정보에 의하면', '타임스탬프' 등의 부자연스러운 기계적 용어를 절대로 사용하지 마십시오.
대신 "영상에서는~", "화면을 보면~", "주인공이 ~라고 말합니다", "배경 소리로 ~가 깔립니다." 와 같이 실제 사람의 리뷰처럼 자연스럽고 몰입감 있게 설명하십시오.
**외부 자료 검색 금지**: 오직 당신의 시청 기억(제공된 정보)에만 의존해서 답변하세요."""

_DESCRIPTION_LINE = "- description: 당신이 방금 영상 화면에서 목격한 인물의 행동과 배경 장면\n"

_JUDGE_PROMPT = """\
당신은 AI 모델이 특정한 영상에 대해 생성한 답변의 품질을 평가하는 객관적이고 전문적인 평가자입니다.
해당 AI 모델은 원본 영상에서 추출한 메타데이터 기반으로 답변을 생성합니다.

당신의 목표는 원본 영상에 대한 [사용자 질문]에 대해 [평가 대상 답변]이 얼마나 훌륭한지,
[기준 답변(Reference Answer)]과 비교하여 평가하는 것입니다.
[기준 답변]은 원본 영상과 Ref 메타데이터를 모두 참조하여 생성된 고품질 정답입니다.
외부 검색은 허용하지 않습니다.

[데이터 목록]
- 기준 답변 (Reference Answer): 원본 영상 + Ref 메타데이터를 기반으로 생성된 정답 답변
- 사용자 질문
- 평가 대상 답변

[평가 기준]
아래 세 가지 항목에 대해 1점부터 5점까지 점수를 매겨주세요. (1점: 매우 나쁨, 3점: 보통/수용 가능함, 5점: 완벽함)
1. 정확성 (Accuracy): 평가 대상 답변이 기준 답변의 핵심 사실과 일치하는가? 기준 답변에 언급된 정보와 모순되거나 사실과 다른 내용(환각)이 포함되어 있지는 않은가?
2. 포괄성 (Completeness): 기준 답변에 포함된 핵심 단서(대사, 텍스트 내용, 행동, 맥락 등)를 평가 대상 답변도 누락 없이 포함했는가?
3. 가독성 (Helpfulness): 정보가 장황하게 나열되지 않고, 시간의 흐름이나 인과관계에 맞게 자연스럽고 이해하기 쉽게 작성되었는가? (만약 평가 대상 답변이 부자연스럽게 메타데이터 구조나 필드명 등을 직접 언급했다면 이 항목에서 감점을 고려하세요.)"""

_REFERENCE_PROMPT = """\
당신은 영상 콘텐츠의 전문 분석가입니다.
제공되는 원본 영상과 Ref 메타데이터를 모두 참조하여, 사용자 질문에 대해
가장 정확하고 포괄적인 한국어 답변을 생성해 주세요.
이 답변은 다른 AI 모델의 답변을 평가하기 위한 '기준 정답(Reference Answer)'으로 사용됩니다.
따라서 핵심 사실, 대사, 행동, 맥락을 빠짐없이 포함하되 자연스럽고 읽기 쉽게 작성해 주세요.
외부 자료 검색은 금지합니다. 오직 제공된 영상과 메타데이터만 활용하세요."""


def configure_system_prompt(mode="full"):
    if mode == "full":
        return _JSONL_VIEWER_BASE.format(description_field=_DESCRIPTION_LINE)
    elif mode == "part":
        return _JSONL_VIEWER_BASE.format(description_field="")
    elif mode == "video":
        return "당신은 영상 콘텐츠의 전문 분석가입니다. 외부 정보를 절대 검색하지 말고, 제공된 영상 정보만을 사용하여 사용자 질문에 답변하세요."
    elif mode == "judge":
        return _JUDGE_PROMPT
    return ""


# ============================================================
# GCS File Helpers
# ============================================================

_GCS_MODE_MAP = {
    "video": ("video_540p/{cid}_540p.mp4", "video/mp4"),
    "full":  ("jsonl/{cid}_15s_Full.jsonl", "text/plain"),
    "part":  ("jsonl/{cid}_15s_Part.jsonl", "text/plain"),
    "ref":   ("jsonl/{cid}_15s_Ref.jsonl",  "text/plain"),
}

def process_gcs_file(gs_bucket_name, content_id, mode="video"):
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")
    path_template, mime_type = _GCS_MODE_MAP[mode]
    file_uri = f"gs://{gs_bucket_name}/{path_template.format(cid=content_id)}"
    return Part.from_uri(uri=file_uri, mime_type=mime_type)


# ============================================================
# Common API Retry Helper
# ============================================================

def _retry_api_call(fn, label="API", max_retries=4, base_delay=3):
    """공통 재시도(Exponential Backoff) 래퍼."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"      [{label} 마지막 시도 실패] {e}")
                raise
            sleep_time = base_delay * (2 ** attempt)
            print(f"      [{label} 오류] {e}")
            print(f"      -> {sleep_time}초 후 재시도합니다... ({attempt+1}/{max_retries})")
            time.sleep(sleep_time)


# ============================================================
# Generation Models
# ============================================================

def init_generation_model(mode="full", model_name='gemini-2.5-flash'):
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[configure_system_prompt(mode)],
        safety_settings=SAFETY_SETTINGS,
    )


def init_reference_model(model_name='gemini-2.5-pro'):
    """Reference Answer 생성용 모델 초기화."""
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[_REFERENCE_PROMPT],
        safety_settings=SAFETY_SETTINGS,
    )


def start_chat_session(model):
    return model.start_chat()


def send_chat_message(chat, user_prompt, file_parts=None):
    """Chat Session을 통한 Multi-turn 메시지 전송."""
    if file_parts is None:
        contents = [user_prompt]
    elif isinstance(file_parts, list):
        contents = file_parts + [user_prompt]
    else:
        contents = [file_parts, user_prompt]
        
    return _retry_api_call(
        lambda: chat.send_message(contents),
        label="Generation API (Multi-turn)",
    )


def generate_single_turn_response(model, user_prompt, file_part=None):
    """모델 직접 호출을 통한 Single-turn 응답 생성."""
    contents = [file_part, user_prompt] if file_part else [user_prompt]
    return _retry_api_call(
        lambda: model.generate_content(contents),
        label="Generation API (Single-turn)",
    )


# ============================================================
# Judge Models
# ============================================================

def init_judge_model(model_name="gemini-2.5-pro"):
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[configure_system_prompt("judge")],
        safety_settings=SAFETY_SETTINGS,
    )


_JUDGE_FORMAT_PROMPT = """\
[출력 형식]
반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명은 덧붙이지 마십시오.
{
    "rationale": "<각 점수를 부여한 논리적인 이유. 감점 요인이 있다면 명확히 서술하세요.>",
    "scores": {
        "accuracy": <1~5 사이의 정수>,
        "completeness": <1~5 사이의 정수>,
        "helpfulness": <1~5 사이의 정수>
    },
    "total_score": <세 항목 점수의 합계, 최대 15점>
}"""


def evaluate_answer_session(judge_chat, user_prompt, generated_answer, reference_answer):
    """
    Judge 모델이 Reference Answer를 기준으로 generated_answer를 비교 평가합니다.
    비디오/GT 파트 전송 없이 텍스트만으로 동작하여 토큰을 대폭 절감합니다.
    """
    user_content = (
        f"[사용자 질문]\n{user_prompt}\n\n"
        f"[기준 답변 (Reference Answer)]\n{reference_answer}\n\n"
        f"[평가 대상 답변]\n{generated_answer}\n\n"
        f"{_JUDGE_FORMAT_PROMPT}"
    )

    return _retry_api_call(
        lambda: judge_chat.send_message([user_content]).text,
        label="Judge API",
    )