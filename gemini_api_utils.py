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

def _retry_api_call(fn, label="API", delay=10):
    """공통 재시도(Infinite Loop for 429/5xx) 래퍼."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            err_msg = str(e)
            # Terminal Errors: 재시도해도 해결되지 않는 오류들
            # 400 (Invalid Argument), 403 (Permission Denied), 404 (Not Found)
            # Safety Rating/Block (차단됨)
            is_terminal = any(code in err_msg for code in ["400", "401", "403", "404", "Safety"])
            
            if is_terminal:
                print(f"      [{label} 치명적 오류] {err_msg}")
                raise
            
            # Retryable: 429 (Quota), 500, 503, 504 등
            print(f"      [{label} 오류] {err_msg}")
            print(f"      -> {delay}초 후 재시도합니다... (시도 횟수: {attempt})")
            time.sleep(delay)


# ============================================================
# API Interaction Helpers
# ============================================================

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
