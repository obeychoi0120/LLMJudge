import os
import sys
import time
import json
from google import genai
from google.genai import types
from google.cloud import storage



# ============================================================
# config.json 지원 키 목록
# ============================================================

_CONFIG_KEYS = [
    "query_gen_model", "response_gen_model", "judge_model",
    "location", "reference_model", "reference_use_ref",
    "keypoint_model", "query_judge_model",
    "summary_gen_model", "bubble_query_model", "user_query_model",
    "bubble_thinking_budget", "response_thinking_budget"
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

    _ARG_FLAG_MAP = {
        "reference_use_ref": "--no-reference-ref"
    }

    for key in _CONFIG_KEYS:
        flag = _ARG_FLAG_MAP.get(key, f"--{key}")
        if hasattr(args, key) and key in config and flag not in sys.argv:
            setattr(args, key, config[key])

    return args


# ============================================================
# Client Factory
# ============================================================

def create_client(project_id: str, location: str) -> genai.Client:
    """Vertex AI 백엔드를 사용하는 genai.Client를 생성합니다."""
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )


# ============================================================
# GenerateContentConfig Helpers
# ============================================================

def make_generate_config(
    system_instruction: str = None,
    thinking_budget: int = None,
) -> types.GenerateContentConfig:
    """GenerateContentConfig 객체를 생성합니다.

    Args:
        system_instruction: 시스템 프롬프트 문자열
        thinking_budget: Thinking 토큰 수 (0=off, -1=dynamic, 양수=지정). None이면 미설정.
    """
    kwargs = {}

    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction

    if thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    return types.GenerateContentConfig(**kwargs)


# ============================================================
# JSON Parsing
# ============================================================

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


# ============================================================
# GCS File Helpers
# ============================================================

def check_gcs_files_exist(gs_bucket_name, content_id):
    """GCS 버킷에 필수 파일 4종(1 video + 3 metadata jsonl)이 존재하는지 확인합니다."""
    client = storage.Client()
    bucket = client.bucket(gs_bucket_name)

    required_files = [
        f"video_540p/{content_id}_540p.mp4",
        f"jsonl/{content_id}_Full.jsonl",
        f"jsonl/{content_id}_Part.jsonl",
        f"jsonl/{content_id}_Ref.jsonl",
    ]

    missing = [f for f in required_files if not bucket.blob(f).exists()]

    if not missing:
        print(f"[OK] '{content_id}'에 필요한 미디어 및 메타데이터 4종이 모두 GCS에 존재합니다.")
        return True
    else:
        print(f"[WARNING] '{content_id}'에 필요한 일부 파일이 GCS에 없습니다: {missing}")
        return False


_GCS_MODE_MAP = {
    "video": ("video_540p/{cid}_540p.mp4", "video/mp4"),
    "full":  ("jsonl/{cid}_Full.jsonl", "text/plain"),
    "part":  ("jsonl/{cid}_Part.jsonl", "text/plain"),
    "ref":   ("jsonl/{cid}_Ref.jsonl",  "text/plain"),
}


def process_gcs_file(gs_bucket_name, content_id, mode="video"):
    """GCS 파일 전체를 참조하는 Part를 반환합니다."""
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")
    path_template, mime_type = _GCS_MODE_MAP[mode]
    file_uri = f"gs://{gs_bucket_name}/{path_template.format(cid=content_id)}"
    return types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)


# ============================================================
# Truncation Helpers
# ============================================================

def download_gcs_text(gs_bucket_name, blob_path):
    """GCS 버킷에서 텍스트 파일 내용을 다운로드합니다."""
    client = storage.Client()
    bucket = client.bucket(gs_bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_text(encoding="utf-8")


def truncate_jsonl_range(jsonl_text, start_time, end_time):
    """JSONL 텍스트에서 [start_time, end_time] 구간의 Scene만 추출합니다."""
    truncated_lines = []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            scene = json.loads(line)
            scene_end = scene.get("end_time", float("inf"))
            if start_time < scene_end <= end_time:
                truncated_lines.append(line)
        except json.JSONDecodeError:
            continue
    return "\n".join(truncated_lines)


def process_gcs_file_range(gs_bucket_name, content_id, mode, start_time, end_time):
    """[start_time, end_time] 구간 데이터 Part를 반환합니다.

    - video 모드: VideoMetadata의 start_offset/end_offset으로 구간 클리핑
    - jsonl 모드: GCS에서 다운로드 후 해당 구간만 추출하여 인라인 Part 반환
    """
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")

    path_template, mime_type = _GCS_MODE_MAP[mode]
    blob_path = path_template.format(cid=content_id)
    file_uri = f"gs://{gs_bucket_name}/{blob_path}"

    if mode == "video":
        return types.Part.from_uri(
            file_uri=file_uri,
            mime_type=mime_type,
            video_metadata=types.VideoMetadata(
                start_offset=f"{int(start_time)}s",
                end_offset=f"{int(end_time)}s",
            ),
        )
    else:
        jsonl_text = download_gcs_text(gs_bucket_name, blob_path)
        range_text = truncate_jsonl_range(jsonl_text, start_time, end_time)
        return types.Part.from_bytes(data=range_text.encode("utf-8"), mime_type="text/plain")


def process_gcs_file_truncated(gs_bucket_name, content_id, mode, end_time):
    """[0, end_time]까지 Truncation된 GCS 파일 Part를 반환합니다."""
    return process_gcs_file_range(gs_bucket_name, content_id, mode, 0.0, end_time)


# ============================================================
# Common API Retry Helper
# ============================================================

def _retry_api_call(fn, label="API", delay=10):
    """공통 재시도(Infinite Loop for 429/5xx) 래퍼.

    재시도 대상: 429 (Quota Exceeded), 500 / 503 / 504 (서버 일시 오류)
    그 외 모든 오류는 즉시 raise.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            err_msg = str(e)
            is_retryable = any(code in err_msg for code in ["429", "500", "503", "504"])

            if not is_retryable:
                print(f"      [{label} 치명적 오류] {err_msg}")
                raise

            print(f"      [{label} 오류] {err_msg}")
            print(f"      -> {delay}초 후 재시도합니다... (시도 횟수: {attempt})")
            time.sleep(delay)


# ============================================================
# API Interaction Helpers (신 SDK Chat 방식)
# ============================================================

def start_chat_session(client: genai.Client, model: str, config: types.GenerateContentConfig):
    """Chat 세션을 생성합니다."""
    return client.chats.create(model=model, config=config)


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


def generate_single_turn_response(client: genai.Client, model: str, config: types.GenerateContentConfig, user_prompt, file_part=None):
    """Single-turn 응답 생성."""
    if file_part is None:
        contents = [user_prompt]
    elif isinstance(file_part, list):
        contents = file_part + [user_prompt]
    else:
        contents = [file_part, user_prompt]

    return _retry_api_call(
        lambda: client.models.generate_content(model=model, contents=contents, config=config),
        label="Generation API (Single-turn)",
    )
