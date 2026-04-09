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
    # 공통
    "location", "keypoint_model", "keypoint_thinking_budget",
    # A-track: Voice Hint
    "vh_gen_model", "keyscene_summary_model", "vh_judge_model",
    "vh_thinking_budget", "keyscene_summary_thinking_budget", "vh_judge_thinking_budget",
    "use_ref_for_keyscene_summary",
    # B-track: User Query
    "uq_gen_model", "uq_response_model", "uq_reference_model", "uq_judge_model",
    "uq_gen_thinking_budget", "uq_response_thinking_budget", "uq_reference_thinking_budget", "uq_judge_thinking_budget",
    "use_ref_for_uq_reference",
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

    for key in _CONFIG_KEYS:
        flag = f"--{key}"
        if hasattr(args, key) and key in config and flag not in sys.argv:
            setattr(args, key, config[key])

    return args

def get_common_argparser(description=""):
    """모든 파이프라인 스크립트에서 공통으로 사용하는 인자를 포함한 ArgumentParser를 반환합니다."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    
    # GCP 공통
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")

    # 모델 공통 (A-track)
    parser.add_argument("--keypoint_model", default="gemini-2.5-flash", help="Keypoint 식별에 사용할 모델명")
    parser.add_argument("--keypoint_thinking_budget", type=int, default=512, help="Keypoint 식별 모델의 Thinking Budget")
    parser.add_argument("--vh_gen_model", default="gemini-2.5-flash", help="Voice Hint 생성 모델명")
    parser.add_argument("--vh_thinking_budget", type=int, default=128, help="Voice Hint 모델의 Thinking Budget")
    parser.add_argument("--keyscene_summary_model", default="gemini-2.5-flash", help="KeyScene Summary 생성 모델명")
    parser.add_argument("--keyscene_summary_thinking_budget", type=int, default=512, help="KeyScene Summary 모델의 Thinking Budget")
    parser.add_argument("--use_ref_for_keyscene_summary", type=lambda x: str(x).lower() == 'true', default=False, help="Summary 생성 시 Ref JSONL 참조 여부")
    parser.add_argument("--vh_judge_model", default="gemini-3.1-pro-preview", help="Voice Hint 질문 Judge 모델명")
    parser.add_argument("--vh_judge_thinking_budget", type=int, default=1024, help="Voice Hint Judge 모델의 Thinking Budget")

    # 모델 공통 (B-track)
    parser.add_argument("--uq_gen_model", default="gemini-3.1-pro-preview", help="User Query 생성 모델명")
    parser.add_argument("--uq_gen_thinking_budget", type=int, default=1024, help="UQ 생성 모델의 Thinking Budget")
    parser.add_argument("--uq_reference_model", default="gemini-3.1-pro-preview", help="User Query Reference Answer 생성 모델명")
    parser.add_argument("--uq_reference_thinking_budget", type=int, default=2048, help="UQ Reference Answer 생성 모델의 Thinking Budget")
    parser.add_argument("--use_ref_for_uq_reference", type=lambda x: str(x).lower() == 'true', default=False, help="Reference 생성 시 Ref JSONL 참조 여부")
    parser.add_argument("--uq_response_model", default="gemini-2.5-flash", help="User Query 답변 생성 모델명")
    parser.add_argument("--uq_response_thinking_budget", type=int, default=-1, help="UQ Response 생성 모델의 Thinking Budget (-1=동적)")
    parser.add_argument("--uq_judge_model", default="gemini-3.1-pro-preview", help="User Query 답변 평가 모델명")
    parser.add_argument("--uq_judge_thinking_budget", type=int, default=1024, help="UQ Response Judge 모델의 Thinking Budget")

    return parser


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
# Common File / Progress Helpers
# ============================================================

def ensure_output_dir(file_path):
    """파일 경로의 디렉토리가 없으면 생성합니다."""
    odir = os.path.dirname(file_path)
    if odir and not os.path.exists(odir):
        os.makedirs(odir, exist_ok=True)


def load_processed_content_ids(jsonl_path):
    """JSONL 파일에서 이미 처리된 content_id 집합을 반환합니다."""
    processed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed.add(json.loads(line)["content_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return processed


def load_processed_pairs(jsonl_path, key_fields=("content_id", "query")):
    """JSONL 파일에서 이미 처리된 (key1, key2) 쌍의 집합을 반환합니다."""
    processed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    values = tuple(obj.get(k) for k in key_fields)
                    if all(v is not None for v in values):
                        processed.add(values)
                except (json.JSONDecodeError, KeyError):
                    pass
    return processed


def load_jsonl(path):
    """JSONL 파일을 읽어 딕셔너리 리스트로 반환합니다 (빈줄/파싱 에러 무시)."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def append_jsonl(path, record, lock=None):
    """JSONL 파일에 레코드 1건을 append합니다 (thread-safe).

    Args:
        path: 출력 파일 경로
        record: JSON 직렬화 가능한 딕셔너리
        lock: threading.Lock 객체 (멀티스레드 환경에서 사용)
    """
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if lock:
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def init_pipeline(args):
    """CLI 초기화를 일괄 수행합니다: config 병합 → 필수 인자 검증 → client 생성.

    Returns:
        (args, client) 튜플. 검증 실패 시 SystemExit을 raise합니다.
    """
    args = load_config(args)

    if not getattr(args, "gcp_project_id", None) or not getattr(args, "gs_bucket_name", None):
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. "
              "(--gcp_project_id 인자를 주입하거나 config.json을 생성하세요)")
        sys.exit(1)

    location = getattr(args, "location", "global")
    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {location}...")
    client = create_client(args.gcp_project_id, location)

    return args, client


def retry_parse_json(fn, label="API", max_retries=3):
    """API 호출 함수 fn()을 실행하고 JSON 파싱까지 수행합니다.

    JSON 파싱 실패 시 최대 max_retries회 재시도합니다.

    Args:
        fn: 호출 시 텍스트를 반환하는 callable
        label: 로깅용 라벨
        max_retries: JSON 파싱 재시도 횟수

    Returns:
        파싱된 딕셔너리. 최종 실패 시 {"raw_response": ...} fallback을 반환합니다.
    """
    score_text = None
    for attempt in range(max_retries):
        try:
            score_text = fn()
            return parse_json_response(score_text)
        except json.JSONDecodeError:
            print(f"      [{label}] JSON 파싱 실패 ({attempt+1}/{max_retries}), 재시도...")
            if score_text:
                print(f"      [{label}] [Raw]: {score_text[:100]}...")
            time.sleep(2)
        except Exception as e:
            print(f"      [{label}] 오류: {e}")
            raise
    print(f"      [{label}] JSON 파싱 최종 실패.")
    return {"raw_response": score_text if score_text else "Error"}


# ============================================================
# GCS File Helpers
# ============================================================

def check_gcs_files_exist(gs_bucket_name, content_id):
    """GCS 버킷에 필수 파일 3종(video + Desc + Ref jsonl)이 존재하는지 확인합니다."""
    client = storage.Client()
    bucket = client.bucket(gs_bucket_name)

    required_files = [
        f"video_540p/{content_id}_540p.mp4",
        f"jsonl/{content_id}_raw.jsonl",
        f"jsonl/{content_id}_imgdesc.jsonl",
        f"jsonl/{content_id}_mmdesc.jsonl",
        f"jsonl/{content_id}_ref.jsonl",
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
    "raw": ("jsonl/{cid}_raw.jsonl", "text/plain"),
    "img_desc": ("jsonl/{cid}_imgdesc.jsonl", "text/plain"),
    "mm_desc": ("jsonl/{cid}_mmdesc.jsonl", "text/plain"),
    "ref": ("jsonl/{cid}_ref.jsonl", "text/plain"),
}


# ============================================================
# GCS 캐시 레이어
# ============================================================

_gcs_text_cache: dict[str, str] = {}
_gcs_part_cache: dict[str, types.Part] = {}


def download_gcs_text(gs_bucket_name, blob_path):
    """GCS 버킷에서 텍스트 파일 내용을 다운로드합니다 (캐시 적용)."""
    cache_key = f"{gs_bucket_name}/{blob_path}"
    if cache_key in _gcs_text_cache:
        return _gcs_text_cache[cache_key]
    client = storage.Client()
    bucket = client.bucket(gs_bucket_name)
    blob = bucket.blob(blob_path)
    text = blob.download_as_text(encoding="utf-8")
    _gcs_text_cache[cache_key] = text
    return text


def clear_gcs_cache():
    """GCS 텍스트/Part 캐시를 초기화합니다 (메모리 관리용)."""
    _gcs_text_cache.clear()
    _gcs_part_cache.clear()


def preload_content_metadata(gs_bucket_name, content_id):
    """content_id에 해당하는 메타데이터 JSONL을 한 번에 캐시에 로드합니다."""
    modes = ["raw", "img_desc", "mm_desc", "ref"]
    to_download = []
    for mode in modes:
        path_template, _ = _GCS_MODE_MAP[mode]
        blob_path = path_template.format(cid=content_id)
        cache_key = f"{gs_bucket_name}/{blob_path}"
        if cache_key not in _gcs_text_cache:
            to_download.append((mode, blob_path))

    if not to_download:
        return
    for mode, blob_path in to_download:
        download_gcs_text(gs_bucket_name, blob_path)


def load_scenes(gs_bucket_name, content_id, mode="ref"):
    """지정한 mode의 JSONL을 파싱하여 Scene 리스트로 반환합니다 (캐시 자동 활용).

    Args:
        mode: 'ref', 'img_desc', 'mm_desc', 'raw' 중 하나
    """
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")
    path_template, _ = _GCS_MODE_MAP[mode]
    blob_path = path_template.format(cid=content_id)
    text = download_gcs_text(gs_bucket_name, blob_path)
    return [json.loads(l) for l in text.strip().split("\n") if l.strip()]


def process_gcs_file(gs_bucket_name, content_id, mode="video"):
    """GCS 파일 전체를 참조하는 Part를 반환합니다 (캐시 적용)."""
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")

    cache_key = f"{gs_bucket_name}/{content_id}/{mode}/full"
    if cache_key in _gcs_part_cache:
        return _gcs_part_cache[cache_key]

    path_template, mime_type = _GCS_MODE_MAP[mode]
    file_uri = f"gs://{gs_bucket_name}/{path_template.format(cid=content_id)}"
    part = types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)
    _gcs_part_cache[cache_key] = part
    return part


# ============================================================
# Truncation Helpers
# ============================================================

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


def get_gcs_text_range(gs_bucket_name, content_id, mode, start_time, end_time):
    """지정된 mode의 JSONL 파일에서 [start_time, end_time] 구간의 텍스트를 반환합니다."""
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")
    path_template, _ = _GCS_MODE_MAP[mode]
    blob_path = path_template.format(cid=content_id)
    jsonl_text = download_gcs_text(gs_bucket_name, blob_path)
    return truncate_jsonl_range(jsonl_text, start_time, end_time)


def process_gcs_file_range(gs_bucket_name, content_id, mode, start_time, end_time):
    """[start_time, end_time] 구간 데이터 Part를 반환합니다.

    - video 모드: VideoMetadata의 start_offset/end_offset으로 구간 클리핑 (캐시 적용)
    - jsonl 모드: GCS에서 다운로드 후 해당 구간만 추출하여 인라인 Part 반환
    """
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")

    path_template, mime_type = _GCS_MODE_MAP[mode]
    blob_path = path_template.format(cid=content_id)
    file_uri = f"gs://{gs_bucket_name}/{blob_path}"

    if mode == "video":
        # 비디오 Part 객체 캐시: 동일 범위의 Part 재생성 방지
        cache_key = f"{gs_bucket_name}/{content_id}/video/{int(start_time)}-{int(end_time)}"
        if cache_key in _gcs_part_cache:
            return _gcs_part_cache[cache_key]
        part = types.Part(
            file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
            video_metadata=types.VideoMetadata(
                start_offset=f"{int(start_time)}s",
                end_offset=f"{int(end_time)}s",
            ),
        )
        _gcs_part_cache[cache_key] = part
        return part
    else:
        # JSONL 텍스트 다운로드는 download_gcs_text 캐시가 자동 적용됨
        range_text = get_gcs_text_range(gs_bucket_name, content_id, mode, start_time, end_time)
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
