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
    "location", "keypoint_model", "keypoint_thinking_level",
    # A-track: Voice Hint
    "vh_gen_model", "vh_judge_model",
    "vh_thinking_level", "vh_judge_thinking_level",
    "vh_gen_past_scenes_size",
    # A-track: KeyScene Summary
    "kss_past_summary_model", "kss_past_summary_thinking_level",
    "kss_current_scene_model", "kss_current_scene_thinking_level",
    "use_ref_for_keyscene_summary",
    # B-track: VH Response
    "vh_response_model", "vh_response_thinking_level",
    "vh_response_past_scenes_size",
    "vh_response_judge_model", "vh_response_judge_thinking_level",
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
    parser.add_argument("--keypoint_model", default="gemini-3.1-flash-lite-preview", help="Keypoint 식별에 사용할 모델명")
    parser.add_argument("--keypoint_thinking_level", default="low", help="Keypoint 식별 모델의 Thinking Level (low/medium/high)")
    parser.add_argument("--vh_gen_model", default="gemini-3.1-flash-lite-preview", help="Voice Hint 생성 모델명")
    parser.add_argument("--vh_thinking_level", default="low", help="Voice Hint 모델의 Thinking Level (low/medium/high)")
    parser.add_argument("--vh_gen_past_scenes_size", type=int, default=5, help="Voice Hint 과거 맥락에 포함할 최대 Scene 개수")
    parser.add_argument("--kss_past_summary_model", default="gemini-3.1-flash-lite-preview", help="[Session 1] 과거 요약 생성 모델명")
    parser.add_argument("--kss_past_summary_thinking_level", default="medium", help="[Session 1] 과거 요약 모델의 Thinking Level (low/medium/high)")
    parser.add_argument("--kss_current_scene_model", default="gemini-3.1-pro-preview", help="[Session 2] 현재 장면 묘사 모델명")
    parser.add_argument("--kss_current_scene_thinking_level", default="high", help="[Session 2] 현재 장면 묘사 모델의 Thinking Level (low/medium/high)")
    parser.add_argument("--use_ref_for_keyscene_summary", type=lambda x: str(x).lower() == 'true', default=False, help="Summary 생성 시 Ref JSONL 참조 여부")
    parser.add_argument("--vh_judge_model", default="gemini-3.1-pro-preview", help="Voice Hint 질문 Judge 모델명")
    parser.add_argument("--vh_judge_thinking_level", default="high", help="Voice Hint Judge 모델의 Thinking Level (low/medium/high)")

    # 모델 공통 (B-track: VH Response)
    parser.add_argument("--vh_response_model", default="gemini-3.1-flash-lite-preview", help="VH Response 생성 모델명")
    parser.add_argument("--vh_response_thinking_level", default="low", help="VH Response 생성 모델의 Thinking Level (low/medium/high)")
    parser.add_argument("--vh_response_past_scenes_size", type=int, default=5, help="VH Response 생성 시 과거 맥락에 포함할 최대 Scene 개수")
    parser.add_argument("--vh_response_judge_model", default="gemini-3.1-pro-preview", help="VH Response Judge 모델명")
    parser.add_argument("--vh_response_judge_thinking_level", default="high", help="VH Response Judge 모델의 Thinking Level (low/medium/high)")

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
    thinking_level: str = None,
) -> types.GenerateContentConfig:
    """GenerateContentConfig 객체를 생성합니다.

    Args:
        system_instruction: 시스템 프롬프트 문자열
        thinking_level: Thinking 토큰 레벨 ("low", "medium", "high"). None이면 미설정.
    """
    kwargs = {}

    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction

    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level
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
    """GCS 버킷에 필수 파일 3종(video, final jsonl, ref jsonl)이 존재하는지 확인합니다."""
    client = storage.Client()
    bucket = client.bucket(gs_bucket_name)

    required_files = [
        f"video_540p/{content_id}_540p.mp4",
        f"jsonl/{content_id}_final.jsonl",
        f"jsonl/{content_id}_ref.jsonl",
    ]

    missing = [f for f in required_files if not bucket.blob(f).exists()]

    if not missing:
        print(f"[OK] '{content_id}'에 필요한 미디어 및 메타데이터가 모두 GCS에 존재합니다.")
        return True
    else:
        print(f"[WARNING] '{content_id}'에 필요한 일부 파일이 GCS에 없습니다: {missing}")
        return False


_GCS_MODE_MAP = {
    "video": ("video_540p/{cid}_540p.mp4", "video/mp4"),
    "final": ("jsonl/{cid}_final.jsonl", "text/plain"),
    "ref":   ("jsonl/{cid}_ref.jsonl", "text/plain"),
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
    modes = ["final", "ref"]
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
        mode: 'video', 'final', 'ref' 중 하나
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

def truncate_jsonl_by_scene_idx(jsonl_text, start_idx, end_idx):
    """JSONL 텍스트에서 scene_idx가 [start_idx, end_idx] 범위인 Scene만 추출합니다."""
    truncated_lines = []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            scene = json.loads(line)
            s_idx = scene.get("scene_idx")
            if s_idx is not None and start_idx <= s_idx <= end_idx:
                truncated_lines.append(line)
        except json.JSONDecodeError:
            continue
    return "\n".join(truncated_lines)


def get_gcs_descriptions_by_scene_idx(gs_bucket_name, content_id, mode, start_idx, end_idx):
    """scene_idx 구간의 description 필드만 추출하여 [Scene N] 태그와 함께 반환합니다."""
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")
    path_template, _ = _GCS_MODE_MAP[mode]
    blob_path = path_template.format(cid=content_id)
    jsonl_text = download_gcs_text(gs_bucket_name, blob_path)
    lines = []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            scene = json.loads(line)
            s_idx = scene.get("scene_idx")
            if s_idx is not None and start_idx <= s_idx <= end_idx:
                desc = scene.get("description", "")
                lines.append(f"[Scene {s_idx}]\n{desc}")
        except json.JSONDecodeError:
            continue
    return "\n\n".join(lines)


def get_gcs_text_by_scene_idx(gs_bucket_name, content_id, mode, start_idx, end_idx):
    """scene_idx 구간의 전체 JSONL 텍스트를 반환합니다."""
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")
    path_template, _ = _GCS_MODE_MAP[mode]
    blob_path = path_template.format(cid=content_id)
    jsonl_text = download_gcs_text(gs_bucket_name, blob_path)
    return truncate_jsonl_by_scene_idx(jsonl_text, start_idx, end_idx)


def get_gcs_raw_fields_by_scene_idx(gs_bucket_name, content_id, start_idx, end_idx):
    """*_raw.jsonl(또는 *_ref.jsonl)에서 Shot별 raw_speech/raw_ocr을
    Scene 단위로 병합하여 정제된 JSON Lines 텍스트로 반환합니다.

    'raw' 모드 Source로 사용됩니다.
    - speech: 모든 Shot의 raw_speech를 시간 순서대로 이어 붙인 통합 텍스트
    - on_screen_text: 모든 Shot의 raw_ocr에서 중복 제거한 고유 텍스트 목록
    """
    path_template, _ = _GCS_MODE_MAP["final"]
    blob_path = path_template.format(cid=content_id)
    jsonl_text = download_gcs_text(gs_bucket_name, blob_path)

    lines = []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            scene = json.loads(line)
            s_idx = scene.get("scene_idx")
            if s_idx is None or not (start_idx <= s_idx <= end_idx):
                continue

            timeline = scene.get("timeline", [])

            # Shot별 raw_asr를 시간 순서대로 concat
            speech_parts = []
            for shot in timeline:
                text = (shot.get("raw_asr") or "").strip()
                if text:
                    speech_parts.append(text)
            speech = " ".join(speech_parts)

            # Shot별 raw_ocr를 중복 제거 (쉼표로 구분된 복수 항목도 분리)
            seen_ocr = set()
            ocr_list = []
            for shot in timeline:
                raw_ocr = (shot.get("raw_ocr") or "").strip()
                if not raw_ocr:
                    continue
                for item in raw_ocr.split(","):
                    item = item.strip()
                    if item and item not in seen_ocr:
                        seen_ocr.add(item)
                        ocr_list.append(item)

            filtered = {"scene_idx": s_idx, "duration": scene.get("duration", "")}
            if speech:
                filtered["speech"] = speech
            if ocr_list:
                filtered["on_screen_text"] = ocr_list
            lines.append(json.dumps(filtered, ensure_ascii=False))
        except json.JSONDecodeError:
            continue
    return "\n".join(lines)


# ============================================================
# Processed JSONL Helpers (저작권 안전 파편화 데이터)
# ============================================================

def parse_duration_to_times(duration):
    """duration 값을 (start_time, end_time) float 튜플로 파싱합니다.
    
    지원 형식:
    - 리스트: [0.0, 35.97]
    - 문자열: '0.0 - 35.97'
    """
    if isinstance(duration, (list, tuple)) and len(duration) >= 2:
        return float(duration[0]), float(duration[1])
    parts = str(duration).split(" - ")
    return float(parts[0]), float(parts[1])


def format_vlm_structure_as_text(vlm_struct):
    """vlm_img_structure 또는 vlm_mm_structure 딕셔너리를 읽기 좋은 텍스트로 변환합니다.

    신규 포맷 (subjects, actions, contexts) 및 구 포맷 (subject, environment, actions, context) 모두 지원합니다.

    예시 출력 (신규):
      Subjects: him. The; with various; background features; A man; ...
      Actions: the man.; his chin.; he listens; ...
      Contexts: of the; the chat; (A Man; ...

    예시 출력 (구):
      Subject: A marine animal swimming through ice-covered waters
      Environment: An Arctic sea with floating ice floes
      Actions: The animal is moving through the water; creating ripples
      Context: animals, bears, ice, melting, sea, warming, water
    """
    if not vlm_struct or not isinstance(vlm_struct, dict):
        return ""
    lines = []
    # 신규 포맷: subjects, actions, contexts
    if vlm_struct.get("subjects"):
        lines.append(f"Subjects: {'; '.join(vlm_struct['subjects'])}")
    elif vlm_struct.get("subject"):
        lines.append(f"Subject: {vlm_struct['subject']}")
    if vlm_struct.get("environment"):
        lines.append(f"Environment: {vlm_struct['environment']}")
    if vlm_struct.get("actions"):
        lines.append(f"Actions: {'; '.join(vlm_struct['actions'])}")
    if vlm_struct.get("contexts"):
        lines.append(f"Contexts: {'; '.join(vlm_struct['contexts'])}")
    elif vlm_struct.get("context"):
        lines.append(f"Context: {', '.join(vlm_struct['context'])}")
    return "\n".join(lines)


def format_vlm_graph_as_text(vlm_graph):
    """vlm_graph 리스트를 읽기 좋은 텍스트로 변환합니다.

    예시 출력:
      (narwhal) -[DOING]-> (swimming)
      (narwhal) -[AT]-> (ocean)
      (ice) -[AT]-> (river)
    """
    if not vlm_graph or not isinstance(vlm_graph, list):
        return ""
    lines = []
    for triple in vlm_graph:
        if isinstance(triple, dict):
            s = triple.get("subject", "?")
            r = triple.get("relation", "?")
            o = triple.get("object", "?")
            lines.append(f"({s}) -[{r}]-> ({o})")
    return "\n".join(lines)


def get_processed_vlm_descriptions_by_scene_idx(gs_bucket_name, content_id, vlm_key, start_idx, end_idx):
    """*_final.jsonl에서 지정된 VLM 구조 데이터를 추출하여
    [Scene N] 태그와 함께 텍스트 형태로 반환합니다.

    Args:
        vlm_key: 'vlm_img_structure_chunk2', 'vlm_graph' 등 추출할 VLM 필드 키
    """
    path_template, _ = _GCS_MODE_MAP["final"]
    blob_path = path_template.format(cid=content_id)
    jsonl_text = download_gcs_text(gs_bucket_name, blob_path)
    lines = []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            scene = json.loads(line)
            s_idx = scene.get("scene_idx")
            if s_idx is not None and start_idx <= s_idx <= end_idx:
                vlm_data = scene.get(vlm_key, {})
                if isinstance(vlm_data, str):
                    desc = vlm_data
                elif isinstance(vlm_data, list):
                    desc = format_vlm_graph_as_text(vlm_data)
                else:
                    desc = format_vlm_structure_as_text(vlm_data)
                
                if desc:
                    lines.append(f"[Scene {s_idx}]\n{desc}")
        except json.JSONDecodeError:
            continue
    return "\n\n".join(lines)





def get_gcs_raw_with_mmvlm_by_scene_idx(gs_bucket_name, content_id, start_idx, end_idx):
    """*_final.jsonl에서 raw_asr/raw_ocr과 vlm_mm_description을
    Scene 단위로 결합하여 정제된 JSON Lines 텍스트로 반환합니다.

    'raw_with_mmvlm' 모드 Source로 사용됩니다.
    """
    path_template, _ = _GCS_MODE_MAP["final"]
    blob_path = path_template.format(cid=content_id)
    jsonl_text = download_gcs_text(gs_bucket_name, blob_path)

    lines = []
    for line in jsonl_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            scene = json.loads(line)
            s_idx = scene.get("scene_idx")
            if s_idx is None or not (start_idx <= s_idx <= end_idx):
                continue

            timeline = scene.get("timeline", [])

            # Shot별 raw_asr concat
            speech_parts = []
            for shot in timeline:
                text = (shot.get("raw_asr") or "").strip()
                if text:
                    speech_parts.append(text)
            speech = " ".join(speech_parts)

            # Shot별 raw_ocr 중복 제거
            seen_ocr = set()
            ocr_list = []
            for shot in timeline:
                raw_ocr = (shot.get("raw_ocr") or "").strip()
                if not raw_ocr:
                    continue
                for item in raw_ocr.split(","):
                    item = item.strip()
                    if item and item not in seen_ocr:
                        seen_ocr.add(item)
                        ocr_list.append(item)

            filtered = {"scene_idx": s_idx, "duration": scene.get("duration", "")}

            # vlm_mm_description 먼저: 장면 개요 (context anchor)
            mm_desc = scene.get("vlm_mm_description", "")
            if mm_desc:
                filtered["vlm_mm_description"] = mm_desc

            # speech/on_screen_text 나중: 사실 정보로 보정 (recency bias 활용)
            if speech:
                filtered["speech"] = speech
            if ocr_list:
                filtered["on_screen_text"] = ocr_list

            lines.append(json.dumps(filtered, ensure_ascii=False))
        except json.JSONDecodeError:
            continue
    return "\n".join(lines)


# ───────────────────────────────────────────────
# 공통 Source 빌드 헬퍼
# ───────────────────────────────────────────────

# 텍스트 모드별 fetch 함수 매핑
_TEXT_MODE_FETCHERS = {
    "raw":              lambda gs, cid, s, e: get_gcs_raw_fields_by_scene_idx(gs, cid, s, e),
    "imgvlm_chunk2":    lambda gs, cid, s, e: get_processed_vlm_descriptions_by_scene_idx(gs, cid, "vlm_img_structure_chunk2", s, e),
    "imgvlm_graph":     lambda gs, cid, s, e: get_processed_vlm_descriptions_by_scene_idx(gs, cid, "vlm_graph", s, e),
    "raw_with_mmvlm":   lambda gs, cid, s, e: get_gcs_raw_with_mmvlm_by_scene_idx(gs, cid, s, e),
}


def _text_to_part(text):
    """텍스트를 genai Part로 변환합니다. 빈 문자열이면 빈 문자열 그대로 반환."""
    if not text:
        return ""
    return types.Part.from_bytes(data=text.encode("utf-8"), mime_type="text/plain")


def build_mode_parts(gs_bucket_name, content_id, target_modes,
                     current_start_idx, current_end_idx,
                     past_start_idx=None, past_end_idx=None,
                     current_start_time=None, current_end_time=None,
                     past_start_time=None, past_end_time=None):
    """target_modes에 대해 현재/과거 데이터 Part를 빌드합니다.

    Args:
        gs_bucket_name: GCS 버킷명
        content_id: 콘텐츠 ID
        target_modes: 빌드할 모드 리스트 (예: ["video", "raw", "raw_with_mmvlm", "imgvlm_chunk2", "imgvlm_graph"])
        current_start_idx, current_end_idx: 현재 Scene 구간
        past_start_idx, past_end_idx: 과거 Scene 구간 (None이면 과거 없음)
        current_start_time, current_end_time: 현재 Scene의 타임스탬프 오버라이드
        past_start_time, past_end_time: 과거 Scene의 타임스탬프 오버라이드

    Returns:
        (past_parts, current_parts) 딕셔너리 튜플
    """
    has_past = past_start_idx is not None and past_end_idx is not None
    past_parts = {}
    current_parts = {}

    if "video" in target_modes:
        if has_past:
            past_parts["video"] = process_gcs_file_by_scene_idx(
                gs_bucket_name, content_id, "video", past_start_idx, past_end_idx,
                start_time_override=past_start_time, end_time_override=past_end_time
            )
        current_parts["video"] = process_gcs_file_by_scene_idx(
            gs_bucket_name, content_id, "video", current_start_idx, current_end_idx,
            start_time_override=current_start_time, end_time_override=current_end_time
        )

    for mode, fetcher in _TEXT_MODE_FETCHERS.items():
        if mode not in target_modes:
            continue
        if has_past:
            past_parts[mode] = _text_to_part(fetcher(gs_bucket_name, content_id, past_start_idx, past_end_idx))
        current_parts[mode] = _text_to_part(fetcher(gs_bucket_name, content_id, current_start_idx, current_end_idx))

    return past_parts, current_parts


def process_gcs_video_part(gs_bucket_name, content_id, start_time, end_time):
    """[start_time, end_time] 구간으로 클리핑된 비디오 Part를 반환합니다. (VideoMetadata 기반, 캐시 적용)"""
    path_template, mime_type = _GCS_MODE_MAP["video"]
    blob_path = path_template.format(cid=content_id)
    file_uri = f"gs://{gs_bucket_name}/{blob_path}"

    cache_key = f"{gs_bucket_name}/{content_id}/video/{start_time:.3f}-{end_time:.3f}"
    if cache_key in _gcs_part_cache:
        return _gcs_part_cache[cache_key]
    part = types.Part(
        file_data=types.FileData(file_uri=file_uri, mime_type=mime_type),
        video_metadata=types.VideoMetadata(
            start_offset=f"{start_time:.3f}s",
            end_offset=f"{end_time:.3f}s",
        ),
    )
    _gcs_part_cache[cache_key] = part
    return part


def process_gcs_file_by_scene_idx(gs_bucket_name, content_id, mode, start_idx, end_idx, start_time_override=None, end_time_override=None):
    """scene_idx 구간 데이터 Part를 반환합니다.

    - video 모드: ref JSONL에서 scene_idx로 start/end time lookup 후 VideoMetadata 클리핑 (override 지정 시 우선 사용)
    - jsonl 모드: GCS에서 다운로드 후 scene_idx 구간만 추출하여 인라인 Part 반환
    """
    if mode not in _GCS_MODE_MAP:
        raise ValueError(f"mode should be one of {list(_GCS_MODE_MAP.keys())}, got '{mode}'.")

    if mode == "video":
        if start_time_override is not None and end_time_override is not None:
            start_time = start_time_override
            end_time = end_time_override
        else:
            ref_scenes = load_scenes(gs_bucket_name, content_id, mode="ref")
            start_scene = next((s for s in ref_scenes if s.get("scene_idx") == start_idx), None)
            end_scene   = next((s for s in ref_scenes if s.get("scene_idx") == end_idx), None)
            # ref JSONL은 "duration": "0.0 - 35.97" 형태이므로 파싱하여 사용
            start_time = parse_duration_to_times(start_scene["duration"])[0] if start_scene and start_scene.get("duration") else 0.0
            end_time   = parse_duration_to_times(end_scene["duration"])[1]   if end_scene   and end_scene.get("duration")   else 0.0
        return process_gcs_video_part(gs_bucket_name, content_id, start_time, end_time)
    else:
        range_text = get_gcs_text_by_scene_idx(gs_bucket_name, content_id, mode, start_idx, end_idx)
        return types.Part.from_bytes(data=range_text.encode("utf-8"), mime_type="text/plain")


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
            is_retryable = any(code in err_msg for code in ["429", "500", "503", "504", "403"])

            if not is_retryable:
                print(f"      [{label} 치명적 오류] {err_msg}")
                raise

            print(f"      [{label} 오류] {err_msg}")
            print(f"      -> {delay}초 후 재시도합니다... (시도 횟수: {attempt})")
            time.sleep(delay)





# ============================================================
# Post-processing / Validation
# ============================================================

def sort_and_validate_jsonl(file_path, keypoints_by_content, expected_modes=None, mode_key=None):
    """JSONL 파일을 content_id, scene_idx, mode 순으로 정렬하고 누락된 Scene/모드을 점검합니다.
    - 기존의 'pipeline_done' 시그널은 정렬 과정에서 제거됩니다.
    - 각 줄이 (content_id, scene_idx, mode) 단위의 flat 포맷임을 가정합니다.

    Returns:
        (missing_scenes, data_records) 튜플. missing_scenes는 누락 항목 리스트, data_records는 전체 레코드 리스트.
    """
    if not os.path.exists(file_path):
        print(f"[Warning] 파일을 찾을 수 없습니다: {file_path}")
        return []

    # 정렬 수행
    sort_jsonl_file(file_path)

    print(f"\n{'='*50}")
    print(f"결과 파일 검증: {file_path}")

    data_records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if not obj.get("pipeline_done"):
                        data_records.append(obj)
                except json.JSONDecodeError:
                    pass

    # 누락 점검
    print("\n[최종 누락분 점검]")

    if expected_modes is not None and len(expected_modes) == 0:
        # 모드 없는 파일 (KSS 등): (content_id, scene_idx) 단위로만 점검
        done_set = {
            (x.get("content_id"), x.get("scene_idx"))
            for x in data_records
        }
        missing_scenes = []
        for c_id, kps in keypoints_by_content.items():
            for kp in kps:
                s_idx = kp.get("scene_idx", kps.index(kp))
                if (c_id, s_idx) not in done_set:
                    missing_scenes.append((c_id, s_idx, None))

        if missing_scenes:
            print(f"-> 총 {len(missing_scenes)}개의 Scene 처리가 누락되었습니다.")
            for c_id, s_idx, _ in missing_scenes:
                print(f"    - ({c_id}, scene={s_idx})")
        else:
            print("-> 모든 Scene이 누락 없이 정상적으로 생성되었습니다.")
    else:
        # 모드 기반 파일 (KSD, VH 등): (content_id, scene_idx, mode) 3-tuple 단위
        modes_to_check = expected_modes or ["video", "raw", "raw_with_mmvlm", "imgvlm"]
        done_set_modes = {
            (x.get("content_id"), x.get("scene_idx"), x.get("mode"))
            for x in data_records
        }

        missing_scenes = []
        for c_id, kps in keypoints_by_content.items():
            for kp in kps:
                s_idx = kp.get("scene_idx", kps.index(kp))
                for m in modes_to_check:
                    if (c_id, s_idx, m) not in done_set_modes:
                        missing_scenes.append((c_id, s_idx, m))

        if missing_scenes:
            print(f"-> 총 {len(missing_scenes)}개의 (Scene, Mode) 처리가 누락되었습니다.")
            for c_id, s_idx, m in missing_scenes:
                print(f"    - ({c_id}, scene={s_idx}, mode={m})")
        else:
            print("-> 모든 Scene/Mode가 누락 없이 정상적으로 생성되었습니다.")
    print("=" * 50 + "\n")
    
    return missing_scenes, data_records


# ============================================================
# Common Pipeline Helpers
# ============================================================

def load_keypoints_by_content(jsonl_path):
    """Keypoint JSONL 파일을 파싱하여 {content_id: [kp, ...]} 딕셔너리로 반환합니다."""
    keypoints_by_content = {}
    for data in load_jsonl(jsonl_path):
        c_id = data.get("content_id")
        kps = data.get("keypoints", [])
        if c_id and kps:
            keypoints_by_content[c_id] = kps
    return keypoints_by_content


def load_summary_map(jsonl_path):
    """KeyScene Summary JSONL을 파싱하여 {(content_id, scene_idx): summary_text} 맵을 반환합니다."""
    summary_map = {}
    if not os.path.exists(jsonl_path):
        return summary_map
    for rec in load_jsonl(jsonl_path):
        c_id = rec.get("content_id")
        s_idx = rec.get("scene_idx")
        if c_id and s_idx is not None:
            summary_map[(c_id, s_idx)] = rec.get("summary", "")
    return summary_map


def check_input_file(path, hint=""):
    """입력 파일의 존재 여부를 확인합니다. 없으면 에러 출력 후 False 반환.

    Args:
        path: 확인할 파일 경로
        hint: 에러 메시지에 참고로 출력할 선행 스크립트 안내 (예: "먼저 identify_keypoint.py를 실행하세요.")
    """
    if os.path.exists(path):
        return True
    msg = f"Error: {path} 파일이 존재하지 않습니다."
    if hint:
        msg += f" {hint}"
    print(msg)
    return False


def print_pipeline_banner(title):
    """파이프라인 시작 배너를 출력합니다."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def print_pipeline_done(output_path):
    """파이프라인 완료 배너를 출력합니다."""
    print("\n" + "=" * 50)
    print(f"모든 작업이 완료되었습니다. 저장 위치: {output_path}")
    print("=" * 50)


_MODE_SORT_ORDER = {"video": 0, "raw": 1, "raw_with_mmvlm": 2, "imgvlm_chunk2": 3, "imgvlm_graph": 4, "kss": 5}

def sort_jsonl_file(filepath):
    """JSONL 파일을 (content_id, scene_idx, mode, query) 순으로 정렬합니다."""
    if not os.path.exists(filepath):
        return
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if not obj.get("pipeline_done"):
                        records.append(obj)
                except json.JSONDecodeError:
                    pass
    if not records:
        return
    records.sort(key=lambda x: (
        x.get("content_id", ""),
        x.get("scene_idx", 0),
        _MODE_SORT_ORDER.get(x.get("mode", ""), 99),
        x.get("query", ""),
    ))
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[정렬] {filepath} ({len(records)}개 항목 정렬 완료)")
