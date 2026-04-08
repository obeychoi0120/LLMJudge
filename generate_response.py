import os
import time
import argparse
import json
import subprocess
import sys
import concurrent.futures
import threading
from gemini_api_utils import (
    create_client, make_generate_config,
    process_gcs_file, process_gcs_file_range,
    check_gcs_files_exist, load_config,
    _retry_api_call, load_scenes,
    ensure_output_dir, preload_content_metadata,
)

# ============================================================
# System Prompts (Local)
# ============================================================

_JSONL_VIEWER_BASE = """당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
당신에게는 두 가지 종류의 시청 기억(Memory)이 제공됩니다:
1. **과거 정보 (Past Information)**: 지금까지 시청하며 습득한 맥락입니다.
2. **현재 정보 (Current Information)**: 시청자가 지금 집중해서 보고 있는 구간입니다.

이 정보를 바탕으로 사용자 질문에 대해 가장 자연스럽고 정확한 한국어 답변을 제공해 주세요.

[Description 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- description: 해당 Scene의 시각적 상황, 인물 행동, 대사, 화면 자막, 환경음 등을 종합한 자세한 묘사

[분석 및 지시사항]
- **현재 장면에 집중**: 답변 시 "현재 정보(Current Information)" 구간에서 일어나는 일들에 우선순위를 두어 답변하세요. 과거 정보는 맥락을 설명하는 데 활용하세요.
- **자연스러운 시청자 관점**: "JSON", "타임스탬프" 등 기계적인 용어 대신 "영상에서는~", "자막에 ~라고 나옵니다"와 같이 실제 시청자처럼 말하세요.
- **외부 자료 검색 금지**: 오직 당신의 시청 기억(제공된 정보)에만 의존하세요."""

_REFERENCE_PROMPT = """당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
제공되는 원본 영상과 Reference용 메타데이터를 모두 참조하여, 사용자 질문에 대해 가장 정확하고 포괄적인 한국어 답변을 생성해 주세요.

[Reference 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- speech: 등장인물들의 대사
- texts: 화면 속 자막, 간판 정보 등
- sounds: 환경음 및 효과음

[메타데이터 사용 시 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 잘못 전사될 수 있습니다.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식되거나, 의미 없는 워터마크/로고가 포함될 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다 (예: 괴물 소리를 고양이 골골송으로 인식하는 등). 
반드시 비디오 프레임의 시각 정보를 우선적으로 참고하고, 메타데이터는 보조 자료로만 활용하세요.

이 답변은 다른 AI 모델의 답변을 평가하기 위한 '기준 답변(Reference Answer)'으로 사용됩니다.
따라서 핵심 사실, 대사, 행동, 맥락을 빠짐없이 포함하되 자연스럽고 읽기 쉽게 작성해 주세요.
외부 자료 검색은 금지합니다. 오직 제공된 영상과 메타데이터만 활용하세요."""


def make_generation_config(mode="desc", thinking_budget=None):
    """Response 생성용 GenerateContentConfig를 반환합니다."""
    if mode == "desc":
        prompt = _JSONL_VIEWER_BASE
    elif mode == "video":
        prompt = "당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다. 외부 정보를 절대 검색하지 말고, 제공된 영상 정보만을 사용하여 사용자 질문에 답변하세요."
    else:
        prompt = ""
    return make_generate_config(system_instruction=prompt, thinking_budget=thinking_budget)


def make_reference_config(thinking_budget=None):
    """Reference Answer 생성용 config를 반환합니다."""
    return make_generate_config(system_instruction=_REFERENCE_PROMPT, thinking_budget=thinking_budget)


def main():
    parser = argparse.ArgumentParser(description="Generate Responses using Gemini models")
    parser.add_argument("--json_file", default="assets/user_query.jsonl", help="질문 목록 JSONL 파일 경로 (generate_user_query.py 출력)")
    parser.add_argument("--output_file", default="assets/uq_responses.jsonl", help="통합 답변 목록을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--reference_file", default="assets/uq_references.jsonl", help="Reference 답변을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--uq_response_model", default="gemini-2.5-flash", help="사용할 생성 모델명")
    parser.add_argument("--uq_reference_model", default="gemini-2.5-pro", help="Reference Answer 생성 모델명")
    parser.add_argument("--no-uq-reference-ref", dest="uq_reference_use_ref", action="store_false", help="Reference 생성 시 Ref JSONL 미참조 (Video만 사용)")
    parser.set_defaults(uq_reference_use_ref=True)
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")
    parser.add_argument("--uq_response_thinking_budget", type=int, default=-1,
                        help="UQ Response 생성 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")
    parser.add_argument("--uq_reference_thinking_budget", type=int, default=4096,
                        help="UQ Reference Answer 생성 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}")
    client = create_client(args.gcp_project_id, args.location)
    
    # 출력 폴더 생성 (responses, references)
    for fpath in [args.output_file, args.reference_file]:
        ensure_output_dir(fpath)

    print("\n" + "=" * 50)
    print("Gemini Inference 프로세스를 시작합니다 (Session-based, JSONL Pipeline).")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다. 다른 터미널의 출력을 기다리며 지속 처리합니다.")
    print("=" * 50)

    try:
        while True:
            # 1. Output 진행률 읽기 - (content_id, query) 쌍 단위로 추적
            processed_pairs = set()  # (content_id, query) 튜플
            
            # 1-1. 유효한 Reference 목록 먼저 수집
            valid_references = set()
            if os.path.exists(args.reference_file):
                with open(args.reference_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            ref = json.loads(line)
                            c_id = ref.get("content_id")
                            query = ref.get("query")
                            ref_text = ref.get("reference", "")
                            if c_id and query and not str(ref_text).startswith("Error"):
                                valid_references.add((c_id, query))
                        except json.JSONDecodeError:
                            pass

            # 1-2. Responses 읽기하여 최종 완료 상태 확인
            if os.path.exists(args.output_file):
                with open(args.output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            ans = json.loads(line)
                            c_id = ans.get("content_id")
                            query = ans.get("query")
                            if c_id and query and (c_id, query) in valid_references:
                                # 3개 모드에 유효한 답변이 모두 있으면 완료로 간주
                                answers = ans.get("answers", {})
                                is_complete = all(
                                    answers.get(m) and not str(answers.get(m, "")).startswith("Error")
                                    for m in ["video", "desc"]
                                )
                                if is_complete:
                                    processed_pairs.add((c_id, query))
                        except json.JSONDecodeError:
                            pass

            # 2. Input 읽기
            query_dict = {}
            if os.path.exists(args.json_file):
                with open(args.json_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "content_id" in data:
                                    query_dict[data["content_id"]] = data
                            except json.JSONDecodeError:
                                pass
                query_list = list(query_dict.values())
            else:
                query_list = []
                if not args.continuous:
                    print(f"Error: {args.json_file} 파일이 존재하지 않습니다.")
                    return

            new_data_processed = False

            # Resume Plan 계산 및 출력
            # 새 포맷: queries = [{query, end_time, scene_idx}]
            # pending_work: content_id -> [q_item, ...]
            pending_work = {}
            for item in query_list:
                c_id = item["content_id"]
                raw_queries = item.get("queries", [])
                c_pending = []
                for q_item in raw_queries:
                    # 새 포맷(dict) 및 구 포맷(str) 모두 처리
                    q_str = q_item["query"] if isinstance(q_item, dict) else q_item
                    if (c_id, q_str) not in processed_pairs:
                        c_pending.append(q_item)
                if c_pending:
                    pending_work[c_id] = c_pending

            if pending_work:
                print("\n[TODO] 작업 목록:")
                for c_id, q_items in pending_work.items():
                    print(f"- content_id '{c_id}':")
                    for q_item in q_items:
                        q_str = q_item["query"] if isinstance(q_item, dict) else q_item
                        end_val = q_item.get("end_time") if isinstance(q_item, dict) else None
                        end_str = f" [end={end_val:.1f}s]" if end_val is not None else ""
                        print(f"    - query \"{q_str}\"{end_str}")
                print("-" * 50)

            file_write_lock = threading.Lock()

            def process_item(item):
                content_id = item["content_id"]
                if content_id not in pending_work:
                    return False

                pending_queries = pending_work[content_id]

                print(f"\nProcessing Content: '{content_id}'")

                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    return False

                # Reference 메타데이터 프리로드 & Scene 로드
                preload_content_metadata(args.gs_bucket_name, content_id)
                try:
                    ref_scenes = load_scenes(args.gs_bucket_name, content_id, mode="desc")
                except Exception as e:
                    print(f"[{content_id}] Warning: Desc JSONL 로드 실패 ({e}). start_time이 0으로 설정될 수 있습니다.")
                    ref_scenes = []

                print(f"[{content_id}] Initializing Generation configs ({args.uq_response_model})...")
                gen_configs = {}
                for mode in ["video", "desc"]:
                    gen_configs[mode] = make_generation_config(mode=mode, thinking_budget=args.uq_response_thinking_budget)

                print(f"[{content_id}] Initializing Reference config ({args.uq_reference_model}, Ref={'ON' if args.uq_reference_use_ref else 'OFF'})...")
                ref_config = make_reference_config(thinking_budget=args.uq_reference_thinking_budget)

                for q_item in pending_queries:
                    # 새 포맷(dict) / 구 포맷(str) 호환 처리
                    if isinstance(q_item, dict):
                        user_prompt = q_item["query"]
                        end_time = float(q_item.get("end_time", 0))
                        scene_idx = q_item.get("scene_idx", -1)
                        # start_time이 있으면 사용, 없으면 scene_idx로 찾기
                        start_time = q_item.get("start_time")
                        if start_time is None and scene_idx != -1:
                            target_scene = next((s for s in ref_scenes if s.get("scene_idx") == scene_idx), None)
                            start_time = target_scene.get("start_time", 0.0) if target_scene else 0.0
                        else:
                            start_time = float(start_time or 0.0)
                        
                        has_end_time = end_time > 0
                    else:
                        user_prompt = q_item
                        end_time = 0.0
                        start_time = 0.0
                        scene_idx = -1
                        has_end_time = False

                    if (content_id, user_prompt) in processed_pairs:
                        print(f"[{content_id}] Query: '{user_prompt[:40]}...' -> already completed (skip)")
                        continue

                    end_label = f"Range=[{start_time:.1f}s ~ {end_time:.1f}s]" if has_end_time else "full"
                    print(f"[{content_id}] Processing Query [{end_label}]: '{user_prompt}'")

                    # Past/Current Parts 데이터 준비
                    if has_end_time:
                        # 2. Past Data (0 ~ start_time)
                        past_parts = {
                            "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", 0.0, start_time),
                            "desc":  process_gcs_file_range(args.gs_bucket_name, content_id, "desc",  0.0, start_time),
                            "ref":   process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   0.0, start_time),
                        }
                        # 3. Current Data (start_time ~ end_time)
                        curr_parts = {
                            "video": process_gcs_file_range(args.gs_bucket_name, content_id, "video", start_time, end_time),
                            "desc":  process_gcs_file_range(args.gs_bucket_name, content_id, "desc",  start_time, end_time),
                            "ref":   process_gcs_file_range(args.gs_bucket_name, content_id, "ref",   start_time, end_time),
                        }
                    else:
                        # 통 데이터 (하위 호환성)
                        past_parts = {"video": None, "desc": None, "ref": None}
                        curr_parts = {
                            "video": process_gcs_file(args.gs_bucket_name, content_id, mode="video"),
                            "desc":  process_gcs_file(args.gs_bucket_name, content_id, mode="desc"),
                            "ref":   process_gcs_file(args.gs_bucket_name, content_id, mode="ref"),
                        }

                    # 1. Reference Answer 생성
                    print(f"[{content_id}]  Generating [reference]...")
                    try:
                        if has_end_time:
                            ref_contents = [
                                "--- Past Information ---", past_parts["video"], past_parts["ref"],
                                "--- Current Information ---", curr_parts["video"], curr_parts["ref"]
                            ]
                        else:
                            ref_contents = [curr_parts["video"], curr_parts["ref"]]
                        
                        response_text = _retry_api_call(
                            lambda: client.models.generate_content(
                                model=args.uq_reference_model,
                                contents=[*ref_contents, user_prompt],
                                config=ref_config
                            ).text,
                            label=f"[{content_id}] Reference 생성"
                        )
                        reference_answer = response_text
                        print(f"[{content_id}]  Reference answer generated ({len(reference_answer.split())} words)")
                    except Exception as e:
                        print(f"[{content_id}]  Generating [reference] Error: {e}")
                        reference_answer = f"Error: {str(e)}"

                    # 2. 3개 Mode 답변 생성
                    answers_for_query = {}

                    def generate_for_mode(mode):
                        print(f"[{content_id}]  Generating [{mode}]...")
                        try:
                            time.sleep(1)
                            if has_end_time:
                                mode_contents = [
                                    "--- Past Information ---", past_parts[mode],
                                    "--- Current Information ---", curr_parts[mode]
                                ]
                            else:
                                mode_contents = [curr_parts[mode]]
                                
                            mode_contents_with_prompt = [*mode_contents, user_prompt]
                            answer_text = _retry_api_call(
                                lambda: client.models.generate_content(
                                    model=args.uq_response_model,
                                    contents=mode_contents_with_prompt,
                                    config=gen_configs[mode]
                                ).text,
                                label=f"[{content_id}] [{mode}] 생성"
                            )
                            return mode, answer_text
                        except Exception as e:
                            print(f"[{content_id}]  Generating [{mode}] Error: {e}")
                            return mode, f"Error: {str(e)}"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as mode_executor:
                        futures_mode = [mode_executor.submit(generate_for_mode, m) for m in ["video", "desc"]]
                        for future in concurrent.futures.as_completed(futures_mode):
                            m, text = future.result()
                            answers_for_query[m] = text

                    ordered_answers = {m: answers_for_query[m] for m in ["video", "desc"] if m in answers_for_query}

                    ref_record = {
                        "content_id": content_id,
                        "scene_idx": scene_idx if has_end_time else None,
                        "start_time": start_time if has_end_time else None,
                        "end_time": end_time if has_end_time else None,
                        "query": user_prompt,
                        "reference": reference_answer
                    }
                    response_record = {
                        "content_id": content_id,
                        "scene_idx": scene_idx if has_end_time else None,
                        "start_time": start_time if has_end_time else None,
                        "end_time": end_time if has_end_time else None,
                        "query": user_prompt,
                        "answers": ordered_answers
                    }

                    with file_write_lock:
                        with open(args.reference_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(ref_record, ensure_ascii=False) + "\n")
                        with open(args.output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(response_record, ensure_ascii=False) + "\n")

                    processed_pairs.add((content_id, user_prompt))
                    print(f"[{content_id}]  -> Reference/Response 저장 완료")
                    print("-" * 50)

                return True

            for item in query_list:
                if process_item(item):
                    new_data_processed = True

            if not args.continuous:
                break
            
            if not new_data_processed:
                # 새 데이터가 없으면 잠시 대기
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 모니터링 루프가 중단되었습니다.")

    if not args.continuous:
        print("\n[Aggregation] JSONL 결과를 분석용 JSON 형식으로 병합합니다...")
        output_dir = os.path.dirname(args.output_file) or "assets"
        subprocess.run([sys.executable, "jsonl_to_json.py", "--input_dir", output_dir])

    print("\n생성 프로세스가 완료/종료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
