import os
import time
import argparse
import json
import subprocess
import sys
import concurrent.futures
import threading
from gemini_api_utils import (
    make_generate_config,
    process_gcs_file, process_gcs_file_range,
    check_gcs_files_exist,
    _retry_api_call, load_scenes,
    ensure_output_dir, preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
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

def make_generation_config(mode="img_desc", thinking_budget=None):
    """Response 생성용 GenerateContentConfig를 반환합니다."""
    if mode in ["raw", "img_desc", "mm_desc"]:
        prompt = _JSONL_VIEWER_BASE
    elif mode == "video":
        prompt = "당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다. 외부 정보를 절대 검색하지 말고, 제공된 영상 정보만을 사용하여 사용자 질문에 답변하세요."
    else:
        prompt = ""
    return make_generate_config(system_instruction=prompt, thinking_budget=thinking_budget)



# ============================================================
# Input / Progress Helpers
# ============================================================

def _load_query_items(json_file):
    """JSONL 입력 파일을 파싱하여 content_id별 query dict 리스트로 반환합니다."""
    query_dict = {}
    for data in load_jsonl(json_file):
        if "content_id" not in data:
            continue
        c_id = data["content_id"]
        if c_id not in query_dict:
            query_dict[c_id] = {"content_id": c_id, "queries": []}
        if "queries" in data:
            # scene별 레코드: queries + scene context를 각 query에 주입
            scene_ctx = {
                "scene_idx": data.get("scene_idx", -1),
                "start_time": data.get("start_time"),
                "end_time": data.get("end_time"),
            }
            for q in data["queries"]:
                if isinstance(q, dict):
                    # 이미 dict 형태면 scene_ctx 보완만
                    enriched = {**scene_ctx, **q}
                    query_dict[c_id]["queries"].append(enriched)
                else:
                    query_dict[c_id]["queries"].append({
                        "query": q, **scene_ctx
                    })
    return list(query_dict.values())


def _load_completed_pairs(responses_path):
    """responses 파일로부터 완료된 (content_id, query) 쌍을 반환합니다."""
    completed = set()
    for ans in load_jsonl(responses_path):
        c_id = ans.get("content_id")
        query = ans.get("query")
        if c_id and query:
            answers = ans.get("answers", {})
            is_complete = all(
                answers.get(m) and not str(answers.get(m, "")).startswith("Error")
                for m in ["video", "raw", "img_desc", "mm_desc"]
            )
            if is_complete:
                completed.add((c_id, query))

    return completed


def _build_parts(gs_bucket_name, content_id, start_time, end_time, has_end_time):
    """Past/Current Parts를 빌드합니다."""
    modes_to_fetch = ["video", "raw", "img_desc", "mm_desc"]
    if has_end_time:
        past_parts = {m: process_gcs_file_range(gs_bucket_name, content_id, m, 0.0, start_time)
                      for m in modes_to_fetch}
        curr_parts = {m: process_gcs_file_range(gs_bucket_name, content_id, m, start_time, end_time)
                      for m in modes_to_fetch}
    else:
        past_parts = {m: None for m in modes_to_fetch}
        curr_parts = {m: process_gcs_file(gs_bucket_name, content_id, mode=m)
                      for m in modes_to_fetch}
    return past_parts, curr_parts


def _build_mode_contents(past_parts, curr_parts, mode, has_end_time):
    """지정된 mode의 API 호출용 contents 리스트를 빌드합니다."""
    if has_end_time:
        return [
            "--- Past Information ---", past_parts[mode],
            "--- Current Information ---", curr_parts[mode],
        ]
    else:
        return [curr_parts[mode]]



# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Responses using Gemini models")
    parser.add_argument("--json_file", default="assets/user_query.jsonl", help="질문 목록 JSONL 파일 경로 (generate_user_query.py 출력)")
    parser.add_argument("--output_file", default="assets/uq_responses.jsonl", help="통합 답변 목록을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--uq_response_gen_model", default="gemini-2.5-flash", help="사용할 생성 모델명")
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")
    parser.add_argument("--skip_aggregate", action="store_true", help="수행 완료 후 자동 집계 로직을 건너뜁니다.")
    parser.add_argument("--uq_response_gen_thinking_budget", type=int, default=-1,
                        help="UQ Response 생성 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args, client = init_pipeline(parser.parse_args())

    # 출력 폴더 생성
    for fpath in [args.output_file]:
        ensure_output_dir(fpath)

    print("\n" + "=" * 50)
    print("Gemini Inference 프로세스를 시작합니다 (Session-based, JSONL Pipeline).")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다. 다른 터미널의 출력을 기다리며 지속 처리합니다.")
    print("=" * 50)

    try:
        while True:
            # 1. 진행률 읽기
            processed_pairs = _load_completed_pairs(args.output_file)

            # 2. Input 읽기
            query_list = _load_query_items(args.json_file)
            if not query_list and not args.continuous:
                print(f"Error: {args.json_file} 파일이 존재하지 않거나 비어 있습니다.")
                return

            new_data_processed = False

            # Pending 작업 계산
            pending_work = {}
            for item in query_list:
                c_id = item["content_id"]
                c_pending = []
                for q_item in item.get("queries", []):
                    q_str = q_item["query"]
                    if (c_id, q_str) not in processed_pairs:
                        c_pending.append(q_item)
                if c_pending:
                    pending_work[c_id] = c_pending

            if pending_work:
                print("\n[TODO] 작업 목록:")
                for c_id, q_items in pending_work.items():
                    print(f"- content_id '{c_id}':")
                    for q_item in q_items:
                        end_val = q_item.get("end_time")
                        end_str = f" [end={end_val:.1f}s]" if end_val is not None else ""
                        print(f"    - query \"{q_item['query']}\"{end_str}")
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
                    ref_scenes = load_scenes(args.gs_bucket_name, content_id, mode="img_desc")
                except Exception as e:
                    print(f"[{content_id}] Warning: JSONL 로드 실패 ({e}). start_time이 0으로 설정될 수 있습니다.")
                    ref_scenes = []

                print(f"[{content_id}] Initializing Generation configs ({args.uq_response_gen_model})...")
                gen_configs = {mode: make_generation_config(mode=mode, thinking_budget=args.uq_response_gen_thinking_budget)
                               for mode in ["video", "raw", "img_desc", "mm_desc"]}

                for q_item in pending_queries:
                    user_prompt = q_item["query"]
                    end_time = float(q_item.get("end_time") or 0)
                    scene_idx = q_item.get("scene_idx", -1)
                    start_time = q_item.get("start_time")
                    if start_time is None and scene_idx != -1:
                        target_scene = next((s for s in ref_scenes if s.get("scene_idx") == scene_idx), None)
                        start_time = target_scene.get("start_time", 0.0) if target_scene else 0.0
                    else:
                        start_time = float(start_time or 0.0)

                    has_end_time = end_time > 0

                    if (content_id, user_prompt) in processed_pairs:
                        print(f"[{content_id}] Query: '{user_prompt[:40]}...' -> already completed (skip)")
                        continue

                    end_label = f"Range=[{start_time:.1f}s ~ {end_time:.1f}s]" if has_end_time else "full"
                    print(f"[{content_id}] Processing Query [{end_label}]: '{user_prompt}'")

                    # Parts 준비
                    past_parts, curr_parts = _build_parts(
                        args.gs_bucket_name, content_id, start_time, end_time, has_end_time)

                    # 1. 2개 Mode 답변 생성
                    answers_for_query = {}

                    def generate_for_mode(mode):
                        print(f"[{content_id}]  Generating [{mode}]...")
                        try:
                            time.sleep(1)
                            mode_contents = _build_mode_contents(
                                past_parts, curr_parts, mode, has_end_time)
                            answer_text = _retry_api_call(
                                lambda: client.models.generate_content(
                                    model=args.uq_response_gen_model,
                                    contents=[*mode_contents, user_prompt],
                                    config=gen_configs[mode]
                                ).text,
                                label=f"[{content_id}] [{mode}] 생성"
                            )
                            return mode, answer_text
                        except Exception as e:
                            print(f"[{content_id}]  Generating [{mode}] Error: {e}")
                            return mode, f"Error: {str(e)}"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as mode_executor:
                        futures_mode = [mode_executor.submit(generate_for_mode, m) for m in ["video", "raw", "img_desc", "mm_desc"]]
                        for future in concurrent.futures.as_completed(futures_mode):
                            m, text = future.result()
                            answers_for_query[m] = text

                    ordered_answers = {m: answers_for_query[m] for m in ["video", "raw", "img_desc", "mm_desc"] if m in answers_for_query}

                    time_ctx = {
                        "scene_idx": scene_idx if has_end_time else None,
                        "start_time": start_time if has_end_time else None,
                        "end_time": end_time if has_end_time else None,
                    }

                    response_record = {"content_id": content_id, **time_ctx, "query": user_prompt, "answers": ordered_answers}

                    append_jsonl(args.output_file, response_record, lock=file_write_lock)

                    processed_pairs.add((content_id, user_prompt))
                    print(f"[{content_id}]  -> Response 저장 완료")
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
        os._exit(1)

    if not args.continuous:
        if not args.skip_aggregate:
            print("\n[Aggregation] JSONL 결과를 분석용 JSON 형식으로 병합합니다...")
            output_dir = os.path.dirname(args.output_file) or "assets"
            subprocess.run([sys.executable, "jsonl_to_json.py", "--input_dir", output_dir])

    print("\n생성 프로세스가 완료/종료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
