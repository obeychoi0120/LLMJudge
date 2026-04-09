import os
import time
import argparse
import threading
import sys
from gemini_api_utils import (
    make_generate_config,
    process_gcs_file, process_gcs_file_range,
    check_gcs_files_exist,
    _retry_api_call, load_scenes,
    ensure_output_dir, preload_content_metadata,
    init_pipeline, load_jsonl, append_jsonl,
)

_REFERENCE_PROMPT = """당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
제공되는 원본 영상과 Reference용 메타데이터를 모두 참조하여, 사용자 질문에 대해 가장 정확하고 포괄적인 한국어 답변을 생성해 주세요.

[Reference 메타데이터의 필드 설명]
- scene_idx: 영상 Scene 인덱스
- start_time: 영상 Scene 시작 시간 (초)
- end_time: 영상 Scene 종료 시간 (초)
- duration: 영상 Scene의 길이 (초)
- sounds: 환경음 및 효과음
- texts: 화면 속 자막, 간판 정보 등
- speech: 등장인물들의 대사 (영어 또는 한국어)

[메타데이터 사용 시 주의사항]
speech, texts, sounds 필드는 자동 추출된 값으로, 부정확할 수 있습니다.
- sounds: 효과음 분류 오류가 빈번합니다. 반드시 비디오 프레임의 시각 정보를 우선 참고하세요.
- texts: OCR 오류로 인해 화면 텍스트가 잘못 인식될 수 있습니다.
- speech: 음성 인식 오류로 인해 대사가 누락되거나 철자가 틀릴 수 있습니다.

이 답변은 다른 AI 모델의 답변을 평가하기 위한 '기준 답변(Reference Answer)'으로 사용됩니다.
따라서 핵심 사실, 대사, 행동, 맥락을 빠짐없이 포함하되 자연스럽고 읽기 쉽게 작성해 주세요.
외부 자료 검색은 금지합니다. 오직 제공된 영상과 메타데이터만 활용하세요."""

def make_reference_config(thinking_budget=None):
    return make_generate_config(system_instruction=_REFERENCE_PROMPT, thinking_budget=thinking_budget)

def _load_query_items(json_file):
    query_dict = {}
    for data in load_jsonl(json_file):
        if "content_id" not in data:
            continue
        c_id = data["content_id"]
        if c_id not in query_dict:
            query_dict[c_id] = {"content_id": c_id, "queries": []}
        if "queries" in data:
            scene_ctx = {
                "scene_idx": data.get("scene_idx", -1),
                "start_time": data.get("start_time"),
                "end_time": data.get("end_time"),
            }
            for q in data["queries"]:
                if isinstance(q, dict):
                    enriched = {**scene_ctx, **q}
                    query_dict[c_id]["queries"].append(enriched)
                else:
                    query_dict[c_id]["queries"].append({
                        "query": q, **scene_ctx
                    })
    return list(query_dict.values())

def _load_completed_pairs(references_path):
    valid_references = set()
    for ref in load_jsonl(references_path):
        c_id = ref.get("content_id")
        query = ref.get("query")
        ref_text = ref.get("reference", "")
        if c_id and query and not str(ref_text).startswith("Error"):
            valid_references.add((c_id, query))
    return valid_references

def _build_parts(gs_bucket_name, content_id, start_time, end_time, has_end_time):
    # Only need video and ref for generation
    modes_to_fetch = ["video", "ref"]
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

def _build_ref_contents(past_parts, curr_parts, has_end_time, use_ref):
    if has_end_time:
        contents = ["--- Past Information ---", past_parts["video"]]
        if use_ref:
            contents.append(past_parts["ref"])
        contents += ["--- Current Information ---", curr_parts["video"]]
        if use_ref:
            contents.append(curr_parts["ref"])
    else:
        contents = [curr_parts["video"]]
        if use_ref:
            contents.append(curr_parts["ref"])
    return contents

def main():
    parser = argparse.ArgumentParser(description="Generate UQ References using Gemini models")
    parser.add_argument("--json_file", default="assets/user_query.jsonl", help="질문 목록 JSONL 파일 경로")
    parser.add_argument("--output_file", default="assets/uq_references.jsonl", help="Reference 답변을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름")
    parser.add_argument("--uq_reference_gen_model", default="gemini-2.5-pro", help="Reference Answer 생성 모델명")
    parser.add_argument("--use_ref_for_uq_reference", type=lambda x: str(x).lower() == 'true', default=False, help="Reference 생성 시 Ref JSONL 참조 여부")
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 처리")
    parser.add_argument("--uq_reference_gen_thinking_budget", type=int, default=4096,
                        help="UQ Reference Answer 생성 모델의 Thinking Budget")

    args, client = init_pipeline(parser.parse_args())

    ensure_output_dir(args.output_file)

    print("\n" + "=" * 50)
    print("UQ Reference 생성을 시작합니다.")
    if args.continuous:
        print("Continuous 모드가 활성화되었습니다.")
    print("=" * 50)

    try:
        while True:
            processed_pairs = _load_completed_pairs(args.output_file)
            query_list = _load_query_items(args.json_file)
            if not query_list and not args.continuous:
                print(f"Error: {args.json_file} 파일이 존재하지 않거나 비어 있습니다.")
                return

            new_data_processed = False
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
                print("\n[TODO] UQ Reference 작업 목록:")
                for c_id, q_items in pending_work.items():
                    print(f"- content_id '{c_id}': {len(q_items)} queries")
                print("-" * 50)

            file_write_lock = threading.Lock()

            def process_item(item):
                content_id = item["content_id"]
                if content_id not in pending_work:
                    return False

                pending_queries = pending_work[content_id]
                print(f"\nProcessing Context for Reference: '{content_id}'")

                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    return False

                preload_content_metadata(args.gs_bucket_name, content_id)
                try:
                    ref_scenes = load_scenes(args.gs_bucket_name, content_id, mode="img_desc")
                except Exception as e:
                    print(f"[{content_id}] Warning: JSONL 로드 실패 ({e}). start_time이 0으로 설정될 수 있습니다.")
                    ref_scenes = []

                print(f"[{content_id}] Initializing Reference config ({args.uq_reference_gen_model}, Ref={'ON' if args.use_ref_for_uq_reference else 'OFF'})...")
                ref_config = make_reference_config(thinking_budget=args.uq_reference_gen_thinking_budget)

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
                        continue

                    end_label = f"Range=[{start_time:.1f}s ~ {end_time:.1f}s]" if has_end_time else "full"
                    print(f"[{content_id}] Reference Query [{end_label}]: '{user_prompt}'")

                    past_parts, curr_parts = _build_parts(
                        args.gs_bucket_name, content_id, start_time, end_time, has_end_time)

                    print(f"[{content_id}]  Generating [reference]...")
                    try:
                        ref_contents = _build_ref_contents(
                            past_parts, curr_parts, has_end_time, args.use_ref_for_uq_reference)

                        reference_answer = _retry_api_call(
                            lambda: client.models.generate_content(
                                model=args.uq_reference_gen_model,
                                contents=[*ref_contents, user_prompt],
                                config=ref_config
                            ).text,
                            label=f"[{content_id}] Reference 생성"
                        )
                        print(f"[{content_id}]  Reference answer generated ({len(reference_answer.split())} words)")
                    except Exception as e:
                        print(f"[{content_id}]  Generating [reference] Error: {e}")
                        reference_answer = f"Error: {str(e)}"

                    time_ctx = {
                        "scene_idx": scene_idx if has_end_time else None,
                        "start_time": start_time if has_end_time else None,
                        "end_time": end_time if has_end_time else None,
                    }

                    ref_record = {"content_id": content_id, **time_ctx, "query": user_prompt, "reference": reference_answer}

                    append_jsonl(args.output_file, ref_record, lock=file_write_lock)
                    processed_pairs.add((content_id, user_prompt))
                    print(f"[{content_id}]  -> Reference 저장 완료")
                    print("-" * 50)

                return True

            for item in query_list:
                if process_item(item):
                    new_data_processed = True

            if not args.continuous:
                break

            if not new_data_processed:
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 모니터링 루프가 중단되었습니다.")
        os._exit(1)

    print("\nReference 생성 완료\n" + "=" * 50)

if __name__ == "__main__":
    main()
