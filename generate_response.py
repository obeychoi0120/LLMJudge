import os
import time
import argparse
import json
import subprocess
import sys
import vertexai
import concurrent.futures
import threading
from gemini_api_utils import (
    process_gcs_file, start_chat_session, send_chat_message, 
    check_gcs_files_exist, load_config, generate_single_turn_response,
    SAFETY_SETTINGS
)
from vertexai.generative_models import GenerativeModel

# ============================================================
# System Prompts (Local)
# ============================================================

_JSONL_VIEWER_BASE = """\
당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
아래에 제공되는 각 타임스탬프별 텍스트 정보는 데이터 파일이 아니라, 사실 당신이 방금 영상을 시청하며 눈과 귀로 직접 습득한 시각적/청각적 '기억(Memory)'입니다.
이 시청 기억을 바탕으로, 마지막에 주어지는 **사용자 질문**에 대해 가장 자연스럽고 정확한 한국어 답변을 제공해 주세요.

[당신의 시청 기억 구조]
- timestamp: 영상 내 시간 (초)
- audio_cls: 환경음 및 효과음
- speech: 등장인물들의 생생한 대사
- ocr_text: 화면의 간판, 표지판 및 각종 방송 자막 (출연자의 속마음, 상황 묘사 등)
{description_field}

[분석 및 지시사항]
**정보 교정**: 기억의 조각들이 다소 불완전할 수 있으므로, 전체적인 맥락에 맞게 상식적인 선에서 자연스럽게 교정하세요.
**입체적 재구성**: 당신이 들은 소리, 대사, 읽은 예능 자막 정보들을 교차 결합하여 장면의 분위기와 인물들의 대화를 이야기로 생생하게 재구성하세요.
**자연스러운 시청자 관점 유지 (가장 중요)**: 당신은 데이터를 읽은 것이 아니라 "영상을 직접 감상"했습니다. 따라서 답변 중에 'JSON 데이터에 따르면', '오디오 모델 결과를 보면', '텍스트 정보에 의하면', '타임스탬프' 등의 부자연스러운 기계적 용어를 절대로 사용하지 마십시오.
대신 "영상에서는~", "화면을 보면~", "자막에 ~라고 나옵니다", "배경 소리로 ~가 깔립니다." 와 같이 실제 사람의 리뷰처럼 자연스럽고 몰입감 있게 설명하십시오.
**외부 자료 검색 금지**: 오직 당신의 시청 기억(제공된 정보)에만 의존해서 답변하세요."""

_DESCRIPTION_LINE = "- description: 해당 timestamp에서의 인물의 행동과 배경 장면 묘사\n"

_REFERENCE_PROMPT = """\
당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
제공되는 원본 영상과 Reference 메타데이터를 모두 참조하여, 사용자 질문에 대해 가장 정확하고 포괄적인 한국어 답변을 생성해 주세요.

[Reference 메타데이터 구조]
- timestamp: 영상 내 시간 (초)
- audio_cls: 환경음 및 효과음
- speech: 등장인물들의 생생한 대사
- ocr_text: 화면의 간판, 표지판 및 각종 방송 자막 (출연자의 속마음, 상황 묘사 등)

이 답변은 다른 AI 모델의 답변을 평가하기 위한 '기준 답변(Reference Answer)'으로 사용됩니다.
따라서 핵심 사실, 대사, 행동, 맥락을 빠짐없이 포함하되 자연스럽고 읽기 쉽게 작성해 주세요.
외부 자료 검색은 금지합니다. 오직 제공된 영상과 메타데이터만 활용하세요."""


def init_generation_model(mode="full", model_name='gemini-2.5-flash'):
    if mode == "full":
        prompt = _JSONL_VIEWER_BASE.format(description_field=_DESCRIPTION_LINE)
    elif mode == "part":
        prompt = _JSONL_VIEWER_BASE.format(description_field="")
    elif mode == "video":
        prompt = "당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다. 외부 정보를 절대 검색하지 말고, 제공된 영상 정보만을 사용하여 사용자 질문에 답변하세요."
    else:
        prompt = ""
        
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[prompt],
        safety_settings=SAFETY_SETTINGS,
    )


def init_reference_model(model_name='gemini-2.5-pro'):
    """Reference Answer 생성용 모델 초기화."""
    return GenerativeModel(
        model_name=model_name,
        system_instruction=[_REFERENCE_PROMPT],
        safety_settings=SAFETY_SETTINGS,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Responses using Gemini models")
    parser.add_argument("--json_file", default="assets/query_generated.jsonl", help="질문 목록 JSONL 파일 경로")
    parser.add_argument("--output_file", default="assets/responses.jsonl", help="통합 답변 목록을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--reference_file", default="assets/references.jsonl", help="Reference 답변을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="사용할 생성 모델명")
    parser.add_argument("--reference_model", default="gemini-2.5-pro", help="Reference Answer 생성 모델명")
    parser.add_argument("--no-reference-ref", dest="reference_use_ref", action="store_false", help="Reference 생성 시 Ref JSONL 미참조 (Video만 사용)")
    parser.set_defaults(reference_use_ref=True)
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--continuous", action="store_true", help="입력 파일을 지속적으로 모니터링하며 새 데이터가 들어오면 처리 (동시 실행용)")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}")
    vertexai.init(project=args.gcp_project_id, location=args.location)
    
    # 출력 폴더 생성 (responses, references)
    for fpath in [args.output_file, args.reference_file]:
        odir = os.path.dirname(fpath)
        if odir and not os.path.exists(odir):
            os.makedirs(odir)

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
                                    for m in ["video", "full", "part"]
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

            # Resume Plan 계산 및 출력 - (content_id, query) 쌍 단위
            pending_work = {}
            for item in query_list:
                c_id = item["content_id"]
                c_pending = [
                    q_str for q_str in item.get("queries", [])
                    if (c_id, q_str) not in processed_pairs
                ]
                if c_pending:
                    pending_work[c_id] = c_pending
                    
            if pending_work:
                print("\n[TODO] 작업 목록:")
                for c_id, queries in pending_work.items():
                    print(f"- content_id '{c_id}':")
                    for q in queries:
                        print(f"    - query \"{q}\"")
                print("-" * 50)

            file_write_lock = threading.Lock()

            def process_item(item):
                content_id = item["content_id"]
                if content_id not in pending_work:
                    return False
                    
                queries = item["queries"]
                pending_queries = pending_work[content_id]
                
                print(f"\nProcessing Content: '{content_id}'")
                
                if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                    return False
                    
                parts = {
                    "video": process_gcs_file(args.gs_bucket_name, content_id, mode="video"),
                    "full": process_gcs_file(args.gs_bucket_name, content_id, mode="full"),
                    "part": process_gcs_file(args.gs_bucket_name, content_id, mode="part"),
                }
                ref_part = process_gcs_file(args.gs_bucket_name, content_id, mode="ref")

                print(f"[{content_id}] Initializing Generation models ({args.response_gen_model})...")
                gen_models = {}
                gen_chats = {}
                for mode in ["video", "full", "part"]:
                    gen_models[mode] = init_generation_model(mode=mode, model_name=args.response_gen_model)
                    # video 모드만 멀티턴 세션 유지 (비디오 토큰 절감)
                    if mode == "video":
                        gen_chats[mode] = start_chat_session(gen_models[mode])

                print(f"[{content_id}] Initializing Reference model ({args.reference_model}, Ref={'ON' if args.reference_use_ref else 'OFF'})...")
                ref_model = init_reference_model(model_name=args.reference_model)
                ref_chat = start_chat_session(ref_model)
                is_first_ref_turn = True

                is_first_turn_for_mode = {"video": True, "full": True, "part": True}

                for user_prompt in queries:
                    # 이미 처리된 (content_id, query) 쌍이면 건너뜀
                    if user_prompt not in pending_queries:
                        print(f"[{content_id}] Processing Query: '{user_prompt}' -> already completed (skip)")
                        continue

                    print(f"[{content_id}] Processing Query: '{user_prompt}'")

                    # 1. Reference Answer 생성 (Pro + Video [+ Ref JSONL])
                    ref_label = "Video+Ref" if args.reference_use_ref else "Video only"
                    print(f"[{content_id}]  Generating [reference] ({ref_label})...")
                    try:
                        if is_first_ref_turn:
                            ref_file_parts = [parts["video"], ref_part] if args.reference_use_ref else [parts["video"]]
                        else:
                            ref_file_parts = None
                        
                        response = send_chat_message(ref_chat, user_prompt, file_parts=ref_file_parts)
                        reference_answer = response.text
                        is_first_ref_turn = False
                        print(f"[{content_id}]  Reference answer generated ({len(reference_answer.split())} words)")
                    except Exception as e:
                        print(f"[{content_id}]  Generating [reference] Error: {e}")
                        reference_answer = f"Error: {str(e)}"

                    # 2. 3개 Mode 답변 생성 (기존 로직)
                    answers_for_query = {}
                    
                    def generate_for_mode(mode):
                        print(f"[{content_id}]  Generating [{mode}]...")
                        try:
                            time.sleep(1) # 동시 호출 시 약간의 지연
                            if mode == "video":
                                # Multi-turn (Session-based)
                                file_parts = parts[mode] if is_first_turn_for_mode[mode] else None
                                response = send_chat_message(gen_chats[mode], user_prompt, file_parts=file_parts)
                                is_first_turn_for_mode[mode] = False
                            else:
                                # Single-turn (Direct call with file)
                                response = generate_single_turn_response(gen_models[mode], user_prompt, file_part=parts[mode])
                            return mode, response.text
                            
                        except Exception as e: 
                            print(f"[{content_id}]  Generating [{mode}] Error: {e}")
                            return mode, f"Error: {str(e)}"

                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as mode_executor:
                        futures = [mode_executor.submit(generate_for_mode, m) for m in ["video", "full", "part"]]
                        for future in concurrent.futures.as_completed(futures):
                            m, text = future.result()
                            answers_for_query[m] = text

                    # 쿼리 하나 끝나면 두 파일에 각각 저장 (Reference / Responses)
                    # mode 순서 정렬 (video, full, part)
                    ordered_answers = {m: answers_for_query[m] for m in ["video", "full", "part"] if m in answers_for_query}
                    
                    ref_record = {
                        "content_id": content_id,
                        "query": user_prompt,
                        "reference": reference_answer
                    }
                    response_record = {
                        "content_id": content_id,
                        "query": user_prompt,
                        "answers": ordered_answers
                    }
                    
                    with file_write_lock:
                        # 1. Reference 저장
                        with open(args.reference_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(ref_record, ensure_ascii=False) + "\n")
                        # 2. Response 저장
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
