import os
import argparse
import json
import vertexai
from gemini_api_utils import (
    process_gcs_file, start_chat_session, send_chat_message, 
    init_generation_model, check_gcs_files_exist
)

def main():
    parser = argparse.ArgumentParser(description="Generate Responses using Gemini models")
    parser.add_argument("--json_file", default="user_query_list.json", help="질문 목록 JSON 파일 경로")
    parser.add_argument("--output_file", default="output/responses.json", help="통합 답변 목록을 저장할 파일 경로")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="사용할 생성 모델명")
    parser.add_argument("--location", default="asia-northeast3", help="GCP Location")

    args = parser.parse_args()

    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                args.gcp_project_id = args.gcp_project_id or config.get("gcp_project_id")
                args.gs_bucket_name = args.gs_bucket_name or config.get("gs_bucket_name")
            except json.JSONDecodeError:
                pass

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}")
    vertexai.init(project=args.gcp_project_id, location=args.location)
    
    if not os.path.exists(args.json_file):
        print(f"Error: {args.json_file} 파일이 존재하지 않습니다.")
        return
        
    with open(args.json_file, "r", encoding="utf-8") as f:
        query_list = json.load(f)

    # Resume logic
    all_answers = []
    processed_ids = set()
    existing_answers_dict = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            try:
                loaded_answers = json.load(f)
                query_len_dict = {item["content_id"]: len(item["queries"]) for item in query_list}
                
                for ans in loaded_answers:
                    c_id = ans["content_id"]
                    all_answers.append(ans)
                    existing_answers_dict[c_id] = ans
                    
                    is_complete = True
                    if c_id not in query_len_dict or len(ans.get("queries", [])) < query_len_dict[c_id]:
                        is_complete = False
                    else:
                        for q in ans.get("queries", []):
                            answers = q.get("answers", {})
                            for mode in ["video", "full", "part"]:
                                mode_ans = answers.get(mode, "")
                                if not mode_ans or str(mode_ans).startswith("Error"):
                                    is_complete = False
                                    break
                            if not is_complete:
                                break
                    if is_complete:
                        processed_ids.add(c_id)
                        
                print(f"[{len(processed_ids)}] 개의 콘텐츠가 완전히 처리되어 건너뜁니다.")
            except json.JSONDecodeError:
                print(f"Warning: {args.output_file} 파일을 읽는 중 오류가 발생했습니다. 새로 시작합니다.")

    # 출력 폴더 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "=" * 50)
    print("Gemini Inference 프로세스를 시작합니다 (Session-based).")
    print("=" * 50)

    for item in query_list:
        content_id = item["content_id"]
        if content_id in processed_ids:
            continue
            
        queries = item["queries"]
        
        print(f"\nProcessing Content: '{content_id}'")
        
        if not check_gcs_files_exist(args.gs_bucket_name, content_id):
            continue
            
        parts = {
            "video": process_gcs_file(args.gs_bucket_name, content_id, mode="video"),
            "full": process_gcs_file(args.gs_bucket_name, content_id, mode="full"),
            "part": process_gcs_file(args.gs_bucket_name, content_id, mode="part"),
        }

        print(f"Initializing Generation models ({args.response_gen_model})...")
        gen_chats = {}
        for mode in ["video", "full", "part"]:
            gen_model = init_generation_model(mode=mode, model_name=args.response_gen_model)
            gen_chats[mode] = start_chat_session(gen_model)

        is_first_turn_for_mode = {"video": True, "full": True, "part": True}
        answers_dict = {"content_id": content_id, "queries": []}
        
        existing_ans_data = existing_answers_dict.get(content_id, {})
        existing_queries = existing_ans_data.get("queries", [])
        existing_query_map = {q["query"]: q.get("answers", {}) for q in existing_queries}

        for user_prompt in queries:
            print(f"Processing Query: '{user_prompt}'")
            answers_for_query = {}
            existing_answers = existing_query_map.get(user_prompt, {})
            
            for mode in ["video", "full", "part"]:
                prev_ans = existing_answers.get(mode, "")
                if prev_ans and not str(prev_ans).startswith("Error"):
                    print(f"  [{mode}] mode 이미 완료됨 (건너뜀)")
                    answers_for_query[mode] = prev_ans
                    continue
                
                print(f"  [{mode}] mode 생성 중...")
                file_part = parts[mode] if is_first_turn_for_mode[mode] else None
                
                try:
                    response = send_chat_message(gen_chats[mode], user_prompt, file_part=file_part)
                    answers_for_query[mode] = response.text
                    is_first_turn_for_mode[mode] = False
                except Exception as e:
                    print(f"  [{mode}] mode 오류 발생: {e}")
                    answers_for_query[mode] = f"Error: {str(e)}"

            answers_dict["queries"].append({
                "query": user_prompt,
                "answers": answers_for_query
            })
            
            # Save partial progress per query
            existing_idx = next((i for i, ans in enumerate(all_answers) if ans["content_id"] == content_id), None)
            if existing_idx is not None:
                all_answers[existing_idx] = answers_dict
            else:
                all_answers.append(answers_dict)

            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(all_answers, f, indent=4, ensure_ascii=False)
            print(f"  -> 쿼리 단위 임시 저장 완료: {args.output_file}")
            print("-" * 50)
            
        processed_ids.add(content_id)

    print("\n모든 생성 처리가 완료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
