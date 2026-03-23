import os
import argparse
import json
import vertexai
from run_gemini_cli import (
    init_gemini_client, process_gcs_file, start_chat_session, send_chat_message, 
    init_generation_model, check_gcs_files_exist
)

def main():
    parser = argparse.ArgumentParser(description="Generate Responses using Gemini models")
    parser.add_argument("--json_file", default="user_query_list.json", help="질문 목록 JSON 파일 경로")
    parser.add_argument("--output_file", default="output/responses.json", help="통합 답변 목록을 저장할 파일 경로")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름")
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="사용할 생성 모델명")
    parser.add_argument("--location", default="asia-northeast3", help="GCP Location")

    args = parser.parse_args()

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}")
    init_gemini_client(args.gcp_project_id, location=args.location)
    
    if not os.path.exists(args.json_file):
        print(f"Error: {args.json_file} 파일이 존재하지 않습니다.")
        return
        
    with open(args.json_file, "r", encoding="utf-8") as f:
        query_list = json.load(f)

    # Resume logic
    all_answers = []
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            try:
                all_answers = json.load(f)
                processed_ids = {item["content_id"] for item in all_answers}
                print(f"[{len(processed_ids)}] 개의 콘텐츠가 이미 {args.output_file}에 존재하여 건너뜁니다.")
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
        for mode in ["full", "part", "video"]:
            gen_model = init_generation_model(mode=mode, model_name=args.response_gen_model)
            gen_chats[mode] = start_chat_session(gen_model)

        is_first_turn = True
        answers_dict = {"content_id": content_id, "queries": []}

        for user_prompt in queries:
            print(f"Processing Query: '{user_prompt}'")
            answers_for_query = {}
            
            for mode in ["full", "part", "video"]:
                print(f"  [{mode}] mode 생성 중...")
                file_part = parts[mode] if is_first_turn else None
                
                try:
                    response = send_chat_message(gen_chats[mode], user_prompt, file_part=file_part)
                    answers_for_query[mode] = response.text
                except Exception as e:
                    print(f"  [{mode}] mode 오류 발생: {e}")
                    answers_for_query[mode] = f"Error: {str(e)}"

            answers_dict["queries"].append({
                "query": user_prompt,
                "answers": answers_for_query
            })

            is_first_turn = False
            print("-" * 50)
            
        all_answers.append(answers_dict)
        processed_ids.add(content_id)

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_answers, f, indent=4, ensure_ascii=False)
        print(f"답변이 업데이트 되었습니다: {args.output_file}")

    print("\n모든 생성 처리가 완료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
