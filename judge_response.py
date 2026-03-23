import os
import argparse
import json
import time
import vertexai
from run_gemini_cli import (
    init_gemini_client, process_gcs_file, start_chat_session, 
    evaluate_answer_session, init_judge_model, check_gcs_files_exist
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Responses using Judge model")
    parser.add_argument("--answers_file", default="output/responses.json", help="답변 목록 JSON 파일 경로")
    parser.add_argument("--output_file", default="output/scores.json", help="최종 평가 결과 저장 경로")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--judge_model", default="gemini-2.5-pro", help="사용할 평가 모델명")
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

    init_gemini_client(args.gcp_project_id, location=args.location)
    
    if not os.path.exists(args.answers_file):
        print(f"Error: {args.answers_file} 파일이 존재하지 않습니다. 먼저 generate_response.py를 실행하세요.")
        return
        
    with open(args.answers_file, "r", encoding="utf-8") as f:
        content_answers_list = json.load(f)

    # Resume logic
    all_scores = []
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            try:
                all_scores = json.load(f)
                processed_ids = {item["content_id"] for item in all_scores}
                print(f"[{len(processed_ids)}] 개의 콘텐츠가 이미 {args.output_file}에 존재하여 건너뜁니다.")
            except json.JSONDecodeError:
                print(f"Warning: {args.output_file} 파일을 읽는 중 오류가 발생했습니다. 새로 시작합니다.")

    # 출력 폴더 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "=" * 50)
    print("Gemini Evaluation 프로세스를 시작합니다 (Session-based).")
    print("=" * 50)

    for content_answers in content_answers_list:
        content_id = content_answers["content_id"]
        if content_id in processed_ids:
            continue
            
        print(f"\nEvaluating Content: '{content_id}'")
        
        if not check_gcs_files_exist(args.gs_bucket_name, content_id):
            continue
            
        video_part = process_gcs_file(args.gs_bucket_name, content_id, mode="video")
        gt_part = process_gcs_file(args.gs_bucket_name, content_id, mode="gt")

        print(f"Initializing Judge model ({args.judge_model})...")
        judge_model = init_judge_model(model_name=args.judge_model)
        
        judge_chats = {}
        for mode in ["full", "part", "video"]:
            judge_chats[mode] = start_chat_session(judge_model)
        
        is_first_turn = True
        content_scores = []

        for query_item in content_answers["queries"]:
            user_prompt = query_item["query"]
            answers = query_item["answers"]
            
            print(f"Scoring Query: '{user_prompt}'")
            
            for mode in ["full", "part", "video"]:
                generated_answer = answers.get(mode)
                if not generated_answer or generated_answer.startswith("Error"):
                    print(f"  [{mode}] 유효한 답변이 없습니다 (건너뜀).")
                    continue
                    
                time.sleep(1) # 평가 루프 과부하 방지
                
                try:
                    score_text = evaluate_answer_session(
                        judge_chat=judge_chats[mode], 
                        user_prompt=user_prompt, 
                        generated_answer=generated_answer,
                        is_first_turn=is_first_turn,
                        video_part=video_part,
                        gt_json_part=gt_part
                    )
                    
                    # JSON 파싱 정제부
                    clean_text = score_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]
                        
                    try:
                        score_dict = json.loads(clean_text)
                    except json.JSONDecodeError:
                        print(f"  [Warning] Failed to parse JSON from judge response.")
                        score_dict = {"raw_response": score_text}
                    
                    score_entry = {
                        "query": user_prompt,
                        "mode": mode,
                        "judge": score_dict 
                    }   
                    
                    content_scores.append(score_entry)
                    print(f"  [{mode}] Evaluation completed.")
                    
                except Exception as e:
                    print(f"  [{mode}] Evaluation error: {e}")
            
            is_first_turn = False
            print("-" * 50)
            
        all_scores.append({
            "content_id": content_id,
            "scores": content_scores
        })
        processed_ids.add(content_id)

        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, indent=4, ensure_ascii=False)
        print(f"평가 결과가 업데이트 되었습니다: {args.output_file}")
        
    print("\n모든 평가 처리가 완료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
