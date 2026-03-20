import os
import argparse
import json
import time
import vertexai
from run_gemini_cli import (
    init_gemini_client, process_gcs_file, start_chat_session, send_chat_message, 
    evaluate_answer_session, init_generation_model, init_judge_model
)

def main():
    parser = argparse.ArgumentParser(description="Gemini Multi-turn Chat CLI with Session-based Evaluation")
    parser.add_argument("--json_file", default="user_query_list.json", help="질문 목록 JSON 파일 경로")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (환경 변수 GCP_PROJECT_ID가 없을 경우 필수)")
    parser.add_argument("--gs_bucket", help="GCS 버킷 이름")
    
    args = parser.parse_args()

    project_id = args.gcp_project_id
    if not project_id:
        print("Error: GCP Project ID가 설정되지 않았습니다. --project_id 인자를 사용하거나 GCP_PROJECT_ID 환경 변수를 설정하세요.")
        return

    gs_bucket_name = args.gs_bucket
    if not gs_bucket_name:
        print("Error: GCS 버킷 이름이 설정되지 않았습니다. --gs_bucket 인자를 사용하거나 GS_BUCKET_NAME 환경 변수를 설정하세요.")
        return

    print(f"Initializing Gemini client for project: {project_id}...")
    init_gemini_client(project_id)
    
    if not os.path.exists(args.json_file):
        print(f"Error: {args.json_file} 파일이 존재하지 않습니다.")
        return
        
    with open(args.json_file, "r", encoding="utf-8") as f:
        query_list = json.load(f)

    for output_dir in ["response", "scores"]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print("\n" + "=" * 50)
    print("Gemini Inference 및 Evaluation 프로세스를 시작합니다 (Session-based).")
    print("=" * 50)

    try:
        for item in query_list:
            content_id = item["content_id"]
            queries = item["queries"]
            
            print(f"\nProcessing Content: '{content_id}'")
            print(f"Preparing GCS files from bucket: {gs_bucket_name}")
            
            parts = {
                "video": process_gcs_file(gs_bucket_name, content_id, mode="video"),
                "full": process_gcs_file(gs_bucket_name, content_id, mode="full"),
                "part": process_gcs_file(gs_bucket_name, content_id, mode="part"),
                "gt": process_gcs_file(gs_bucket_name, content_id, mode="gt")
            }

            print(f"Initializing Generation models (us-central1)...")
            vertexai.init(project=project_id, location="us-central1")
            
            gen_chats = {}
            for mode in ["full", "part", "video"]:
                gen_model = init_generation_model(mode=mode, model_name='gemini-2.5-flash')
                gen_chats[mode] = start_chat_session(gen_model)

            print("Initializing Judge model (global)...")
            # 3.1 Pro Preview가 원활하게 지원되는 글로벌 리전 사용
            vertexai.init(project=project_id, location="global")
            judge_model = init_judge_model(model_name="gemini-2.5-pro")
            
            # --- Judge 모델도 모드별로 각각의 Chat Session을 열어 속도 향상 달성 ---
            judge_chats = {}
            for mode in ["full", "part", "video"]:
                judge_chats[mode] = start_chat_session(judge_model)
            
            is_first_turn = True
            all_scores = []

            for user_prompt in queries:
                print(f"Processing Query: '{user_prompt}'\n")
                answers = {}
                
                # 1. Generation Step
                for mode in ["full", "part", "video"]:
                    print(f"Generating answer for [{mode}] mode...")
                    file_part = parts[mode] if is_first_turn else None
                    
                    response = send_chat_message(gen_chats[mode], user_prompt, file_part=file_part)
                    answers[mode] = response.text
                    
                    txt_filename = f"response/{content_id}_response_{mode}.txt"
                    with open(txt_filename, "a", encoding="utf-8") as f:
                        f.write(f"Q:\n{user_prompt}\n\nA:\n{answers[mode]}\n" + "="*50 + "\n")
                        
                    print(f"[{mode}] 답변이 저장되었습니다: {txt_filename}")
                
                # 2. Evaluation Step
                print("\nJudge 모델을 통해 각 답변을 평가합니다...")
                for mode in ["full", "part", "video"]:
                    print(f"Scoring [{mode}] answer...")
                    time.sleep(1) # 평가 루프 과부하 방지 딜레이
                    
                    try:
                        score_text = evaluate_answer_session(
                            judge_chat=judge_chats[mode], 
                            user_prompt=user_prompt, 
                            generated_answer=answers[mode],
                            is_first_turn=is_first_turn,
                            video_part=parts["video"],
                            gt_json_part=parts["gt"]
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
                            print(f"[Warning] Failed to parse JSON from judge response.")
                            score_dict = {"raw_response": score_text}
                        
                        score_entry = {
                            "query": user_prompt,
                            "mode": mode,
                            "judge": score_dict 
                        }   
                        
                        all_scores.append(score_entry)
                        print(f"--- [{mode}] Evaluation Result ---")
                        print(json.dumps(score_entry, indent=2, ensure_ascii=False))
                        
                    except Exception as e:
                        print(f"Evaluation error for {mode}: {e}")
                
                is_first_turn = False
                print("\n" + "-" * 50)
                
            merged_score_filename = f"scores/{content_id}_all_scores.json"
            with open(merged_score_filename, "w", encoding="utf-8") as f:
                json.dump(all_scores, f, indent=4, ensure_ascii=False)
            print(f"모든 쿼리와 모드에 대한 평가 결과가 하나로 저장되었습니다: {merged_score_filename}")
                
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 실행이 중단되었습니다. 종료합니다.")
        
    print("\n모든 모드의 처리 및 평가가 완료되었습니다.\n")
    print("=" * 50)

if __name__ == "__main__":
    main()