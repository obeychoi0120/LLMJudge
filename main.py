import os
import argparse
import json
from run_gemini_cli import init_gemini_client, process_gcs_file, start_chat_session, send_chat_message, evaluate_answer, init_generation_model, init_judge_model

GS_BUCKET_NAME = "insight-youtubevideodataset"

def main():
    os.environ["GCP_PROJECT_ID"] = "insight-dev-490002"
    # 0. Argument Parser 설정
    parser = argparse.ArgumentParser(description="Gemini Multi-turn Chat CLI with GCS File Support")
    parser.add_argument("--json_file", default="user_query_list.json", help="질문 목록 JSON 파일 경로")
    parser.add_argument("--project_id", help="GCP 프로젝트 ID (환경 변수 GCP_PROJECT_ID가 없을 경우 필수)")
    
    args = parser.parse_args()

    # GCP 프로젝트 ID 설정 (명령행 인자 우선, 없으면 환경 변수)
    project_id = args.project_id or os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        print("Error: GCP Project ID가 설정되지 않았습니다. --project_id 인자를 사용하거나 GCP_PROJECT_ID 환경 변수를 설정하세요.")
        return

    print(f"Initializing Gemini client for project: {project_id}...")
    client = init_gemini_client(project_id)
    
    print("\nInitializing and testing generative models...")

    gen_models = {}
    for mode in ["video", "full", "nodesc"]:
        gen_models[mode] = init_generation_model(mode=mode, model_name='gemini-2.5-flash')
    
    # JSON 파일 읽기
    if not os.path.exists(args.json_file):
        print(f"Error: {args.json_file} 파일이 존재하지 않습니다.")
        return
        
    with open(args.json_file, "r", encoding="utf-8") as f:
        query_list = json.load(f)

    if not os.path.exists("response"):
        os.makedirs("response")
    if not os.path.exists("scores"):
        os.makedirs("scores")

    print("\n"+"="*50)
    print("Gemini Inference 및 Evaluation 프로세스를 시작합니다.")
    print("="*50)

    try:
        for item in query_list:
            content_id = item["content_id"]
            queries = item["queries"]
            
            print(f"\nProcessing Content: '{content_id}'")
            
            # GCS 파일 준비 (세 가지 모드 모두)
            print(f"Preparing GCS files from bucket: {GS_BUCKET_NAME} for {content_id}...")
            parts = {}
            parts["video"] = process_gcs_file(GS_BUCKET_NAME, content_id, mode="video")
            parts["full"] = process_gcs_file(GS_BUCKET_NAME, content_id, mode="full")
            parts["nodesc"] = process_gcs_file(GS_BUCKET_NAME, content_id, mode="nodesc")
            parts["gt"] = process_gcs_file(GS_BUCKET_NAME, content_id, mode="gt")

            # 각 모드별 멀티 턴 채팅 세션 시작
            chats = {}
            for mode in ["video", "full", "nodesc"]:
                chats[mode] = start_chat_session(gen_models[mode])
            
            is_first_turn = True
            
            # 해당 content_id의 모든 평가 결과를 담을 리스트
            all_scores = []

            for user_prompt in queries:
                print(f"\nProcessing Query: '{user_prompt}'\n")
                
                # 모드별 답변 생성 및 텍스트 파일 저장             
                answers = {}
                for mode in ["video", "full", "nodesc"]:
                    print(f"Generating answer for [{mode}] mode...")
                    if is_first_turn:
                        response = send_chat_message(chats[mode], user_prompt, parts[mode])
                    else:
                        response = send_chat_message(chats[mode], user_prompt)

                    answers[mode] = response.text
                    
                    txt_filename = f"response/{content_id}_response_{mode}.txt"
                    with open(txt_filename, "a", encoding="utf-8") as f:
                        f.write(f"Q: {user_prompt}\n\nA: {response.text}\n" + "="*50 + "\n")
                        
                    print(f"[{mode}] 답변이 저장되었습니다: {txt_filename}")
                
                is_first_turn = False
                
                # Pro 모델을 통한 자동 평가
                print("-" * 50)
                print("Judge 모델을 통해 각 답변을 평가합니다...")
                for mode in ["video", "full", "nodesc"]:
                    print(f"Scoring [{mode}] answer...")
                    try:
                        # 독립된 평가를 위해 매 쿼리마다 새로운 Judge 모델 인스턴스 생성
                        current_judge_model = init_judge_model(model_name="gemini-3.1-pro-preview")
                        score_text = evaluate_answer(current_judge_model, parts["video"], user_prompt, answers[mode], parts["gt"])
                        
                        # 모델 응답에서 JSON 파싱 (마크다운 코드 블록 등이 포함될 수 있으므로 정제 시도 가능)
                        try:
                            # 만약 평론가가 ```json ... ``` 형태로 답변을 시작하면 제거
                            clean_text = score_text.strip()
                            if clean_text.startswith("```json"):
                                clean_text = clean_text[7:]
                            if clean_text.endswith("```"):
                                clean_text = clean_text[:-3]
                            
                            score_dict = json.loads(clean_text)
                        except json.JSONDecodeError:
                            print(f"[Warning] Failed to parse JSON from judge response. Using raw text.")
                            score_dict = {"raw_response": score_text}
                        
                        # query 및 mode 정보 추가
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
                        
                print("\n" + "-" * 50)
                
            # 하나의 content_id 처리가 끝나면, 누적된 평가 결과를 하나의 JSON 파일로 저장
            merged_score_filename = f"scores/{content_id}_all_scores.json"
            with open(merged_score_filename, "w", encoding="utf-8") as f:
                json.dump(all_scores, f, indent=4, ensure_ascii=False)
            print(f"\n모든 쿼리와 모드에 대한 평가 결과가 하나로 저장되었습니다: {merged_score_filename}")
                
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 채팅이 중단되었습니다. 종료합니다.")
        
    print("\n모든 모드의 처리 및 평가가 완료되었습니다.\n")
    print("=" * 50)

if __name__ == "__main__":
    main()