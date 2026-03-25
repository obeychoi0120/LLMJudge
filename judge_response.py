import os
import argparse
import json
import time
import vertexai
from gemini_api_utils import (
    process_gcs_file, start_chat_session, 
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

    vertexai.init(project=args.gcp_project_id, location=args.location)
    
    if not os.path.exists(args.answers_file):
        print(f"Error: {args.answers_file} 파일이 존재하지 않습니다. 먼저 generate_response.py를 실행하세요.")
        return
        
    with open(args.answers_file, "r", encoding="utf-8") as f:
        content_answers_list = json.load(f)

    # Resume logic
    all_scores = []
    processed_ids = set()
    existing_scores_dict = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            try:
                loaded_scores = json.load(f)
                
                # count how many queries we need per content_id
                query_len_dict = {item["content_id"]: len(item["queries"]) for item in content_answers_list}
                
                for sc in loaded_scores:
                    c_id = sc["content_id"]
                    all_scores.append(sc)
                    existing_scores_dict[c_id] = sc
                    
                    is_complete = True
                    target_query_count = query_len_dict.get(c_id, 0)
                    scores_list = sc.get("scores", [])
                    
                    # We expect 3 valid scores (one for each mode) per query
                    valid_counts = {}
                    for entry in scores_list:
                        q_text = entry.get("query")
                        mode = entry.get("mode")
                        judge_data = entry.get("judge", {})
                        if "raw_response" not in judge_data and "scores" in judge_data:
                            valid_counts.setdefault(q_text, set()).add(mode)
                            
                    if len(valid_counts) < target_query_count:
                        is_complete = False
                    else:
                        for q_text, modes in valid_counts.items():
                            if len(modes) < 3: # video, full, part
                                is_complete = False
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
        for mode in ["video", "full", "part"]:
            judge_chats[mode] = start_chat_session(judge_model)
        
        is_first_turn_for_mode = {"video": True, "full": True, "part": True}
        
        # Load existing scores
        existing_sc_data = existing_scores_dict.get(content_id, {})
        old_content_scores = existing_sc_data.get("scores", [])
        
        # Keep only valid scores and build lookup map
        content_scores = []
        existing_score_map = {}
        for entry in old_content_scores:
            q_text = entry.get("query")
            m = entry.get("mode")
            judge_data = entry.get("judge", {})
            if "raw_response" not in judge_data and "scores" in judge_data:
                content_scores.append(entry)
                existing_score_map.setdefault(q_text, {})[m] = entry

        scores_dict = {"content_id": content_id, "scores": content_scores}

        for query_item in content_answers["queries"]:
            user_prompt = query_item["query"]
            answers = query_item["answers"]
            
            print(f"Scoring Query: '{user_prompt}'")
            
            for mode in ["video", "full", "part"]:
                generated_answer = answers.get(mode)
                if not generated_answer or str(generated_answer).startswith("Error"):
                    print(f"  [{mode}] 유효한 답변이 없습니다 (건너뜀).")
                    continue
                
                if mode in existing_score_map.get(user_prompt, {}):
                    print(f"  [{mode}] mode 평가 이미 완료됨 (건너뜀).")
                    continue
                
                time.sleep(1) # 평가 루프 과부하 방지
                
                try:
                    score_text = evaluate_answer_session(
                        judge_chat=judge_chats[mode], 
                        user_prompt=user_prompt, 
                        generated_answer=generated_answer,
                        is_first_turn=is_first_turn_for_mode[mode],
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
                    is_first_turn_for_mode[mode] = False
                    
                except Exception as e:
                    print(f"  [{mode}] Evaluation error: {e}")
            
            print("-" * 50)
            
            # Save partial progress per query
            existing_idx = next((i for i, sc in enumerate(all_scores) if sc["content_id"] == content_id), None)
            if existing_idx is not None:
                all_scores[existing_idx] = scores_dict
            else:
                all_scores.append(scores_dict)

            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(all_scores, f, indent=4, ensure_ascii=False)
            print(f"  -> 평가 쿼리 단위 임시 저장 완료: {args.output_file}")
            
        processed_ids.add(content_id)
        
    print("\n모든 평가 처리가 완료되었습니다.\n" + "=" * 50)

if __name__ == "__main__":
    main()
