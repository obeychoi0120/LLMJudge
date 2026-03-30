import pandas as pd
import json
import os

def export_details(input_dir, output_dir):
    # 1. 파일 읽기
    references_path = os.path.join(input_dir, "references.json")
    responses_path = os.path.join(input_dir, "responses.json")
    scores_path = os.path.join(input_dir, "scores.json")
    
    with open(references_path, "r", encoding="utf-8") as f:
        references = json.load(f)
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    with open(scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)
        
    # 2. 데이터 매핑용 딕셔너리 생성
    # 구조: data_map[(content_id, query)] = { "ref_video": ..., "resp_video": ..., "judge_video": ..., "score_video": ... }
    data_map = {}
    
    for item in references:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            if key not in data_map:
                data_map[key] = {}
            # references.json은 'reference' 키를 직접 가짐 (문자열)
            ref_text = q.get("reference", "")
            data_map[key]["ref_video"] = ref_text
            data_map[key]["ref_full"] = ref_text
            data_map[key]["ref_part"] = ref_text
            
    for item in responses:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            if key not in data_map:
                continue
            answers = q.get("answers", {})
            data_map[key]["resp_video"] = answers.get("video", "")
            data_map[key]["resp_full"] = answers.get("full", "")
            data_map[key]["resp_part"] = answers.get("part", "")
            
    for item in scores:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            if key not in data_map:
                continue
            judge_data = q.get("judge", {})
            for mode in ["video", "full", "part"]:
                mode_judge = judge_data.get(mode, {})
                data_map[key][f"judge_{mode}"] = mode_judge.get("rationale", "")
                data_map[key][f"score_{mode}"] = mode_judge.get("total_score", "")
                
    # 3. 데이터프레임 구조로 평탄화 (Flatten)
    flat_data = []
    # 데이터의 순서를 유지하기 위해 다시 순회
    for item in references:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            
            for mode in ["video", "full", "part"]:
                row = {
                    "content_id": cid,
                    "query": query,
                    "mode": mode,
                    "reference": data_map[key].get(f"ref_{mode}", ""),
                    "response": data_map[key].get(f"resp_{mode}", ""),
                    "judge": data_map[key].get(f"judge_{mode}", ""),
                    "score": data_map[key].get(f"score_{mode}", "")
                }
                flat_data.append(row)
                
    # 4. 엑셀 저장
    df = pd.DataFrame(flat_data)
    out_path = os.path.join(output_dir, "details.xlsx")
    
    # 엑셀 서식화 기능 (너비 자동 조절)
    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Details')
    # 컬럼 너비를 약간 넓게 셋팅
    worksheet = writer.sheets['Details']
    for idx, col in enumerate(df.columns):
        series = df[col]
        worksheet.column_dimensions[chr(65+idx)].width = 25
    writer.close()
    
    print(f"Created: {out_path}")

def export_scores(input_dir, output_dir):
    agg_path = os.path.join(input_dir, "scores_aggregated.json")
    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)
        
    flat_agg = []
    for cid, modes in agg.get("by_video", {}).items():
        for mode, metrics in modes.items():
            row = {"content_id": cid, "mode": mode}
            row.update(metrics)
            flat_agg.append(row)
            
    for mode, metrics in agg.get("overall", {}).items():
        row = {"content_id": "OVERALL", "mode": mode}
        row.update(metrics)
        flat_agg.append(row)
        
    df_scores = pd.DataFrame(flat_agg)
    out_path = os.path.join(output_dir, "scores.xlsx")
    
    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df_scores.to_excel(writer, index=False, sheet_name='Scores')
    worksheet = writer.sheets['Scores']
    for idx, col in enumerate(df_scores.columns):
        worksheet.column_dimensions[chr(65+idx)].width = 15
    writer.close()
        
    print(f"Created: {out_path}")

if __name__ == "__main__":
    assets_dir = "assets"
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    export_details(assets_dir, results_dir)
    export_scores(assets_dir, results_dir)