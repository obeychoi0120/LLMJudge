import pandas as pd
import json
import os
from utils import load_jsonl

def aggregate_uq_scores(input_dir):
    """UQ Score를 집계하여 JSON으로 저장합니다."""
    scores_file = os.path.join(input_dir, "uq_response_scores.json")
    output_file = os.path.join(input_dir, "uq_response_scores_aggregated.json")
    
    if not os.path.exists(scores_file):
        print(f"[Skip] {scores_file} 파일이 없어 UQ Aggregation을 건너뜁니다.")
        return

    print(f"Aggregating UQ scores from {scores_file}...")
    with open(scores_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {scores_file}: {e}")
            return

    metrics = ["accuracy", "completeness", "helpfulness", "total_score"]
    modes = ["video", "raw", "img_desc", "mm_desc"]

    results_by_video = {}
    overall_raw = {m: {met: [] for met in metrics} for m in modes}

    for video_item in data:
        c_id = video_item.get("content_id")
        if not c_id: continue
        
        video_raw = {m: {met: [] for met in metrics} for m in modes}
        
        for query_item in video_item.get("queries", []):
            judge_data = query_item.get("judge", {})
            for m in modes:
                if m in judge_data:
                    m_data = judge_data[m]
                    s_data = m_data.get("scores", {})
                    ts = m_data.get("total_score")
                    
                    for met in metrics[:-1]:
                        if met in s_data:
                            val = s_data[met]
                            if isinstance(val, (int, float)):
                                video_raw[m][met].append(val)
                                overall_raw[m][met].append(val)
                    
                    if ts is not None and isinstance(ts, (int, float)):
                        video_raw[m]["total_score"].append(ts)
                        overall_raw[m]["total_score"].append(ts)

        avg_video = {}
        for m in modes:
            avg_mode = {}
            for met in metrics:
                s_list = video_raw[m][met]
                if s_list:
                    avg_mode[met] = round(sum(s_list) / len(s_list), 2)
            if avg_mode:
                avg_video[m] = avg_mode
        
        if avg_video:
            results_by_video[c_id] = avg_video

    results_overall = {}
    for m in modes:
        avg_mode = {}
        for met in metrics:
            s_list = overall_raw[m][met]
            if s_list:
                avg_mode[met] = round(sum(s_list) / len(s_list), 2)
        if avg_mode:
            results_overall[m] = avg_mode

    final_output = {"by_video": results_by_video, "overall": results_overall}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"[Done] UQ Score Aggregation: {output_file}")


def aggregate_vh_scores(input_dir):
    """VH Score를 집계하여 JSON으로 저장합니다."""
    scores_file = os.path.join(input_dir, "voice_hint_scores.jsonl")
    output_file = os.path.join(input_dir, "voice_hint_scores_aggregated.json")

    if not os.path.exists(scores_file):
        print(f"[Skip] {scores_file} 파일이 없어 VH Aggregation을 건너뜁니다.")
        return

    print(f"Aggregating VH scores from {scores_file}...")
    try:
        data = load_jsonl(scores_file)
    except Exception as e:
        print(f"Error: Failed to parse {scores_file}: {e}")
        return

    metrics = ["curiosity_and_hook", "temporal_immersion", "total_score"]
    modes = []
    results_by_video = {}
    overall_raw = {}

    for video_item in data:
        c_id = video_item.get("content_id")
        if not c_id: continue
        
        video_raw = {}
        
        items = video_item.get("queries", [])
        if not items and "query" in video_item: 
            items = [video_item]

        for query_item in items:
            m = query_item.get("mode", "unknown")
            if m not in video_raw:
                video_raw[m] = {met: [] for met in metrics}
            if m not in overall_raw:
                overall_raw[m] = {met: [] for met in metrics}
                if m not in modes:
                    modes.append(m)

            judge_data = query_item.get("judge", {})
            ts = query_item.get("total_score")

            for met in metrics[:-1]:
                if met in judge_data:
                    val = judge_data[met].get("score")
                    if isinstance(val, (int, float)):
                        video_raw[m][met].append(val)
                        overall_raw[m][met].append(val)
            
            if isinstance(ts, (int, float)):
                video_raw[m]["total_score"].append(ts)
                overall_raw[m]["total_score"].append(ts)

        avg_video = {}
        for m in video_raw:
            avg_mode = {}
            for met in metrics:
                s_list = video_raw[m][met]
                if s_list:
                    avg_mode[met] = round(sum(s_list) / len(s_list), 2)
            if avg_mode:
                avg_video[m] = avg_mode
        
        if avg_video:
            results_by_video[c_id] = avg_video

    results_overall = {}
    for m in overall_raw:
        avg_mode = {}
        for met in metrics:
            s_list = overall_raw[m][met]
            if s_list:
                avg_mode[met] = round(sum(s_list) / len(s_list), 2)
        if avg_mode:
            results_overall[m] = avg_mode

    final_output = {"by_video": results_by_video, "overall": results_overall}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"[Done] VH Score Aggregation: {output_file}")


def export_uq_details(input_dir, output_dir):
    """User Query의 Response/Reference/Score를 상세 Excel로 내보냅니다."""
    # 1. 파일 읽기
    import glob
    summary_files = glob.glob(os.path.join(input_dir, "keyscene_summary*.json"))
    references_path = summary_files[0] if summary_files else os.path.join(input_dir, "keyscene_summary.json")

    responses_path = os.path.join(input_dir, "uq_responses.json")
    scores_path = os.path.join(input_dir, "uq_response_scores.json")

    for path in [references_path, responses_path, scores_path]:
        if not os.path.exists(path):
            print(f"[Skip] {path} 파일이 없어 UQ Details 내보내기를 건너뜁니다.")
            return

    with open(references_path, "r", encoding="utf-8") as f:
        keyscene_summaries = json.load(f)
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    with open(scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    modes = ["video", "raw", "img_desc", "mm_desc"]

    # 2. 데이터 매핑용 딕셔너리 생성
    summary_map = {}
    for group in keyscene_summaries:
        if "items" in group:
            for item in group["items"]:
                cid = item.get("content_id")
                idx = item.get("scene_idx")
                if cid and idx is not None:
                    summary_map[(cid, idx)] = item.get("summary", "")
        else:
            cid = group.get("content_id")
            idx = group.get("scene_idx")
            if cid and idx is not None:
                summary_map[(cid, idx)] = group.get("summary", "")

    data_map = {}

    for item in responses:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            scene_idx = q.get("scene_idx")
            key = (cid, query)
            if key not in data_map:
                data_map[key] = {}
                
            ref_text = summary_map.get((cid, scene_idx), "") if scene_idx is not None else ""
            for mode in modes:
                data_map[key][f"ref_{mode}"] = ref_text
                
            answers = q.get("answers", {})
            for mode in modes:
                data_map[key][f"resp_{mode}"] = answers.get(mode, "")

    for item in scores:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            if key not in data_map:
                continue
            judge_data = q.get("judge", {})
            for mode in modes:
                mode_judge = judge_data.get(mode, {})
                data_map[key][f"judge_{mode}"] = mode_judge.get("rationale", "")
                data_map[key][f"score_{mode}"] = mode_judge.get("total_score", "")

    # 3. 데이터프레임 구조로 평탄화
    flat_data = []
    for item in responses:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            for mode in modes:
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
    out_path = os.path.join(output_dir, "uq_response_details.xlsx")

    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Details')
    worksheet = writer.sheets['Details']
    for idx, col in enumerate(df.columns):
        worksheet.column_dimensions[chr(65 + idx)].width = 25
    writer.close()

    print(f"Created: {out_path}")

def export_uq_scores(input_dir, output_dir):
    """User Query의 집계 점수를 Excel로 내보냅니다."""
    agg_path = os.path.join(input_dir, "uq_response_scores_aggregated.json")
    if not os.path.exists(agg_path):
        print(f"[Skip] {agg_path} 파일이 없어 UQ Scores 내보내기를 건너뜁니다.")
        return

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
    out_path = os.path.join(output_dir, "uq_response_scores.xlsx")

    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df_scores.to_excel(writer, index=False, sheet_name='Scores')
    worksheet = writer.sheets['Scores']
    for idx, col in enumerate(df_scores.columns):
        worksheet.column_dimensions[chr(65 + idx)].width = 15
    writer.close()

    print(f"Created: {out_path}")


def export_vh_details(input_dir, output_dir):
    """Voice Hint Judge 상세 결과를 Excel로 내보냅니다."""
    vh_scores_path = os.path.join(input_dir, "voice_hint_scores.jsonl")
    if not os.path.exists(vh_scores_path):
        print(f"[Skip] {vh_scores_path} 파일이 없어 VH Details 내보내기를 건너뜁니다.")
        return

    vh_data = load_jsonl(vh_scores_path)

    flat_vh = []
    metrics = ["curiosity_and_hook", "temporal_immersion"]

    for item in vh_data:
        cid = item.get("content_id", "")
        item_scene_idx = item.get("scene_idx")

        if "query" in item and "judge" in item:
            judge = item.get("judge", {})
            total = item.get("total_score", "")
            row = {
                "content_id": cid,
                "scene_idx": item_scene_idx,
                "mode": item.get("mode", ""),
                "query": item.get("query", ""),
                "rationale_curiosity": judge.get("curiosity_and_hook", {}).get("rationale", ""),
                "rationale_temporal": judge.get("temporal_immersion", {}).get("rationale", ""),
            }
            for met in metrics:
                row[met] = judge.get(met, {}).get("score", "")
            row["total_score"] = total
            flat_vh.append(row)
        elif "queries" in item:
            for q_entry in item.get("queries", []):
                judge = q_entry.get("judge", {})
                total = q_entry.get("total_score", "")
                scene_idx = q_entry.get("scene_idx", item_scene_idx)
                row = {
                    "content_id": cid,
                    "scene_idx": scene_idx,
                    "mode": q_entry.get("mode", ""),
                    "query": q_entry.get("query", ""),
                    "rationale_curiosity": judge.get("curiosity_and_hook", {}).get("rationale", ""),
                    "rationale_temporal": judge.get("temporal_immersion", {}).get("rationale", ""),
                }
                for met in metrics:
                    row[met] = judge.get(met, {}).get("score", "")
                row["total_score"] = total
                flat_vh.append(row)

    if not flat_vh:
        print("[Skip] Voice Hint Score 데이터가 비어 있습니다.")
        return

    df = pd.DataFrame(flat_vh)
    out_path = os.path.join(output_dir, "vh_details.xlsx")

    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='VH Details')
    worksheet = writer.sheets['VH Details']
    
    col_widths = {
        "content_id": 20, "scene_idx": 10, "mode": 10, "query": 40,
        "rationale_curiosity": 40, "curiosity_and_hook": 12,
        "rationale_temporal": 40, "temporal_immersion": 12,
        "total_score": 12
    }
    for idx, col in enumerate(df.columns):
        width = col_widths.get(col, 15)
        worksheet.column_dimensions[chr(65 + idx)].width = width
    writer.close()

    print(f"Created: {out_path}")

def export_vh_scores(input_dir, output_dir):
    """Voice Hint Judge 집계 점수를 Excel로 내보냅니다."""
    agg_path = os.path.join(input_dir, "voice_hint_scores_aggregated.json")
    if not os.path.exists(agg_path):
        print(f"[Skip] {agg_path} 파일이 없어 VH Scores 내보내기를 건너뜁니다.")
        return

    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)

    data_by_mode = {}
    
    for cid, modes_data in agg.get("by_video", {}).items():
        for mode, metrics in modes_data.items():
            if mode not in data_by_mode:
                data_by_mode[mode] = []
            row = {"content_id": cid}
            row.update(metrics)
            data_by_mode[mode].append(row)

    for mode, metrics in agg.get("overall", {}).items():
        if mode not in data_by_mode:
            data_by_mode[mode] = []
        row = {"content_id": "OVERALL"}
        row.update(metrics)
        data_by_mode[mode].append(row)

    exported_count = 0
    for mode, rows in data_by_mode.items():
        if not rows:
            continue

        df_scores = pd.DataFrame(rows)
        safe_mode = mode.replace("_", "") if "_" in mode else mode
        out_path = os.path.join(output_dir, f"vh_scores_{safe_mode}.xlsx")

        writer = pd.ExcelWriter(out_path, engine='openpyxl')
        df_scores.to_excel(writer, index=False, sheet_name='VH Scores')
        worksheet = writer.sheets['VH Scores']
        for idx, col in enumerate(df_scores.columns):
            worksheet.column_dimensions[chr(65 + idx)].width = 15
        writer.close()

        print(f"Created: {out_path}")
        exported_count += 1

    if exported_count == 0:
        print("[Skip] Voice Hint Aggregated Score 데이터가 비어 있습니다.")

if __name__ == "__main__":
    assets_dir = "assets"
    results_dir = "results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("="*60)
    print("1. Data Aggregation Phase")
    print("="*60)
    aggregate_uq_scores(assets_dir)
    aggregate_vh_scores(assets_dir)

    print("\n" + "="*60)
    print("2. Excel Export Phase")
    print("="*60)
    export_uq_details(assets_dir, results_dir)
    export_uq_scores(assets_dir, results_dir)
    export_vh_details(assets_dir, results_dir)
    export_vh_scores(assets_dir, results_dir)

    print("\nAll pipeline tasks completed.")