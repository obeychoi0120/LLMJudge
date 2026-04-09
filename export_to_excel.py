import pandas as pd
import json
import os

def export_details(input_dir, output_dir):
    """User Query의 Response/Reference/Score를 상세 Excel로 내보냅니다."""
    # 1. 파일 읽기
    references_path = os.path.join(input_dir, "uq_references.json")
    responses_path = os.path.join(input_dir, "uq_responses.json")
    scores_path = os.path.join(input_dir, "uq_response_scores.json")

    for path in [references_path, responses_path, scores_path]:
        if not os.path.exists(path):
            print(f"[Skip] {path} 파일이 없어 UQ Details 내보내기를 건너뜁니다.")
            return

    with open(references_path, "r", encoding="utf-8") as f:
        references = json.load(f)
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    with open(scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    modes = ["video", "desc"]

    # 2. 데이터 매핑용 딕셔너리 생성
    data_map = {}

    for item in references:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            if key not in data_map:
                data_map[key] = {}
            ref_text = q.get("reference", "")
            for mode in modes:
                data_map[key][f"ref_{mode}"] = ref_text

    for item in responses:
        cid = item["content_id"]
        for q in item["queries"]:
            query = q["query"]
            key = (cid, query)
            if key not in data_map:
                continue
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
    for item in references:
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
    out_path = os.path.join(output_dir, "uq_details.xlsx")

    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Details')
    worksheet = writer.sheets['Details']
    for idx, col in enumerate(df.columns):
        worksheet.column_dimensions[chr(65 + idx)].width = 25
    writer.close()

    print(f"Created: {out_path}")

def export_scores(input_dir, output_dir):
    """User Query의 집계 점수를 Excel로 내보냅니다."""
    agg_path = os.path.join(input_dir, "scores_aggregated.json")
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
    out_path = os.path.join(output_dir, "uq_scores.xlsx")

    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df_scores.to_excel(writer, index=False, sheet_name='Scores')
    worksheet = writer.sheets['Scores']
    for idx, col in enumerate(df_scores.columns):
        worksheet.column_dimensions[chr(65 + idx)].width = 15
    writer.close()

    print(f"Created: {out_path}")


def export_voice_hint_scores(input_dir, output_dir):
    """Voice Hint Judge 점수를 Excel로 내보냅니다."""
    vh_scores_path = os.path.join(input_dir, "voice_hint_scores.json")
    if not os.path.exists(vh_scores_path):
        # JSONL → JSON 변환 전이면 JSON 파일이 없을 수 있음
        print(f"[Skip] {vh_scores_path} 파일이 없어 VH Scores 내보내기를 건너뜁니다.")
        return

    with open(vh_scores_path, "r", encoding="utf-8") as f:
        vh_data = json.load(f)

    # vh_data 는 content_id별로 그룹핑된 JSON 배열
    # 각 항목: {"content_id": ..., "queries": [{"query": ..., "judge": {...}}]}
    flat_vh = []
    metrics = ["naturalness", "temporal_relevance", "difficulty"]
    overall_sums = {met: [] for met in metrics + ["total_score"]}

    for item in vh_data:
        cid = item.get("content_id", "")
        scene_idx = item.get("scene_idx")

        # flat 형식인 경우 (content_id, scene_idx, query, judge 최상위)
        if "query" in item and "judge" in item:
            judge = item.get("judge", {})
            scores_data = judge.get("scores", {})
            total = judge.get("total_score", "")
            row = {
                "content_id": cid,
                "scene_idx": scene_idx,
                "mode": item.get("mode", ""),
                "query": item.get("query", ""),
                "rationale": judge.get("rationale", ""),
            }
            for met in metrics:
                val = scores_data.get(met, "")
                row[met] = val
                if isinstance(val, (int, float)):
                    overall_sums[met].append(val)
            row["total_score"] = total
            if isinstance(total, (int, float)):
                overall_sums["total_score"].append(total)
            flat_vh.append(row)
        # 그룹핑된 형식
        elif "queries" in item:
            for q_entry in item.get("queries", []):
                judge = q_entry.get("judge", {})
                scores_data = judge.get("scores", {})
                total = judge.get("total_score", "")
                row = {
                    "content_id": cid,
                    "scene_idx": scene_idx,
                    "mode": q_entry.get("mode", ""),
                    "query": q_entry.get("query", ""),
                    "rationale": judge.get("rationale", ""),
                }
                for met in metrics:
                    val = scores_data.get(met, "")
                    row[met] = val
                    if isinstance(val, (int, float)):
                        overall_sums[met].append(val)
                row["total_score"] = total
                if isinstance(total, (int, float)):
                    overall_sums["total_score"].append(total)
                flat_vh.append(row)

    if not flat_vh:
        print("[Skip] Voice Hint Score 데이터가 비어 있습니다.")
        return

    # Overall 행 추가
    overall_row = {"content_id": "OVERALL", "scene_idx": "", "mode": "", "query": "", "rationale": "평균"}
    for met in metrics + ["total_score"]:
        vals = overall_sums[met]
        overall_row[met] = round(sum(vals) / len(vals), 2) if vals else ""
    flat_vh.append(overall_row)

    df = pd.DataFrame(flat_vh)
    out_path = os.path.join(output_dir, "vh_scores.xlsx")

    writer = pd.ExcelWriter(out_path, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='VH Scores')
    worksheet = writer.sheets['VH Scores']
    col_widths = {"content_id": 20, "scene_idx": 10, "mode": 8, "query": 40,
                  "rationale": 50, "naturalness": 14, "temporal_relevance": 18,
                  "difficulty": 12, "total_score": 12}
    for idx, col in enumerate(df.columns):
        width = col_widths.get(col, 15)
        worksheet.column_dimensions[chr(65 + idx)].width = width
    writer.close()

    print(f"Created: {out_path}")


if __name__ == "__main__":
    assets_dir = "assets"
    results_dir = "results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    export_details(assets_dir, results_dir)
    export_scores(assets_dir, results_dir)
    export_voice_hint_scores(assets_dir, results_dir)