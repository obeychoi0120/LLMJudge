import pandas as pd
import json
import os
from utils import load_jsonl



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

    # JSONL은 (content_id, scene_idx, mode, query, judge, total_score) flat 구조
    # by_video_raw[content_id][mode][metric] = [values...]
    by_video_raw = {}
    overall_raw = {}

    for record in data:
        c_id = record.get("content_id")
        if not c_id:
            continue

        m = record.get("mode", "unknown")
        judge_data = record.get("judge", {})
        ts = record.get("total_score")

        # content_id / mode 버킷 초기화
        if c_id not in by_video_raw:
            by_video_raw[c_id] = {}
        if m not in by_video_raw[c_id]:
            by_video_raw[c_id][m] = {met: [] for met in metrics}
        if m not in overall_raw:
            overall_raw[m] = {met: [] for met in metrics}

        # 개별 메트릭 수집
        for met in metrics[:-1]:
            if met in judge_data:
                val = judge_data[met].get("score")
                if isinstance(val, (int, float)):
                    by_video_raw[c_id][m][met].append(val)
                    overall_raw[m][met].append(val)

        if isinstance(ts, (int, float)):
            by_video_raw[c_id][m]["total_score"].append(ts)
            overall_raw[m]["total_score"].append(ts)

    # content_id별 평균 계산
    results_by_video = {}
    for c_id, modes_data in by_video_raw.items():
        avg_video = {}
        for m, raw in modes_data.items():
            avg_mode = {}
            for met in metrics:
                s_list = raw[met]
                if s_list:
                    avg_mode[met] = round(sum(s_list) / len(s_list), 2)
            if avg_mode:
                avg_video[m] = avg_mode
        if avg_video:
            results_by_video[c_id] = avg_video

    # 전체 평균 계산
    results_overall = {}
    for m, raw in overall_raw.items():
        avg_mode = {}
        for met in metrics:
            s_list = raw[met]
            if s_list:
                avg_mode[met] = round(sum(s_list) / len(s_list), 2)
        if avg_mode:
            results_overall[m] = avg_mode

    final_output = {"by_video": results_by_video, "overall": results_overall}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"[Done] VH Score Aggregation: {output_file}")







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



def aggregate_vh_response_scores(input_dir):
    """VH Response Score(vh_response_scores.jsonl)를 집계하여 JSON으로 저장합니다."""
    scores_file = os.path.join(input_dir, "vh_response_scores.jsonl")
    output_file = os.path.join(input_dir, "vh_response_scores_aggregated.json")

    if not os.path.exists(scores_file):
        print(f"[Skip] {scores_file} 파일이 없어 VH Response Score Aggregation을 건너뜁니다.")
        return

    print(f"Aggregating VH Response scores from {scores_file}...")
    try:
        data = load_jsonl(scores_file)
    except Exception as e:
        print(f"Error: Failed to parse {scores_file}: {e}")
        return

    metrics = ["answer_relevance", "factual_precision", "response_quality", "total_score"]

    # by_video_raw[content_id][mode][metric] = [values...]
    by_video_raw = {}
    overall_raw  = {}

    for record in data:
        c_id = record.get("content_id")
        mode = record.get("mode")
        judge = record.get("judge", {})
        if not c_id or not mode or not judge:
            continue

        # total_score: 3개 metric 합계
        ts = sum(
            judge.get(k, {}).get("score", 0)
            for k in metrics[:-1]
            if isinstance(judge.get(k), dict)
        )

        if c_id not in by_video_raw:
            by_video_raw[c_id] = {}
        if mode not in by_video_raw[c_id]:
            by_video_raw[c_id][mode] = {met: [] for met in metrics}
        if mode not in overall_raw:
            overall_raw[mode] = {met: [] for met in metrics}

        for met in metrics[:-1]:
            val = judge.get(met, {}).get("score") if isinstance(judge.get(met), dict) else None
            if isinstance(val, (int, float)):
                by_video_raw[c_id][mode][met].append(val)
                overall_raw[mode][met].append(val)

        by_video_raw[c_id][mode]["total_score"].append(ts)
        overall_raw[mode]["total_score"].append(ts)

    # content_id별 평균 계산
    results_by_video = {}
    for c_id, modes_data in by_video_raw.items():
        avg_video = {}
        for mode, raw in modes_data.items():
            avg_mode = {}
            for met in metrics:
                s_list = raw[met]
                if s_list:
                    avg_mode[met] = round(sum(s_list) / len(s_list), 2)
            if avg_mode:
                avg_video[mode] = avg_mode
        if avg_video:
            results_by_video[c_id] = avg_video

    # 전체 평균 계산
    results_overall = {}
    for mode, raw in overall_raw.items():
        avg_mode = {}
        for met in metrics:
            s_list = raw[met]
            if s_list:
                avg_mode[met] = round(sum(s_list) / len(s_list), 2)
        if avg_mode:
            results_overall[mode] = avg_mode

    final_output = {"by_video": results_by_video, "overall": results_overall}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"[Done] VH Response Score Aggregation: {output_file}")


def export_vh_response_details(input_dir, output_dir):
    """VH Response Judge 상세 결과를 Excel로 내보냅니다."""
    scores_path = os.path.join(input_dir, "vh_response_scores.jsonl")
    if not os.path.exists(scores_path):
        print(f"[Skip] {scores_path} 파일이 없어 VH Response Details 내보내기를 건너뜁니다.")
        return

    # (content_id, scene_idx, mode, query) → response 매핑
    answer_map = {}
    responses_path = os.path.join(input_dir, "vh_responses.jsonl")
    if os.path.exists(responses_path):
        for r in load_jsonl(responses_path):
            c = r.get("content_id")
            s = r.get("scene_idx")
            m = r.get("mode")
            q = r.get("query")
            ans = r.get("answer")
            if c and s is not None and m and q:
                answer_map[(c, s, m, q)] = ans

    data = load_jsonl(scores_path)
    score_keys = ["answer_relevance", "factual_precision", "response_quality"]

    flat_rows = []
    for item in data:
        c_id    = item.get("content_id", "")
        s_idx   = item.get("scene_idx")
        mode    = item.get("mode", "")
        query   = item.get("query", "")
        judge   = item.get("judge", {})

        if not isinstance(judge, dict):
            continue

        ts = sum(
            judge.get(k, {}).get("score", 0)
            for k in score_keys
            if isinstance(judge.get(k), dict)
        )
        response_text = answer_map.get((c_id, s_idx, mode, query), "")
        row = {"content_id": c_id, "scene_idx": s_idx, "mode": mode, "query": query, "response": response_text}
        
        for k in score_keys:
            met_data = judge.get(k, {})
            row[f"rationale_{k}"] = met_data.get("rationale", "") if isinstance(met_data, dict) else ""
            row[k]               = met_data.get("score", "")     if isinstance(met_data, dict) else ""
        row["total_score"] = ts
        flat_rows.append(row)

    if not flat_rows:
        print("[Skip] VH Response Score 데이터가 비어 있습니다.")
        return

    df = pd.DataFrame(flat_rows)
    out_path = os.path.join(output_dir, "vh_response_details.xlsx")

    writer = pd.ExcelWriter(out_path, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="VH Response Details")
    worksheet = writer.sheets["VH Response Details"]
    col_widths = {
        "content_id": 22, "scene_idx": 10, "query": 40, "mode": 12, "response": 60,
        "rationale_answer_relevance": 45,  "answer_relevance": 16,
        "rationale_factual_precision": 45, "factual_precision": 16,
        "rationale_response_quality": 45,  "response_quality": 16,
        "total_score": 12,
    }
    for idx, col in enumerate(df.columns):
        width = col_widths.get(col, 15)
        worksheet.column_dimensions[chr(65 + idx)].width = width
    writer.close()
    print(f"Created: {out_path}")


def export_vh_response_scores(input_dir, output_dir):
    """VH Response Judge 집계 점수를 모드별 Excel로 내보냅니다."""
    agg_path = os.path.join(input_dir, "vh_response_scores_aggregated.json")
    if not os.path.exists(agg_path):
        print(f"[Skip] {agg_path} 파일이 없어 VH Response Scores 내보내기를 건너뜁니다.")
        return

    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)

    data_by_mode = {}

    for c_id, modes_data in agg.get("by_video", {}).items():
        for mode, metrics in modes_data.items():
            if mode not in data_by_mode:
                data_by_mode[mode] = []
            row = {"content_id": c_id}
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
        safe_mode = mode.replace("_", "")
        out_path  = os.path.join(output_dir, f"vh_response_scores_{safe_mode}.xlsx")

        writer = pd.ExcelWriter(out_path, engine="openpyxl")
        df_scores.to_excel(writer, index=False, sheet_name="VH Response Scores")
        worksheet = writer.sheets["VH Response Scores"]
        for idx, col in enumerate(df_scores.columns):
            worksheet.column_dimensions[chr(65 + idx)].width = 15
        writer.close()
        print(f"Created: {out_path}")
        exported_count += 1

    if exported_count == 0:
        print("[Skip] VH Response Aggregated Score 데이터가 비어 있습니다.")


def export_voice_hints(input_dir, output_dir, mode="kss"):
    """Voice Hint 질문을 content_id / scene_idx 기준으로 정리하여 Excel로 내보냅니다.

    컬럼 구조:
    - content_id : 동일 content_id를 가진 첫 번째 행에만 표시, 나머지 행은 빈 칸
    - scene_idx  : 각 scene마다 표시
    - queries    : "1. ~~~\\n2. ~~~" 형태로 한 셀에 번호 붙여 기록
    """
    vh_path = os.path.join(input_dir, "voice_hint.jsonl")
    if not os.path.exists(vh_path):
        print(f"[Skip] {vh_path} 파일이 없어 Voice Hint Export를 건너뜁니다.")
        return

    from collections import defaultdict
    from openpyxl.styles import Alignment

    scenes_by_content = defaultdict(dict)  # {content_id: {scene_idx: queries_text}}

    for rec in load_jsonl(vh_path):
        if rec.get("mode") != mode:
            continue
        c_id  = rec.get("content_id")
        s_idx = rec.get("scene_idx")
        qs    = rec.get("queries", [])
        if not c_id or s_idx is None:
            continue
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
        scenes_by_content[c_id][s_idx] = numbered

    if not scenes_by_content:
        print("[Skip] Voice Hint 데이터가 비어 있습니다.")
        return

    flat_rows = []
    for c_id in sorted(scenes_by_content.keys()):
        first = True
        for s_idx in sorted(scenes_by_content[c_id].keys()):
            flat_rows.append({
                "content_id": c_id if first else "",
                "scene_idx":  s_idx,
                "queries":    scenes_by_content[c_id][s_idx],
            })
            first = False

    df = pd.DataFrame(flat_rows)
    out_path = os.path.join(output_dir, "voice_hints.xlsx")

    writer = pd.ExcelWriter(out_path, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="Voice Hints")
    worksheet = writer.sheets["Voice Hints"]

    col_widths = {"content_id": 28, "scene_idx": 10, "queries": 60}
    for idx, col in enumerate(df.columns):
        worksheet.column_dimensions[chr(65 + idx)].width = col_widths.get(col, 20)

    query_col_letter = chr(65 + list(df.columns).index("queries"))
    for row_cells in worksheet.iter_rows(min_row=2):
        for cell in row_cells:
            if cell.column_letter == query_col_letter:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                cell.alignment = Alignment(vertical="top")

    writer.close()
    print(f"Created: {out_path}")


if __name__ == "__main__":
    assets_dir = "assets"
    results_dir = "results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("="*60)
    print("1. Data Aggregation Phase")
    print("="*60)
    aggregate_vh_scores(assets_dir)
    aggregate_vh_response_scores(assets_dir)

    print("\n" + "="*60)
    print("2. Excel Export Phase")
    print("="*60)
    export_vh_details(assets_dir, results_dir)
    export_vh_scores(assets_dir, results_dir)
    export_vh_response_details(assets_dir, results_dir)
    export_vh_response_scores(assets_dir, results_dir)
    export_voice_hints(assets_dir, results_dir)

    print("\nAll pipeline tasks completed.")