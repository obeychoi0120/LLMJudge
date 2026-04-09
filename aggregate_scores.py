import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Aggregate LLM Judge scores")
    parser.add_argument("--scores_file", default="assets/scores.json", help="Path to scores.json")
    parser.add_argument("--output_file", default="assets/scores_aggregated.json", help="Path to aggregated JSON")
    args = parser.parse_args()

    if not os.path.exists(args.scores_file):
        print(f"Error: {args.scores_file} does not exist.")
        return

    print(f"Reading scores from {args.scores_file}...")
    with open(args.scores_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {args.scores_file}: {e}")
            return

    # metrics: accuracy, completeness, helpfulness, total_score
    # modes: video, raw, img_desc, mm_desc
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
                    
                    # accuracy, completeness, helpfulness
                    for met in metrics[:-1]:
                        if met in s_data:
                            val = s_data[met]
                            if isinstance(val, (int, float)):
                                video_raw[m][met].append(val)
                                overall_raw[m][met].append(val)
                    
                    # total_score
                    if ts is not None and isinstance(ts, (int, float)):
                        video_raw[m]["total_score"].append(ts)
                        overall_raw[m]["total_score"].append(ts)

        # Calculate averages for this video
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

    # Calculate overall averages
    results_overall = {}
    for m in modes:
        avg_mode = {}
        for met in metrics:
            s_list = overall_raw[m][met]
            if s_list:
                avg_mode[met] = round(sum(s_list) / len(s_list), 2)
        if avg_mode:
            results_overall[m] = avg_mode

    final_output = {
        "by_video": results_by_video,
        "overall": results_overall
    }

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"\n[Success] Score Aggregation Complete:")
    print(f"-> Source: {args.scores_file}")
    print(f"-> Results saved to: {args.output_file}")
    print("-" * 50)
    for m in modes:
        if m in results_overall:
            print(f"Overall {m.upper():<5}: Total Score Avg = {results_overall[m].get('total_score', 'N/A')}")
    print("-" * 50)

if __name__ == "__main__":
    main()
