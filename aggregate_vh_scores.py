import json
import os
import argparse
from gemini_api_utils import load_jsonl

def main():
    parser = argparse.ArgumentParser(description="Aggregate Voice Hint scores")
    parser.add_argument("--scores_file", default="assets/voice_hint_scores.jsonl", help="Path to voice_hint_scores.jsonl")
    parser.add_argument("--output_file", default="assets/voice_hint_scores_aggregated.json", help="Path to aggregated JSON")
    args = parser.parse_args()

    if not os.path.exists(args.scores_file):
        print(f"Error: {args.scores_file} does not exist.")
        return

    print(f"Reading scores from {args.scores_file}...")
    try:
        if args.scores_file.endswith(".jsonl"):
            data = load_jsonl(args.scores_file)
        else:
            with open(args.scores_file, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to parse {args.scores_file}: {e}")
        return

    metrics = ["curiosity_and_hook", "temporal_immersion", "total_score"]
    # modes will be discovered dynamically
    modes = []

    results_by_video = {}
    overall_raw = {}

    for video_item in data:
        c_id = video_item.get("content_id")
        if not c_id: continue
        
        video_raw = {}
        
        items = video_item.get("queries", [])
        if not items and "query" in video_item: 
            items = [video_item] # flat format support

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

        # Calculate averages for this video
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

    # Calculate overall averages
    results_overall = {}
    for m in overall_raw:
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

    print(f"\n[Success] Voice Hint Score Aggregation Complete:")
    print(f"-> Source: {args.scores_file}")
    print(f"-> Results saved to: {args.output_file}")
    print("-" * 50)
    for m in modes:
        if m in results_overall:
            print(f"Overall {m.upper():<5}: Total Score Avg = {results_overall[m].get('total_score', 'N/A')}")
    print("-" * 50)

if __name__ == "__main__":
    main()
