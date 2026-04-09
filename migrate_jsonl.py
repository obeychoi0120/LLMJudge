import json
import os
from collections import defaultdict

input_path = r"c:\Users\junsu2.choi\workspace\LLMJudge\assets\user_query.jsonl"
output_path = r"c:\Users\junsu2.choi\workspace\LLMJudge\assets\user_query_migrated.jsonl"

if not os.path.exists(input_path):
    print("File not found.")
    exit(0)

with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line: continue
        data = json.loads(line)
        c_id = data.get("content_id")
        queries = data.get("queries", [])
        
        # If it's already migrated (has scene_idx at the top level), just copy
        if "scene_idx" in data:
            f_out.write(line + "\n")
            continue
            
        scene_dict = defaultdict(list)
        for q in queries:
            s_idx = q["scene_idx"]
            scene_dict[s_idx].append(q)
            
        for s_idx, q_list in scene_dict.items():
            start_time = q_list[0].get("start_time")
            end_time = q_list[0].get("end_time")
            new_data = {
                "content_id": c_id,
                "scene_idx": s_idx,
                "start_time": start_time,
                "end_time": end_time,
                "queries": q_list
            }
            f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print("Migration done")
