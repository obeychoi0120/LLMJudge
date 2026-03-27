import os
import argparse
import json
import glob

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL files back to formatted JSON for human analysis")
    parser.add_argument("--input_dir", default="output", help="모니터링 대상 JSONL 파일들이 있는 디렉토리")
    args = parser.parse_args()

    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"'{args.input_dir}' 디렉토리에 .jsonl 파일이 없습니다.")
        return

    print("=== JSONL to JSON 분석용 추출을 시작합니다 ===")
    
    # query_generated.json(l) 파일로부터 content_id 기본 순서 추출
    reference_order = []
    qg_json_path = os.path.join(args.input_dir, "query_generated.json")
    qg_jsonl_path = os.path.join(args.input_dir, "query_generated.jsonl")
    
    if os.path.exists(qg_json_path):
        try:
            with open(qg_json_path, "r", encoding="utf-8") as f:
                qg_data = json.load(f)
                for item in qg_data:
                    c_id = item.get("content_id")
                    if c_id and c_id not in reference_order:
                        reference_order.append(c_id)
        except Exception:
            pass
    elif os.path.exists(qg_jsonl_path):
        try:
            with open(qg_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        c_id = obj.get("content_id")
                        if c_id and c_id not in reference_order:
                            reference_order.append(c_id)
        except Exception:
            pass
            
    if reference_order:
        print(f"-> 정렬 기준 확인: query_generated 기준 {len(reference_order)}개 항목 순서 적용")
        
    for jsonl_path in jsonl_files:
        base_name = os.path.splitext(jsonl_path)[0]
        json_path = base_name + ".json"
        
        data_dict = {}
        error_count = 0
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Extra data 에러(동시에 쓰여져 줄바꿈이 누락된 경우) 방어 로직
            content = content.replace('}{', '}\n{')
            lines = content.split('\n')
            
            # responses.jsonl / scores.jsonl 여부 판단: 새 포맷은 "query" 키를 최상위로 보유
            is_scores_file = "scores" in base_name
            is_responses_file = "responses" in base_name
            is_flat_format = is_scores_file or is_responses_file  # (content_id, query) 단위 포맷
            
            for line in lines:
                if line.strip():
                    try:
                        obj = json.loads(line.strip())
                        c_id = obj.get("content_id")
                        if c_id:
                            if is_flat_format:
                                # 새 포맷: 키 = (content_id, query)
                                query = obj.get("query")
                                if query:
                                    data_dict[(c_id, query)] = obj
                                elif is_scores_file:
                                    # 구 포맷(scores content_id 단위) 마이그레이션 보조
                                    for entry in obj.get("scores", []):
                                        q = entry.get("query")
                                        m = entry.get("mode")
                                        j = entry.get("judge")
                                        if q and m and j:
                                            key = (c_id, q)
                                            if key not in data_dict:
                                                data_dict[key] = {"content_id": c_id, "query": q, "judge": {}}
                                            data_dict[key]["judge"][m] = j
                                elif is_responses_file:
                                    # 구 포맷(responses content_id 단위) 마이그레이션 보조
                                    for q_entry in obj.get("queries", []):
                                        q = q_entry.get("query")
                                        ans = q_entry.get("answers")
                                        if q and ans:
                                            data_dict[(c_id, q)] = {"content_id": c_id, "query": q, "answers": ans}
                            else:
                                # 일반 JSONL (query_generated 등): content_id 단위로 덮어쓰기
                                data_dict[c_id] = obj
                    except json.JSONDecodeError:
                        error_count += 1
            
            data = list(data_dict.values())
            
            # flat 포맷 파일(responses, scores)일 경우, (content_id, query) 레코드를 content_id별로 재구성
            if is_flat_format:
                content_query_map = {}   # content_id -> [query, ...] (insertion order)
                content_data_map = {}    # content_id -> {query -> payload dict}
                
                data_key = "judge" if is_scores_file else "answers"
                
                for item in data:
                    c_id = item.get("content_id")
                    query = item.get("query")
                    if not c_id or not query:
                        continue
                    if c_id not in content_query_map:
                        content_query_map[c_id] = []
                        content_data_map[c_id] = {}
                    if query not in content_data_map[c_id]:
                        content_query_map[c_id].append(query)
                    content_data_map[c_id][query] = item.get(data_key, {})
                
                desired_mode_order = ["video", "full", "part"]
                reformatted_data = []
                for c_id in content_query_map:
                    queries_list = []
                    for q in content_query_map[c_id]:
                        raw = content_data_map[c_id][q]
                        ordered = {m: raw[m] for m in desired_mode_order if m in raw}
                        for m, val in raw.items():
                            if m not in desired_mode_order:
                                ordered[m] = val
                        queries_list.append({"query": q, data_key: ordered})
                    reformatted_data.append({"content_id": c_id, "queries": queries_list})
                data = reformatted_data
            
            # reference_order(입력 메타데이터) 순서에 맞춰서 최종 정렬
            if reference_order:
                def get_sort_key(item):
                    c_id = item.get("content_id")
                    try:
                        return reference_order.index(c_id)
                    except ValueError:
                        return len(reference_order) # 리스트에 없으면 맨 뒤로
                data.sort(key=get_sort_key)
            
            # 사람이 보기 편하도록 indent=4 속성 추가
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            msg = f"[성공] '{jsonl_path}' (고유항목 {len(data)} 건) -> '{json_path}'"
            if error_count > 0:
                msg += f" (깨진 조각 {error_count}개 무시됨)"
            print(msg)
            
        except Exception as e:
            print(f"[실패] '{jsonl_path}' 변환 중 오류: {e}")

    print("=======================================")
    print("분석용 JSON 추출이 완료되었습니다. (원본 .jsonl 파이프라인 파일은 그대로 유지됩니다)")

if __name__ == "__main__":
    main()
