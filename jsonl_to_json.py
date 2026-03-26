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
            
            for line in lines:
                if line.strip():
                    try:
                        obj = json.loads(line.strip())
                        c_id = obj.get("content_id")
                        if c_id:
                            # 딕셔너리로 덮어쓰기하여 가장 최신(마지막)의 정상 업데이트만 유지합니다.
                            # 즉, 중간에 Error로 기록된 내역이 있더라도 나중에 Resume을 통해 정상 덮어써졌다면
                            # 최종 파일에는 정상본만 남게 됩니다.
                            data_dict[c_id] = obj
                    except json.JSONDecodeError:
                        error_count += 1
            
            data = list(data_dict.values())
            
            # scores 파일일 경우, query를 기준으로 mode들을 묶는 전처리 수행
            if "scores.jsonl" in jsonl_path or "scores" in base_name:
                reformatted_data = []
                for item in data:
                    c_id = item.get("content_id")
                    original_scores = item.get("scores", [])
                    
                    # 쿼리별로 모드를 모음
                    query_map = {}
                    for entry in original_scores:
                        q = entry.get("query")
                        m = entry.get("mode")
                        j = entry.get("judge")
                        if not q: continue
                        
                        if q not in query_map:
                            query_map[q] = {}
                        query_map[q][m] = j
                        
                    # 최종 딕셔너리 구조 생성 시 지정된 mode 순서 적용
                    queries_list = []
                    desired_order = ["video", "full", "part"]
                    for q, s in query_map.items():
                        ordered_scores = {}
                        for m_key in desired_order:
                            if m_key in s:
                                ordered_scores[m_key] = s[m_key]
                        for m_key, val in s.items():
                            if m_key not in desired_order:
                                ordered_scores[m_key] = val
                        queries_list.append({"query": q, "scores": ordered_scores})
                        
                    reformatted_data.append({
                        "content_id": c_id,
                        "queries": queries_list
                    })
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
