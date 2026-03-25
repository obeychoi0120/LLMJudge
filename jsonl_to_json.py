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
