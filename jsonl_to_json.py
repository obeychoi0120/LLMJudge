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
        
        data = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            # 사람이 보기 편하도록 indent=4 속성 추가
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"[성공] '{jsonl_path}' ({len(data)} 건) -> '{json_path}'")
        except Exception as e:
            print(f"[실패] '{jsonl_path}' 변환 중 오류: {e}")

    print("=======================================")
    print("분석용 JSON 추출이 완료되었습니다. (원본 .jsonl 파이프라인 파일은 그대로 유지됩니다)")

if __name__ == "__main__":
    main()
