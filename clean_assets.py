import os
import json
import argparse
import tempfile
import shutil

# 4개의 기본 에셋 파일 경로
ASSET_FILES = [
    "assets/interactive_queries.jsonl",
    "assets/interactive_query_scores.jsonl",
    "assets/interactive_query_responses.jsonl",
    "assets/interactive_query_response_scores.jsonl"
]

def clean_modes_from_file(file_path, target_modes):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return

    print(f"Processing {file_path}...")
    deleted_count = 0
    kept_count = 0

    # 동일한 디렉토리에 임시 파일 생성
    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(file_path), text=True)
    
    with open(file_path, 'r', encoding='utf-8') as infile, \
         os.fdopen(fd, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                # mode가 target_modes에 포함되어 있으면 건너뛰기(삭제)
                if record.get("mode") in target_modes:
                    deleted_count += 1
                else:
                    outfile.write(line + "\n")
                    kept_count += 1
            except json.JSONDecodeError:
                # JSON 파싱에 실패한 줄은 안전하게 그대로 유지
                outfile.write(line + "\n")
                kept_count += 1

    # 임시 파일을 원본 파일로 덮어쓰기
    shutil.move(temp_path, file_path)
    print(f" -> [완료] 삭제됨: {deleted_count}개 | 유지됨: {kept_count}개")

def main():
    parser = argparse.ArgumentParser(description="특정 mode의 레코드를 4개의 asset 파일에서 일괄 삭제합니다.")
    parser.add_argument("--modes", required=True, nargs="+", help="삭제할 대상 mode 이름들 (예: imgvlm_chunk2 video)")
    parser.add_argument("--files", nargs="*", default=ASSET_FILES, help="처리할 JSONL 파일 목록 (기본값: 4개의 주요 asset 파일)")
    
    args = parser.parse_args()

    print(f"=== 대상 모드 {args.modes} 삭제 정리 작업 시작 ===")
    for fpath in args.files:
        clean_modes_from_file(fpath, args.modes)
    print("=== 모든 파일 처리 완료 ===")

if __name__ == "__main__":
    main()
