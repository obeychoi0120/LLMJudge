import json
import argparse
import os

def clean_file(filepath, remove_modes):
    if not os.path.exists(filepath):
        print(f"[Skip] {filepath} 파일이 존재하지 않습니다.")
        return

    print(f"[{filepath}] 클리닝 시작...")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    removed_mode_count = 0
    removed_pipeline_done = False
    new_lines = []

    for line in lines:
        if not line.strip(): continue
        try:
            obj = json.loads(line)
        except:
            new_lines.append(line)
            continue
            
        # pipeline_done 시그널 제거 (generation 재실행을 위해)
        if "pipeline_done" in obj:
            removed_pipeline_done = True
            continue
            
        # voice_hint.jsonl (flat 포맷) 및 voice_hint_scores.jsonl 모두
        # 최상위 'mode' 필드로 판별하여 제거
        if "mode" in obj:
            if obj["mode"] in remove_modes:
                removed_mode_count += 1
                continue

        new_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    # 원본 파일 직접 덮어쓰기 (백업 X)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    print(f"  -> 완료: 삭제건수 {removed_mode_count}건. (삭제 모드: {remove_modes})", end="")
    if removed_pipeline_done:
        print(" [pipeline_done 시그널 제거완료]")
    else:
        print()

def main():
    parser = argparse.ArgumentParser(description="Clean specific modes from Voice Hint JSONL")
    parser.add_argument("--targets", nargs="+", default=["assets/voice_hint.jsonl", "assets/voice_hint_scores.jsonl"], help="클리닝할 대상 파일 목록")
    parser.add_argument("--remove_modes", nargs="+", default=[], help="제거할 모드 목록 (예: frag frag_with_vlm)")
    args = parser.parse_args()

    for target in args.targets:
        clean_file(target, args.remove_modes)
        
    print("\n모든 클리닝 작업이 종료되었습니다.")

if __name__ == "__main__":
    main()
