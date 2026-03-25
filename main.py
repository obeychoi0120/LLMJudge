import argparse
import subprocess
import sys
import os
import json
from gemini_api_utils import load_config

def main():
    parser = argparse.ArgumentParser(description="End-to-End LLM Judge Pipeline Orchestrator")
    parser.add_argument("--input_file", default="content_list.json", help="최초 컨텐츠 목록(JSON) 파일 경로")
    parser.add_argument("--generated_queries_file", default="output/query_generated.jsonl", help="생성된 질문 목록을 저장할 파일 경로")
    parser.add_argument("--responses_file", default="output/responses.jsonl", help="생성/통합된 답변 목록을 저장할 파일 경로")
    parser.add_argument("--scores_file", default="output/scores.jsonl", help="최종 평가 결과를 저장할 파일 경로")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="us-central1", help="GCP Location (Set to 'global' for gemini-3-pro-preview)")
    parser.add_argument("--query_gen_model", default="gemini-2.5-pro", help="사용할 질문 생성 모델명")
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="사용할 답변 생성 모델명")
    parser.add_argument("--judge_model", default="gemini-2.5-pro", help="사용할 평가 모델명")
    parser.add_argument("--generate-query", action="store_true", help="질문 자동 생성을 우선 수행")
    parser.add_argument("--skip-response", action="store_true", help="답변 생성을 건너뛰기")
    parser.add_argument("--skip-judge", action="store_true", help="평가를 건너뛰기")
    parser.add_argument("--skip-aggregate", action="store_true", help="최종 JSON 변환 건너뛰기")
    
    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (--gcp_project_id 인자를 주입하거나 config.json을 생성하세요)")
        return
        
    common_project_args = [
        "--gcp_project_id", args.gcp_project_id,
        "--location", args.location
    ]
    
    current_input_file = args.input_file
    
    # 입력 파일 형식 감지 (단순 리스트 vs 쿼리 포함 객체 리스트 vs JSONL)
    has_queries = False
    if os.path.exists(current_input_file):
        if current_input_file.endswith(".jsonl"):
            with open(current_input_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.strip():
                    try:
                        first_item = json.loads(first_line)
                        if isinstance(first_item, dict) and "queries" in first_item:
                            has_queries = True
                    except json.JSONDecodeError:
                        pass
        else:
            with open(current_input_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        first_item = data[0]
                        # 문자열 리스트거나 queries 키가 없으면 'content-only'로 간주
                        if isinstance(first_item, dict) and "queries" in first_item:
                            has_queries = True
                except json.JSONDecodeError:
                    pass

    # 쿼리가 없는데 --generate-query도 설정되지 않은 경우, 자동으로 쿼리 생성을 시도하거나 경고
    should_run_query_gen = args.generate_query
    if not has_queries and not args.generate_query:
        print(f"\n[Notice] '{current_input_file}'에 쿼리 정보가 없어 질문 생성을 자동으로 시작합니다.")
        should_run_query_gen = True

    if should_run_query_gen:
        print("\n" + "="*60)
        print(">>> 0. Generating Queries")
        print("="*60)
        cmd = [
            sys.executable, "generate_query.py", 
            "--input_file", current_input_file, 
            "--output_file", args.generated_queries_file,
            "--gs_bucket_name", args.gs_bucket_name,
            "--query_gen_model", args.query_gen_model
        ] + common_project_args
        subprocess.run(cmd, check=True)
        # 생성된 쿼리 파일 경로를 다음 파이프라인(Generation)의 입력으로 전환
        current_input_file = args.generated_queries_file
        print(f"-> Updated JSON input to {current_input_file} for subsequent steps.")
    else:
        print(f"\n[Info] '{current_input_file}'의 기존 쿼리 정보를 사용하여 프로세스를 진행합니다.")
        
    if not args.skip_response:
        print("\n" + "="*60)
        print(">>> 1. Generating Responses")
        print("="*60)
        cmd = [
            sys.executable, "generate_response.py", 
            "--json_file", current_input_file,
            "--output_file", args.responses_file,
            "--gs_bucket_name", args.gs_bucket_name,
            "--response_gen_model", args.response_gen_model
        ] + common_project_args
        subprocess.run(cmd, check=True)
        
    if not args.skip_judge:
        print("\n" + "="*60)
        print(">>> 2. Judging Responses")
        print("="*60)
        cmd = [
            sys.executable, "judge_response.py", 
            "--answers_file", args.responses_file,
            "--output_file", args.scores_file,
            "--gs_bucket_name", args.gs_bucket_name,
            "--judge_model", args.judge_model
        ] + common_project_args
        subprocess.run(cmd, check=True)
        
    if not args.skip_aggregate:
        print("\n" + "="*60)
        print(">>> 3. Aggregating JSONL to JSON")
        print("="*60)
        output_dir = os.path.dirname(args.scores_file) or "output"
        cmd = [
            sys.executable, "jsonl_to_json.py",
            "--input_dir", output_dir
        ]
        subprocess.run(cmd, check=True)
        
    print("\n\nEnd-to-End Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()