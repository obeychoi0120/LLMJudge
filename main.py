import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="End-to-End LLM Judge Pipeline Orchestrator")
    parser.add_argument("--input_file", default="user_query_list_sample.json", help="최초 컨텐츠/질문 목록 JSON 파일 경로")
    parser.add_argument("--generated_queries_file", default="output/query_generated.json", help="생성된 질문 목록을 저장할 파일 경로")
    parser.add_argument("--responses_file", default="output/responses.json", help="생성/통합된 답변 목록을 저장할 파일 경로")
    parser.add_argument("--scores_file", default="output/scores.json", help="최종 평가 결과를 저장할 파일 경로")
    parser.add_argument("--gcp_project_id", default="insight-dev-490002", help="GCP 프로젝트 ID")
    parser.add_argument("--gs_bucket_name", default="insight-youtubevideodataset-us", help="GCS 버킷 이름")
    parser.add_argument("--location", default="us-central1", help="GCP Location (Set to 'global' for gemini-3-pro-preview)")
    parser.add_argument("--query_gen_model", default="gemini-2.5-pro", help="사용할 질문 생성 모델명")
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="사용할 답변 생성 모델명")
    parser.add_argument("--judge_model", default="gemini-2.5-pro", help="사용할 평가 모델명")
    parser.add_argument("--generate-query", action="store_true", help="질문 자동 생성을 우선 수행")
    parser.add_argument("--skip-response", action="store_true", help="답변 생성을 건너뛰기")
    parser.add_argument("--skip-judge", action="store_true", help="평가를 건너뛰기")
    
    args = parser.parse_args()
    
    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    common_project_args = [
        "--gcp_project_id", args.gcp_project_id,
        "--location", args.location
    ]
    
    current_input_file = args.input_file
    
    if args.generate_query:
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
        # 생성된 쿼리 파일 경로를 다음 파이프라인(Generation)의 입력으로 덮어씀
        current_input_file = args.generated_queries_file
        print(f"-> Updated JSON input to {current_input_file} for subsequent steps.")
        
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
            "--gs_bucket", args.gs_bucket,
            "--judge_model", args.judge_model
        ] + common_project_args
        subprocess.run(cmd, check=True)
        
    print("\n\nEnd-to-End Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()