import argparse
import subprocess
import sys
import os
import json
from gemini_api_utils import load_config

def main():
    parser = argparse.ArgumentParser(description="End-to-End LLM Judge Pipeline Orchestrator (Bubble & User Query)")
    parser.add_argument("--input_file", default="content_list.json", help="최초 컨텐츠 목록(JSON) 파일 경로")
    
    # Bubble Query (현재 장면 관련 질문) 입출력
    parser.add_argument("--bubble_queries_file", default="assets/bubble_query_generated.jsonl", help="Bubble Query 생성된 질문 목록 경로")
    parser.add_argument("--bubble_judged_queries_file", default="assets/bubble_query_judged.jsonl", help="Bubble Query Judge 통과한 질문 목록 경로")
    parser.add_argument("--bubble_query_scores_file", default="assets/bubble_query_scores.jsonl", help="Bubble Query 질문별 Judge 점수 파일 경로")
    
    # User Query (전체 시청 내역 기반 질문 및 답변) 입출력
    parser.add_argument("--user_queries_file", default="assets/user_query_generated.jsonl", help="User Query 생성된 질문 목록 경로")
    parser.add_argument("--responses_file", default="assets/responses.jsonl", help="User Query 생성/통합된 답변 목록 경로")
    parser.add_argument("--references_file", default="assets/references.jsonl", help="User Query Reference 답변 목록 경로")
    parser.add_argument("--scores_file", default="assets/scores.jsonl", help="User Query 최종 답변 평가 결과 경로")
    
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--location", default="global", help="GCP Location")
    
    parser.add_argument("--keypoint_model", default="gemini-2.5-flash", help="Keypoint Scene 식별에 사용할 Budget 모델명")
    parser.add_argument("--query_gen_model", default="gemini-2.5-flash", help="질문 생성(Bubble/User)에 사용할 Budget 모델명")
    parser.add_argument("--summary_gen_model", default="gemini-2.5-pro", help="Detailed Summary 생성에 사용할 Premium 모델명")
    parser.add_argument("--query_judge_model", default="gemini-2.5-pro", help="Bubble Query 질문 Judge에 사용할 Premium 모델명")
    
    parser.add_argument("--response_gen_model", default="gemini-2.5-flash", help="User Query 답변 생성 모델명")
    parser.add_argument("--reference_model", default="gemini-2.5-pro", help="User Query Reference Answer 생성 모델명")
    parser.add_argument("--no-reference-ref", dest="reference_use_ref", action="store_false", help="Reference 생성 시 Ref JSONL 미참조 (Video만 사용)")
    parser.set_defaults(reference_use_ref=True)
    parser.add_argument("--judge_model", default="gemini-2.5-pro", help="User Query 답변 평가 모델명")
    
    parser.add_argument("--skip-query-gen", action="store_true", help="질문 생성(Bubble/User)을 건너뛰기")
    parser.add_argument("--skip-query-judge", action="store_true", help="Bubble Query 질문 Judge를 건너뛰기")
    parser.add_argument("--skip-response", action="store_true", help="User Query 답변 생성을 건너뛰기")
    parser.add_argument("--skip-judge", action="store_true", help="User Query 답변 평가를 건너뛰기")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (--gcp_project_id 인자를 주입하거나 config.json을 생성하세요)")
        return

    common_project_args = [
        "--gcp_project_id", args.gcp_project_id,
        "--gs_bucket_name", args.gs_bucket_name,
        "--location", args.location
    ]

    # ───────────────────────────────────────────────
    # Step 0: Keypoint Scene 식별 + Bubble/User Query 생성
    # ───────────────────────────────────────────────
    if not args.skip_query_gen:
        print("\n" + "="*60)
        print(">>> 0-1. Keypoint Scene 식별 + Bubble Query 생성 (generate_bubble_query.py)")
        print("="*60)
        cmd_bubble = [
            sys.executable, "generate_bubble_query.py",
            "--input_file", args.input_file,
            "--output_file", args.bubble_queries_file,
            "--keypoint_model", args.keypoint_model,
            "--query_gen_model", args.query_gen_model,
            "--summary_gen_model", args.summary_gen_model,
        ] + common_project_args
        subprocess.run(cmd_bubble, check=True)
        print(f"-> Bubble Query 저장 완료: {args.bubble_queries_file}")

        print("\n" + "="*60)
        print(">>> 0-2. 파생 Scene 기반 User Query 생성 (generate_user_query.py)")
        print("="*60)
        cmd_user = [
            sys.executable, "generate_user_query.py",
            "--input_file", args.bubble_queries_file,
            "--output_file", args.user_queries_file,
            "--query_gen_model", args.query_gen_model,
        ] + common_project_args
        subprocess.run(cmd_user, check=True)
        print(f"-> User Query 저장 완료: {args.user_queries_file}")
    else:
        print(f"\n[Skip] 질문 생성 건너뜀.")

    # ───────────────────────────────────────────────
    # Step 1: Bubble Query 질문 품질 Judge (judge_query.py)
    # ───────────────────────────────────────────────
    if not args.skip_query_judge:
        print("\n" + "="*60)
        print(">>> 1. Bubble Query 질문 통과 여부 Judge (judge_query.py)")
        print("="*60)
        cmd = [
            sys.executable, "judge_query.py",
            "--input_file", args.bubble_queries_file,
            "--output_file", args.bubble_judged_queries_file,
            "--scores_file", args.bubble_query_scores_file,
            "--query_judge_model", args.query_judge_model,
        ] + common_project_args
        subprocess.run(cmd, check=True)
        print(f"-> Bubble Query Judge 통과 질문 저장 완료: {args.bubble_judged_queries_file}")
    else:
        print(f"\n[Skip] Bubble Query 질문 Judge 건너뜀.")

    # ───────────────────────────────────────────────
    # Step 2: User Query 답변 생성 (generate_response.py)
    # ───────────────────────────────────────────────
    if not args.skip_response:
        print("\n" + "="*60)
        print(">>> 2. User Query 답변 생성 (generate_response.py)")
        print("="*60)
        # User Query의 쿼리를 입력으로 사용 (사전 Judge 없음)
        cmd = [
            sys.executable, "generate_response.py",
            "--json_file", args.user_queries_file,
            "--output_file", args.responses_file,
            "--reference_file", args.references_file,
            "--response_gen_model", args.response_gen_model,
            "--reference_model", args.reference_model,
        ]
        if not args.reference_use_ref:
            cmd.append("--no-reference-ref")
        cmd += common_project_args
        subprocess.run(cmd, check=True)
    else:
        print(f"\n[Skip] User Query 답변 생성 건너뜀.")
 
    # ───────────────────────────────────────────────
    # Step 3: User Query 답변 Judge (judge_response.py)
    # ───────────────────────────────────────────────
    if not args.skip_judge:
        print("\n" + "="*60)
        print(">>> 3. User Query 답변 Judge (judge_response.py)")
        print("="*60)
        cmd = [
            sys.executable, "judge_response.py",
            "--answers_file", args.responses_file,
            "--references_file", args.references_file,
            "--output_file", args.scores_file,
            "--judge_model", args.judge_model,
        ] + common_project_args
        subprocess.run(cmd, check=True)
    else:
        print(f"\n[Skip] User Query 답변 Judge 건너뜀.")

    # ───────────────────────────────────────────────
    # Step 4: JSONL → JSON 집계 + 엑셀 내보내기
    # ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> 4. JSONL 집계 및 엑셀 내보내기")
    print("="*60)
    output_dir = os.path.dirname(args.scores_file) or "assets"
    subprocess.run([sys.executable, "jsonl_to_json.py", "--input_dir", output_dir], check=True)

    scores_json = os.path.join(output_dir, "scores.json")
    if os.path.exists(scores_json):
        subprocess.run([sys.executable, "aggregate_scores.py", "--scores_file", scores_json])
        subprocess.run([sys.executable, "export_to_excel.py"])

    print("\n\nEnd-to-End Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()