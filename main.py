import argparse
import subprocess
import sys
import os
import json
from gemini_api_utils import load_config, get_common_argparser

def main():
    parser = get_common_argparser("End-to-End LLM Judge Pipeline Orchestrator (Voice Hint & User Query)")
    parser.add_argument("--input_file", default="content_list.json", help="최초 컨텐츠 목록(JSON) 파일 경로")

    # Voice Hint 입출력 설정
    parser.add_argument("--voice_hints_file", default="assets/voice_hint.jsonl", help="Voice Hint 생성된 질문 목록 경로")
    parser.add_argument("--keyscene_summary_file", help="KeyScene Detailed Summary 별도 저장 경로")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="Keypoint Scene 목록 저장 경로")
    parser.add_argument("--voice_hint_scores_file", default="assets/voice_hint_scores.jsonl", help="Voice Hint 질문별 Judge 점수 파일 경로")
        
    # User Query 입출력 설정
    parser.add_argument("--user_queries_file", default="assets/user_query.jsonl", help="User Query 생성된 질문 목록 경로")
    parser.add_argument("--responses_file", default="assets/uq_responses.jsonl", help="User Query 생성/통합된 답변 목록 경로")

    parser.add_argument("--scores_file", default="assets/uq_response_scores.jsonl", help="User Query 최종 답변 평가 결과 경로")
    
    # 파이프라인 설정
    parser.add_argument("--skip-keypoint", action="store_true", help="A-1. Keypoint 식별을 건너뛰기")
    parser.add_argument("--skip-keyscene-summary", action="store_true", help="A-2. KeyScene Summary 생성을 건너뛰기")
    parser.add_argument("--skip-voice-hint-gen", action="store_true", help="A-3. Voice Hint 생성을 건너뛰기")
    parser.add_argument("--skip-query-judge", action="store_true", help="A-4. Voice Hint 품질 평가를 건너뛰기")
    parser.add_argument("--skip-user-query-gen", action="store_true", help="B-1. User Query 생성을 건너뛰기")

    parser.add_argument("--skip-response", action="store_true", help="B-3. User Query 답변 생성을 건너뛰기")
    parser.add_argument("--skip-judge", action="store_true", help="B-4. User Query 답변 평가를 건너뛰기")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (--gcp_project_id 인자를 주입하거나 config.json을 생성하세요)")
        return

    # 기본값 설정 로직 (config 적용 후)
    if args.keyscene_summary_file is None:
        args.keyscene_summary_file = "assets/keyscene_summary_withref.jsonl" if args.use_ref_for_keyscene_summary else "assets/keyscene_summary_noref.jsonl"


    common_project_args = [
        "--gcp_project_id", args.gcp_project_id,
        "--gs_bucket_name", args.gs_bucket_name,
        "--location", args.location
    ]

    # ───────────────────────────────────────────────
    # A-1: Keypoint Scene 식별 (identify_keypoint.py)
    # ───────────────────────────────────────────────
    if not args.skip_keypoint:
        print("\n" + "="*60)
        print(">>> A-1. Keypoint Scene 식별 (identify_keypoint.py)")
        print("="*60)
        cmd_kp = [
            sys.executable, "identify_keypoint.py",
            "--input_file", args.input_file,
            "--output_file", args.keypoints_file,
            "--keypoint_model", args.keypoint_model,
            "--keypoint_thinking_level", str(args.keypoint_thinking_level),
        ] + common_project_args
        subprocess.run(cmd_kp, check=True)
        print(f"-> Keypoint Scenes 저장 완료: {args.keypoints_file}")
    else:
        print(f"\n[Skip] Keypoint 식별 건너뜀.")

    # ───────────────────────────────────────────────
    # A-2: KeyScene Summary 생성 (generate_keyscene_summary.py)
    # ───────────────────────────────────────────────
    if not args.skip_keyscene_summary:
        print("\n" + "="*60)
        print(">>> A-2. KeyScene Summary 생성 (generate_keyscene_summary.py)")
        print("="*60)
        cmd_ks = [
            sys.executable, "generate_keyscene_summary.py",
            "--input_file", args.keypoints_file,
            "--keyscene_summary_file", args.keyscene_summary_file,
            "--kss_past_summary_model", args.kss_past_summary_model,
            "--kss_past_summary_thinking_level", str(args.kss_past_summary_thinking_level),
            "--kss_current_scene_model", args.kss_current_scene_model,
            "--kss_current_scene_thinking_level", str(args.kss_current_scene_thinking_level),
            "--use_ref_for_keyscene_summary", str(args.use_ref_for_keyscene_summary),
        ] + common_project_args
        subprocess.run(cmd_ks, check=True)
        print(f"-> KeyScene Summary 저장 완료: {args.keyscene_summary_file}")
    else:
        print(f"\n[Skip] KeyScene Summary 생성 건너뜀.")

    # ───────────────────────────────────────────────
    # A-3: Voice Hint 생성 (generate_voice_hint.py)
    # ───────────────────────────────────────────────
    if not args.skip_voice_hint_gen:
        print("\n" + "="*60)
        print(">>> A-3. Voice Hint 생성 (generate_voice_hint.py)")
        print("="*60)
        cmd_vh = [
            sys.executable, "generate_voice_hint.py",
            "--input_file", args.keypoints_file,
            "--output_file", args.voice_hints_file,
            "--vh_gen_model", args.vh_gen_model,
            "--vh_thinking_level", str(args.vh_thinking_level),
        ] + common_project_args
        subprocess.run(cmd_vh, check=True)
        print(f"-> Voice Hint 저장 완료: {args.voice_hints_file}")
    else:
        print(f"\n[Skip] Voice Hint 생성 건너뜀.")

    # ───────────────────────────────────────────────
    # A-4: Voice Hint 질문 품질 Judge (judge_voice_hint.py)
    # ───────────────────────────────────────────────
    if not args.skip_query_judge:
        print("\n" + "="*60)
        print(">>> A-4. Voice Hint 질문 품질 평가 (judge_voice_hint.py)")
        print("="*60)
        cmd = [
            sys.executable, "judge_voice_hint.py",
            "--input_file", args.voice_hints_file,
            "--keyscene_summary_file", args.keyscene_summary_file,
            "--scores_file", args.voice_hint_scores_file,
            "--vh_judge_model", args.vh_judge_model,
            "--vh_judge_thinking_level", str(args.vh_judge_thinking_level),
        ] + common_project_args
        subprocess.run(cmd, check=True)
        print(f"-> Voice Hint 평가 점수 저장 완료: {args.voice_hint_scores_file}")
    else:
        print(f"\n[Skip] Voice Hint 질문 평가 건너뜀.")

    # ───────────────────────────────────────────────
    # B-1: User Query 생성 (generate_user_query.py)
    # ───────────────────────────────────────────────
    if not args.skip_user_query_gen:
        print("\n" + "="*60)
        print(">>> B-1. User Query 생성 (generate_user_query.py)")
        print("="*60)
        cmd_user = [
            sys.executable, "generate_user_query.py",
            "--keypoints_file", args.keypoints_file,
            "--output_file", args.user_queries_file,
            "--uq_gen_model", args.uq_gen_model,
            "--uq_gen_thinking_level", str(args.uq_gen_thinking_level),
        ] + common_project_args
        subprocess.run(cmd_user, check=True)
        print(f"-> User Query 저장 완료: {args.user_queries_file}")
    else:
        print(f"\n[Skip] User Query 생성 건너뜀.")



    # ───────────────────────────────────────────────
    # B-2: User Query 답변 생성 (generate_response.py)
    # ───────────────────────────────────────────────
    if not args.skip_response:
        print("\n" + "="*60)
        print(">>> B-3. User Query 답변 생성 (generate_response.py)")
        print("="*60)
        cmd = [
            sys.executable, "generate_uq_response.py",
            "--json_file", args.user_queries_file,
            "--output_file", args.responses_file,
            "--uq_response_model", args.uq_response_model,
            "--uq_response_thinking_level", str(args.uq_response_thinking_level),
            "--skip_aggregate"
        ]
        cmd += common_project_args
        subprocess.run(cmd, check=True)
    else:
        print(f"\n[Skip] User Query 답변 생성 건너뜀.")

    # ───────────────────────────────────────────────
    # B-3: User Query 답변 Judge (judge_response.py)
    # ───────────────────────────────────────────────
    if not args.skip_judge:
        print("\n" + "="*60)
        print(">>> B-4. User Query 답변 Judge (judge_response.py)")
        print("="*60)
        cmd = [
            sys.executable, "judge_response.py",
            "--answers_file", args.responses_file,
            "--keyscene_summary_file", args.keyscene_summary_file,
            "--output_file", args.scores_file,
            "--uq_judge_model", args.uq_judge_model,
            "--uq_judge_thinking_level", str(args.uq_judge_thinking_level),
            "--skip_aggregate"
        ] + common_project_args
        subprocess.run(cmd, check=True)
    else:
        print(f"\n[Skip] User Query 답변 Judge 건너뜀.")

    # ───────────────────────────────────────────────
    # B-5: JSONL → JSON 집계 + 엑셀 내보내기
    # ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> B-5. JSONL 집계 및 엑셀 내보내기")
    print("="*60)
    output_dir = os.path.dirname(args.scores_file) or "assets"
    subprocess.run([sys.executable, "jsonl_to_json.py", "--input_dir", output_dir], check=True)

    scores_json = os.path.join(output_dir, "scores.json")
    if os.path.exists(scores_json):
        subprocess.run([sys.executable, "aggregate_scores.py", "--scores_file", scores_json])
        subprocess.run([sys.executable, "export_to_excel.py"])

    print("\n\nEnd-to-End Pipeline Completed Successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 파이프라인이 즉시 강제 종료됩니다.")
        import os
        os._exit(1)