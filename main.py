import argparse
import subprocess
import sys
import os
import json
from utils import load_config, get_common_argparser

def main():
    parser = get_common_argparser("End-to-End LLM Judge Pipeline Orchestrator (Voice Hint & VH Response)")
    parser.add_argument("--input_file", default="content_list.json", help="최초 컨텐츠 목록(JSON) 파일 경로")

    # Voice Hint 입출력 설정
    parser.add_argument("--voice_hints_file", default="assets/voice_hint.jsonl", help="Voice Hint 생성된 질문 목록 경로")
    parser.add_argument("--keyscene_summary_file", help="KeyScene Detailed Summary 별도 저장 경로")
    parser.add_argument("--keypoints_file", default="assets/keypoint_scenes.jsonl", help="KeyScene 목록 저장 경로")
    parser.add_argument("--voice_hint_scores_file", default="assets/voice_hint_scores.jsonl", help="Voice Hint 질문별 Judge 점수 파일 경로")
        
    # VH Response 입출력 설정 (B-track)
    parser.add_argument("--vh_responses_file",  default="assets/vh_responses.jsonl",      help="VH Response 저장 경로 (generate_vh_response.py 출력)")
    parser.add_argument("--vh_scores_file",      default="assets/vh_response_scores.jsonl", help="VH Response Judge 점수 저장 경로")
    parser.add_argument("--query_source", choices=["kss", "sourcewise"], default="kss",
                        help="Response 생성에 사용할 Voice Hint 질문의 출처 (kss: KSS 기반 공통 질문, sourcewise: 각 모드별로 생성된 질문)")

    args = parser.parse_args()
    args = load_config(args)
    # KSS 생성 시 항상 reference metadata를 참조하도록 고정
    args.use_ref_for_keyscene_summary = True

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (--gcp_project_id 인자를 주입하거나 config.json을 생성하세요)")
        return

    # 기본값 설정 로직 (config 적용 후)
    if args.keyscene_summary_file is None:
        args.keyscene_summary_file = "assets/keyscene_summary.jsonl"

    common_project_args = [
        "--gcp_project_id", args.gcp_project_id,
        "--gs_bucket_name", args.gs_bucket_name,
        "--location", args.location
    ]

    # ───────────────────────────────────────────────
    # O-1: KeyScene 식별 (identify_keyscene.py)
    # ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> O-1. KeyScene 식별 (identify_keyscene.py)")
    print("="*60)
    cmd_kp = [
        sys.executable, "identify_keyscene.py",
        "--input_file", args.input_file,
        "--output_file", args.keypoints_file,
        "--keypoint_model", args.keypoint_model,
        "--keypoint_thinking_level", str(args.keypoint_thinking_level),
    ] + common_project_args
    subprocess.run(cmd_kp, check=True)
    print(f"-> KeyScene 저장 완료: {args.keypoints_file}")

    # ───────────────────────────────────────────────
    # O-2: KeyScene Summary 생성 (generate_keyscene_summary.py)
    # ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> O-2. KeyScene Summary 생성 (generate_keyscene_summary.py)")
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

    # ───────────────────────────────────────────────
    # A-1: Voice Hint 생성 (generate_voice_hint.py)
    # ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> A-1. Voice Hint 생성 (generate_voice_hint.py)")
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

    # ───────────────────────────────────────────────
    # A-2: Voice Hint 질문 품질 Judge (judge_voice_hint.py)
    # ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> A-2. Voice Hint 질문 품질 평가 (judge_voice_hint.py)")
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

    # ───────────────────────────────────────────────
    # B-1: VH(KSS) Response 생성 (generate_vh_response.py)
    # ───────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> B-1. VH Response 생성 (generate_vh_response.py)")
    print("="*60)
    cmd = [
        sys.executable, "generate_vh_response.py",
        "--input_file",      args.voice_hints_file,
        "--output_file",     args.vh_responses_file,
        "--keypoints_file",  args.keypoints_file,
        "--vh_response_model",           args.vh_response_model,
        "--vh_response_thinking_level",  str(args.vh_response_thinking_level),
        "--query_source",                args.query_source,
    ] + common_project_args
    subprocess.run(cmd, check=True)
    print(f"-> VH Response 저장 완료: {args.vh_responses_file}")

    # ──────────────────────────────────────────────
    # B-2: VH Response Judge (judge_vh_response.py)
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> B-2. VH Response Judge (judge_vh_response.py)")
    print("="*60)
    cmd = [
        sys.executable, "judge_vh_response.py",
        "--keyscene_summary_file", args.keyscene_summary_file,
        "--output_file", args.vh_scores_file,
        "--vh_response_judge_model", args.vh_response_judge_model,
        "--vh_response_judge_thinking_level", str(args.vh_response_judge_thinking_level),
        "--query_source", args.query_source,
    ] + common_project_args
    subprocess.run(cmd, check=True)
    print(f"-> VH Response 점수 저장 완료: {args.vh_scores_file}")

    # ──────────────────────────────────────────────
    # B-3: 엑셀 리포트 내보내기
    # ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(">>> B-3. 엑셀 리포트 내보내기")
    print("="*60)
    subprocess.run([sys.executable, "export_to_excel.py"])

    print("\n\nEnd-to-End Pipeline Completed Successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] 파이프라인이 즉시 강제 종료됩니다.")
        import os
        os._exit(1)