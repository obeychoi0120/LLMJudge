import os
import argparse
import json
import time
import concurrent.futures
import threading
from gemini_api_utils import (
    make_generate_config,
    parse_json_response,
    _retry_api_call, retry_parse_json, start_chat_session,
    ensure_output_dir, load_processed_pairs,
    init_pipeline, load_jsonl, append_jsonl,
)

# ───────────────────────────────────────────────
# Voice Hint Judge 프롬프트 (Text-based)
# ───────────────────────────────────────────────

_QUERY_JUDGE_PROMPT = """\
당신은 AI가 자동 생성한 '현재 장면 중심의 질문' 품질을 평가하는 전문 평가자입니다.
해당 질문은 영상의 특정 시점을 보고 있는 시청자가 남길 법한 질문으로 설계되었습니다.

평가 시에는 해당 질문을 생성할 당시 영상 맥락 전체를 아우르는 **상세 요약(Detailed Summary)** 이 Reference로 제공됩니다.
비디오 원본 대신 제공되는 이 텍스트 요약을 충실히 바탕으로, 아래 3가지 항목에 대해 각 1~5점으로 평가하세요.

[평가 항목]
1. 자연스러움 (Naturalness): 실제 시청자가 해당 장면에서 인터넷 커뮤니티나 친구에게
   물어볼 법한 캐주얼한 구어체 중심의 질문인가?
   - 5점: 매우 자연스럽고 현실적인 질문
   - 3점: 다소 어색하지만 억지스럽지는 않음
   - 1점: 인위적이거나 비현실적인 질문

2. 시점 적합성 (Temporal Relevance): 질문이 미래 내용(스포일러)을 미리 알고 있거나 전제하지 않고,
   오로지 제공된 현재 맥락 내에서 발생할 수 있는 합당한 탐색적 질문인가?
   - 5점: 해당 시점까지 겪은 내용이나 화면을 근거로 자연히 할 수 있는 질문
   - 3점: 다소 조기 질문이지만 큰 문제 없음
   - 1점: 해당 시점에서는 알 수 없는 내용을 전제한 질문

3. 난이도 (Difficulty): 
   단순히 구글링으로 해결되거나 영상 요소(등장인물 성향, 구체적 사물, 진행 상황)와 완전히 무관하여 너무 명백한 질문은 낮게 평가하세요.
   - 5점: 영상을 어느 정도 관찰해야만 의미가 있는 호기심
   - 3점: 무난한 영상 기반 질문
   - 1점: 영상 없이도 쉽게 답할 수 있거나 내용과 유리된 질문\
"""

_QUERY_JUDGE_FORMAT_PROMPT = """\
[출력 형식]
반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명은 덧붙이지 마십시오.
{
    "rationale": "<각 점수를 부여한 논리적인 이유>",
    "scores": {
        "naturalness": <1~5 사이의 정수>,
        "temporal_relevance": <1~5 사이의 정수>,
        "difficulty": <1~5 사이의 정수>
    },
    "total_score": <세 항목 점수의 합계, 최대 15점>
}\
"""


def make_query_judge_config(thinking_budget=None):
    return make_generate_config(system_instruction=_QUERY_JUDGE_PROMPT, thinking_budget=thinking_budget)


def evaluate_query(client, model_name, judge_config, detailed_summary, query_text):
    """생성된 질문 하나를 텍스트 요약본 기준으로 평가합니다."""
    user_content = (
        f"[평가 대상 질문]\n{query_text}\n\n"
        f"[Reference: 상세 요약(Detailed Summary)]\n{detailed_summary}\n\n"
        f"위 상세 요약을 근거로, 평가 대상 질문의 품질을 평가하세요.\n\n"
        f"{_QUERY_JUDGE_FORMAT_PROMPT}"
    )

    judge_chat = start_chat_session(client, model_name, judge_config)
    return _retry_api_call(
        lambda: judge_chat.send_message(user_content).text,
        label="Query Judge API (Text)",
    )

def judge_one(q_item, content_id, scene_idx, detailed_summary,
              client, args, judge_config, file_write_lock):
    query_text = q_item["query"]

    print(f"  Judging: \"{query_text[:40]}...\"")

    if not detailed_summary:
        print(f"    [Warning] Scene {scene_idx}에 Summary가 없습니다. 스킵합니다.")
        return

    try:
        time.sleep(1)

        score_dict = retry_parse_json(
            lambda: evaluate_query(
                client, args.vh_judge_model, judge_config,
                detailed_summary, query_text
            ),
            label=f"VH Judge (Scene {scene_idx})",
        )

        total = score_dict.get("total_score", 0) if score_dict else 0

        score_record = {
            "content_id": content_id,
            "scene_idx": scene_idx,
            "mode": q_item.get("mode"),
            "query": query_text,
            "judge": score_dict,
        }

        print(f"    -> Score: {total}/15 | {score_dict.get('rationale', '')[:100] if score_dict else 'N/A'}")

        append_jsonl(args.scores_file, score_record, lock=file_write_lock)

    except Exception as e:
        print(f"    [Error] Judge 최종 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="Voice Hint 질문을 텍스트 요약 기반으로 품질 평가")
    parser.add_argument("--input_file", default="assets/voice_hint.jsonl", help="Voice Hint 질문 목록 JSONL 경로")
    parser.add_argument("--summary_file", default="assets/vh_summary.jsonl", help="Detailed Summary JSONL 경로")
    parser.add_argument("--scores_file", default="assets/voice_hint_scores.jsonl", help="Voice Hint 질문별 Judge 점수 저장 경로")
    
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--vh_judge_model", default="gemini-2.5-pro", help="질문 평가에 사용할 Premium 모델명")
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--vh_judge_thinking_budget", type=int, default=2048, help="Voice Hint Judge 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args, client = init_pipeline(parser.parse_args())
    judge_config = make_query_judge_config(thinking_budget=args.vh_judge_thinking_budget)

    ensure_output_dir(args.scores_file)

    processed_pairs = load_processed_pairs(args.scores_file)
    if processed_pairs:
        print(f"[{len(processed_pairs)}] 개의 (content_id, query) 쌍이 이미 처리됨.")

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_voice_hint.py를 실행하세요.")
        return

    # Summary 맵 로드: (content_id, scene_idx) -> summary_text
    summary_map = {}
    for rec in load_jsonl(args.summary_file):
        key = (rec.get("content_id"), rec.get("scene_idx"))
        if key[0] and key[1] is not None:
            summary_map[key] = rec.get("summary", "")
    if summary_map:
        print(f"[Summary] {len(summary_map)}개 Scene의 Summary 로드됨 ({args.summary_file})")
    elif os.path.exists(args.summary_file):
        pass  # 파일은 있지만 비어 있음
    else:
        print(f"[Warning] Summary 파일을 찾을 수 없습니다: {args.summary_file}")

    # 입력: 각 줄이 scene 단위 레코드 {content_id, scene_idx, queries: [...]}
    scene_list = load_jsonl(args.input_file)

    print(f"\n{'='*50}")
    print(f"Voice Hint 질문 품질 평가 프로세스 (Text-only) 시작")
    print(f"{'='*50}")

    file_write_lock = threading.Lock()

    try:
        for scene_item in scene_list:
            content_id = scene_item.get("content_id")
            scene_idx  = scene_item.get("scene_idx")
            query_groups = scene_item.get("queries", [])

            if not content_id or scene_idx is None or not query_groups:
                continue

            detailed_summary = summary_map.get((content_id, scene_idx), "")

            # 그룹화된 포맷을 펼쳐서 (mode, query) 개별 항목 리스트로 변환
            flat_queries = []
            for group in query_groups:
                mode = group.get("mode", "")
                for q_text in group.get("queries", []):
                    flat_queries.append({"mode": mode, "query": q_text})

            pending = [q for q in flat_queries
                       if (content_id, q["query"]) not in processed_pairs]
            if not pending:
                print(f"\n[Skip] '{content_id}' Scene {scene_idx}: 모든 질문 평가 완료")
                continue

            print(f"\nEvaluating '{content_id}' Scene {scene_idx} ({len(pending)}개 질문)")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        judge_one, q, content_id, scene_idx, detailed_summary,
                        client, args, judge_config, file_write_lock
                    )
                    for q in pending
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    print(f"\n{'='*50}")
    print(f"Voice Hint 질문 평가 완료. 점수 기록: {args.scores_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
