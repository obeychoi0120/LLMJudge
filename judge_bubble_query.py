import os
import argparse
import json
import time
import concurrent.futures
import threading
from gemini_api_utils import (
    create_client, make_generate_config,
    load_config, parse_json_response,
    _retry_api_call, start_chat_session,
    ensure_output_dir, load_processed_pairs,
)

# ───────────────────────────────────────────────
# Bubble Query Judge 프롬프트 (Text-based)
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

def judge_one(q_item):
    query_text = q_item["query"]
    detailed_summary = q_item.get("detailed_summary", "")
    scene_idx = q_item.get("scene_idx", -1)

    print(f"  Judging: \"{query_text[:40]}...\"")

    if not detailed_summary:
        print(f"    [Warning] 이 질문에는 detailed_summary가 없습니다. 스킵합니다.")
        return

    try:
        time.sleep(1)

        max_parse_retries = 3
        score_dict = None
        for attempt in range(max_parse_retries):
            try:
                score_text = evaluate_query(
                    client, args.bq_judge_model, judge_config,
                    detailed_summary, query_text
                )
                score_dict = parse_json_response(score_text)
                break
            except json.JSONDecodeError:
                print(f"    [Warning] JSON 파싱 실패 ({attempt+1}/{max_parse_retries}), 재시도...")
                time.sleep(2)
            except Exception as e:
                print(f"    [Error] Judge 실패: {e}")
                break

        total = score_dict.get("total_score", 0) if score_dict else 0

        score_record = {
            "content_id": content_id,
            "query": query_text,
            "scene_idx": scene_idx,
            "start_time": q_item.get("start_time"),
            "end_time": q_item.get("end_time"),
            "judge": score_dict,
        }

        print(f"    -> Score: {total}/15 | {score_dict.get('rationale', '')[:100] if score_dict else 'N/A'}")

        with file_write_lock:
            with open(args.scores_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(score_record, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"    [Error] Judge 최종 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description="Bubble Query 질문을 텍스트 요약 기반으로 품질 평가")
    parser.add_argument("--input_file", default="assets/bubble_query.jsonl", help="Bubble Query 질문 목록 JSONL 경로")
    parser.add_argument("--scores_file", default="assets/bubble_query_scores.jsonl", help="Bubble Query 질문별 Judge 점수 저장 경로")
    
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--bq_judge_model", default="gemini-2.5-pro", help="질문 평가에 사용할 Premium 모델명")
    parser.add_argument("--location", default="global", help="GCP Location")
    parser.add_argument("--bq_judge_thinking_budget", type=int, default=2048,
                        help="BQ Judge 모델의 Thinking Budget (0=비활성화, -1=동적, 1~24576=지정 토큰 수)")

    args = parser.parse_args()
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다.")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}")
    client = create_client(args.gcp_project_id, args.location)
    judge_config = make_query_judge_config(thinking_budget=args.bq_judge_thinking_budget)

    ensure_output_dir(args.scores_file)

    processed_pairs = load_processed_pairs(args.scores_file)
    if processed_pairs:
        print(f"[{len(processed_pairs)}] 개의 (content_id, query) 쌍이 이미 처리됨.")

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_bubble_query.py를 실행하세요.")
        return

    content_list = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                content_list.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"\n{'='*50}")
    print(f"Bubble Query 질문 품질 평가 프로세스 (Text-only) 시작")
    print(f"{'='*50}")

    file_write_lock = threading.Lock()

    try:
        for content_item in content_list:
            content_id = content_item.get("content_id")
            queries = content_item.get("queries", [])

            if not content_id or not queries:
                continue

            print(f"\nEvaluating Content: '{content_id}'")

            pending = [q for q in queries if isinstance(q, dict) and (content_id, q["query"]) not in processed_pairs]
            if not pending:
                print(f"  -> 모든 질문이 이미 평가됨. Skip.")
                continue

            print(f"  -> {len(pending)}개 질문 평가 예정")

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(judge_one, q) for q in pending]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

            scored_count = len([q for q in pending if (content_id, q["query"]) not in processed_pairs])
            print(f"\n  -> '{content_id}': {scored_count}개 질문 평가 완료")

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")

    print(f"\n{'='*50}")
    print(f"Bubble Query 질문 평가 완료. 점수 기록: {args.scores_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
