import os
import argparse
import json
import time
import concurrent.futures
import threading
from gemini_api_utils import (
    get_common_argparser,
    make_generate_config,
    parse_json_response,
    _retry_api_call, retry_parse_json, start_chat_session,
    ensure_output_dir, load_processed_pairs,
    init_pipeline, load_jsonl, append_jsonl,
)

# ───────────────────────────────────────────────
# Voice Hint Judge 프롬프트 (Text-based)
# ───────────────────────────────────────────────

_QUERY_JUDGE_PROMPT = """당신은 스마트 TV 플랫폼의 '개인화된 인터랙티브 시청 경험'을 기획하고 AI 생성 질문의 비즈니스 가치를 평가하는 최고 수준의 CX(Customer Experience) 전문가이자 프로덕트 매니저입니다.

이 예상 질문들의 궁극적인 비즈니스 목표는 수동적인 시청자를 자극하여 **1) 리모컨 조작(상호작용)을 즉각 유도**하고, **2) 스마트폰 검색으로 인한 시선 이탈을 방지**하며, **3) TV 플랫폼 내의 추가적인 콘텐츠 탐색(체류 시간 증대)으로 연결**하는 것입니다.

평가 시에는 해당 질문이 생성된 시점의 맥락을 담은 **[1. 과거 장면 요약]**과 **[2. 현재 장면]**이 Reference로 제공됩니다. 비디오 원본 대신 제공되는 이 텍스트 요약을 바탕으로, 아래 4가지 항목에 대해 각 1~5점으로 평가하세요.

[평가 항목]
1. 호기심 및 상호작용 유도력 (Curiosity & Hook): 질문이 시청자의 흥미를 강렬하게 자극하여 당장 리모컨을 눌러 '답변'을 확인하고 싶게 만드는가?
   - 5점: 시청자가 무의식적으로 속으로 품었을 법한 포인트(가려운 곳)를 정확히 짚어내어, "나도 방금 저거 궁금했는데!"라는 공감과 함께 무조건 클릭하게 만드는 매력적인 질문.
   - 3점: 영상 내용과 관련된 무난한 질문이나, 굳이 상호작용을 하면서까지 답을 확인하고 싶은 강력한 동기 부여는 부족함.
   - 1점: 이미 답을 알 수 있는 뻔한 사실을 묻거나, 다큐멘터리 해설처럼 지나치게 딱딱하고 학술적이어서 시청자를 피로하게 만드는 질문.

2. 시점 몰입도 (Temporal Immersion): 질문의 타이밍이 **[2. 현재 장면]**과 완벽히 동기화되어 극의 감정선과 시청 몰입을 방해하지 않는가?
   - 5점: [2. 현재 장면]의 특정 시각적/청각적 단서를 보는 바로 그 순간 화면에 떴을 때, 씬의 분위기를 깨지 않고 자연스럽게 스며드는 완벽한 타이밍의 질문.
   - 3점: 현재 장면의 맥락이긴 하나 타이밍이 살짝 지연되었거나, 질문의 초점이 다소 포괄적임.
   - 1점: 앞선 [1. 과거 장면 요약]만 묻는 '뒷북' 질문이거나, 긴장감 넘치는 씬에서 흐름을 깨거나, 추후 전개될 내용을 암시하는 스포일러성 질문.

3. 플랫폼 체류 확장성 (Platform Extensibility): 이 질문에 대한 답변이 플랫폼 내 추가 탐색(관련 VOD 추천, 배우/촬영지 정보, OST 검색, 세계관 지식 등)으로 이어져 체류 시간을 늘릴 잠재력이 있는가?
   - 5점: 답변이 단순 예/아니오로 끝나지 않고, 시청자가 관련된 배경지식이나 타 작품 등 부가적인 TV 생태계 콘텐츠를 더 소비하게 만들어 스마트폰 이탈을 완벽히 방어하는 확장성 높은 질문.
   - 3점: 흥미로운 정보를 주지만, 해당 지식을 얻는 것에서 상호작용이 종료될 가능성이 큰 1차원적인 질문.
   - 1점: 시청자의 시선을 오히려 스마트폰 구글링으로 이탈하게 만들거나, 체류 시간 연장에 전혀 도움이 되지 않는 질문.

4. UI 적합성 및 대화형 간결성 (UI Fitness & Conciseness): TV 화면에서 멀리서 보아도 한눈에 직관적으로 읽히며, 옆 사람에게 무심코 툭 던질 법한 자연스러운 구어체인가? (인물/고유명사 원어 병기 허용)
   - 5점: 텍스트가 군더더기 없이 간결하여 직관적이며(TV 화면 UI 최적화), 기계적인 느낌 없이 입에 착 붙는 캐주얼하고 친근한 대화체.
   - 3점: 어투는 자연스러우나 질문 문장이 다소 길고 복잡하여 읽고 이해하는 데 시간이 약간 소요됨.
   - 1점: 기계 번역투이거나 작위적인 문어체로 작성되어, 화면에 떴을 때 텍스트 공해(Spam)처럼 느껴지는 질문.
"""

_QUERY_JUDGE_FORMAT_PROMPT = """[출력 형식]
반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명은 덧붙이지 마십시오.
평가 점수를 매기기 전에, 'rationale' 객체 내에 각 항목에 대한 평가 논리를 먼저 작성하세요.
{
    "rationale": {
        "hook_and_curiosity": "<항목 1. 호기심 및 상호작용 유도력 에 대한 구체적인 평가 이유>",
        "extensibility": "<항목 3. 플랫폼 체류 확장성 에 대한 구체적인 평가 이유>",
        "overall_business_value": "<이 질문이 최종적으로 플랫폼 체류 시간 증대와 이탈 방어에 어떻게 기여하는지 종합 의견>"
    },
    "scores": {
        "curiosity_and_hook": <1~5 사이의 정수>,
        "temporal_immersion": <1~5 사이의 정수>,
        "platform_extensibility": <1~5 사이의 정수>,
        "ui_fitness_and_conciseness": <1~5 사이의 정수>
    },
    "total_score": <네 항목 점수의 합계, 최소 4점 ~ 최대 20점>
}"""

def make_query_judge_config(thinking_budget=None):
    return make_generate_config(system_instruction=_QUERY_JUDGE_PROMPT, thinking_budget=thinking_budget)


def evaluate_query(client, model_name, judge_config, detailed_summary, query_text):
    """생성된 질문 하나를 텍스트 요약본 기준으로 평가합니다."""
    user_content = (
        f"[평가 대상 질문]\n{query_text}\n\n"
        f"[Reference: 상세 요약]\n{detailed_summary}\n\n"
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
    if not detailed_summary:
        print(f"[Warning] Scene {scene_idx}에 Summary가 없습니다. 스킵합니다.")
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

        print(f"\nQuery ({score_record['mode']}): {query_text}\n-> Score: {total}/20 | {score_dict.get('rationale', '') if score_dict else 'N/A'}")

        append_jsonl(args.scores_file, score_record, lock=file_write_lock)

    except Exception as e:
        print(f"[Error] Judge 최종 실패: {e}")

def main():
    parser = get_common_argparser(description="Voice Hint 질문을 텍스트 요약 기반으로 품질 평가")
    parser.add_argument("--input_file", default="assets/voice_hint.jsonl", help="Voice Hint 질문 목록 JSONL 경로")
    parser.add_argument("--keyscene_summary_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--scores_file", default="assets/voice_hint_scores.jsonl", help="Voice Hint 질문별 Judge 점수 저장 경로")
    

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
    for rec in load_jsonl(args.keyscene_summary_file):
        key = (rec.get("content_id"), rec.get("scene_idx"))
        if key[0] and key[1] is not None:
            summary_map[key] = rec.get("summary", "")
    if summary_map:
        print(f"[Summary] {len(summary_map)}개 Scene의 Summary 로드됨 ({args.keyscene_summary_file})")
    elif os.path.exists(args.keyscene_summary_file):
        pass  # 파일은 있지만 비어 있음
    else:
        print(f"[Warning] Summary 파일을 찾을 수 없습니다: {args.keyscene_summary_file}")

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
