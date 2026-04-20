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
    _retry_api_call, retry_parse_json,
    ensure_output_dir, load_processed_pairs,
    init_pipeline, load_jsonl, append_jsonl,
)

# ───────────────────────────────────────────────
# Voice Hint Judge 프롬프트 (Text-based)
# ───────────────────────────────────────────────

_QUERY_JUDGE_PROMPT = """당신은 스마트 TV 플랫폼의 '개인화된 인터랙티브 시청 경험'을 기획하고 AI 생성 질문의 비즈니스 가치를 평가하는 최고 수준의 CX(Customer Experience) 전문가이자 프로덕트 매니저입니다.
이 예상 질문들의 궁극적인 비즈니스 목표는 수동적인 시청자를 자극하여 **1) 리모콘 조작(상호작용)을 즉각 유도**하고, **2) 스마트폰 검색으로 인한 시선 이탈을 방지**하며, **3) TV 플랫폼 내의 추가적인 콘텐츠 탐색(체류 시간 증대)으로 연결**하는 것입니다.
평가 시에는 해당 질문이 생성된 시점의 맥락을 담은 **[Reference Scene Summary (과거 요약 및 현재 장면 묘사)]**가 참고용으로 제공됩니다. 제공되는 텍스트 요약을 기반으로, 아래 2가지 항목에 대해 각 1~5점으로 평가하세요.

[평가 항목]
1. 시점 몰입 및 현재 답변 가능성 (Temporal Immersion & Answerability): 질문의 타이밍이 **[현재 장면]**과 완벽히 동기화되며, 플랫폼 시스템이 '현재 시점' 기준에서 즉시 답변해 줄 수 있는 정보(미래 전개 제외)로 국한되었는가?
   - 5점: [현재 장면]의 특정 시각적/청각적 단서를 기반으로, "결과가 어떨까요?" 같은 미래 추측이 완벽히 배제된 채 오직 즉시 파악할 수 있는 사물의 정체나 숨어있는 배경지식만을 묻는 타이밍 완벽한 질문.
   - 3점: 현재 장면의 정보이긴 하나, 단서가 너무 흐릿하거나 질문의 초점이 다소 포괄적이라 즉각적인 호기심 해소를 유도하기엔 매력이 떨어지는 질문.
   - 1점: "과연 어떻게 될까요?", "결과는 어떨까요?" 처럼 영상을 계속 봐야만 알 수 있는 스토리를 묻는 **시스템이 즉답 불가한** '미래 추측 질문'이거나, 혹은 반대로 Reference의 [과거 장면 요약]에 이미 밝혀진 사실을 또 묻는 '뒷북 질문'인 경우.

2. 호기심 및 상호작용 유도력 (Curiosity & Hook): 질문이 시청자의 흥미를 강렬하게 자극하여 당장 리모컨을 눌러 '답변'을 확인하고 싶게 만드는가?
   - 5점: 시청자가 무의식적으로 속으로 품었을 법한 포인트(가려운 곳)를 정확히 짚어내며, '해당 장르/컨텐츠의 전문 평론가'가 말을 건네듯 정중하고 세련된 톤으로 호기심을 부드럽게 자극하는 매력적인 질문.
   - 3점: 영상 내용과 관련된 무난한 질문이나, 굳이 상호작용을 하면서까지 답을 확인하고 싶은 강력한 동기 부여는 부족함.
   - 1점: 이미 영상 속에서 설명되었거나, 영상을 시청하지 않아도 일반적인 상식으로 답을 쉽게 알 수 있는 뻔한 질문."""

_QUERY_JUDGE_FORMAT_PROMPT = """[출력 형식]
반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명은 덧붙이지 마십시오.
각 평가 항목에 대해 평가 논리(rationale)를 먼저 작성한 후 점수(score)를 매기세요.
{
    "temporal_immersion": {
        "rationale": "<항목 1. 시점 몰입도 에 대한 구체적인 평가 이유>",
        "score": <1~5 사이의 정수>
    },
    "curiosity_and_hook": {
        "rationale": "<항목 2. 호기심 및 상호작용 유도력 에 대한 구체적인 평가 이유>",
        "score": <1~5 사이의 정수>
    }
}"""

def make_query_judge_config(thinking_level=None):
    return make_generate_config(system_instruction=_QUERY_JUDGE_PROMPT, thinking_level=thinking_level)


def evaluate_query(client, model_name, judge_config, detailed_summary, query_text):
    """생성된 질문을 KeyScene Summary 기반으로 평가합니다."""
    contents = []
    if detailed_summary:
        contents += ["--- [Reference Scene Summary (과거 요약 및 현재 장면 묘사)] ---", detailed_summary]
    contents += [
        f"[평가 대상 질문]\n{query_text}\n\n",
        "위 [Reference Scene Summary] 와 앞서 정의된 [평가 항목]을 기준으로, 평가 대상 질문의 품질을 평가하세요.\n\n" + _QUERY_JUDGE_FORMAT_PROMPT
    ]

    return _retry_api_call(
        lambda: client.models.generate_content(
            model=model_name, contents=contents, config=judge_config
        ).text,
        label="Query Judge API",
    )

def judge_one(q_item, content_id, scene_idx, detailed_summary,
              client, args, judge_config, file_write_lock):
    query_text = q_item["query"]
    if not detailed_summary:
        print(f"[Warning] Scene {scene_idx}에 KeyScene Summary가 없습니다. 스킵합니다.")
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

        _SCORE_KEYS = ["temporal_immersion", "curiosity_and_hook"]
        total = sum(
            (score_dict.get(k, {}).get("score", 0) if isinstance(score_dict.get(k), dict) else 0)
            for k in _SCORE_KEYS
        ) if score_dict else 0

        score_record = {
            "content_id": content_id,
            "scene_idx": scene_idx,
            "mode": q_item.get("mode"),
            "query": query_text,
            "judge": score_dict,
            "total_score": total,
        }

        _ITEM_LABELS = [
            ("temporal_immersion", "시점 몰입도"),
            ("curiosity_and_hook", "호기심 및 상호작용 유도력"),
        ]
        print(f"\nQuery ({score_record['mode']}): {query_text}")
        if score_dict:
            print("[Rationale]")
            for i, (key, label) in enumerate(_ITEM_LABELS, 1):
                item = score_dict.get(key, {})
                rationale = item.get("rationale", "N/A") if isinstance(item, dict) else "N/A"
                score = item.get("score", "N/A") if isinstance(item, dict) else "N/A"
                print(f"{i}. {label}({score}점): {rationale}")
        print(f"-> Total Score: {total}/10")

        append_jsonl(args.scores_file, score_record, lock=file_write_lock)

    except Exception as e:
        print(f"[Error] Judge 최종 실패: {e}")

def main():
    parser = get_common_argparser(description="Voice Hint 질문을 KeyScene Summary 기반으로 품질 평가")
    parser.add_argument("--input_file", default="assets/voice_hint.jsonl", help="Voice Hint 질문 목록 JSONL 경로")
    parser.add_argument("--kss_file", default="assets/keyscene_summary.jsonl", help="KeyScene Summary JSONL 경로")
    parser.add_argument("--scores_file", default="assets/voice_hint_scores.jsonl", help="Voice Hint 질문별 Judge 점수 저장 경로")
    parser.add_argument("--watch", action="store_true", help="파일을 계속 모니터링하여 새 질문을 실시간으로 평가합니다.")
    parser.add_argument("--modes", nargs="+", default=["img_desc", "mm_desc", "kss"], choices=["img_desc", "mm_desc", "kss"], help="평가할 모드 직접 지정 (기본값: 모두 평가)")
    

    args, client = init_pipeline(parser.parse_args())
    judge_config = make_query_judge_config(thinking_level=args.vh_judge_thinking_level)

    ensure_output_dir(args.scores_file)

    processed_pairs = load_processed_pairs(args.scores_file)
    if processed_pairs:
        print(f"[{len(processed_pairs)}] 개의 (content_id, query) 쌍이 이미 처리됨.")

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다. 먼저 generate_voice_hint.py를 실행하세요.")
        return

    # Summary 맵 로드: (content_id, scene_idx) -> summary_text
    summary_map = {}
    for rec in load_jsonl(args.kss_file):
        key = (rec.get("content_id"), rec.get("scene_idx"))
        if key[0] and key[1] is not None:
            summary_map[key] = rec.get("summary", "")
    if summary_map:
        print(f"[Summary] {len(summary_map)}개 Scene의 Summary 로드됨 ({args.kss_file})")
    elif os.path.exists(args.kss_file):
        pass  # 파일은 있지만 비어 있음
    else:
        print(f"[Warning] Summary 파일을 찾을 수 없습니다: {args.kss_file}")

    print(f"\n{'='*50}")
    print(f"Voice Hint 질문 품질 평가 프로세스 시작 (Watch 모드: {args.watch})")
    print(f"{'='*50}")

    file_write_lock = threading.Lock()
    last_position = 0
    pipeline_done = False
    
    total_generated = 0
    total_evaluated = len(processed_pairs)

    try:
        while True:
            if not os.path.exists(args.input_file):
                if not args.watch:
                    print(f"Error: {args.input_file} 파일이 존재하지 않습니다.")
                    return
                time.sleep(2)
                continue
                
            with open(args.input_file, "r", encoding="utf-8") as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
                
            new_items = []
            for line in new_lines:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        if obj.get("pipeline_done"):
                            pipeline_done = True
                        else:
                            new_items.append(obj)
                    except json.JSONDecodeError:
                        pass
                        
            if new_items:
                for scene_item in new_items:
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
                        if mode not in args.modes:
                            continue
                        for q_text in group.get("queries", []):
                            flat_queries.append({"mode": mode, "query": q_text})

                    total_generated += len(flat_queries)
                    pending = [q for q in flat_queries if (content_id, q["query"]) not in processed_pairs]
                    
                    if not pending:
                        continue

                    start_time = float(scene_item.get("start_time", 0.0))
                    end_time = float(scene_item.get("end_time", 0.0))
                    print(f"\nEvaluating '{content_id}' Scene {scene_idx} | Range=[{start_time:.1f}s ~ {end_time:.1f}s] ({len(pending)}개 질문)")

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
                            total_evaluated += 1
                            
                    processed_pairs.update((content_id, q["query"]) for q in pending)

                # TODO List 현황 출력 (한 사이클 처리 후)
                pending_count = total_generated - total_evaluated
                print(f"\n▶ [Judging TODO List] Total Generated: {total_generated} | Evaluated: {total_evaluated} | Pending: {pending_count}")

            if args.watch:
                if pipeline_done:
                    print("\n[Watch] Generation 파이프라인의 종료 시그널(pipeline_done)을 감지했습니다. 모든 처리를 완료하고 종료합니다.")
                    break
                time.sleep(3)
            else:
                break

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        os._exit(1)

    print(f"\n{'='*50}")
    print(f"Voice Hint 질문 평가 완료. 점수 기록: {args.scores_file}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
