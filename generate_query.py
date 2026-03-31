import os
import time
import argparse
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from gemini_api_utils import (
    process_gcs_file, check_gcs_files_exist, SAFETY_SETTINGS,
    load_config, parse_json_response, _retry_api_call,
)

def init_query_generator_model(model_name):
    system_prompt = """
    당신은 영상 콘텐츠와 메타데이터를 분석하여 인간 시청자들이 남길 법한 자연스럽고 현실적인 질문들을 생성하는 데이터 생성 전문가입니다.
    사용자는 원본 비디오 프레임과 그 내용이 요약된 Reference 메타데이터(JSONL)를 함께 제공합니다.

    [목표]
    영상을 주의 깊게 시청한 일반인이 실제로 궁금해할 만한 자연스러운 질문을 5~10개 생성하세요.

    [질문 카테고리 가이드]
    생성하는 질문들은 아래 4가지 유형이 고르게 섞여야 합니다.
    1. 포괄적/전체적 질문 (Global Context): 영상의 전체 흐름, 분위기, 스토리 요약을 묻는 내용
       - 예) "여기 전체적으로 무슨 내용이야?", "이번 영상 결말이 대체 뭐야?"
    2. 핵심 사건/장면 (Core Events): 극의 전개상 중요한 특정 사건이나 인물의 행동 이유를 묻는 내용
       - 예) "주인공이 아까 왜 갑자기 화낸 거야?", "영상 초반에 둘이 왜 싸운 거야?"
    3. 세부 디테일 (Fine-Grained Details): 영상에 짧게 스쳐 지나가는 글귀, 소품, 옷차림 등 지엽적인 디테일을 확인하는 내용
       - 예) "아까 칠판에 적혀 있던 글씨 뭐라고 쓰여 있는 거임?", "주인공 책상 위에 엎어져 있던 책 제목이 뭐야?"
    4. 정보 탐색 (Information Search): 영상 내 등장하는 물건, 장소, 브랜드 등 호기심을 자극하는 실물 정보
       - 예) "저기 찍힌 바다 엄청 예쁜데 어디 관광지야?", "영상 중간에 주인공이 입고 있는 저 자켓 디자인 특이한데 어디 거야?"

    [작성 규칙]
    - 어투: 실제 시청자가 인터넷 커뮤니티나 친구에게 대충 물어보는 듯한 매우 캐주얼하고 격식 없는 구어체(반말 위주)를 사용하세요. AI가 번역한 듯한 딱딱한 문어체는 피하십시오.
    - 양식: 질문 외의 다른 서론이나 설명은 절대 추가하지 말고, 아래 예시와 같은 순수 JSON 배열(Array) 형식으로만 출력하세요.

    [출력 형태 예시]
    [
        "이 영상 전체적으로 어떤 분위기야?",
        "주인공이 왜 저렇게 행동하는지 이유가 뭐야?",
        "마지막에 상황이 어떻게 마무리되는지 요약 좀 해줄래?",
        "영상 중간에 주인공이 입고 있는 저 자켓 어디 거야?",
        "배경으로 나오는 저 야경 예쁜 곳 관광지 이름이 뭐야?"
    ]
    """
    return GenerativeModel(
        model_name=model_name, 
        system_instruction=[system_prompt],
        safety_settings=SAFETY_SETTINGS
    )

def generate_queries_for_content(model, video_part, ref_part):
    prompt = (
        "제공된 비디오와 Reference 메타데이터를 바탕으로, "
        "포괄적인 질문, 사건 중심 질문, 세부 디테일 질문, 정보 탐색형 질문 등 4가지 유형을 골고루 섞어서 "
        "구어체의 캐주얼한 질문 5~10개를 생성해 주세요."
    )
    contents = [video_part, ref_part, prompt]
    return _retry_api_call(
        lambda: model.generate_content(contents).text,
        label="Query Generation",
    )

def main():
    parser = argparse.ArgumentParser(description="Generate User Queries using Gemini Pro")
    parser.add_argument("--input_file", default="content_list.json", help="입력 JSON 파일 경로 (content_id 리스트)")
    parser.add_argument("--output_file", default="assets/query_generated.jsonl", help="생성된 질문 목록을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--query_gen_model", default="gemini-2.5-pro", help="질문 생성에 사용할 모델명")
    parser.add_argument("--location", default="global", help="GCP Location")
    
    args = parser.parse_args()
    
    args = load_config(args)

    if not args.gcp_project_id or not args.gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (config.json을 생성하세요)")
        return

    print(f"Initializing Gemini client for project: {args.gcp_project_id}, location: {args.location}...")
    vertexai.init(project=args.gcp_project_id, location=args.location)
    
    print(f"Initializing Query Generator Model ({args.query_gen_model})...")
    generator_model = init_query_generator_model(model_name=args.query_gen_model)
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다.")
        return  
        
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_list = json.load(f)
        
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        processed_ids.add(item["content_id"])
                    except json.JSONDecodeError:
                        pass
        print(f"[{len(processed_ids)}] 개의 콘텐츠가 이미 {args.output_file}에 존재하여 건너뜁니다.")
        
    # 출력 폴더 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*50)
    print("사용자 질문 자동 생성 프로세스를 시작합니다.")
    print("="*50)

    try:
        for item in input_list:
            # Handle both simple string list and list of objects
            content_id = item if isinstance(item, str) else item.get("content_id")
            if not content_id:
                continue
            if content_id in processed_ids:
                continue
                
            print(f"\nProcessing Content: '{content_id}'")
            
            if not check_gcs_files_exist(args.gs_bucket_name, content_id):
                continue
                
            # Use the video and ref files
            print(f"Preparing GCS files (mode: video, ref) for {content_id}...")
            video_part = process_gcs_file(args.gs_bucket_name, content_id, mode="video")
            ref_part = process_gcs_file(args.gs_bucket_name, content_id, mode="ref")
            
            print("Generating queries...")
            try:
                time.sleep(2) # API Rate Limit 방지
                response_text = generate_queries_for_content(generator_model, video_part, ref_part)
                generated_queries = parse_json_response(response_text)
                print(f"Successfully generated {len(generated_queries)} queries for '{content_id}':")
                
                for i, q in enumerate(generated_queries, 1):
                    print(f"  {i}. {q}")
                
                if generated_queries:
                    output_entry = {
                        "content_id": content_id,
                        "queries": generated_queries
                    }
                    processed_ids.add(content_id)
                    # JSONL Append
                    with open(args.output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  [{content_id}] 질문 생성 최종 실패로 건너뜁니다: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        
    print("\n" + "="*50)
    print(f"모든 콘텐츠의 질문 생성이 완료되었습니다. 저장 경로: {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
