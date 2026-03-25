import os
import time
import argparse
import json
import vertexai
from vertexai.generative_models import GenerativeModel
from gemini_api_utils import process_gcs_file, check_gcs_files_exist, SAFETY_SETTINGS, load_config, parse_json_response

def init_query_generator_model(model_name):
    system_prompt = """
    당신은 영상 콘텐츠와 정답 메타데이터(GT JSONL)를 분석하여 해당 영상을 시청한 일반 사용자가 던질 법한 질문들을 자연스럽게 생성하는 전문가입니다.
    첨부된 파일은 원본 비디오 영상과 영상의 핵심 정답(GT) 내용이 포함된 JSONL 파일입니다.

    [지시사항]
    1. 이 영상을 시청하는 일반인이 실제로 물어볼 만한 자연스러운 질문을 5~10개 생성해 주세요.
    2. 너무 지엽적이거나 세세한 디테일(예: 의미 없는 사물의 위치 등)만 묻는 질문은 지양하세요.
    3. 대신, 영상의 전체적인 흐름, 스토리, 사건의 맥락 등 '두루뭉술하고 포괄적인 질문(예: 여기 전체적으로 무슨 내용이야?, 분위기가 어때?)'을 절반 정도 포함하세요.
    4. 실제 시청자가 대충 물어보는 듯한 매우 캐주얼하고 격식 없는 구어체(예: "이번 영상 결말이 대체 뭐야?", "저 사람 왜 저러는 거야?")로 질문을 작성해 주세요.
    5. 비디오 분석 도중 시청자의 호기심을 자극할 만한 매력적인 물건(예: 예쁜 옷, 특정 연예인이 착용한 아이템, 전자기기)이나 장소(예: 유명한 관광지, 예쁜 식당)가 포착된다면, 그 정보에 대해 묻는 질문(예: "저기 나오는 예쁜 원피스 어디 브랜드야?", "저기 찍힌 바다 엄청 예쁜데 어디 관광지야?")을 1~2개 포함해 주세요.

    [출력 형태 예시]
    반드시 아래의 JSON 배열(Array of strings) 형식으로만 출력하세요. 다른 말은 절대 추가하지 마세요.
    [
        "이 영상 전체적으로 어떤 분위기야?",
        "주인공이 왜 저렇게 행동하는지 이유가 뭐야?",
        "마지막에 상황이 어떻게 마무리되는지 요약 좀 해줄래?",
        "영상 중간에 주인공이 입고 있는 저 자켓 디자인 특이한데 어디 거야?",
        "배경으로 나오는 저 야경 예쁜 곳 관광지 이름이 뭐야?"
    ]   
    """
    return GenerativeModel(
        model_name=model_name, 
        system_instruction=[system_prompt],
        safety_settings=SAFETY_SETTINGS
    )

def generate_queries_for_content(model, video_part, gt_part, max_retries=4, base_delay=3):
    prompt = f"제공된 영상 콘텐츠에 대해, 포괄적인 흐름을 묻는 질문과 구체적인 핵심을 묻는 질문을 섞어, 실제 시청자가 할 법한 질문 5-10개를 캐주얼한 어투로 생성해 주세요."
    contents = [video_part, gt_part, prompt]
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"      [Query Generation 마지막 시도 실패] {e}")
                raise e
            sleep_time = base_delay * (2 ** attempt)
            print(f"      [Query Generation 오류] {e}")
            print(f"      -> {sleep_time}초 후 재시도합니다... ({attempt+1}/{max_retries})")
            time.sleep(sleep_time)

def main():
    parser = argparse.ArgumentParser(description="Generate User Queries using Gemini Pro")
    parser.add_argument("--input_file", default="content_list.json", help="입력 JSON 파일 경로 (content_id 리스트)")
    parser.add_argument("--output_file", default="output/query_generated.jsonl", help="생성된 질문 목록을 저장할 파일 경로 (.jsonl)")
    parser.add_argument("--gcp_project_id", help="GCP 프로젝트 ID (기본값: config.json 사용)")
    parser.add_argument("--gs_bucket_name", help="GCS 버킷 이름 (기본값: config.json 사용)")
    parser.add_argument("--query_gen_model", default="gemini-2.5-pro", help="질문 생성에 사용할 모델명")
    parser.add_argument("--location", default="us-central1", help="GCP Location")
    
    args = parser.parse_args()
    
    args = load_config(args)

    project_id = args.gcp_project_id
    gs_bucket_name = args.gs_bucket_name
    
    if not project_id or not gs_bucket_name:
        print("Error: GCP Project ID 및 GCS 버킷 이름이 필요합니다. (config.json을 생성하세요)")
        return

    print(f"Initializing Gemini client for project: {project_id}, location: {args.location}...")
    vertexai.init(project=project_id, location=args.location)
    
    print(f"Initializing Query Generator Model ({args.query_gen_model})...")
    generator_model = init_query_generator_model(model_name=args.query_gen_model)
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다.")
        return  
        
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_list = json.load(f)
        
    output_list = []
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
            
            if not check_gcs_files_exist(gs_bucket_name, content_id):
                continue
                
            # Use the video and gt files
            print(f"Preparing GCS files (mode: video, gt) for {content_id}...")
            video_part = process_gcs_file(gs_bucket_name, content_id, mode="video")
            gt_part = process_gcs_file(gs_bucket_name, content_id, mode="gt")
            
            print("Generating queries...")
            try:
                time.sleep(2) # API Rate Limit 방지
                response_text = generate_queries_for_content(generator_model, video_part, gt_part)
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
