import os
import argparse
import json
from vertexai.generative_models import GenerativeModel
from run_gemini_cli import init_gemini_client, process_gcs_file, check_gcs_files_exist

def init_query_generator_model(model_name):
    system_prompt = """
    당신은 영상 콘텐츠와 정답 메타데이터(GT JSONL)를 분석하여 해당 영상을 시청한 일반 사용자가 궁금해할 만한 핵심적이고 흥미로운 질문들을 생성하는 전문가입니다.
    첨부된 파일은 원본 비디오 영상과 영상의 핵심 정답(GT) 내용이 포함된 JSONL 파일입니다.

    [지시사항]
    1. 전체 내용을 파악한 뒤, 이 영상을 보는 시청자가 실제로 궁금해할 법한 자연스럽고 흥미로운 질문을 5~10개 생성해 주세요.
    2. 영상의 핵심 사건, 인물의 행위 이유, 중요한 디테일, 흥미로운 장면에 대한 질문을 포함하세요.
    3. 영상에서 정확하게 나오지 않더라도 사용자가 궁금해할 만한 질문도 포함하세요.
    4. 실제 TV 시청자가 사용할 만한 캐주얼하고 격식 없는 어투로 질문을 생성해 주세요.

    [출력 형태 예시]
    반드시 아래의 JSON 배열(Array of strings) 형식으로만 출력하세요. 다른 말은 절대 추가하지 마세요.
    [
        "영상에서 주인공이 가장 처음 먹은 음식은 무엇인가요?",
        "두 번째 등장인물이 화를 낸 이유는 영상의 전후 맥락상 무엇 때문인가요?",
        "영상 후반부에 나타난 붉은색 자동차의 역할은 무엇인가요?"
    ]
    """
    return GenerativeModel(model_name=model_name, system_instruction=[system_prompt])

def generate_queries_for_content(model, video_part, gt_part):
    prompt = f"제공된 컨텐츠에 대한 시청자 질문 5-10개를 실제 사용자가 쓸 법한 캐주얼하고 격식 없는 어투로 생성해 주세요."
    response = model.generate_content([video_part, gt_part, prompt])
    return response.text

def main():
    parser = argparse.ArgumentParser(description="Generate User Queries using Gemini Pro")
    parser.add_argument("--input_file", default="user_query_list.json", help="입력 JSON 파일 경로 (content_id 참조용)")
    parser.add_argument("--output_file", default="output/query_generated.json", help="생성된 질문 목록을 저장할 파일 경로")
    parser.add_argument("--gcp_project_id", required=True, help="GCP 프로젝트 ID (필수)")
    parser.add_argument("--gs_bucket_name", required=True, help="GCS 버킷 이름 (필수)")
    parser.add_argument("--query_gen_model", default="gemini-2.5-pro", help="질문 생성에 사용할 모델명")
    parser.add_argument("--location", default="us-central1", help="GCP Location")
    
    args = parser.parse_args()
    
    project_id = args.gcp_project_id
    gs_bucket_name = args.gs_bucket_name

    print(f"Initializing Gemini client for project: {project_id}, location: {args.location}...")
    init_gemini_client(project_id, location=args.location)
    
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
            try:
                output_list = json.load(f)
                processed_ids = {item["content_id"] for item in output_list}
                print(f"[{len(processed_ids)}] 개의 콘텐츠가 이미 {args.output_file}에 존재하여 건너뜁니다.")
            except json.JSONDecodeError:
                print(f"Warning: {args.output_file} 파일을 읽는 중 오류가 발생했습니다. 새로 시작합니다.")
        
    # 출력 폴더 생성
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*50)
    print("사용자 질문 자동 생성 프로세스를 시작합니다.")
    print("="*50)

    try:
        for item in input_list:
            content_id = item["content_id"]
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
            response_text = generate_queries_for_content(generator_model, video_part, gt_part)
            
            # JSON parsing
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
                
            generated_queries = json.loads(clean_text)
            print(f"Successfully generated {len(generated_queries)} queries for '{content_id}':")
            for i, q in enumerate(generated_queries, 1):
                print(f"  {i}. {q}")
            
            if generated_queries:
                output_entry = {
                    "content_id": content_id,
                    "queries": generated_queries
                }   
                output_list.append(output_entry)
                processed_ids.add(content_id)
            
            # Intermediate saving
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(output_list, f, indent=4, ensure_ascii=False)
                
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        
    print("\n" + "="*50)
    print(f"모든 콘텐츠의 질문 생성이 완료되었습니다. 저장 경로: {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
