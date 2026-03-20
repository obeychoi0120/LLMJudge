import os
import argparse
import json
from vertexai.generative_models import GenerativeModel
from run_gemini_cli import init_gemini_client, process_gcs_file

GS_BUCKET_NAME = "insight-youtubevideodataset"

def init_query_generator_model(model_name="gemini-3.1-pro-preview"):
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

def generate_queries_for_content(model, video_part, gt_part, content_id):
    prompt = f"콘텐츠 ID '{content_id}'에 대한 흥미로운 시청자 질문 5-10개를 생성해 주세요."
    response = model.generate_content([video_part, gt_part, prompt])
    return response.text

def main():
    os.environ["GCP_PROJECT_ID"] = "insight-dev-490002"
    parser = argparse.ArgumentParser(description="Generate User Queries using Gemini Pro")
    parser.add_argument("--input_file", default="user_query_list.json", help="입력 JSON 파일 경로 (content_id 참조용)")
    parser.add_argument("--output_file", default="generated_query_list.json", help="생성된 질문 목록을 저장할 파일 경로")
    parser.add_argument("--project_id", help="GCP 프로젝트 ID")
    
    args = parser.parse_args()
    
    project_id = args.project_id or os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        print("Error: GCP Project ID가 설정되지 않았습니다.")
        return

    print(f"Initializing Gemini client for project: {project_id}...")
    init_gemini_client(project_id)
    
    print("Initializing Query Generator Model (gemini-3.1-pro-preview)...")
    generator_model = init_query_generator_model()
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} 파일이 존재하지 않습니다.")
        return  
        
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_list = json.load(f)
        
    output_list = []
    
    print("\n" + "="*50)
    print("사용자 질문 자동 생성 프로세스를 시작합니다.")
    print("="*50)

    try:
        for item in input_list:
            content_id = item["content_id"]
            print(f"\nProcessing Content: '{content_id}'")
            
            # Use the video and gt files
            print(f"Preparing GCS files (mode: video, gt) for {content_id}...")
            video_part = process_gcs_file(GS_BUCKET_NAME, content_id, mode="video")
            gt_part = process_gcs_file(GS_BUCKET_NAME, content_id, mode="gt")
            
            print("Generating queries...")
            try:
                response_text = generate_queries_for_content(generator_model, video_part, gt_part, content_id)
                
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
            except json.JSONDecodeError:
                print(f"[Warning] Failed to parse JSON from response. Using raw text fallback for '{content_id}'.")
                generated_queries = [line.strip('- "*,') for line in clean_text.split('\n') if line.strip() and not line.strip() in ['[', ']']]
            except Exception as e:
                print(f"[Error] Failed to generate queries for '{content_id}': {e}")
                generated_queries = []
            
            if generated_queries:
                output_entry = {
                    "content_id": content_id,
                    "queries": generated_queries
                }   
                output_list.append(output_entry)
            
            # Intermediate saving
            # with open("generated_queries/" + args.output_file, "w", encoding="utf-8") as f:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(output_list, f, indent=4, ensure_ascii=False)
                
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        
    print("\n" + "="*50)
    print(f"모든 콘텐츠의 질문 생성이 완료되었습니다. 저장 경로: {args.output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
