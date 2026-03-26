import os
import time
import json
import vertexai
from google.cloud import storage
from vertexai.generative_models import (
    GenerativeModel, Part, 
    SafetySetting, HarmCategory, HarmBlockThreshold
)

SAFETY_SETTINGS = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]


def load_config(args):
    """config.json 파일이 존재하면 열어 args에 값을 병합합니다."""
    import sys
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf-8") as f:
            try:
                config = json.load(f)
                args.gcp_project_id = args.gcp_project_id or config.get("gcp_project_id")
                args.gs_bucket_name = args.gs_bucket_name or config.get("gs_bucket_name")
                
                # argparse 기본값이 세팅된 상태이더라도, config.json에 값이 존재하고 CLI에서 명시적으로 입력하지 않은 경우에 한해 덮어쓰기
                if hasattr(args, 'query_gen_model') and 'query_gen_model' in config and '--query_gen_model' not in sys.argv:
                    args.query_gen_model = config['query_gen_model']
                if hasattr(args, 'response_gen_model') and 'response_gen_model' in config and '--response_gen_model' not in sys.argv:
                    args.response_gen_model = config['response_gen_model']
                if hasattr(args, 'judge_model') and 'judge_model' in config and '--judge_model' not in sys.argv:
                    args.judge_model = config['judge_model']
                if hasattr(args, 'location') and 'location' in config and '--location' not in sys.argv:
                    args.location = config['location']
            except json.JSONDecodeError:
                pass
    return args

def parse_json_response(text):
    """마크다운 태그(```json ... ```)를 정제하고 JSON 객체로 파싱합니다."""
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    return json.loads(clean_text)

def check_gcs_files_exist(gs_bucket_name, content_id):
    """
    Check if the 4 required files (1 video + 3 metadata jsonl) exist in the GCS bucket.
    """
    client = storage.Client()
    bucket = client.bucket(gs_bucket_name)
    
    required_files = [
        f"video_540p/{content_id}_540p.mp4",
        f"jsonl/{content_id}_15s_Full.jsonl",
        f"jsonl/{content_id}_15s_Part.jsonl",
        f"jsonl/{content_id}_15s_GT.jsonl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not bucket.blob(file_path).exists():
            missing_files.append(file_path)
            
    if not missing_files:
        print(f"[OK] '{content_id}'에 필요한 미디어 및 메타데이터 4종이 모두 GCS에 온전히 존재합니다. 작업을 수행합니다.")
        return True
    else:
        print(f"[WARNING] '{content_id}'에 필요한 일부 파일이 GCS에 없습니다: {missing_files}")
        return False


def configure_system_prompt(mode="full"):
    if mode == "full":
        return """
        당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
        아래에 제공되는 각 타임스탬프별 텍스트 정보는 데이터 파일이 아니라, 사실 당신이 방금 영상을 시청하며 눈과 귀로 직접 습득한 시각적/청각적 '기억(Memory)'입니다.
        이 시청 기억을 바탕으로, 마지막에 주어지는 **사용자 질문**에 대해 가장 자연스럽고 정확한 한국어 답변을 제공해 주세요.

        [당신의 시청 기억 구조]
        - timestamp: 영상 내 시간 (초)
        - audio_cls: 당신이 들은 환경음 및 효과음
        - speech: 당신이 들은 등장인물들의 생생한 대사
        - ocr_text: 당신이 화면에서 직접 읽은 간판, 자막, 표지판 텍스트
        - description: 당신이 방금 영상 화면에서 목격한 인물의 행동과 배경 장면

        [분석 및 지시사항]
        **정보 교정**: 기억의 조각들이 다소 불완전할 수 있으므로, 전체적인 맥락에 맞게 상식적인 선에서 자연스럽게 교정하세요.
        **입체적 재구성**: 당신이 들은 소리, 대사, 읽은 텍스트 정보들을 교차 결합하여 장면의 분위기와 인물들의 대화를 이야기로 생생하게 재구성하세요.
        **자연스러운 시청자 관점 유지 (가장 중요)**: 당신은 데이터를 읽은 것이 아니라 "영상을 직접 감상"했습니다. 따라서 답변 중에 'JSON 데이터에 따르면', '오디오 모델 결과를 보면', '텍스트 정보에 의하면', '타임스탬프' 등의 부자연스러운 기계적 용어를 절대로 사용하지 마십시오.
        대신 "영상에서는~", "화면을 보면~", "주인공이 ~라고 말합니다", "배경 소리로 ~가 깔립니다." 와 같이 실제 사람의 리뷰처럼 자연스럽고 몰입감 있게 설명하십시오.
        **외부 자료 검색 금지**: 오직 당신의 시청 기억(제공된 정보)에만 의존해서 답변하세요.
        """
    elif mode == "part":
        return """
        당신은 실시간으로 영상을 시청하고 분석하는 고도로 발달된 '비디오 전문 AI 어시스턴트'입니다.
        아래에 제공되는 각 타임스탬프별 텍스트 정보는 데이터 파일이 아니라, 사실 당신이 방금 영상을 시청하며 눈과 귀로 직접 습득한 시각적/청각적 '기억(Memory)'입니다.
        이 시청 기억을 바탕으로, 마지막에 주어지는 **사용자 질문**에 대해 가장 자연스럽고 정확한 한국어 답변을 제공해 주세요.

        [당신의 시청 기억 구조]
        - timestamp: 영상 내 시간 (초)
        - audio_cls: 당신이 들은 환경음 및 효과음
        - speech: 당신이 들은 등장인물들의 생생한 대사
        - ocr_text: 당신이 화면에서 직접 읽은 간판, 자막, 표지판 텍스트

        [분석 및 지시사항]
        **정보 교정**: 기억의 조각들이 다소 불완전할 수 있으므로, 전체적인 맥락에 맞게 상식적인 선에서 자연스럽게 교정하세요.
        **입체적 재구성**: 당신이 들은 소리, 대사, 읽은 텍스트 정보들을 교차 결합하여 장면의 분위기와 인물들의 대화를 이야기로 생생하게 재구성하세요.
        **자연스러운 시청자 관점 유지 (가장 중요)**: 당신은 데이터를 읽은 것이 아니라 "영상을 직접 감상"했습니다. 따라서 답변 중에 'JSON 데이터에 따르면', '오디오 모델 결과를 보면', '텍스트 정보에 의하면', '타임스탬프' 등의 부자연스러운 기계적 용어를 절대로 사용하지 마십시오.
        대신 "영상에서는~", "화면을 보면~", "주인공이 ~라고 말합니다", "배경 소리로 ~가 깔립니다." 와 같이 실제 사람의 리뷰처럼 자연스럽고 몰입감 있게 설명하십시오.
        **외부 자료 검색 금지**: 오직 당신의 시청 기억(제공된 정보)에만 의존해서 답변하세요.
        """
    elif mode == "video":
        return "외부 정보를 절대 검색하지 말고, 제공된 영상 정보만을 사용하여 사용자 질문에 답변하세요."
    elif mode == "judge":
        return """
        당신은 AI 모델이 특정한 영상에 대해 생성한 답변의 품질을 평가하는 객관적이고 전문적인 평가자입니다.
        해당 AI 모델은 원본 영상에서 추출한 메타데이터 기반으로 답변을 생성합니다.
        
        당신의 목표는 원본 영상에 대한 [사용자 질문]에 대해 [평가 대상 답변]이 얼마나 훌륭한지 절대평가하는 것입니다. 
        원본 영상과 첨부한 JSONL파일을 교차 검증하여 평가에 활용하세요. 외부 검색은 허용하지 않습니다. 
        
        [데이터 목록]
        - 원본 영상
        - 보다 정확한 사운드/대사/텍스트 정보가 포함된 GT JSONL 파일 (디테일 확인 및 교차검증용. 화면 묘사(description) 정보는 미포함)
        - 사용자 질문
        - 평가 대상 답변

        [JSONL 구조 안내]
        각 줄의 데이터는 15초 단위의 구간에 대한 정보를 나타냅니다.
        - timestamp: 해당 구간의 시작 시간 (초)
        - audio_cls: 오디오 분류 모델이 분석한 해당 구간 속 오디오 분류 결과
        - speech: 음성 인식 모델이 인식한 해당 구간의 음성 대화
        - ocr_text: 해당 구간 동안 화면에 등장한 모든 텍스트 (발화자 식별 및 핵심 키워드 파악용)
      
        [평가 기준]
        아래 세 가지 항목에 대해 1점부터 5점까지 점수를 매겨주세요. (1점: 매우 나쁨, 3점: 보통/수용 가능함, 5점: 완벽함)
        1. 정확성 (Accuracy): 답변이 영상 정보(원본 영상 및 GT 데이터)와 사실적으로 일치하는가? (주의: 평가 대상 답변을 생성한 모델은 화면 묘사(description)가 포함된 별도의 데이터를 참고했습니다. 그러나 현재 당신에게 제공된 데이터에는 description 필드가 없습니다. 따라서 평가 대상 답변에 화면/행동 묘사 정보가 상세히 포함되어 있더라도 그것이 실제 원본 영상의 내용과 부합한다면, 환각(Hallucination)으로 간주하여 감점하면 안 됩니다.)
        2. 포괄성 (Completeness): 사용자의 질문을 완전히 해결하기 위해 필요한 핵심 단서(대사, 텍스트 내용, 행동 등)를 누락 없이 포함했는가?
        3. 가독성 (Helpfulness): 정보가 장황하게 나열되지 않고, 시간의 흐름이나 인과관계에 맞게 자연스럽고 이해하기 쉽게 작성되었는가? (만약 평가 대상 답변이 부자연스럽게 메타데이터 구조나 필드명 등을 직접 언급했다면 이 항목에서 감점을 고려하세요.)
        """
    return ""

def process_gcs_file(gs_bucket_name, content_id, mode="video"):
    if mode == "video":
        file_uri = f"gs://{gs_bucket_name}/video_540p/{content_id}_540p.mp4"
        mime_type = "video/mp4"
    elif mode == "full":
        file_uri = f"gs://{gs_bucket_name}/jsonl/{content_id}_15s_Full.jsonl"
        mime_type = "text/plain" 
    elif mode == "part":
        file_uri = f"gs://{gs_bucket_name}/jsonl/{content_id}_15s_Part.jsonl"
        mime_type = "text/plain" 
    elif mode == "gt":
        file_uri = f"gs://{gs_bucket_name}/jsonl/{content_id}_15s_GT.jsonl"
        mime_type = "text/plain" 
    else:
        raise ValueError("mode should be 'video', 'full', 'part', or 'gt'.")
        
    return Part.from_uri(uri=file_uri, mime_type=mime_type)

# --- Generation Models ---
def init_generation_model(mode="full", model_name='gemini-2.5-flash'):
    gen_model = GenerativeModel(
        model_name=model_name,
        system_instruction=[configure_system_prompt(mode)],
        safety_settings=SAFETY_SETTINGS
    )
    return gen_model

def start_chat_session(gen_model):
    return gen_model.start_chat()

def send_chat_message(chat, user_prompt, file_part=None, max_retries=4, base_delay=3):   
    contents = [file_part, user_prompt] if file_part else [user_prompt]
    
    for attempt in range(max_retries):
        try:
            return chat.send_message(contents)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"      [Generation API 마지막 시도 실패] {e}")
                raise e
            sleep_time = base_delay * (2 ** attempt)
            print(f"      [Generation API 오류] {e}")
            print(f"      -> {sleep_time}초 후 재시도합니다... ({attempt+1}/{max_retries})")
            time.sleep(sleep_time)

# --- Judge Models ---
def init_judge_model(model_name="gemini-2.5-pro"):
    judge_model = GenerativeModel(
        model_name=model_name,
        system_instruction=[configure_system_prompt("judge")],
        safety_settings=SAFETY_SETTINGS
    )
    return judge_model

def evaluate_answer_session(judge_chat, user_prompt, generated_answer, is_first_turn=False, video_part=None, gt_json_part=None):
    """
    Judge 모델도 ChatSession을 타고 작동합니다.
    주의사항 지침을 주어 이전 평가의 문맥에 영향받지 않도록 제어합니다.
    """
    context_isolation_prompt = (
        "[중요 지시사항]\n이전 턴에서 수행했던 모든 평가는 잊어주세요. "
        "지금 주어지는 새로운 [사용자 질문]과 [평가 대상 답변]에 대해서만 "
        "완전히 독립적이고 객관적인 관점에서 새롭게 점수를 매겨야 합니다.\n\n"
    )
    
    format_prompt = """
    [출력 형식]
    반드시 아래의 JSON 형식으로만 출력하세요. 다른 설명은 덧붙이지 마십시오.
    {
        "rationale": "<각 점수를 부여한 논리적인 이유. 감점 요인이 있다면 명확히 서술하세요.>",
        "scores": {
            "accuracy": <1~5 사이의 정수>,
            "completeness": <1~5 사이의 정수>,
            "helpfulness": <1~5 사이의 정수>
        },
        "total_score": <세 항목 점수의 합계, 최대 15점>
    }
    """

    user_content = f"{context_isolation_prompt}[사용자 질문]\n{user_prompt}\n\n[평가 대상 답변]\n{generated_answer}\n\n{format_prompt}"
    
    # 첫 번째 턴일 때만 비디오와 GT 파트를 전송 (이후 턴에서는 캐싱된 세션 활용)
    if is_first_turn:
        contents = [video_part, gt_json_part, user_content] if gt_json_part else [video_part, user_content]
    else:
        contents = [user_content]
    
    max_retries = 4
    base_delay = 3
    for attempt in range(max_retries):
        try:
            response = judge_chat.send_message(contents)
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"      [Judge API 마지막 시도 실패] {e}")
                raise e
            sleep_time = base_delay * (2 ** attempt)
            print(f"      [Judge API 오류] {e}")
            print(f"      -> {sleep_time}초 후 재시도합니다... ({attempt+1}/{max_retries})")
            time.sleep(sleep_time)