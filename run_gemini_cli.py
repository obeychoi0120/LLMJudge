import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part

def init_gemini_client(gcp_project_id):
    # 기본적으로 빠른 응답을 위해 서울 리전을 우선 초기화합니다.
    vertexai.init(project=gcp_project_id, location="asia-northeast3") 
    return None

def configure_system_prompt(mode="full"):
    if mode == "full":
        return """
        첨부한 파일은 긴 비디오 영상을 15초 단위의 세그먼트로 나누어 데이터를 추출한 JSONL (JSON Lines) 파일입니다. 이 영상 데이터를 전체적으로 이해한 후, 마지막에 주어지는 **사용자 질문**에 대해 가장 정확하고 종합적인 답변을 한국어로 작성해 주세요.

        [JSONL 구조 안내]
        각 줄의 데이터는 15초 단위의 구간에 대한 정보를 나타냅니다.
        - timestamp: 해당 구간의 시작 시간 (초)
        - audio_cls: 오디오 분류 모델이 분석한 해당 구간 속 오디오 분류 결과
        - speech: 음성 인식 모델이 인식한 해당 구간의 음성 대화
        - ocr_text: 해당 화면에 등장한 자막 텍스트 (발화자 식별 및 핵심 키워드 파악용)
        - description: 시각적 관찰 모델이 영어로 작성한 해당 구간의 화면/행동 묘사

        [분석 및 지시사항]
        **정보 교정**: 제공된 정보가 부정확하거나 불완전할 수 있으므로, 당신이 판단하여 적절히 교정하여야 합니다.
        **입체적 추론**: 제공한 <audio_cls, speech, ocr_text, description> 정보를 교차 검증하여, 등장인물이 언제 어떤 표정으로 어떤 말을 하며 어떤 행동을 했는지 분석한 후 자연스럽게 재구성하세요.
        **맥락의 연결 (Stitching)**: 개별 세그먼트를 분절해서 보지 말고, 시간에 따른 대화의 주제 변화와 인물들의 행동 흐름을 하나의 스토리로 연결하여 이해하세요.
        **외부 자료 검색 금지**: 외부 자료 검색을 금지하며, 오직 제공된 정보만 사용하여 답변하세요.
        """
    elif mode == "nodesc":
        return """
        첨부한 파일은 긴 비디오 영상을 15초 단위의 세그먼트로 나누어 데이터를 추출한 JSONL (JSON Lines) 파일입니다. 이 영상 데이터를 전체적으로 이해한 후, 마지막에 주어지는 **사용자 질문**에 대해 가장 정확하고 종합적인 답변을 한국어로 작성해 주세요.

        [JSONL 구조 안내]
        각 줄의 데이터는 15초 단위의 구간에 대한 정보를 나타냅니다.
        - timestamp: 해당 구간의 시작 시간 (초)
        - audio_cls: 오디오 분류 모델이 분석한 해당 구간 속 오디오 분류 결과
        - speech: 음성 인식 모델이 인식한 해당 구간의 음성 대화
        - ocr_text: 해당 화면에 등장한 자막 텍스트 (발화자 식별 및 핵심 키워드 파악용)

        [분석 및 지시사항]
        **정보 교정**: 제공된 정보가 부정확하거나 불완전할 수 있으므로, 당신이 판단하여 적절히 교정하여야 합니다.
        **입체적 추론**: 제공한 <audio_cls, speech, ocr_text> 정보를 교차 검증하여, 등장인물이 언제 어떤 표정으로 어떤 말을 하며 어떤 행동을 했는지 분석한 후 자연스럽게 재구성하세요.
        **맥락의 연결 (Stitching)**: 개별 세그먼트를 분절해서 보지 말고, 시간에 따른 대화의 주제 변화와 인물들의 행동 흐름을 하나의 스토리로 연결하여 이해하세요.
        **외부 자료 검색 금지**: 외부 자료 검색을 금지하며, 오직 제공된 정보만 사용하여 답변하세요.
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
        - 사운드/대사/텍스트 정보가 포함된 JSONL 파일 (디테일 확인 및 교차검증용)
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
        1. 정확성 (Accuracy): 답변이 영상 정보와 사실적으로 일치하는가? 영상에 없는 내용을 지어내거나(Hallucination) 외부 지식을 무리하게 섞지 않았는가?
        2. 포괄성 (Completeness): 사용자의 질문을 완전히 해결하기 위해 필요한 핵심 단서(대사, 텍스트 내용, 행동 등)를 누락 없이 포함했는가?
        3. 가독성 (Helpfulness): 정보가 장황하게 나열되지 않고, 시간의 흐름이나 인과관계에 맞게 자연스럽고 이해하기 쉽게 작성되었는가?

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
    return ""

def process_gcs_file(gs_bucket_name, content_id, mode="video"):
    if mode == "video":
        file_uri = f"gs://{gs_bucket_name}/video_540p/{content_id}_540p.mp4"
        mime_type = "video/mp4"
    elif mode == "full":
        file_uri = f"gs://{gs_bucket_name}/jsonl/{content_id}_15s.jsonl"
        mime_type = "text/plain" 
    elif mode == "nodesc":
        file_uri = f"gs://{gs_bucket_name}/jsonl/{content_id}_15s_NoDesc.jsonl"
        mime_type = "text/plain" 
    elif mode == "gt":
        file_uri = f"gs://{gs_bucket_name}/jsonl/{content_id}_15s_GT.jsonl"
        mime_type = "text/plain" 
    else:
        raise ValueError("mode should be 'video', 'full', 'nodesc', or 'gt'.")
        
    return Part.from_uri(uri=file_uri, mime_type=mime_type)

# --- Generation Models ---
def init_generation_model(mode="full", model_name='gemini-2.5-flash'):
    gen_model = GenerativeModel(
        model_name=model_name,
        system_instruction=[configure_system_prompt(mode)]
    )
    return gen_model

def start_chat_session(gen_model):
    return gen_model.start_chat()

def send_chat_message(chat, user_prompt, file_part=None):   
    contents = [file_part, user_prompt] if file_part else [user_prompt]
    return chat.send_message(contents)

# --- Judge Models ---
def init_judge_model(model_name="gemini-3.1-pro-preview"):
    judge_model = GenerativeModel(
        model_name=model_name,
        system_instruction=[configure_system_prompt("judge")]
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
    
    user_content = f"{context_isolation_prompt}[사용자 질문]\n{user_prompt}\n\n[평가 대상 답변]\n{generated_answer}"
    
    # 첫 번째 턴일 때만 비디오와 GT 파트를 전송 (이후 턴에서는 캐싱된 세션 활용)
    if is_first_turn:
        contents = [video_part, gt_json_part, user_content] if gt_json_part else [video_part, user_content]
    else:
        contents = [user_content]
    
    response = judge_chat.send_message(contents)
    return response.text