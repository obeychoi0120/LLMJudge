# LLMJudge

Google Cloud Storage(GCS)에 저장된 영상 및 메타데이터를 활용하여 Google Gemini 모델 기반 질의응답을 수행하고, 또 다른 강력한 Gemini 모델을 통해 답변의 품질을 자동 평가하는 파이프라인(CLI)입니다.

## 📝 프로젝트 개요

이 프로젝트는 긴 비디오 영상 데이터와 그에서 추출한 15초 단위의 멀티모달 메타데이터(JSONL)를 Gemini 모델(`gemini-2.5-flash`)에 입력하여 사용자 질문에 대한 다중 턴(Multi-turn) 답변을 생성합니다. 
생성된 답변은 Judge 모델(`gemini-3.1-pro-preview`)을 통해 **정확성(Accuracy)**, **포괄성(Completeness)**, **가독성(Helpfulness)** 의 3가지 기준으로 절대평가되며 자동으로 채점됩니다.

## ✨ 주요 기능

- **시청자 질문 자동 생성**: `gemini-3.1-pro-preview` 모델이 원본 비디오와 정답(GT) 메타데이터를 분석하여 해당 콘텐츠를 시청한 사용자가 실제 궁금해할 만한 핵심 문항(5~10개)을 자동으로 생성합니다.
- **다중 모드(Multi-mode) 추론**: 영상 정보를 제공하는 방식에 따라 3가지 모드로 추론을 진행합니다.
  - `video`: 원본 비디오 파일(.mp4)만을 제공하여 답변을 생성.
  - `full`: 오디오 분류, 음성 인식, 자막(OCR), 시각적 행동 묘사(Description)가 모두 포함된 15초 단위 JSONL 제공.
  - `nodesc`: 시각적 행동 묘사를 제외한 나머지 메타데이터 JSONL 제공.
- **최적화된 Session-based 추론 및 평가**: 대용량 파라미터(Video, JSONL)를 매번 재업로드하는 병목을 제거하기 위해 Chat Session을 활용하여 최초 1회만 업로드합니다. 특히 평가(Judge) 파이프라인에도 세션을 도입하되 인과관계 오염을 막는 독립화 프롬프트를 주입하여, 객관성을 유지하면서도 압도적으로 빠른 평가 속도를 보장합니다.
- **다중 리전(Multi-region) 스위칭 아키텍처**: 네트워크 지연 속도를 최소화하기 위해 답변 생성(Generation)은 서울 리전(`asia-northeast3`)에서 수행하고, 최신 모델 지원이 필요한 답변 평가(Judge)는 글로벌 리전(`global`)으로 런타임에 자동 전환하여 수행합니다.
- **LLM-as-a-Judge 자동 평가 파이프라인**: `gemini-3.1-pro-preview` 판정 모델을 활용해 앞서 생성된 3가지 모드의 답변을 평가합니다. 각 1~5점 척도로 세분화된 점수와 논리적인 평가 사유(`rationale`)를 반환합니다.
- **자동화된 결과 저장**:
  - `response/` 디렉토리에 각 모드별 모델의 텍스트 답변(.txt)이 저장됩니다.
  - `scores/` 디렉토리에 Judge 모델이 채점한 최종 평가 결과가 통합된 JSON 파일로 저장됩니다.

## 🗂 파일 구조

```text
LLMJudge/
├── main.py                    # 전체 추론 및 평가 파이프라인을 실행하는 메인 스크립트
├── run_gemini_cli.py          # Gemini API 초기화, GCS 파일 로드, 프롬프트 구성 및 채팅/평가 함수 모음
├── generate_query.py          # Pro 모델을 사용하여 시청자 질문을 자동 생성하는 스크립트
├── user_query_list.json       # (실제 실행용) 평가를 진행할 콘텐츠 ID와 사용자 질문 리스트
├── user_query_list_sample.json# (참고용) 질문 리스트 샘플 포맷
├── generated_query_list.json  # 자동 생성된 사용자 질문 목록이 저장되는 JSON 파일
├── response/                  # 생성된 LLM 답변 텍스트 파일이 저장되는 폴더
└── scores/                    # Judge 모델의 평가 점수 및 사유 메타데이터(JSON)가 저장되는 폴더
```

## 🚀 설치 및 사전 준비

1. **Python 환경 설정 및 패키지 설치**
   이 프로젝트는 `vertexai`, `google-cloud-aiplatform` 등의 Google Cloud 모듈을 사용합니다.
   ```bash
   pip install google-cloud-aiplatform vertexai
   ```

2. **GCP (Google Cloud Platform) 인증**
   GCS 버킷 접근 및 Vertex AI API 사용을 위해 GCP 인증이 필요합니다.
   ```bash
   gcloud auth application-default login
   ```

## 🎯 실행 방법

### 1. 사용자 질문 자동 생성 (`generate_query.py`)
평가 파이프라인 시작 전, 영상 및 GT 메타데이터 기반으로 다수의 평가용 질문(Queries)을 자동으로 생성할 수 있습니다. 

```bash
python generate_query.py --input_file user_query_list_sample.json --output_file generated_query_list.json --gcp_project_id <YOUR_GCP_PROJECT_ID> --gs_bucket <YOUR_GS_BUCKET>
```
명령어를 실행하면 `input_file`에 명시된 `content_id`를 불러오고, `gemini-3.1-pro-preview` 판정 모델이 각 영상(video)과 정답 메타데이터(GT)를 분석하여 질문 5~10개를 생성 및 터미널에 로깅합니다. 생성된 모든 결과는 `output_file`로 지정된 경로에 JSON 형식으로 저장됩니다.

### 2. 추론 및 모델 답변 평가 (`main.py`)
질문 세트가 준비되면 `main.py`를 실행하여 3가지 모드에 대한 답변 추론과 G-Eval 방식의 자동 평가 파이프라인을 시작합니다. 

```bash
python main.py --json_file user_query_list.json --gcp_project_id <YOUR_GCP_PROJECT_ID> --gs_bucket <YOUR_GS_BUCKET>
```

### Query 데이터 포맷 (`user_query_list.json`)
아래와 같이 `content_id`와 해당 콘텐츠에 수행할 `queries` 목록을 포함한 JSON 배열 형태로 작성해야 합니다.

```json
[
    {
        "content_id": "001_NatGeoKR_Narwhal_6m",
        "queries": [
            "이 영상에서 일어나는 일들을 설명해 줘.",
            "이 영상에서 일각돌고래가 먹잇감을 어떻게 사냥하는지 묘사해 줘."
        ]
    }
]
```

## 📊 평가 결과 (`scores/`)

스크립트 실행이 완료되면 `scores` 폴더에 `{content_id}_all_scores.json` 형태의 파일이 생성됩니다. 이 파일에는 각 쿼리 당 `video`, `full`, `nodesc` 모드의 모델 평가가 담겨 있습니다.

**결과 예시:**
```json
{
  "query": "이 영상에서 일어나는 일들을 설명해 줘.",
  "mode": "full",
  "judge": {
    "rationale": "답변이 제공된 메타데이터에 충실하게 작성되었으며, 맥락에 맞게 자연스럽게 연결되었습니다...",
    "scores": {
      "accuracy": 5,
      "completeness": 5,
      "helpfulness": 4
    },
    "total_score": 14
  }
}
```
