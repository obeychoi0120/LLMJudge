# LLMJudge

Google Cloud Storage(GCS)에 저장된 영상 및 메타데이터를 활용하여 Google Gemini 모델 기반 질의응답을 수행하고, 또 다른 강력한 Gemini 모델을 통해 답변의 품질을 자동 평가하는 파이프라인(CLI)입니다.

## 📝 프로젝트 개요

이 프로젝트는 긴 비디오 영상 데이터와 그에서 추출한 15초 단위의 멀티모달 메타데이터(JSONL)를 Gemini 모델(`gemini-2.5-flash`)에 입력하여 사용자 질문에 대한 다중 턴(Multi-turn) 답변을 생성합니다. 
생성된 답변은 Judge 모델(`gemini-2.5-pro`)을 통해 **정확성(Accuracy)**, **포괄성(Completeness)**, **가독성(Helpfulness)** 의 3가지 기준으로 절대평가되며 자동으로 채점됩니다.

## ✨ 주요 기능

- **시청자 질문 자동 생성**: `gemini-2.5-pro` 모델이 원본 비디오와 정답(GT) 메타데이터를 분석하여 해당 콘텐츠를 시청한 사용자가 실제 궁금해할 만한 핵심 문항(5~10개)을 자동으로 생성합니다.
- **다중 모드(Multi-mode) 추론**: 영상 정보를 제공하는 방식에 따라 3가지 모드로 추론을 진행합니다.
  - `video`: 원본 비디오 파일(.mp4)만을 제공하여 답변을 생성.
  - `full`: 오디오 분류, 음성 인식, 자막(OCR), 시각적 행동 묘사(Description)가 모두 포함된 15초 단위 JSONL 제공.
  - `part`: 시각적 행동 묘사를 제외한 나머지 메타데이터 JSONL 제공.
- **최적화된 Session-based 추론 및 평가**: 대용량 파라미터(Video, JSONL)를 매번 재업로드하는 병목을 제거하기 위해 Chat Session을 활용하여 최초 1회만 업로드합니다. 특히 평가(Judge) 파이프라인에도 세션을 도입하되 인과관계 오염을 막는 독립화 프롬프트를 주입하여, 객관성을 유지하면서도 압도적으로 빠른 평가 속도를 보장합니다.
- **안정적인 기본 리전 및 모델 설정**: 안정적인 멀티모달 처리를 위해 기본 리전은 `us-central1`로 설정되어 있으며, 고성능 추론 및 평가를 위해 `gemini-2.5-pro` 모델을 기본으로 사용합니다. 필요 시 최신 `gemini-3-pro-preview` 모델과 `global` 엔드포인트를 조합하여 사용할 수 있습니다.
- **LLM-as-a-Judge 자동 평가 파이프라인**: `gemini-2.5-pro` 판정 모델을 활용해 앞서 생성된 3가지 모드의 답변을 평가합니다. 각 1~5점 척도로 세분화된 점수와 논리적인 평가 사유(`rationale`)를 반환합니다.
  - Judge 모델은 원본 video와 GT JSONL을 기준으로 평가를 진행합니다.
  - GT JSONL 파일은  `YoutubeDataCollection` 에서 추출한 메타데이터 중 하나입니다.

- **결과 저장**:
  - `output/` 디렉토리에 모든 결과물이 저장됩니다.

## 🗂 파일 구조

```text
LLMJudge/
├── main.py                     # 전체 파이프라인 분기점을 관리하는 오케스트레이터
├── generate_query.py           # 질문 생성 모듈 (`--generate-query` 옵션 시 동작)
├── generate_response.py        # 모드별 답변 생성(Inference) 모듈 
├── judge_response.py           # 프롬프트 기반 평가(Judge) 모듈
├── run_gemini_cli.py           # Gemini SDK 초기화, GCS 데이터 검증 및 프롬프트 등 공통 헬퍼 
├── sample_config.json          # 설정 파일 샘플 (복사하여 config.json으로 사용)
├── sample_user_query_list.json # 기본 입력 파일: 평가를 진행할 콘텐츠 ID (및 선별적 질문) 명시
└── output/                     # 파이프라인의 결과물이 통합 저장되는 디렉토리
    ├── query_generated.json    # 1️⃣ 자동 생성된 질문 목록 (생략 가능)
    ├── responses.json          # 2️⃣ 각 모드(full, part, video)별 모델 추론 답변
    └── scores.json             # 3️⃣ 최종 평가 점수(1~5점 척도) 및 Rationale
```

## 🚀 설치 및 사전 준비

1. **Python 환경 설정 및 패키지 설치**
   이 프로젝트는 `vertexai`, `google-cloud-aiplatform` 등의 Google Cloud 모듈을 사용합니다.
   ```bash
   pip install google-cloud-aiplatform google-cloud-storage vertexai
   ```

2. **GCP (Google Cloud Platform) 인증**

   GCS 버킷 접근 및 Vertex AI API 사용을 위해 GCP 인증이 필요합니다.
   ```bash
   gcloud auth application-default login
   ```
3. **GCP 프로젝트 및 버킷 설정 (`config.json`)**
   이 프로젝트는 프로젝트 ID 및 버킷 이름을 `config.json`에서 읽어옵니다. 루트 디렉토리에 있는 `sample_config.json`을 복사하여 `config.json`을 생성하고 본인의 정보를 입력해 주세요.
   ```bash
   cp sample_config.json config.json
   # 이후 config.json의 내용을 본인의 GCP Project ID 및 Bucket Name으로 수정합니다.
   ```
   *(참고: `config.json`은 `.gitignore`에 등록되어 있어 외부 저장소에 유출되지 않습니다.)*

4. **GCS 버킷에 데이터 업로드**

    기본적으로 `gs://insight-youtubevideodataset-us` 버킷에 데이터를 업로드합니다. (us-central1 리전)

    필요 시 개인이 버킷 생성하여 사용 가능합니다.

    컨텐츠 하나 당 4개의 데이터가 필요하며 다음과 같이 구성하여야 합니다.

    ```bash
    {BUCKET_NAME}/video_540p/{content_id}.mp4 # 원본 540p 영상
    {BUCKET_NAME}/jsonl/{content_id}_15s.jsonl  # Description이 포함된 JSONL
    {BUCKET_NAME}/jsonl/{content_id}_15s_NoDesc.jsonl # Description이 제외된 JSONL
    {BUCKET_NAME}/jsonl/{content_id}_15s_GT.jsonl # Judge model 용 GT JSONL
    ```
    예시:
    ```bash
    gs://insight-youtubevideodataset-us/video_540p/001_NatGeoKR_Narwhal_6m.mp4
    gs://insight-youtubevideodataset-us/jsonl/001_NatGeoKR_Narwhal_6m_15s.jsonl
    gs://insight-youtubevideodataset-us/jsonl/001_NatGeoKR_Narwhal_6m_15s_NoDesc.jsonl
    gs://insight-youtubevideodataset-us/jsonl/001_NatGeoKR_Narwhal_6m_15s_GT.jsonl
    ```
    로컬 터미널에서 다음과 같이 업로드:
    ```bash
    gcloud storage buckets create gs://{BUCKET_NAME}
    gcloud storage cp -r {LOCAL_DATA_PATH} gs://{BUCKET_NAME}/
    ```



## 🎯 실행 방법

`main.py`는 오케스트레이터 역할을 하며, 옵션에 따라 필요한 파이프라인을 유연하게 제어할 수 있습니다. 스크립트는 실행 중 네트워크 중단 등으로 종료되더라도 `output/` 폴더 내 기존 결과물을 인식하여 누락된 `content_id`부터 작업을 재개(Resume)합니다.

### 1. 기본 사용법 (Response 생성 및 Judge 평가)
이미 `sample_user_query_list.json` 형태(질문이 이미 포함된 JSON)를 입력으로 넣어 바로 답변 생성과 평가를 수행하는 방식입니다.

```bash
python main.py \
  --input_file <YOUR_QUERY_LIST_JSON_FILE> \
  --gcp_project_id <YOUR_GCP_PROJECT_ID> \
  --gs_bucket_name <YOUR_GS_BUCKET_NAME>
```

### 2. E2E 사용법 (Query 생성 -> Response 생성 -> Judge 평가)
입력 파일(`content_list.json`)에 `content_id` 목록(단순 문자열 리스트)만 있는 경우, 파이프라인이 자동으로 질문 생성부터 시작합니다.

```bash
python main.py \
  --input_file <YOUR_CONTENT_LIST_JSON_FILE> \
  --gcp_project_id <YOUR_GCP_PROJECT_ID> \
  --gs_bucket_name <YOUR_GS_BUCKET_NAME>
```
*과정: 1) 질문 생성 (`output/query_generated.json`) -> 2) 답변 생성 (`output/responses.json`) -> 3) 최종 평가 (`output/scores.json`)*

### 3. 모델 커스텀 및 최신 3 Pro 적용법
특정 단계에서 구동되는 모델을 변경하거나, 리전(Location)을 설정해야 될 때 다음과 같은 파라미터를 추가 조절할 수 있습니다:
- `--query_gen_model`: 질문 자동 생성 모델 (기본: `gemini-2.5-pro`)
- `--response_gen_model`: 답변 생성(Inference) 모델 (기본: `gemini-2.5-flash`)
- `--judge_model`: 평가 모델 (기본: `gemini-2.5-pro`)
- `--location`: GCP 리전 설정 (기본: `us-central1`)

**예시: Gemini 3.0 Pro 미리보기 모델을 사용해 E2E 파이프라인 가동하기**
```bash
python main.py \
  --generate-query \
  --query_gen_model gemini-3-pro-preview \
  --judge_model gemini-3-pro-preview \
  --location global \
  --gcp_project_id <YOUR_GCP_PROJECT_ID> \
  --gs_bucket_name <YOUR_GS_BUCKET_NAME>
```

## 📊 평가 결과 (`output/scores.json`)

파이프라인이 모두 순회되면 `output/scores.json`에 아래 형태와 같이 종합 평가 파일이 생성됩니다.

**결과 예시:**
```json
[
  {
    "content_id": "001_NatGeoKR_Narwhal_6m",
    "scores": [
      {
        "query": "이 영상에서 일각돌고래가 사냥하는 모습을 묘사해 줘.",
        "mode": "full",
        "judge": {
          "rationale": "답변이 제공된 메타데이터에 충실하게 작성되었으며...",
          "scores": {
            "accuracy": 5,
            "completeness": 5,
            "helpfulness": 4
          },
          "total_score": 14
        }
      }
    ]
  }
]
```
