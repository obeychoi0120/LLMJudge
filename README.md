# LLMJudge

Google Cloud Storage(GCS)에 저장된 영상 및 메타데이터를 활용하여 Google Gemini 모델 기반 질의응답을 수행하고, 또 다른 강력한 Gemini 모델을 통해 답변의 품질을 자동 평가하는 파이프라인(CLI)입니다.

## 📝 프로젝트 개요

이 프로젝트는 긴 비디오 영상 데이터와 그에서 추출한 15초 단위의 멀티모달 메타데이터(JSONL)를 Gemini 모델(`gemini-2.5-flash`)에 입력하여 사용자 질문에 대한 다중 턴(Multi-turn) 답변을 생성합니다. 
생성된 답변은 Judge 모델(`gemini-2.5-pro`)을 통해 **정확성(Accuracy)**, **포괄성(Completeness)**, **가독성(Helpfulness)** 의 3가지 기준으로 절대평가되며 자동으로 채점됩니다.

## ✨ 주요 특징 및 아키텍처

- **시청자 질문 자동 생성**: `gemini-2.5-pro` 모델이 원본 비디오와 정답(GT) 메타데이터를 분석하여 해당 콘텐츠를 시청한 사용자가 실제 궁금해할 만한 핵심 문항(5~10개)을 자동으로 생성합니다.
- **다중 모드(Multi-mode) 추론**: 영상 정보를 제공하는 방식에 따라 3가지 모드로 추론을 진행합니다.
  - `video`: 원본 비디오 파일(.mp4)만을 제공하여 답변을 생성.
  - `full`: 오디오 분류, 음성 인식, 자막(OCR), 시각적 행동 묘사(Description)가 모두 포함된 15초 단위 JSONL 제공.
  - `part`: 시각적 행동 묘사를 제외한 나머지 메타데이터 JSONL 제공.
- **최적화된 Session-based 추론 및 평가**: 대용량 파라미터(Video, JSONL)를 매번 재업로드하는 병목을 제거하기 위해 Chat Session을 활용하여 최초 1회만 업로드합니다.
- **안정적인 재시도(Exponential Backoff)**: API Rate Limit(429) 등 일시적 리소스 고갈 발생 시, 지수 백오프 기반 재시도 로직이 동작하여 프로세스가 강제 종료되지 않고 안전하게 복구됩니다.
- **쿼리 단위 부분 저장(Partial Checkpoint) 및 이어하기(Resume)**: 기존의 단일 비디오(전체 쿼리) 완료 후 저장 방식에서 탈피하여, **단일 쿼리(질문 1개)** 처리가 끝날 때마다 JSONL 파일에 실시간으로 Append(누적) 저장합니다. 스크립트 중단 시, 시작할 때 이전 이력을 완벽히 복원하고 잔여 작업(Resume Plan)만 효율적으로 연산합니다.
- **비동기 연속 파이프라인 모니터링 (`--continuous`)**: 각 스크립트(`generate_response`, `judge_response`)는 `--continuous` 플래그를 통해 이전 단계의 출력을 실시간으로 모니터링하며 병렬 파이프라인 형태로 구동될 수 있습니다.

## 🗂 파일 구조

```text
LLMJudge/
├── main.py                     # 전체 파이프라인 분기점을 관리하는 오케스트레이터
├── generate_query.py           # 질문 생성 모듈 (`--generate-query` 옵션 시 동작)
├── generate_response.py        # 모드별 답변 생성(Inference) 모듈 
├── judge_response.py           # 프롬프트 기반 평가(Judge) 모듈
├── gemini_api_utils.py           # Gemini SDK 초기화, GCS 데이터 검증 등 헬퍼 
├── jsonl_to_json.py            # [분석 도구] 생성된 JSONL 로그를 읽기 편한 JSON으로 변환
├── sample_config.json          # 설정 파일 샘플 (복사하여 config.json으로 사용)
├── sample_user_query_list.json # 기본 입력 파일 샘플
└── output/                     # 파이프라인의 결과물이 통합 저장되는 디렉토리
    ├── query_generated.jsonl   # 1️⃣ 자동 생성된 질문 목록 (실시간 누적)
    ├── responses.jsonl         # 2️⃣ 각 모드별 추론 답변 (실시간 누적)
    └── scores.jsonl            # 3️⃣ 최종 평가 점수 및 Rationale (실 실시간 누적)
```

## 🚀 설치 및 사전 준비

1. **Python 환경 설정 및 패키지 설치**
   ```bash
   pip install google-cloud-aiplatform google-cloud-storage vertexai
   ```

2. **GCP 인증 및 설정**
   ```bash
   gcloud auth application-default login
   ```

3. **설정 파일 복사 (`config.json`)**
   ```bash
   cp sample_config.json config.json
   # 본인의 GCP Project ID 및 Bucket Name으로 수정
   ```

4. **GCS 데이터 구조**
    컨텐츠 하나 당 4개의 데이터가 필요합니다.
    ```text
    gs://{BUCKET_NAME}/video_540p/{content_id}.mp4
    gs://{BUCKET_NAME}/jsonl/{content_id}_15s_Full.jsonl
    gs://{BUCKET_NAME}/jsonl/{content_id}_15s_Part.jsonl
    gs://{BUCKET_NAME}/jsonl/{content_id}_15s_GT.jsonl
    ```

## 🎯 실행 방법

### 방법 A. 메인 오케스트레이터 일괄 실행 (순차 진행)
`main.py`를 실행하면 옵션에 따라 필요한 파이프라인을 유연하게 제어할 수 있습니다. 스크립트는 중단 후 재실행 시 누락된 단위(쿼리)부터 자동으로 이어하기(Resume)를 알립니다.

**1. 기본 사용법 (입력 파일에 Query가 이미 있는 경우)**
```bash
python main.py \
  --input_file <YOUR_QUERY_LIST_JSONL_FILE>
```

**2. E2E 사용법 (입력 파일에 Content ID만 있는 경우)**
지정된 입력이 빈 리스트일 때 `--generate-query` 플래그를 추가하면 파이프라인의 맨 앞단인 '질문 생성' 부터 시작됩니다.
```bash
python main.py \
  --generate-query \
  --input_file <YOUR_CONTENT_LIST_JSON_FILE> \
```

### 방법 B. 병렬 모니터링 파이프라인 실행 (`--continuous`)
가장 권장되는 대규모 파이프라인 방식입니다. 3개의 시스템 (또는 터미널 창)을 열고 스크립트를 각각 독립적으로 켜두면, 앞선 단계의 결과물이 파일에 기록되는 즉시 다음 단계 스크립트가 실시간으로 데이터를 이어받아 연쇄 처리합니다.

- **터미널 1 (질문 생성 완료 후 JSONL 출력만 관리)**
  ```bash
  python generate_query.py --input_file content_list.json
  ```
- **터미널 2 (답변 생성 모니터링)**
  ```bash
  python generate_response.py --continuous
  ```
- **터미널 3 (답변 평가 모니터링)**
  ```bash
  python judge_response.py --continuous
  ```

### 💡 분석용 JSON 실시간 변환 도구
파이프라인이 `.jsonl` 형태로 데이터를 실시간 누적(Append)하므로 일반 텍스트 편집기에서 읽기에는 다소 불편할 수 있습니다. 
파이프라인 동작 도중이든 완료 후든 **언제든지** 아래 명령으로 오류 처리 이력이 깔끔하게 자동으로 병합된 상태의 분석용 `.json` 파일들을 추출해 낼 수 있습니다.
```bash
python jsonl_to_json.py
```

## 📊 평가 결과 (`output/scores.jsonl`)

평가가 완료된 후 `jsonl_to_json.py`를 통해 추출되는 `scores.json` 결과 예시입니다.
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
