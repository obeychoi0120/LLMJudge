# LLMJudge (Multimodal Interactive Evaluation)

Google Cloud Storage(GCS)에 저장된 영상 및 메타데이터를 활용하여 **멀티모달 대화 시나리오 품질을** 자동으로 생성하고 평가 할 수 있는 자동화 파이프라인(CLI)입니다.

## 핵심 아키텍처 개요

이 파이프라인은 동영상의 시간적 구조를 기반으로 핵심 장면 구간 별 주요 전환 점(KeyScene)을 식별하고, 각각의 장면에서 **과거 맥락(Past)** 과 **현재 초점(Current Focus)** 을 결합하여 분석합니다. 2개의 트랙(A/B)으로 구성된 자동 평가프레임을 통해, 서로 다른 데이터소스 조건에서 동일한 질문을 던지고 답변의 품질을 비교하는 통제변인 실험을 수행합니다.

### 2-Track 평가체계

- **A-Track (Voice Hint)**: 6개 모드(kss, video, raw, raw_with_mmvlm, imgvlm_chunk2, imgvlm_graph)의 Source를 기반으로 시청자의 호기심을 유발하는 질문을, KSS Anchor 기준으로 품질 평가
- **B-Track (VH Response)**: A-Track의 **kss 모드로 생성된 공통 Query**를 기준, 각 모드별 Source로 답변 생성 → KSS + World Knowledge 기반 품질 비교. 동일한 질문으로 데이터소스 간 **통제 실험**을 수행

---

## 저작권 안전한 데이터 구조 (Copyright-Safe Architecture)

이 파이프라인은 외부 모델 호출 시 원본의 **직접 재현이 불가능한 형태로 변환**합니다. 원저작 LLM이 생성한 설명문을 그대로 전송할 경우 저작물의 표현을 침해할 가능성이 있으므로, 구조화와 단편화 처리를 거쳐 전송 위험을 최소화합니다.

### 저작권 안전 처리 흐름: VLM 구조화 + 단편화 + 마스킹 (imgvlm)

`imgvlm` 모드는 온디바이스 VLM이 추출한 시각 메타데이터에 다단계 변환을 **비가역적으로 적용하여**, 원본 서술문/문장을 복원할 수 없는 형태로 가공합니다. 3단계 처리 프로세스는 다음과 같습니다:

1. **구조화 (Structuring)**: VLM 추출 원문을 의미 단위로 Subjects / Actions / Contexts 3개 카테고리로 분류
2. **2워드 단편화 (Bigram Fragmentation)**: 각 항목을 무작위로 2워드 단위로 잘라 내 원문 순서와의 대응을 완전히 제거
3. **고유명사 마스킹**: 작품명, 캐릭터명, 지명 등 식별자를 `[MASKED]` 토큰으로 대체

```
VLM 원문:  "A narwhal swims gracefully through the Arctic waters near ice floes"

구조화 + 단편화 후:
  Subjects: ["near ice", "A narwhal"]                    (셔플됨)
  Actions:  ["gracefully through", "swims [MASKED]"]     (단편화 + 마스킹)
  Contexts: ["ice floes", "the Arctic", "waters near"]   (셔플됨)
```

> **`raw` / `raw_with_mmvlm` 모드**는 원본 ASR/OCR를 그대로 사용하므로, **저작권 계약 완료된 컨텐츠에만 사용 가능한** 모드입니다.

## 전체 파이프라인 구성

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        CL["content_list.json"]
        GCS["GCS Assets\n(Video, JSONL)"]
    end

    subgraph INFRA["공통 인프라"]
        A1["A-1. identify_keyscene.py"]
        KP["keypoint_scenes.jsonl"]
        A2["A-2. generate_keyscene_summary.py\n[Phase 1] 과거 요약 (Pro)\n[Phase 2] 현재 생성 (Pro)"]
        KSS["keyscene_summary.jsonl\n(Ground Truth Anchor)"]
    end

    subgraph TRACKS[" "]
        direction LR
        subgraph BTRACK["B-Track: VH Response"]
            B1["B-1. generate_vh_response.py\nKSS에서 생성된 고품질 VH를 공통 Query로 사용\n각 모드별 Source로 답변 생성"]
            VHR["vh_responses.jsonl"]
            B2["B-2. judge_vh_response.py\n(KSS + World Knowledge, 3기준 15점)"]
            VHRS["vh_response_scores.jsonl"]
        end
        subgraph ATRACK["A-Track: Voice Hint"]
            A3["A-3. generate_voice_hint.py\nkss / video / raw / raw_with_mmvlm\nimgvlm_chunk2 / imgvlm_graph"]
            VH["voice_hint.jsonl"]
            A4["A-4. judge_voice_hint.py\n(KSS Anchor 기준, 2기준 10점)"]
            VHS["voice_hint_scores.jsonl"]
        end
    end

    CL --> A1
    GCS -.-> A1
    A1 --> KP
    KP --> A2
    GCS -.-> A2
    A2 --> KSS

    KSS --> A3
    KP --> A3
    GCS -.-> A3
    A3 --> VH
    VH --> A4
    KSS --> A4
    A4 --> VHS

    KSS --> B1
    KP --> B1
    GCS -.-> B1
    B1 --> VHR
    VHR --> B2
    KSS --> B2
    B2 --> VHRS
```

### 파이프라인 요약

| Step | 스크립트 | Output | 모델 |
|------|----------|--------|------|
| A-1 | `identify_keyscene.py` | `keypoint_scenes.jsonl` | Flash Lite |
| A-2 | `generate_keyscene_summary.py` | `keyscene_summary.jsonl` | Pro |
| A-3 | `generate_voice_hint.py` | `voice_hint.jsonl` | Flash Lite |
| A-4 | `judge_voice_hint.py` | `voice_hint_scores.jsonl` | Pro |
| B-1 | `generate_vh_response.py` | `vh_responses.jsonl` | Flash Lite |
| B-2 | `judge_vh_response.py` | `vh_response_scores.jsonl` | Pro |
| - | `export_to_excel.py` | Excel 리포트 | - |

---

## 핵심 설계 기술 상세

### 1. KeyScene 식별: Scene 규모별 적응형 3단계 전략

| Scene 수 | 구분 | Stage 1 | Stage 2 |
|----------|------|---------|---------|
| ≤ 8개 | A | 전체를 KeyScene으로 그대로 사용 | - |
| 9 ~ 17개 | B | 2배수 → 각 그룹에서 4개씩 축소 | 최종 통합 |
| ≥ 18개 | C | 3배수 → Candidate 리스트 생성 | Selector가 8개 최종 선별 |

### 2. KeyScene Summary: 2-Phase 순차 생성전략

```
[Phase 1: 과거 맥락 요약] → Pro (thinking: high)
  입력: 기존 KSS + Gap 구간 Ref 메타데이터 (텍스트 only)
       ↓
[Phase 2: 현재 장면 생성] → Pro (thinking: high)
  입력: 과거 요약 + 현재 Ref JSONL + 현재 비디오클립 (멀티모달)
```

생성된 KSS는 이후 모든 Judge 스크립트에서 **Ground Truth Anchor**로 참조됩니다.

### 3. 평가 기준(Judge): 채점 프레임워크

모든 Judge는 **rationale + score의 flat JSON 포맷**을 출력합니다.

#### A-Track: Voice Hint Judge (2-Criteria, 10점 만점)

KSS(KeyScene Summary)와 대조해 평가하되, 시청자의 몰입감 유지와 호기심 유발이라는 두 가지 축으로 채점합니다.

| 기준 | 평가 포인트 |
|------|----------|
| **Temporal Immersion** | 시청자의 현재 시청시점 대비 자연스러운, 과거 맥락 활용과 미래의 전개를 암시 (시제 혼동없이는 감점) |
| **Curiosity & Hook** | 호기심을 유발하는 구체적이고 흥미로운 질문을 제공하면서도 스포일러를 회피하는지 |

> 대조 비교 평가 대상: `kss`, `video`, `raw`, `raw_with_mmvlm`, `imgvlm_chunk2`, `imgvlm_graph`

#### B-Track: VH Response Judge (3-Criteria, 15점 만점)

| 기준 | 평가 포인트 |
|------|----------|
| **Answer Relevance** | 시청자의 질문에 직접적으로 응답하는지 |
| **Factual Precision** | 사실 정확성 + 원본 정보 활용의 적절한지 |
| **Response Quality** | 가독성과 구조, 자연스러운 흐름과 완결성 |

#### ~~C-Track: Description Judge~~ (Deprecated)

> KeyScene Description 평가는 현 파이프라인에서 제외 되었습니다. 관련 스크립트는 `archived/`에 보관되어있습니다.

### 4. Voice Hint 생성 및 평가 철학: Tangential World Knowledge (곁다리 지식)

Voice Hint 파이프라인(A-Track)은 시청자가 스마트 TV 리모컨을 눌러 능동적으로 상호작용하도록 유도하기 위해 **'곁다리 지식(Tangential World Knowledge)'**을 핵심 전략으로 사용합니다. 

단순히 영상의 '줄거리(Narrative)'나 '팩트'를 묻는 것을 넘어, 장르별 특성에 맞춰 화면에 등장한 시각/청각적 요소를 트리거로 삼아 다음과 같은 확장 지식을 질문합니다. 평가지표(Judge) 역시 KSS 원문에 해당 내용이 없더라도 이러한 확장된 지식 기반 질문을 훌륭한 '호기심(Curiosity & Hook) 자극 요소'로 간주하여 고평가(5점)하도록 설계되었습니다:

*   **드라마/예능:** 허구적 플롯 대신 화면 속 소품의 기원이나 촬영 장소의 문화적/역사적 배경
*   **스포츠:** 결과 예측 대신 방금 활약한 선수의 폼, 이적 비하인드, 뉴비(초보자)를 위한 팀 역사 및 라이벌 구도
*   **게임:** 단순 상황 묘사 대신 플레이어의 전략, 캐릭터 특성, 아이템 메타 변화, 최신 패치 소식
*   **뉴스/시사/다큐:** 단순 멘트 요약 대신 보도 이면의 역사적 맥락, 경제적 파급 효과, 정책의 비하인드, 과거 유사 사례

---

## 디렉토리 구조

```text
LLMJudge/
├── main.py                          # E2E 파이프라인 오케스트레이터
├── identify_keyscene.py             # KeyScene Scene 식별 (A-1)
├── generate_keyscene_summary.py     # KeyScene Summary 생성 (A-2)
├── generate_voice_hint.py           # Voice Hint 생성 (A-3)
├── judge_voice_hint.py              # Voice Hint 품질 Judge (A-4)
├── generate_vh_response.py          # VH Response 다모드 답변 생성 (B-1)
├── judge_vh_response.py             # VH Response Judge (B-2)
├── utils.py                         # Gemini SDK, GCS 연동, 공통 유틸
├── export_to_excel.py               # Excel 리포트 생성
├── jsonl_to_json.py                 # JSONL → Pretty JSON 변환 유틸
├── clean_assets.py                  # JSONL 내 특정 모드 제거 유틸
├── config.json                      # 실행 설정 (GCP, 모델명 등)
├── content_list.json                # 평가 대상 Content ID 목록
├── sample_config.json               # config.json 템플릿
├── sample_content_list.json         # content_list.json 템플릿
├── sample_data/                     # 예시 JSONL (파이프라인 참고용)
├── archived/                        # Deprecated 스크립트
│   ├── generate_keyscene_description.py  # (Deprecated) KeyScene Description 생성
│   ├── judge_descriptions.py             # (Deprecated) Description Judge
│   ├── generate_user_query.py            # (Deprecated) User Query 생성
│   └── generate_uq_response.py           # (Deprecated) UQ Response 생성
└── assets/                          # 파이프라인 중간 산출 및 최종 결과 저장소
    ├── keypoint_scenes.jsonl
    ├── keyscene_summary.jsonl
    ├── voice_hint.jsonl
    ├── voice_hint_scores.jsonl
    ├── vh_responses.jsonl
    └── vh_response_scores.jsonl
```

## 사전 준비 및 환경 설정

1. **Python 패키지 설치**
   ```bash
   pip install google-genai google-cloud-storage pandas openpyxl
   ```

2. **GCP 인증 및 설정**
   ```bash
   gcloud auth application-default login
   ```
   `config.json`에 GCP 프로젝트 ID와 GCS 버킷 이름을 설정합니다.

3. **GCS 디렉토리 구조 및 데이터 업로드**

   파이프라인은 GCS 버킷에서 3종의 파일을 읽습니다. 아래 경로 규칙을 **반드시** 준수해야 합니다.

   ```text
   gs://{bucket}/
   ├── video_540p/
   │   └── {content_id}_540p.mp4          # 540p 다운스케일 비디오
   └── jsonl/
       ├── {content_id}_final.jsonl       # Scene별 메타데이터 (VLM 구조 포함)
       └── {content_id}_ref.jsonl         # Scene별 참조 메타데이터 (speech, texts, sounds)
   ```

   | 파일 | 설명 | 필수 |
   |------|------|:----:|
   | `{content_id}_540p.mp4` | Gemini 모델에 입력되는 비디오. 540p 해상도 권장 | ✅ |
   | `{content_id}_final.jsonl` | 각 Scene의 전체 메타데이터 (VLM 이미지 구조, 타임라인 등) | ✅ |
   | `{content_id}_ref.jsonl` | 각 Scene의 참조 메타데이터 (speech, texts, sounds, duration 등) | ✅ |

   > **참고**: `content_id`는 `content_list.json`에 등록하는 ID와 동일해야 합니다.

   #### GCS 버킷 생성

   자신만의 버킷을 생성하려면 아래 명령어를 사용합니다.
   ```bash
   # 버킷 생성 (리전: us-central1)
   gcloud storage buckets create gs://my-llmjudge-bucket \
       --project=your-gcp-project-id \
       --location=us-central1\
       --uniform-bucket-level-access

   # 생성 확인
   gcloud storage buckets describe gs://my-llmjudge-bucket
   ```

   생성 후 `config.json`의 `gs_bucket_name`을 생성한 버킷 이름으로 변경합니다.
   ```json
   {
       "gs_bucket_name": "my-llmjudge-bucket"
   }
   ```

   > **참고**: 버킷 이름은 전역적으로 고유해야 합니다. 팀 내 충돌을 방지하려면 `{팀명}-llmjudge-{이니셜}` 같은 네이밍 컨벤션을 권장합니다.

   #### 데이터 업로드 방법

   **gcloud storage를 사용한 업로드:**
   ```bash
   gcloud storage cp /path/to/{content_id}_540p.mp4   gs://{bucket}/video_540p/
   gcloud storage cp /path/to/{content_id}_final.jsonl gs://{bucket}/jsonl/
   gcloud storage cp /path/to/{content_id}_ref.jsonl   gs://{bucket}/jsonl/
   ```

   **업로드 후 검증:**
   ```bash
   # 특정 content_id의 파일 존재 확인
   gcloud storage ls gs://{bucket}/video_540p/{content_id}_540p.mp4
   gcloud storage ls gs://{bucket}/jsonl/{content_id}_final.jsonl
   gcloud storage ls gs://{bucket}/jsonl/{content_id}_ref.jsonl

   # 또는 파이프라인을 실행하면 자동으로 3종 파일 존재 여부를 검증합니다.
   # → [OK] '{content_id}'에 필요한 미디어 및 메타데이터가 모두 GCS에 존재합니다.
   ```

   #### content_list.json 등록

   업로드한 콘텐츠를 파이프라인에서 처리하려면 `content_list.json`에 해당 `content_id`를 추가합니다.
   ```json
   [
       "my_video_001",
       "my_video_002"
   ]
   ```

4. **`config.json` 주요 설정 키** (sample_config.json 참조)
```json
{
    "gcp_project_id": "your-gcp-project-id",
    "gs_bucket_name": "your-gcs-bucket-name",
    "location": "global",
    "keypoint_model": "gemini-3.1-flash-lite-preview",
    "keypoint_thinking_level": "medium",
    "kss_past_summary_model": "gemini-3.1-pro-preview",
    "kss_past_summary_thinking_level": "high",
    "kss_current_scene_model": "gemini-3.1-pro-preview",
    "kss_current_scene_thinking_level": "high",
    "use_ref_for_keyscene_summary": true,
    "vh_gen_model": "gemini-3.1-flash-lite-preview",
    "vh_gen_past_scenes_size": 5,
    "vh_thinking_level": "medium",
    "vh_judge_model": "gemini-3.1-pro-preview",
    "vh_judge_thinking_level": "high",
    "vh_response_model": "gemini-3.1-flash-lite-preview",
    "vh_response_past_scenes_size": 5,
    "vh_response_thinking_level": "medium",
    "vh_response_judge_model": "gemini-3.1-pro-preview",
    "vh_response_judge_thinking_level": "high"
}
```

## 실행 가이드

### 공통 인프라 (모든 Track 공유)

아래 두 스크립트는 **모든 Track의 공통 선행**입니다. A/B Track 실행 전 반드시 완료해야 합니다.

```bash
python identify_keyscene.py                       # O-1: KeyScene 식별 → keypoint_scenes.jsonl
python generate_keyscene_summary.py               # O-2: KeyScene Summary 생성 → keyscene_summary.jsonl (Ground Truth Anchor)
```

### A-Track (Voice Hint)
```bash
python generate_voice_hint.py                     # A-1: Voice Hint 생성 (모드: kss, video, raw, raw_with_mmvlm, imgvlm_chunk2, imgvlm_graph)
python judge_voice_hint.py                        # A-2: Voice Hint Judge (모드: kss, video, raw, raw_with_mmvlm, imgvlm_chunk2, imgvlm_graph)

# 특정 모드만 선택하여 실행할 경우 --modes 인자 사용 (여러 개 지정 가능):
python generate_voice_hint.py --modes imgvlm_chunk2 video
python judge_voice_hint.py --modes imgvlm_chunk2 video
```

### B-Track (VH Response)
```bash
python generate_vh_response.py                    # B-1: 다모드 Response 생성 (video, raw, raw_with_mmvlm, imgvlm_chunk2, imgvlm_graph)
python judge_vh_response.py                       # B-2: Response Judge

# 특정 모드만 선택하여 실행할 경우 --modes 인자 사용 (여러 개 지정 가능):
python generate_vh_response.py --modes imgvlm_chunk2 video
```

### 누락분 자동 재처리 및 Discovery Loop

모든 생성·평가 스크립트는 **재시작 안전(Restart-Safe)** 하게 설계되어 있습니다. 스크립트를 재실행하면 이미 완료된 항목은 건너뛰고 **누락분만 자동으로 재처리**합니다.

또한 `generate_vh_response.py`, `judge_voice_hint.py`, `judge_vh_response.py`는 **Discovery Loop**를 내장하고 있어, 현재 입력 파일에 있는 항목을 모두 처리한 뒤 **입력 파일을 다시 읽어** 새로 추가된 항목이 있는지 확인합니다. 더 이상 처리할 항목이 없을 때 자동 종료됩니다.

```
┌────────────────────────────────────────┐
│          Discovery Loop                │
│                                        │
│  1. 입력 파일 (재)로드                  │
│  2. 미처리 항목 확인                    │
│  3. 없으면 → 종료                       │
│  4. 미처리 항목 처리                    │
│  5. 입력 파일을 다시 읽어 새 항목 감지    │
│     → 있으면 1번으로 돌아감              │
└────────────────────────────────────────┘
```

### 병렬 실행

Discovery Loop 덕분에 파이프라인의 각 단계를 **별도 터미널에서 동시에 실행**할 수 있습니다. 앞 단계에서 생성된 데이터를 뒷 단계가 자동으로 감지하여 처리합니다.

```bash
# Terminal 1 — Voice Hint 생성
python generate_voice_hint.py

# Terminal 2 — VH가 생성되는 대로 평가 (Discovery Loop으로 새 VH 자동 감지)
python judge_voice_hint.py

# Terminal 3 — KSS VH가 생성되는 대로 Response 생성 (Discovery Loop)
python generate_vh_response.py

# Terminal 4 — Response가 생성되는 대로 평가 (Discovery Loop)
python judge_vh_response.py
```

> **참고**: `generate_voice_hint.py`가 선행으로 최소 1개의 KSS 레코드를 생성한 뒤 나머지를 시작하세요. 입력 파일이 아예 존재하지 않으면 오류를 출력하고 종료됩니다.

### 통합 실행 (순차, A+B Track)
```bash
python main.py --input_file content_list.json
```

### Analytics 및 데이터 관리 유틸리티
```bash
# 평가 결과 Excel 리포트 생성
python export_to_excel.py

# 특정 모드 평가 결과 및 생성 데이터를 파일에서 일괄 삭제 (재평가를 위해)
python clean_assets.py --modes imgvlm_chunk2 imgvlm_graph
```

## 주요 산출 데이터 형식 (assets/)

### `vh_response_scores.jsonl` (B-Track 평가 결과)
```json
{
  "content_id": "001_NatGeoKR_Narwhal_6m",
  "query": "일각고래의 엄니가 어떤 역할을 하나요",
  "judge": {
    "video": {
      "answer_relevance": { "rationale": "Directly addresses...", "score": 5 },
      "factual_precision": { "rationale": "Accurate facts...", "score": 4 },
      "response_quality": { "rationale": "Well-structured...", "score": 5 }
    },
    "raw_with_mmvlm": { "...": "..." },
    "imgvlm_chunk2": { "...": "..." },
    "imgvlm_graph": { "...": "..." }
  }
}
```
