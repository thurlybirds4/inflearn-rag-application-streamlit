# Inflearn LLM PPT

인프런 LLM 애플리케이션 과정의 발표를 위한 HTML 슬라이드입니다.

## 슬라이드 열기
**온라인:**
https://thurlybirds4.github.io/inflearn-rag-application-streamlit/01.html

## 깃허브 링크
https://github.com/thurlybirds4/inflearn-rag-application-streamlit

---

# Inflearn Streamlit Project

## 개요 (Overview)

이 저장소는 **LangChain과 Streamlit을 활용하여 RAG(Retrieval Augmented Generation, 검색 증강 생성) 애플리케이션을 구축한 프로젝트**입니다.

이 애플리케이션은 **대한민국 소득세법(소득세법)**을 기반으로 사용자의 질문에 대한 정보와 답변을 제공하는 것을 주요 목적으로 합니다.

또한 **채팅 기록(Chat History)**과 **Few-Shot Learning 템플릿**을 함께 활용하여, 사용자의 질문에 대해 보다 정확하고 문맥에 맞는 답변을 생성하도록 구성했습니다.

## 주요 기능 (Features)

* **LangChain 연동 (LangChain Integration)**
  LangChain을 활용하여 LLM(대규모 언어 모델)을 효율적으로 관리하고 상호작용할 수 있도록 구성했습니다.

* **Streamlit 인터페이스 (Streamlit Interface)**
  Streamlit을 이용하여 사용자가 쉽게 사용할 수 있는 웹 기반 인터페이스를 제공합니다.

* **검색 증강 생성 (Retrieval Augmented Generation, RAG)**
  관련 정보를 먼저 검색한 후 검색된 내용을 LLM에 제공하여, 보다 정확하고 문맥에 맞는 답변을 생성합니다.

* **지식 베이스 (Knowledge Base)**
  대한민국 **소득세법**을 주요 지식 데이터로 활용합니다.

* **대화 기록 (Chat History)**
  이전 대화 내용을 유지하여 사용자의 후속 질문에 대해서도 문맥을 고려한 답변을 제공합니다.

* **Few-Shot Learning 템플릿**
  미리 정의된 예시 및 템플릿을 활용하여 모델의 답변 품질과 일관성을 향상시킵니다.

## 설치 방법 (Installation)

### 1. 저장소 복제

```sh
git clone https://github.com/jasonkang14/inflearn-streamlit.git
cd inflearn-streamlit
```

### 2. 가상환경 생성 및 활성화

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. 필요한 패키지 설치

```sh
pip install -r requirements.txt
```

## 실행 방법 (Usage)

### 1. Streamlit 애플리케이션 실행

```sh
streamlit run chat.py
```

### 2. 웹 브라우저 접속

실행 후 터미널에 표시되는 **로컬 URL(Local URL)**을 웹 브라우저에서 열어 애플리케이션을 사용할 수 있습니다.

## 프로젝트 구조 (Project Structure)

* `chat.py`
  Streamlit 기반의 사용자 인터페이스를 실행하는 **메인 애플리케이션 파일**입니다.

* `llm.py`
  **지식 베이스와 LLM의 상호작용을 처리하는 기능**을 포함하고 있습니다.

* `config.py`
  답변 생성에 사용되는 **Few-Shot Learning 템플릿**이 정의되어 있는 파일입니다.

## 동작 방식 (How It Works)

### 1. 데이터 검색 (Data Retrieval)

사용자의 질문을 분석하고, 소득세법 지식 베이스에서 질문과 관련성이 높은 내용을 검색합니다.

### 2. 문맥 처리 (Contextual Processing)

이전 대화 기록을 함께 활용하여 현재 질문의 맥락을 파악하고, 연속적인 대화가 가능하도록 합니다.

### 3. 템플릿 기반 답변 생성 (Template-Based Generation)

Few-Shot Learning 템플릿을 활용하여 LLM이 보다 정확하고 관련성 높은 답변을 생성할 수 있도록 합니다.

### 4. 사용자 인터페이스 (User Interface)

Streamlit을 기반으로 직관적인 웹 인터페이스를 제공하여 사용자가 LLM 애플리케이션과 쉽게 상호작용할 수 있도록 구성했습니다.

## 기여 (Contributing)

프로젝트에 대한 기여를 환영합니다.

개선 사항이나 버그 수정이 필요한 경우 **Pull Request**를 제출하거나 **Issue**를 생성하여 의견을 공유할 수 있습니다.

## 참고 및 감사 (Acknowledgments)

* **LangChain**
* **Streamlit**
* 프로젝트에 기여해 주신 모든 개발자와 사용자
