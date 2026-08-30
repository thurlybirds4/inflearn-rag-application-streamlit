# 2주차: RAG 이해와 임베딩·Vector DB 개념 정립

날짜: 2026년 7월 24일

<aside>

**목차**

</aside>

# [2주차 학습 목표]

---

🔹 전체 강의의 방향과 학습 내용을 이해하고 실습에 필요한 자료를 활용할 수 있다.

🔹 RAG의 동작 원리를 이해하고 Embedding과 Vector Database가 RAG에서 어떤 역할을 하는지 설명할 수 있다.

# 1. 오늘 한 일

- [x]  강의 방향 및 학습 내용 이해
- [x]  실습 자료 활용
- [x]  RAG의 동작 원리 이해
- [x]  Embedding과 Vector Database가 RAG에서 어떤 역할을 하는지

# 2. 학습 내용 정리

---

## 2.1 강의 방향

- **수강 대상**
    - RAG 구현 과정에서 어려움을 겪고 계신 경우
    - LLM 애플리케이션 개발에 관심이 있는 경우
- **학습 내용**
    - LangChain v0.2 이후 체계를 바탕으로, 프로덕션 환경에 적용 가능한 RAG 파이프라인 설계 
    (강의를 촬영할 당시 기준 v0.2 이였지만 현재는 v1.3x)
    - LangChain + OpenAI 기반 핵심 개발 프레임워크 학습 및 LangSmith를 통한 로깅·모니터링 체계 구축
    - 단순 공식문서 학습을 넘어, 프로덕션 서비스 운영 경험에서 도출된 실전 개발 노하우
- **최종 목표:**
    - 국가법령정보센터 소득세법(docx)을 Knowledge Base로 활용하는 챗봇을 [Streamlit](https://streamlit.io/) 활용하여 개발
    - LLM이 생성한 답변의 정확성과 신뢰성을 검증하기 위해, 답변의 근거가 되는 학습 파일 내 출처를 함께 제공하도록 구성

## 2.2 실습 자료

- **소스 코드**
    - 섹션 1, 2: 코드 실습 없음
    - 섹션 3, 5:  https://github.com/jasonkang14/inflearn-rag-notebook
    - 섹션 4:  https://github.com/jasonkang14/inflearn-streamlit-lecture
- **PPT**
    
    [https://drive.google.com/file/d/1kfjRwMuTlo8FCslpbe-UF7Gnlgmle1fc/view](https://drive.google.com/file/d/1kfjRwMuTlo8FCslpbe-UF7Gnlgmle1fc/view)
    

## 2.3 RAG 동작 원리 이해

- **LLM 모델의 한계**
    - 질문에 대한 정보가 부족한 경우 사실과 다른 내용을 그럴듯하게 생성하는 Hallucination이 발생할 수 있다.
- **RAG의 역할**
    - 외부 문서나 DB에 관련 데이터를 저장해 두고, LLM이 답변을 생성할 때 해당 문서를 먼저 찾아 참고하도록 만든다
- **RAG 각 단어의 의미**
    - Retrieval: 찾아내다 / 질문과 유사한, 관련된 정보를 Vector DB에서 찾아내다.
    - Augmented: 보완하다 / 찾아낸 질문 관련 정보를 프롬프트에 덧붙여서 보완하다.
    - Generation: 생성하다 / 덧붙여진 정보를 참고하여 정확도 높은 답변을 생성한다

<aside>

**[💡 LLM 모델을 만드는 3가지 방법]**

1. 사전 학습: 모델 구조 설정 → 대규모 데이터 준비 → 사전 학습 (대량의 데이터, 충분한 인프라가 필요함. 많은 시간과 비용 소요)
2. 파인 튜닝: 사전학습된 모델에 특정 분야에 특화된 데이터를 추가 학습 (이 또한 많은 시간과 비용 소요)
3. RAG: 특화된 데이터를 Vector DB에 저장한다음, 사전학습된 모델이 사용자의 질문과 관련된 데이터를 DB에서 참고하여 제공해 주는 기법  
(비용 저렴 & 최신 데이터 활용가능 & 할루시네이션 문제 보완)

참고: https://nado-coding.tistory.com/40#(1)%20%EC%82%AC%EC%A0%84%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0%20(Pretraining)-1-2

</aside>

## 2.4 Vector Database과 Embedding이해

- **RAG 역할 복습**
    - 질문 텍스트와 관련된, **유사한 정보를 Vector DB에**서 찾아서 프롬프트에 덧붙여 질문을 보완하여 정확한 답변을 생성하도록 한다.
- **Vector DB란**
    - 텍스트, 문서 등 비정형 데이터를 벡터 임베딩 형태로 저장하고 관리하는 데이터베이스
- **원리 및 역할**
    - **백터 임베딩** 과정을 거치면 유사한 의미를 가진 데이터들이 벡터 공간 내 **가까운 거리에 위치**하게 됨
    - 질문(프롬프트) 입력 시, Vector DB에서 맥락상 가장 가까운 연관 정보를 찾아 프롬프트를 보완(Augment)함

# 2. 기억에 남는 Codes

---

```python
# 오늘 작성한 코드 중 임팩트있었던 부분을 옮겨 적습니다.
2주차 학습내용 (섹션1 & 섹션2)에선 코드 실습 없음 
```

# 3. 이슈 및 해결 과정

---

#### **문제1**

- **설명**
    - 강의 방향 설명에서 랭체인 버전을 v0.2으로 진행함을 알았다 하지만 현재는 v1.3 이상으로 버전 차이가 크다
    - 이는 코드가 그대로 작동하지 않는 문제가 생길 수 있음
- **해결 방법**
    - 프로젝트에서 제공된 `requirements.txt` 파일의 명세에 따라 패키지 버전을 일치시켜 설치하였으며, 라이브러리 간 의존성 충돌 없는 환경 구축을 완료함
    
    ```python
    %pip install -r requirements.txt
    %pip install langchain-google-genai==2.0.1 -q
    ```
    

#### 문제2

- **설명**
    - 제공된 GitHub의 소스코드를 직접 다운로드하는 과정이 번거로움
- **해결 과정**
    - Code IDE에서 `git` 명령어를 사용하여 GitHub 저장소를 직접 가져와 실행한다.
    
    ```python
    # GitHub 저장소 가져오기 (inflearn-rag-notebook의 파일들이 약 5초 내로 로컬에 복붙 된다)
    git clone https://github.com/jasonkang14/inflearn-rag-notebook.git
    ```
    
    ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/2%EC%A3%BC%EC%B0%A8%20RAG%20%EC%9D%B4%ED%95%B4%EC%99%80%20%EC%9E%84%EB%B2%A0%EB%94%A9%C2%B7Vector%20DB%20%EA%B0%9C%EB%85%90%20%EC%A0%95%EB%A6%BD/image.png)
    

# 4. 회의 - 학습 내용 공유

---

### Share Notes & Ideas

- Share Notes:
    - 각 주마다 1명씩 돌아가며 정리내용 발표 (디스코드 활용)
- Ideas:
    - 강의에서 제공하는 GitHub 소스코드를 IDE에서 `git clone`으로 바로 가져와 바로 실행할 수 있도록 공유
    - 강의에서 다루는 랭체인 버전과 현재 버전의 차이가 크다는 문제 공유후 제공 코드 파일에 requirement.txt 파일을

# 5. 다음 학습 계획 수립

---

- [x]  RAG Pipeline 구현
- [x]  Vector Database 변경
- [x]  데이터 전처리