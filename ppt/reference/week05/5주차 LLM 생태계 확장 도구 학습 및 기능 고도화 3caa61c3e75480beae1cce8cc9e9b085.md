# 5주차: LLM 생태계 확장 도구 학습 및 기능 고도화

날짜: 2026년 8월 14일

<aside>

**목차**

</aside>

# [5주차 학습 목표]

---

🔹 RAG와 LLM 애플리케이션을 확장하기 위한 다양한 도구와 방법을 이해한다.

# 1. 오늘 한 일

- [x]  LLM 성능평가

# 2. 학습 내용 정리

---

## 2.1 LLM 성능평가

- **성능평가가 중요한이유:**
    - 서비스를 안정적으로 운영하기 위해서
    - 사용자가 정확한 정보를 받을 수 있는지 평가가 필요함
    - 특정 질의에 따른 정답과 실제 llm이 응답한 답변의 의미가 같은지 평가하는 모델이 존재함
- **실습**
    
    > https://docs.langchain.com/langsmith/evaluation#langsmith-evaluation
    > 
    > 
    > https://smith.langchain.com/
    > 
    - **1. langsmith 회원가입 & api키 발급 & 환경변수 입력**
        
        ```python
        LANGCHAIN_API_KEY=...발급한 키값
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_PROJECT=default
        LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
        ```
        
    - **2. 질의에 따른 정답 데이터 준비**
        
        ```python
        inputs=[
                {"input_question": "제1조에 따른 소득세법의 목적은 무엇인가요?"},
                {"input_question": "'거주자'는 소득세법에서 어떻게 정의되나요?"},
                {"input_question": "'비거주자'는 소득세법에 따라 어떻게 정의되나요?"},
                {"input_question": "소득세법에 따른 '내국법인'은 누구를 의미하나요?"},
        				....
        ]
        outputs=[
                {"output_answer": "소득세법의 목적은 소득의 성격과 납세자의 부담능력에 따라 적정하게 과세함으로써 조세부담의 형평을 도모하고 재정수입의 원활한 조달에 이바지하는 것입니다."},
                {"output_answer": "'거주자'는 한국에 주소를 두거나 183일 이상 거소를 둔 개인을 의미합니다."},
                {"output_answer": "'비거주자'는 거주자가 아닌 개인을 의미합니다."},
                {"output_answer": "'내국법인'은 법인세법 제2조 제1호에 따른 내국법인을 의미합니다."},
                ...
        ```
        
    - **3. langsmith에 해당 질의 응답 저장**
        
        ```python
        from langsmith import Client
        client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
        
        dataset_name = "income_tax_dataset"
        dataset = client.create_dataset(dataset_name)
        client.create_examples(
            inputs=[
                {"input_question": "'비거주자'는 소득세법에 따라 어떻게 정의되나요?"},
        				....
             ]
            outputs=[
                {"output_answer": "'비거주자'는 거주자가 아닌 개인을 의미합니다."},        ....
            ]
            
            metadata= [
                {"contexts": "제1조의2(정의) “비거주자”란 거주자가 아닌 개인을 말한다."},        ....
            ],
            dataset_id=dataset.id,
        )
        ```
        
        ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/5%EC%A3%BC%EC%B0%A8%20LLM%20%EC%83%9D%ED%83%9C%EA%B3%84%20%ED%99%95%EC%9E%A5%20%EB%8F%84%EA%B5%AC%20%ED%95%99%EC%8A%B5%20%EB%B0%8F%20%EA%B8%B0%EB%8A%A5%20%EA%B3%A0%EB%8F%84%ED%99%94/image.png)
        
    - **4. 이전 Rag pipline에서 응답한 답변과 정답 비교**
        
        ```python
        # Rag pipline
        ...
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        index_name = 'tax-markdown-index'
        database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
        retriever = database.as_retriever()
        
        # 래그 봇 클래스 만들기
        import openai
        from langsmith import traceable
        from langsmith.wrappers import wrap_openai
        
        class RagBot:
            def __init__(self, retriever, model: str ='gemini-3.6-flash'): # LLM 모델 생성
            @traceable()
            def retrieve_docs(self, question):    # 질문과 유사한 문장 반환
            @traceable()
            def invoke_llm(self, question, docs): # 프롬프트 생성 및 LLM 응답 반환 -> Evaluators를 활용해서 `answer`와 `contexts`를 평
            @traceable()
            def get_answer(self, question: str):  # 쿼리 응답 반환
            ...
        
        #  평가 함수 생성
        def predict_rag_answer(example: dict):              # 답변만 평가
        def predict_rag_answer_with_context(example: dict): # Context를 활용해서 hallucination을 평가할 때
        def answer_evaluator(run, example) -> dict:         #  RAG 답변 성능을 측정하기 위한 evaluator
        def answer_helpfulness_evaluator(run, example) -> dict:    # 답변이 사용자의 질문에 얼마나 도움되는지 판단하는 Evaluator
        def answer_hallucination_evaluator(run, example) -> dict:  # hallucination 판단을 위한 Evaluator
        
        # 평가 실행 ⚠️⚠️ Gemini API 쿼터 초과로인한 실습 불가 ⚠️⚠️
        from langsmith.evaluation import evaluate
        dataset_name = "income_tax_dataset"
        experiment_results = evaluate(
            predict_rag_answer_with_context, # 어떤 함수를 활용해서 LLM 답변을 확인할지 지정, hallucination 판단 여부에 따라 `with_context` 사용
            data=dataset_name, # Evaluation에 사용될 dataset의 이름
            evaluators=[answer_evaluator, answer_helpfulness_evaluator, answer_hallucination_evaluator], # 실행할 Evaluator의 종류
            experiment_prefix="inflearn-evaluator-lecture-hallucination",
            metadata={"version": "income tax v1, gemini-3.6-flash"}, 
            client=client,  # <--- 필수 추가
            max_concurrency=1,  # <--- 동시 실행 수를 1개로 제한 (API 호출 분산)
        )
        ```
        

# 2. 기억에 남는 Codes

---

```python
# 직접 평가 규칙을 처음부터 작성하지 않고, 검증된 LLM-as-a-Judge 프롬프트를 불러와 자동 점수 매기기(Scoring) 자동화를 구현
## LangChain Hub에서 표준 평가 프롬프트 로드
grade_prompt_answer_accuracy = hub.pull("langchain-ai/rag-answer-vs-reference")

## 프롬프트와 LLM을 결합하여 평가 체인 생성
answer_grader = grade_prompt_answer_accuracy | llm
```

```python
# 단일 실행으로 3가지 핵심 RAG 메트릭을 동시에 측정하는 기능적 구조
evaluators=[
    answer_evaluator,               # 1. 정답 유사도 (Answer Accuracy)
    answer_helpfulness_evaluator,   # 2. 질문 답변 충실도 (Helpfulness)
    answer_hallucination_evaluator  # 3. 환각 발생 여부 (Hallucination)
]
```

---

# 5. 다음 학습 계획 수립

---

- [ ]  ppt 제작