# 4주차: Streamlit 기반 AI 챗봇 개발 및 Web 서비스 배포

날짜: 2026년 8월 7일

<aside>

**목차**

</aside>

# [4주차 학습 목표]

---

🔹 RAG 기반 ChatBot을 구현하고 웹 서비스 형태로 배포할 수 있다.

# 1. 오늘 한 일

- [x]  Rag 기반 챗봇 구현
- [x]  웹 서비스 배포

# 2. 학습 내용 정리

---

## 2.1 Streamlit 설치와 user message 작성

- **1. 환경세팅 및  Streamlit 설치**
    
    ```python
    mkdir inflearn-streamlit
    cd inflearn-streamlit
    
    pyenv virtualenv 3.11.8 inflearn-streamlit
    pyenv local inflearn-streamlit 
    
    git clone https://github.com/jasonkang14/inflearn-streamlit-lecture.git
    pip install -r requirements.txt
    
    # pip install streamlit==1.35.0
    ```
    
- **2. 페이지 레이아웃 및 헤더 설정**
    - st.set_page_config(page_title="소득세 챗봇", page_icon="💬")
    - st.title("💬 소득세 챗봇")
    - st.caption("소득세에 관련된 모든것을 답해드립니다!")
    
    ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/4%EC%A3%BC%EC%B0%A8%20Streamlit%20%EA%B8%B0%EB%B0%98%20AI%20%EC%B1%97%EB%B4%87%20%EA%B0%9C%EB%B0%9C%20%EB%B0%8F%20Web%20%EC%84%9C%EB%B9%84%EC%8A%A4%20%EB%B0%B0%ED%8F%AC/image.png)
    
- **3. 사용자 입력 처리 및 화면 렌더링 + 사용데 데이터 계속 유지**
    
    ```python
    **# 웹 브라우저를 닫기 전까지 사용자의 데이터를 계속 유지(기억)**
    if 'message_list' not in st.session_state:
        st.session_state.message_list = []
    
    **# 이전 채팅 화면에 보이도록 하기**
    for message in st.session_state.message_list:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # **사용자 입력 처리 및 AI 답변 화면 렌더링**
    if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    		# 사용자 입력 레이아웃
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})
        
        # AI 답변 레이아웃
        with st.chat_message("ai"):
            st.write("여기는 AI 메세지")
        st.session_state.message_list.append({"role": "ai", "content": "여기는 AI 메세지"})
    
    ```
    
    ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/4%EC%A3%BC%EC%B0%A8%20Streamlit%20%EA%B8%B0%EB%B0%98%20AI%20%EC%B1%97%EB%B4%87%20%EA%B0%9C%EB%B0%9C%20%EB%B0%8F%20Web%20%EC%84%9C%EB%B9%84%EC%8A%A4%20%EB%B0%B0%ED%8F%AC/image%201.png)
    

## 2.2 **LangChain으로 작성한 코드를 활용한 LLM 답변 생성**

- **1. LangChain 기반 AI 답변 출력 구현**
    
    ```python
    def get_ai_response(user_message):
        
        embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        index_name = 'tax-markdown-index'
        database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    
        llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')
        prompt = hub.pull("rlm/rag-prompt")
        retriever = database.as_retriever(search_kwargs={'k': 4})
    
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
        dictionary = ["사람을 나타내는 표현 -> 거주자"]
    
        prompt = ChatPromptTemplate.from_template(f"""
            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
            그런 경우에는 질문만 리턴해주세요
            사전: {dictionary}
            
            질문: {{question}}
        """)
    
        # 3. 질문 전처리 체인 실행 (사용자 질문 -> 사전 변환된 질문)
        dictionary_chain = prompt | llm | StrOutputParser()
        user_query = dictionary_chain.invoke({"question": user_message})
        
        # 4. RetrievalQA 체인 실행 (반환값은 dict 형태: {'query': ..., 'result': ...})
        ai_message = qa_chain.invoke({"query": user_query})
        
        # 원본 반환 형식 유지 (RetrievalQA의 반환 딕셔너리 그대로 return)
        return ai_message
        
        
    # **사용자 입력 처리 및 AI 답변 화면 렌더링 수정**
    if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    		# 사용자 입력 레이아웃
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.message_list.append({"role": "user", "content": user_question})
        
        # AI 답변 레이아웃
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
    
    ```
    
    ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/4%EC%A3%BC%EC%B0%A8%20Streamlit%20%EA%B8%B0%EB%B0%98%20AI%20%EC%B1%97%EB%B4%87%20%EA%B0%9C%EB%B0%9C%20%EB%B0%8F%20Web%20%EC%84%9C%EB%B9%84%EC%8A%A4%20%EB%B0%B0%ED%8F%AC/image%202.png)
    
- **2. AI 답변 함수 분리**
    
    ```python
    def get_retriever():
        embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        index_name = 'tax-markdown-index'
        database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
        retriever = database.as_retriever(search_kwargs={'k': 4})
        return retriever
    
    def get_llm(model="gemini-3.6-flash"):
        return ChatGoogleGenerativeAI(model=model)
    
    def dictionary_chain():
        dictionary = ["사람을 나타내는 표현 -> 거주자"]
    
        prompt = ChatPromptTemplate.from_template(f"""
            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
            그런 경우에는 질문만 리턴해주세요
            사전: {dictionary}
            
            질문: {{question}}
        """)
    
        # 3. 질문 전처리 체인 실행 (사용자 질문 -> 사전 변환된 질문)
        llm = get_llm()
        dictionary_chain = prompt | llm | StrOutputParser()
        return dictionary_chain
    
    def get_qa_chain():
        llm = get_llm()
        retriever = get_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
        return qa_chain
    
    def get_ai_response(user_message):
        dictionary_chain = get_preprocess_chain()
        qa_chain = get_qa_chain()
        tax_chain = {"query": dictionary_chain} | qa_chain
        ai_message = tax_chain.invoke({"query": user_message})
        return ai_message["result"]
    ```
    

## 2.3 Chat History 추가와 streaming 구현

- **1. 대화 히스토리 관리 및 이전 문맥 유지하기위해 get_qa_chain함수,**
    
    ```python
    # MessagesPlaceholder , create_stuff_documents_chain
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    def get_rag_chain(): # **get_qa_chain ->** get_rag_chain
        llm = get_llm()
        retriever = get_retriever()
    
        # prompt = hub.pull("rlm/rag-prompt")
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
        system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain
    
    ```
    
- **2. AI답변을 깔끔하게 만들기위해 get_ai_response 함수수정**
    
    ```python
    def get_ai_response(user_message):
        dictionary_chain = get_dictionary_chain()
        rag_chain = get_rag_chain()
        tax_chain = ({"input": dictionary_chain} | rag_chain).pick('answer')
        ai_response = tax_chain.stream(
            {"question": user_message},
            config={"configurable": {"session_id": "abc123"}}
          )
        return ai_response["answer"]
    ```
    

## 2.5 지금까지 작성했던 함수들을 리팩토링

- **기존:** 하나의 파이썬 파일에 모든 소스코드 작성
    - chat.py:
        - Streamlit UI/UX 코드
        - **`get_session_history`** (세션별 대화 기록 관리 함수)
        - **`get_retriever`** (Pinecone 벡터 DB 및 임베딩 로드 함수)
        - **`get_llm`** (Gemini 모델 객체 생성 함수)
        - **`get_dictionary_chain`** (사용자 용어 사전 변환 체인 함수)
        - **`get_rag_chain`** (검색 및 대화 기록 결합 RAG 체인 함수)
        - **`get_ai_response`** (스트리밍 답변 생성 메인 함수 함수)
- **리펙토링후:** UI/UX, 로직 기능을 독립된 모듈로 분리
    - chat.py: Streamlit UI/UX 코드
    - llm.py : Rag 파이프라인 함수 정리

## 2.4 Streamlit Cloud를 활용한 서비스 배포

> https://streamlit.io/cloud
> 
- **1. IDE 작업경로에서 기존 가상환경의 버전을 그대로 추출후 requirements.txt에 작성후 git push**
    
    ```python
    pip freeze
    pip freeze > requirements.txt
    
    git add requirements.txt
    git commit -m "Add requirements.txt"
    git push origin main
    ```
    
- **2. github에 streamlit Chatbot 코드 push**
    
    ```python
    # 기본 브랜치 이름을 main으로 변경
    git branch -M main
    
    # 원격 저장소 연결 (YOUR_REPOSITORY_URL 주소 변경)
    git remote add origin YOUR_REPOSITORY_URL
    
    # GitHub로 푸시
    git push -u origin main
    ```
    
- **3. streamlit cloud 회원가입후 (github 연동 계정 추천) → Create app**
- **4. Chatbot코드소스가 담겨있는 github Repository, Branch, Main file path(chat.py), Advanced setting(환경변수 api값을 문자열로) 입력**
    
    ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/4%EC%A3%BC%EC%B0%A8%20Streamlit%20%EA%B8%B0%EB%B0%98%20AI%20%EC%B1%97%EB%B4%87%20%EA%B0%9C%EB%B0%9C%20%EB%B0%8F%20Web%20%EC%84%9C%EB%B9%84%EC%8A%A4%20%EB%B0%B0%ED%8F%AC/image%203.png)
    

# 2. 기억에 남는 Codes

---

```python
# 배경: 전 대화 맥락을 기억하지 못하는 단발성 질의응답 구조였기에, 사용자가 매번 맥락을 포함한 완성형 질문을 재입력해야 하는 UX적 불편함이 존재
## 1. 세션별 대화 기록(History) 메모리 관리
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

## 2. 대화 맥락을 반영한 리트리버 및 히스토리 체인 결합
def get_rag_chain():
    # ... (생략) ...
    # 이전 대화와 최신 질문을 결합해 Standalone Question으로 변환
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # 세션 히스토리를 대화형 RAG 체인에 바인딩
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain
```

# 3. 이슈 및 해결 과정

---

#### **문제1**

- **설명**
    - ModuleNotFoundError: This app has encountered an error. error message is redacted to prevent data leaks …
    - 로컬에서 잘 돌아갔던 코드를 웹에 배포를 하고나니 모듈을 찾을수없다는 오류를 내뱉었다
- **해결과정**
    - 웹에 배포할때 requirements.txt에 langchain_google_genai의 버전 명시를 빼먹었었다
    - langchain_google_genai의 버전만 확인하여 수정할수 있지만 쓰고있는 모든 라이브러리의 버전정보를 freeze하여 다시 requirements.txt에 적재했다

#### 문제2

- **문제 상황:**
    - 스트리밍 Generator 객체에 대한 잘못된 인덱싱 접근 오류
    - `get_ai_response` 함수 내에서 `tax_chain.stream(...)`으로 반환된 스트리밍 데이터를 처리할 때 `ai_response["answer"]` 형태의 딕셔너리 키 접근을 시도하여 `TypeError` 또는 `KeyError`가 발생
- **해결 과정:**
    - `.stream()` 메서드는 전체 결과 딕셔너리가 아닌 **문자열 조각(Chunk)을 순차적으로 내보내는 Iterator(Generator) 객체**를 반환 하기때문에. 이전 단계에서 `.pick('answer')`를 통해 스트림 출력을 답변 텍스트로 이미 필터링했으므로, 딕셔너리처럼 키 접근을 할 수 없음
    - `return ai_response["answer"]` 대신 `return ai_response`로 객체 자체를 반환하도록 수정하여, Streamlit의 `st.write_stream`이 스트림 청크를 한 글자씩 받아 화면에 실시간으로 출력을 완성할 수 있도록 정상화

# 4. 회의 - 학습 내용 공유

---

### Reflection

- 지금까지 소득세법 데이터를 가지고 RAG 챗봇을 구축하였다.
- 이 지식을 확장하여, 지인의 병원 업무 데이터를 결합한 AI 어시스턴트 구축을 할 수 있을 것 같다.
    
    ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/4%EC%A3%BC%EC%B0%A8%20Streamlit%20%EA%B8%B0%EB%B0%98%20AI%20%EC%B1%97%EB%B4%87%20%EA%B0%9C%EB%B0%9C%20%EB%B0%8F%20Web%20%EC%84%9C%EB%B9%84%EC%8A%A4%20%EB%B0%B0%ED%8F%AC/image%204.png)
    

# 5. 다음 학습 계획 수립

---

- [x]  LLM 성능평가
- [x]  Hugging Face 오픈소스 언어모델 활용
- [x]  Hugging Face 오픈소스를 활용한 RAG PipeLine 구성
- [x]  Prompt Templete 활용