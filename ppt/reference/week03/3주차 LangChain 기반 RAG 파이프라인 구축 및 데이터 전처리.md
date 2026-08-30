# 3주차: LangChain 기반 RAG 파이프라인 구축 및 데이터 전처리

날짜: 2026년 7월 31일

<aside>

**목차**

</aside>

# [3주차 학습 목표]

---

🔹 LangChain을 활용해 기본적인 RAG Pipeline을 구현하고, Vector Database 변경 및 데이터 전처리 등을 적용할 수 있다.

# 1. 이번 주차에 한 일

- [x]  RAG Pipeline 구현
- [x]  Vector Database 변경
- [x]  데이터 전처리

# 2. 학습 내용 정리

---

## 2.1 사전 학습 (LangChain 핵심 개념 with 공식문서)

### 2.1.1 L**angChain Integration 및 제미나이 연동**

> https://docs.langchain.com/oss/python/concepts/providers-and-models
> 
> 
> https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
> 
- **LangChain이란?**
    - 초거대 언어 모델(LLM)을 활용해 복잡한 AI 애플리케이션이나 에이전트를 손쉽게 개발할 수 있도록 돕는 **오픈소스 프레임워크**
    - 단순히 AI 모델에 질문을 주고 답변을 받는 수준을 넘어, 외부 데이터(문서, DB), API, 메모리, 검색 엔진 등을 **하나의 사슬(Chain)처럼 연결**할 수 있게 연결 고리 역할을 해줌
- **LangChain Integration 이란?**
    - LLM 회사들의 서비스를 LangChain에서 쓸 수 있게 만든 **실제 연동 모듈/기술**
- **제미나이 연동 예시 (ChatGoogleGenerativeAI)**
    
    ```python
    # 랭체인 제미나이 모듈 설치 
    pip install -U langchain-google-genai
    
    # env파일에 제미나이 API 환경변수이름을 GOOGLE_API_KEY로 하고 키값 할당하기
    import getpass
    import os
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    
    # 모델 호출에 대한 자동 추적을 활성화하려면 LangSmith API 키를 설정
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
    os.environ["LANGSMITH_TRACING"] = "true"
    
    # 제미나이 모델 인스턴스화
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        temperature=1.0,  # 클수록 창의성이 높음 # Gemini 3.0+ defaults to 1.0 
        max_tokens=None, 
        timeout=None,
        max_retries=2,
        # other params...
    )
    ```
    

### **2.1.2 Document Loaders (문서 불러오기)**

> https://docs.langchain.com/oss/python/integrations/document_loaders
> 
- **Document Loaders란?**
    - 텍스트 파일, Word(.docx), PDF, CSV, 등 다양한 데이터를 LangChain에서 다룰 수 있는 Document 객체 표준 형식으로 불러오는 연동 모듈
- **Word(.docx) 불러오기 예시 (Docx2txtLoader)**
    
    ```python
    from langchain_community.document_loaders import Docx2txtLoader
    
    loader = Docx2txtLoader("소득세법.docx")
    docs = loader.load()
    ```
    

### 2.1.3 **Text Splitter (문서 분할하기)**

> https://docs.langchain.com/oss/python/integrations/splitters
> 
- **Text Splitter란?**
    - 로드된 긴 문서를 LLM의 컨텍스트 창 크기와 검색 효율에 맞춰 의미 있는 단위의 작은 조각(Chunk)으로 나누는 모듈
    - Word(.docx)문서의 일반적인 Text Splitter
        
        
        | **구분** | **CharacterTextSplitter** | **RecursiveCharacterTextSplitter** |
        | --- | --- | --- |
        | **분할 방식** | 지정한 1개 구분자로만 문서를 쪼갬
        `"\n\n"` | 다중 구분자 리스트를 순차적으로 시도
        `["\n\n", "\n", " ", ""]`
        
        가장 큰 단위(`\n\n`)부터 쪼개보고, 청크가 너무 크면 다음 단위(`\n`), 또 크면 ( ``) 순으로 내려가며 쪼갬 |
        | **문맥 보존** | 단락이나 문장이 어색하게 잘릴 확률이 높음 | 단락 → 문장 → 단어 순으로 문맥을 최대한 유지함 |
- **문서 분할 예시 (RecursiveCharacterTextSplitter)**
    
    ```python
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)
    ```
    

### 2.1.4 Text Embedding (문서 벡터화하기)

> https://docs.langchain.com/oss/python/integrations/embeddings/
> 
> 
> https://ai.google.dev/gemini-api/docs/embeddings?hl=ko
> 
- **Text Embedding이란?**
    - 텍스트(문장, 문서 조각)를 의미적 유사도가 반영된 숫자의 배열인 벡터(Vector)로 변환하는 기술
    - RAG 시스템에서 질문(Query)과 가장 의미가 가까운 문서 조각을 찾기 위해 필수적임
- **Google 임베딩 모델 사용 예시 (GoogleGenerativeAIEmbeddings)**
    
    ```python
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    # 최신 표준 임베딩 모델 설정 (models/text-embedding-004)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    ```
    

### 2.1.5 Vector Store - Chroma (벡터 DB에 저장하기)

> https://docs.langchain.com/oss/python/integrations/vectorstores
https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
> 

### 2.1.7 Vector Store - Pinecone

> https://docs.langchain.com/oss/python/integrations/vectorstores
https://docs.langchain.com/oss/python/integrations/vectorstores/pinecone
> 
> 
> https://www.pinecone.io/
> 

### 2.1.6 **LangChain Hub**

> https://smith.langchain.com/hub
> 
- **LangChain Hub란?**
    - 잘쓰여진 "AI 프롬프트"를 공유하고 버전 관리하는 공간
        
        ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/3%EC%A3%BC%EC%B0%A8%20LangChain%20%EA%B8%B0%EB%B0%98%20RAG%20%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%98%EB%A6%AC/image.png)
        

## 2.2 실습

### 2.2.1 RAG Pipeline 구현

#### 1) 환경 설정과 LangChain의ChatGoogleGenerativeAI를 활용한 검증

- **1. 프로젝트 생성 및 가상 환경 세팅**
    
    ```python
    # 프로젝트 디렉토리 생성 (in terminal)
    mkdir inflearn-llm-application-ver2
    
    # 파이썬 가상환경 생성 (가상환경 명은 프로젝트명과 동일하게)
    pyenv virtualenv 3.10 inflearn-llm-application-ver2
    
    # 해당 디렉토리에서 사용할 가상환경 지정 # activate 하지 않아도됨
    pyenv local inflearn-llm-application-ver2 
    
    # IDE 환경도 방금만든 가상환경 선택
    # IDE 우측 상위 "select kernel" -> "inflearn-llm-application-ver2"
    ```
    
- **2. 환경 변수 파일 세팅**
    
    ```python
    # 환경변수 파일 생성
    touch .env
    
    # env 파일에 LLM API key 저장하기
    GOOGLE_API_KEY=---------------------------------------------
    ```
    
- **3. 제미나이 답변 생성하기 실습**
    
    ```python
    ######## 제미나이 모델 모듈 다운 ########
    # 1. 제미나이 연동 모듈 다운후 불러오기
    %pip install -U langchain-google-genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # 2. env 파일에 있는 환경변수(제미나이 api) 불러오기
    from dotenv import load_dotenv
    import os
    
    ######## LLM 답변 생성 ########
    # 1. 제미나이 모델 인스턴스 만들기
    llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')
    
    # 2. 모델에 질문을 invoke하여 답변 생성하기
    try:
        ai_message = llm.invoke("상명대는 어떤 대학교 인가요?")
        print(ai_message.text)
    except Exception as e:
        print(e.message)
    
    ```
    

#### 2) LangChain과 Chroma를 활용한 RAG 구성

- **1. LLM의 Knowledge Base가 될 소득세법(DOCX) 문서 준비**
    
    > https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EC%86%8C%EB%93%9D%EC%84%B8%EB%B2%95
    > 
- **2. Docx2txtLoader를 이용해 문서 데이터 로드하기**
    
    ```python
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader('./tax_docs/tax.docx')
    ```
    
- **3. RecursiveCharacterTextSplitter를 이용해 문서를 청크로 쪼개기**
    
    ```python
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    document_list = loader.load_and_split(text_splitter=text_splitter)
    len(document_list)
    ```
    
- **4. GoogleGenerativeAIEmbeddings를 이용해 청크 데이터를 임베딩(백터화)하기**
    
    ```python
    # from dotenv import load_dotenv
    # from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    ```
    
- **5. Chroma에 백터 데이터 저장**
    
    ```python
    from langchain_chroma import Chroma
    import time
    
    database = Chroma(
          embedding_function=embedding,
          # ⭐️ 크로마는 램에 저장되기 때문에 
          # ⭐️ 디스크에 저장하기 위해서는 폴더 지정! 
          # ⭐️ persist_directory 옵션 사용
          persist_directory="./chroma-directory",   # 저장할 폴더이름
          collection_name="chroma-tax-table",       # 그 안의 테이블/서랍 이름 # 디폴트값은 "langchain" 
    )
    
    def add_documents_with_retry(database, documents, max_retries=5):
        for attempt in range(max_retries): # [0, 1, 2, 3, 4]
            try:
                database.add_documents(documents) # DB에 문서를 추가로 넣는 함수
                return True # 정상적으로 넣었으면 True 반환
    
            except Exception as e: # add_documents 중 예외가 나면 # 메시지 문자열로 바꿈
                error_message = str(e) 
                # 429 또는 RESOURCE_EXHAUSTED 메시지가 아닌 에러라면 raise (에러를 그대로 발생시킴, 실행 중단)
                if (
                    "429" not in error_message
                    and "RESOURCE_EXHAUSTED" not in error_message
                ):
                    raise
                # 429 또는 RESOURCE_EXHAUSTED 메시지가 나오면 잠시 대기 후 재시도
                else:
                  wait_seconds = (2 ** attempt) * 10
                  print(f"사용량 제한 발생, {wait_seconds}초 후 재시도")
                  time.sleep(wait_seconds)  # 슬립한다음에 최대 5번 반복
    
        return False
        
        
    batch_size = 5
    for start in range(0, len(document_list), batch_size): # 0~312까지 5씩 증가한 리스트 [0, 5, ..., 310]]
        batch = document_list[start:start + batch_size] # document_chunk_list[0:5], [5:10], ..., [310:315]
    
        # 문서를 넣는 함수
        success = add_documents_with_retry(
            database=database,
            documents=batch  # documents는 5개청크씩 분할해서 넣음!
        )
    
        if not success:
            print(f"{start}번째 청크부터 저장 실패")
            break
    
        print(
            f"{start + 1}번 ~ "
            f"{start + len(batch)}번 청크 저장 완료"
        )
    
        time.sleep(5)
    ```
    
- **6. 질문과 유사한 문맥 정보를 DB에서 추출**
    
    ```python
    query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
    
    # `k` 값을 조절해서 얼마나 많은 데이터를 불러올지 결정
    retrieved_docs = database.similarity_search(query, k=3)
    ```
    
- **7. f-string으로 프롬프트 보완하기**
    
    ```python
    # 1. llm 객체 준비
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')
    
    # 2. 기존 프롬프트  보완
    prompt = f"""
    [Identity]
    - 당신은 최고의 한국 소득세 전문가 입니다
    - [Context]를 참고해서 사용자의 질문에 답변을 주세요
    
    [Context]
    {retrieved_docs}
    
    [Question]
    {query}
    """
    
    llm.invoke(prompt)
    ```
    
- **8. LangChain 체인(Chain) + Hub 활용하기**
    
    ```python
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")
    print(prompt.messages[0].prompt.template)
    #  You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    
    from langchain.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=database.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    ai_message = qa_chain.invoke({"query": query})
    ai_message
    # {'query': '연봉 5천만원인 직장인의 소득세는 얼마인가요?',
    #  'result': '제공된 문서에는 연봉 5천만원에 적용되는 구체적인 근로소득공제액 계산 기준과 종합소득세율 정보가 포함되어 있지 않습니다. 따라서 제공된 정보만으로는 연봉 5천만원인 직장인의 정확한 소득세를 알 수 없습니다.'}
    ```
    

### 2.2.2 Vector Database 변경

#### 1) LangChain을 활용한 Vector Database 변경 (Chroma → Pinecone)

- **~~1. LLM의 Knowledge Base가 될 소득세법(DOCX) 문서 준비~~**
    - 2.2.1 - 2) 와 같음
- **~~2. Docx2txtLoader를 이용해 문서 데이터 로드하기~~**
    - 2.2.1 - 2) 와 같음
- **~~3. RecursiveCharacterTextSplitter를 이용해 문서를 청크로 쪼개기~~**
    - 2.2.1 - 2) 와 같음
- **~~4. GoogleGenerativeAIEmbeddings를 이용해 청크 데이터를 임베딩(백터화)하기~~**
    - 2.2.1 - 2) 와 같음
- **5. Pinecone DB API 발급후 index 만들기**
    
    <aside>
    
    - **index name:** tax-index
    - **Configuration:** text-embedding-3-large 선택 → Dimension 3072로 바꾸기 → 나머진 티폴트
    </aside>
    
- **6. Pinecone에 백터 데이터 저장**
    
    ```python
    database = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding
    )
    
    # 2. 재시도 함수 (Chroma와 동일하게 사용)
    def add_documents_with_retry(database, documents, max_retries=5):
        for attempt in range(max_retries):
            try:
                database.add_documents(documents)
                return True
            except Exception as e:
                error_message = str(e)
                # Pinecone 및 OpenAI API 사용량 초과 에러(429, RESOURCE_EXHAUSTED, Rate limit) 감지
                if (
                    "429" not in error_message
                    and "RESOURCE_EXHAUSTED" not in error_message
                    and "rate_limit" not in error_message.lower()
                ):
                    raise
                else:
                    wait_seconds = (2 ** attempt) * 10
                    print(f"사용량 제한(Rate Limit) 발생, {wait_seconds}초 후 재시도...")
                    time.sleep(wait_seconds)
    
        return False
    
    # 3. 배치 단위로 나누어 저장 실행
    batch_size = 5
    
    for start in range(0, len(document_list), batch_size):
        batch = document_list[start : start + batch_size]
    
        success = add_documents_with_retry(
            database=database,
            documents=batch
        )
    
        if not success:
            print(f"❌ {start}번째 청크부터 저장 실패")
            break
    
        print(f"✅ {start + 1}번 ~ {start + len(batch)}번 청크 저장 완료")
        
        # 파인콘 및 임베딩 API 쿼타 보호를 위해 5초 대기
        time.sleep(5)
    ```
    
- **7.  LangChain 체인(Chain) + Hub 활용하기**
    
    ```python
    # 랭체인 준비물
    ## 1. llm
    llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')
    
    ## 2. 질문과 유사한 문맥정보 k개 추출하는 기능
    retriever = database.as_retriever(search_kwargs={'k': 4})
    
    ## 3. 허브 프롬프트
    prompt = hub.pull("rlm/rag-prompt")
    
    # 랭체인 만들기
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # 랭체인 질문 넣고 답변 생성
    ai_message = qa_chain.invoke({"query": query})
    ```
    

### 2.2.3 데이터 전처리

- **기본 RAG Pipeline 문제점**:
    - 원하는 답변이 나오지 않음
        
        <aside>
        
        - {'query': '연봉 5천만원인 거주자의 종합소득세는?',
        - 'result': '제공된 문서에는 연봉 5천만원에 대한 구체적인 근로소득공제액 및 세율표 등의 정보가 누락되어 있어 정확한 종합소득세를 알 수 없습니다. 따라서 제공된 정보만으로는 해당 질문에 답변할 수 없습니다.'}
        </aside>
        
- **잘못된 답변이 나온이유:**
    - 표를 참고하여 종합소득세를 계산해야하지만 표 데이터를 제대로 인식하지 못하기 때문
- **표데이터를 참고하게끔 하는방법:**
    - 기존 docx문서에서 표 데이터의 경우 마크다운으로 바꾸어준다음 RAG Pipeline 돌리기

# 2. 기억에 남는 Codes

---

```python
# 배경: 구글 공식문서에 적혀있는 임베딩 모델명(gemini-embedding-001)을 사용하여 코드를 돌렸지만 이름이 잘못되었다는 에러와 함께 혼란을 겼었었다
# 밑 코드는 나의 제미나이 API 키로 사용 가능한 모든 임베딩 모델 출력해주는 코드로 모델명 앞에 model/ 를 붙여야 했음을 알수있었다.
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)
```

```python
# 배경: Vector DB(에 다량의 문서 청크를 한 번에 저장할 때 API 호출량 초과로 인해 전체 업로드 파이프라인이 멈추는 에러가 발생하여 혼란을 겼었다
# 대량의 문서를 한번에 임베딩하여 DB에 저장하려고 할때 데이터 적재가 중단되는 현상이 있다고 한다
# 이를 해결하기 위해 데이터를 **5개씩 나누어 넣는 '배치 처리'와**, **에러가 나면 대기 시간을 2배씩 늘려가며 다시 시도하는 '지수 백오프' 방식**을 적용하여 안정적으로 데이터 적재를 완료하였다

def add_documents_with_retry(database, documents, max_retries=5):
		(......)
						(......)
            **else: # 에러가 나면 대기 시간을 2배씩 늘려가며 다시 시도하는  지수 백오프' 방식
              wait_seconds = (2 ** attempt) * 10 
              time.sleep(wait_seconds)**  
    return False
    
    
**# 데이터를 5개씩 나누어 넣는 '배치 처리
batch_size = 5  
for start in range(0, len(document_list), batch_size): # 0~312까지 5씩 증가한 리스트 [0, 5, ..., 310]]
    batch = document_list[start:start + batch_size] # document_chunk_list[0:5], [5:10], ..., [310:315]**

    # 문서를 넣는 함수
    success = add_documents_with_retry(
        database=database,
        **documents=batch  # documents는 5개청크씩 분할해서 넣음!**
    )
		(......)
```

# 3. 이슈 및 해결 과정

---

#### **문제1**

- **설명**
    - _InactiveRpcError: * BatchEmbedContentsRequest.requests[9].model: unexpected model name format…
    - [제미나이 임베딩 모델 명을 공식문서](https://ai.google.dev/gemini-api/docs/embeddings?hl=ko)에서 찾아 gemini-embedding-001라는 임베딩 모델을 사용하였으나 모델이름형식에 대한 오류문을 받았다
- **해결과정**
    - 나의 제미나이API key로 사용가능한 임베딩 모델명을 확인하는 코드를 통해 정확한 모델이름을 확인할 수 있었다
        
        ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/3%EC%A3%BC%EC%B0%A8%20LangChain%20%EA%B8%B0%EB%B0%98%20RAG%20%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%98%EB%A6%AC/image%201.png)
        
    - 

#### **문제2**

- **설명:**
    - GoogleGenerativeAIError: 429 You exceeded your current quota …
    - 임베딩하여 백터화된 데이터를 Vector DB, Chroma에 저장할때 할당량 초과 에러를 받았다
    - Google Generative AI API는 무료 티어 사용 시 분당 요청 와 **분당 토큰 수(TPM)** 제한이 매우 엄격하기때문에
    - 대량의 문서를 한번에 임베딩하여 Chroma DB에 저장하려고 하면 API 요청이 폭주하여 **`429 Quota Exceeded`** 에러가 발생하며 데이터 적재가 중단되는 현상이 있다고 한다
- **실패한 해결 과정:**
    - 사용가능한 임베딩 모델을 바꿔가며 보았으나 이와 관계없이  DB에 적재에선 할당량 초과 에러를 받았다
        
        ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/3%EC%A3%BC%EC%B0%A8%20LangChain%20%EA%B8%B0%EB%B0%98%20RAG%20%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%98%EB%A6%AC/image%202.png)
        
- **성공한 해결 전략 3가지:**
    - **배치 분할 (Batching)**: 전체 문서를 한꺼번에 처리하지 않고 `batch_size = 5`개 단위로 작게 쪼개어 API 요청 1회당 토큰 소모량을 대폭 줄임
    - **지수 백오프 재시도 (Exponential Backoff)**: 429(`RESOURCE_EXHAUSTED`) 에러 발생 시 즉시 실패 처리하지 않고,**10초, 20초, 40초, 80초, 160초처럼 2배씩 늘려가며 다시 시도**하도록 안정성을 확보
    - **간격 딜레이 (Polite Waiting)**: 배치 작업 사이에 `time.sleep(5)`를 부여하여 연속적인 API 호출을 방지하고 분당 요청 수를 안전 범위 내로 유지
        
        ![image.png](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C/3%EC%A3%BC%EC%B0%A8%20LangChain%20%EA%B8%B0%EB%B0%98%20RAG%20%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8%20%EA%B5%AC%EC%B6%95%20%EB%B0%8F%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%98%EB%A6%AC/image%203.png)
        
    

# 4. 회의 - 학습 내용 공유

---

### Reflection

- **좋았던 점 (Keep)**: 새로 이해하게 된 핵심 개념 (예: *“f-string 수동 연동 방식과 `RetrievalQA` 체인의 차이를 명확히 이해해서 RAG 데이터 흐름 개념이 잡혔다.”*)
- **어려웠던 점 / 한계 (Problem)**: 학습이나 구현 과정에서 겪은 병목 (예: *“API Rate Limit 에러로 인해 파이프라인 흐름이 끊기는 현상을 겪었고, 백오프(Exponential Backoff) 재시도 코드의 필요성을 깨달았다.”*)
- **향후 과제 (Try / Action Item)**: 다음에 시도해볼 것 (예: *“다음에는 단순 유사도 검색(`similarity_search`) 외에 MMR(Max Marginal Relevance) 검색 방식을 적용해 답변 다양성을 높여봐야겠다.”*)

# 5. 다음 학습 계획 수립

---

- [x]  Rag 기반 챗봇 구현
- [x]  웹 서비스 배포