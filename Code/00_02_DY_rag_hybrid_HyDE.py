# HyDE 추가
# HyDE prompt 수정
# persona function calling prompt 수정
# model 변경 : sroberta-multitask

import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize Sentence Transformer model (able to create Korean embeddings)
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# Create embeddings using SentenceTransformer
def get_embedding(sentences):
    return model.encode(sentences)

# Create embeddings in batches from a list of documents
def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f'batch {i}')
    return batch_embeddings

# Elasticsearch client setup
es_username = "elastic"
es_password = "5yL4-BZ7LcAr1AFwpI7F"
es = Elasticsearch(['https://localhost:9200'], basic_auth=(es_username, es_password), ca_certs="./elasticsearch-8.8.0/config/certs/http_ca.crt")

# Create a new Elasticsearch index
def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)

# Delete an Elasticsearch index
def delete_es_index(index):
    es.indices.delete(index=index)

# Bulk index documents in Elasticsearch
def bulk_add(index, docs):
    actions = [{'_index': index, '_source': doc} for doc in docs]
    return helpers.bulk(es, actions)

# Retrieve using sparse method
def sparse_retrieve(query_str, size):
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index="test", query=query, size=size)

# Retrieve using dense vector similarity
def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {"field": "embeddings", "query_vector": query_embedding.tolist(), "k": size, "num_candidates": 100}
    return es.search(index="test", knn=knn)

# Combine sparse and dense retrieval to improve results
def hybrid_retrieve(query_str, size):
    dense_results_map = {}

    sparse_results = sparse_retrieve(query_str, size * 2)
    dense_results = dense_retrieve(query_str, size * 2)
    for r in dense_results['hits']['hits']:
        dense_results_map[r["_source"]["docid"]] = r

    mixed_results = []
    only_sparse_reults = []
    for r in sparse_results['hits']['hits']:
        if r["_source"]["docid"] in dense_results_map:
            mixed_results.append(r)
        else:
            only_sparse_reults.append(r)
    combined_results = sorted(mixed_results, key=lambda x: x['_score'], reverse=True)
    combined_results += sorted(only_sparse_reults, key=lambda x: x['_score'], reverse=True)
    
    return combined_results[:size]

# Settings and mappings for the Elasticsearch index
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}
mappings = {
    "properties": {
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "l2_norm"}
    }
}

create_es_index("test", settings, mappings)

# 문서의 content 필드에 대한 임베딩 생성
index_docs = []
with open("../data/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
embeddings = get_embeddings_in_batches(docs)
                
# 생성한 임베딩을 색인할 필드로 추가
for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

# 'test' 인덱스에 대량 문서 추가
ret = bulk_add("test", index_docs)

# 색인이 잘 되었는지 확인 (색인된 총 문서수가 출력되어야 함)
print(ret)

test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"

# 역색인을 사용하는 검색 예제
search_result_retrieve = sparse_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])

# Vector 유사도 사용한 검색 예제
search_result_retrieve = dense_retrieve(test_query, 3)

# 결과 출력 테스트
for rst in search_result_retrieve['hits']['hits']:
    print('score:', rst['_score'], 'source:', rst['_source']["content"])


# 아래부터는 실제 RAG를 구현하는 코드입니다.
from openai import OpenAI
import traceback

# OpenAI API 키를 환경변수에 설정
os.environ["OPENAI_API_KEY"] = "sk-proj-PUEjiW7IbiMhwJSrXouVT3BlbkFJQFSJoHSnnDY8X7HJiulP"

client = OpenAI()
# 사용할 모델을 설정(여기서는 gpt-3.5-turbo-1106 모델 사용)
llm_model = "gpt-3.5-turbo-1106"

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- <context> 정보만을 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 과학 이론, 실험, 사실, 현상, 원리, 역사 등 과학 상식에 관한 질문일 경우 "search function"을 호출한다.
- 그렇지 않은 경우 적절한 대답을 생성한다.

## 과학 상식 질문에 대한 예시

### 과학 이론 질문 예시
- 자연선택 이론이란
- 상대성 이론의 기본 개념은
- 불확정성 원리란 무엇인가?

### 과학 실험 질문 예시
- 밀리컨 기름방울 실험의 발견은?
- 멘델의 유전 법칙 두 가지는?
- 루이 파스퇴르의 살균 실험 결과

### 과학 현상 질문 예시
- 블랙홀의 빛 끌어당김 현상?
- 오로라 발생 원리는?
- 저기압과 고기압의 기후 영향은?

### 과학 원리 질문 예시
- 아치메데스 원리란?
- 파스칼의 법칙
- 광합성의 에너지 변환 과정은?

### 과학 사실 질문 예시
- 수성의 태양계 위치와 특징은?
- DNA 이중나선 구조 발견은?
- 지구 대기의 주요 기체는?

### 과학 역사 질문 예시
- 갈릴레오의 천체망원경 발견은?
- 로잘린드 프랭클린의 DNA 연구 기여는?
- 에디슨과 테슬라의 전기 전쟁 영향은?

"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]

hyde_qa = """
## Role: 과학 상식 전문가

## Instruction
- 사용자의 질문에 대해 한국어로 한 문장으로 요약하여 답변한다.
"""

def hyde_qa(message):
    msg = [{"role": "system", "content": hyde_qa, "role": "user", "content": f"{message}"}]
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return ""

    return result.choices[0].message.content

# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        hyde_query = hyde_qa(standalone_query)

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        # search_result = sparse_retrieve(standalone_query, 3)
        search_result = hybrid_retrieve(hyde_query, 3)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": f"<context>{content}</context>"})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

# 평가 데이터에 대해서 결과 생성 - 파일 포맷은 jsonl이지만 파일명은 csv 사용
eval_rag("../data/eval.jsonl", "sample_submission_sj7.csv")

