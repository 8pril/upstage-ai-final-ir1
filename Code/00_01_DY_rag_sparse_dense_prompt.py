import os
import json
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Initialize Sentence Transformer model (able to create Korean embeddings)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

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
    sparse_results = sparse_retrieve(query_str, size*2)  # Increase initial retrieval size
    dense_candidates = [hit['_source']['content'] for hit in sparse_results['hits']['hits']]
    dense_embeddings = get_embedding(dense_candidates)
    query_embedding = get_embedding([query_str])[0]
    scores = np.dot(dense_embeddings, query_embedding)  # Compute dot product for similarity
    sorted_hits = sorted(zip(scores, sparse_results['hits']['hits']), key=lambda x: x[0], reverse=True)[:size]
    return {'hits': {'hits': [hit for _, hit in sorted_hits]}}

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
- 사용자의 질문과 관련된 모든 이전 메시지 정보를 철저히 검토하고 답변을 생성하십시오.
- 주어진 참조 자료를 사용하여 질문에 대한 정확하고 심층적인 답변을 제공하십시오.
- 만약 참조 자료가 충분하지 않아 답변을 제공할 수 없다면, 더 많은 정보가 필요하다고 명확하게 설명하십시오.
- 검색 결과로부터 얻은 정보를 기반으로 답변을 구성하되, 원본 자료의 정보를 정확히 인용하십시오.
- 답변은 한국어로 명확하고 정확하게 표현하십시오.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 모델은 과학 상식 전문가로서, 과학적 질문에 대하여 정확하고 신뢰할 수 있는 정보를 제공하는 역할을 합니다.
- 검색 API를 사용하여 필요한 모든 과학적 데이터와 정보를 수집하고 이를 기반으로 답변을 구성하십시오.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성하십시오.
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

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        search_result = sparse_retrieve(standalone_query, 3)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
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
eval_rag("../data/eval.jsonl", "sample_submission_5.csv")

