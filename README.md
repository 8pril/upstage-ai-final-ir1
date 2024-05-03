[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/nDCOQnZo)

# 효율적인 RAG 구축 공략 : 과학지식 편(The Goal of Building an Efficient RAG: Scientific Knowledge)


## Team

<table>
<tr>
<td>  <div  align=center> 1 </div>  </td>
<td>  <div  align=center> 2 </div>  </td>
<td>  <div  align=center> 3 </div>  </td>
<td>  <div  align=center> 4 </div>  </td>
<td>  <div  align=center> 5 </div>  </td>
<td>  <div  align=center> 6 </div>  </td>
<td>  <div  align=center> 6 </div>  </td>
</tr>
<tr>
<td>  <div  align=center>  <b>가상민</b>  </div>  </td>
<td>  <div  align=center>  <b>김다운</b>  </div>  </td>
<td>  <div  align=center>  <b>김도연</b>  </div>  </td>
<td>  <div  align=center>  <b>서상혁</b>  </div>  </td>
<td>  <div  align=center>  <b>신동혁</b>  </div>  </td>
<td>  <div  align=center>  <b>이소영</b>  </div>  </td>
<td>  <div  align=center>  <b>장호준</b>  </div>  </td>

<tr>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/76687996/6c21c014-1e77-4ac1-89ac-72b7615c8bf5"  width="250"  height="200"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/0f945311-9828-4e50-a60c-fc4db3fa3b9d"  width="250"  height="200"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/3d913931-5797-4689-aea2-3ef12bc47ef0"  width="250"  height="200"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/a4dbcdb5-1d28-4b91-8555-1168abffc1d0"  width="250"  height="200"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/c4cb11ba-e02f-4776-97c8-9585ae4b9f1d"  width="250"  height="200"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/685b52f9-872e-4456-933f-2bead5efba2b"  width="250"  height="200"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/HojunJ/conventional-repo/assets/76687996/d2bef206-7699-4028-a744-356b1950c4f1"  width="250"  height="200"/>  </td>
</tr>
<tr>
<td>  <div  align=center>  <a  href="https://github.com/3minka">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/HyeokBro">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/d-yeon">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/Daw-ny">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/devhyuk96">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/8pril">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/HojunJ">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
</tr>
</table>

  

## 0. Overview

### Environment

-   AMD Ryzen Threadripper 3960X 24-Core Processor
-   NVIDIA GeForce RTX 3090
-   CUDA Version 12.2

### Requirements

pandas==2.1.4  
numpy==1.23.5  
wandb==0.16.1  
tqdm==4.66.1  
pytorch_lightning==2.1.2  
transformers[torch]==4.35.2  
rouge==1.0.1  
jupyter==1.0.0  
jupyterlab==4.0.9  

## 1. Competiton Info

### Overview

LLM의 등장 이후 여러 산업 분야에서 지식을 다루는 업무들이 점점 고도화되고 있습니다.

특히 정보를 찾기 위해 검색엔진의 입력창에 키워드를 입력하고 결과를 확인하고 원하는 정보가 없으면 다른 키워드로 다시 검색하기를 반복하는 번거로운 과정을 이제 더이상 자주 할 필요가 없어졌습니다.

이제 LLM한테 물어보면 질문의 의도까지 파악해서 필요한 내용만 잘 정리해서 알려 줍니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/78ef61cd-4da4-46e0-ad07-baa2d5ea980d)

그렇지만 LLM이 가진 근본적인 한계도 있습니다.

먼저, 정보라는 것은 의미나 가치가 시간에 따라 계속 변하기 때문에 모델이 이를 실시간으로 학습하기 힘들고 이 때문에 아래 예시처럼 knowledge cutoff 가 자연스럽게 발생합니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/4f85b2f0-cc29-499b-8b6e-7a99b0c2070c)

그리고 LLM이 알려주는 지식이 항상 사실에 기반한 것이 아닌 경우가 종종 있습니다. 특히 특정 도메인이나 문제 영역은 매우 심각한 거짓 정보들을 생성해 내곤 합니다. 아래 예시에서 추천하는 맛집들은 모두 실재하지 않는 장소들입니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/05fbcf71-2119-4c0e-bc87-e5e69bd01ca8)

이러한 환각 현상은 메타인지를 학습하지 않은 LLM의 근본적인 한계라 볼 수 있습니다.

모델은 학습 과정에서 정보를 압축해서 저장하기 때문에 정보의 손실이 발생할 수밖에 없고, 이 때문에 특정 입력 조건에 대해서는 사실 여부보다는 지식를 표현하는 국소적인 패턴이 더 큰 영향을 주면서 답변이 생성될 수 있기 때문입니다.

이러한 문제를 극복하기 위해서는 RAG(Retrieval Augmented Generation) 기술이 필수입니다.

RAG는 질문에 적합한 레퍼런스 추출을 위해 검색엔진을 활용하고 답변 생성을 위해 LLM(Large Language Model)을 활용합니다.

이때 LLM은 스스로 알고 있는 지식을 출력하기보다는 언어 추론 능력을 극대화하는 것에 방점을 둡니다.

이렇게 사실에 기반한 지식 정보를 토대로 질문에 답을 하고 출처 정보도 같이 줄 수 있기 때문에 사용자는 훨씬 더 안심하고 정보를 소비할 수 있게 됩니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/71add39c-6301-4323-8502-a8f98291f5e1)

이번 대회에서는 과학 상식을 질문하는 시나리오를 가정하고 과학 상식 문서 4200여개를 미리 검색엔진에 색인해 둡니다.

대화 메시지 또는 질문이 들어오면 과학 상식에 대한 질문 의도인지 그렇지 않은 지 판단 후에 과학 상식 질문이라면 검색엔진으로부터 적합한 문서들을 추출하고 이를 기반으로 답변을 생성합니다. 

만일 과학 상식 이외의 질문이라면 검색엔진을 활용할 필요 없이 적절한 답을 바로 생성합니다.

마지막으로, 본 프로젝트는 모델링에 중점을 둔 대회가 아니라 RAG(Retrieval Augmented Generation) 시스템의 개발에 집중하고 있습니다. 이 대회는 여러 모델과 다양한 기법, 그리고 앙상블을 활용하여 모델의 성능을 향상시키는 일반적인 모델링 대회와는 다릅니다. 대신에 검색 엔진이 올바른 문서를 색인했는지, 그리고 생성된 답변이 적절한지 직접 확인하는 것이 중요한 대회입니다.

따라서, 참가자들은 작은 규모의 토이 데이터셋(10개 미만)을 사용하여 초기 실험을 진행한 후에 전체 데이터셋에 대한 평가를 진행하는 것을 권장합니다. 실제로 RAG 시스템을 구축할 때에도 이러한 방식이 일반적으로 적용되며, 이를 통해 실험을 더욱 효율적으로 진행할 수 있습니다. 따라서 이번 대회는 2주간 진행되며, 하루에 제출할 수 있는 횟수가 5회로 제한됩니다.

자, 이제 여러분만의 RAG 시스템을 구축하러 가보실까요~?


## Evaluation Metric

사용자가 입력한 질문에 대해서 답변을 얼마나 잘 생성했는지 정량화하는 작업은 매우 고난도의 작업입니다.

어떤 질문에 대해서도 정답이 정해져 있는 것이 아니라 다양한 형태로 표현해 낼 수 있기 때문입니다.

그나마 어느 정도의 객관성을 확보하기 위해서는 다수의 사람이 직접 평가하는 방식을 사용할 수밖에 없습니다.

그렇지만 대회에서는 자동화된 평가 방법을 적용해야 하기 때문에 RAG에 대한 end-to-end 평가 대신 적합한 레퍼런스를 얼마나 잘 추출했는지에 대한 평가만 진행합니다.

이번 평가에서는 MAP(Mean Average Precision)라는 metric을 사용합니다. MAP는 질의 N개에 대한 Average Precision의 평균 값을 구하고, Average Precision은 Precision-recall curve에서 아래쪽 면적을 의미합니다.

계산 과정은 도식화하면 아래 그림과 같습니다.

![image](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/fcc26d42-8929-4271-93e1-438498ffaba2)

그런데 이번 대회에서는 MAP를 약간 변형하여 RAG 평가에 적합하도록 살짝 수정한 형태의 로직을 사용합니다.

대화 메시지가 과학 상식에 대한 질문일 수도 있고 아닐수도 있기 때문에 과학 상식 질문이 아닌 경우는 문서를 추출할 필요가 없습니다. 그래서 검색이 필요없는 ground truth 항목에 대해서는 검색 결과가 없는 경우를 1점으로 주고 그렇지 않는 경우는 0점으로 계산하게 로직을 추가했습니다.

아래 코드의 else 부분이 이에 해당하고 나머지 로직은 원래 MAP 계산 로직을 그대로 따릅니다.

```
def calc_map(gt, pred):    
    sum_average_precision = 0    
    for j in pred:        
        if gt[j["eval_id"]]:            
            hit_count = 0            
            sum_precision = 0            
            for i,docid in enumerate(j["topk"][:3]):                
                if docid in gt[j["eval_id"]]:                    
                    hit_count += 1                    
                    sum_precision += hit_count/(i+1)            
            average_precision = sum_precision / hit_count if hit_count > 0 else 0        
        else:            
            average_precision = 0 if j["topk"] else 1        
        sum_average_precision += average_precision    
    return sum_average_precision/len(pred)
```

다운로드 받은 data.tar.gz 에 포함된 eval.jsonl을 사용하여 결과물 생성하고, 이 결과물을 제출하면 리더보더에 반영됩니다.

## 2. Components

### Directory

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/76687996/17569632-122c-4b30-93d1-3c08717d32e1)

## 3. Strategy

![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (8)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/49c0cc28-6160-4021-b0cb-b727c765be31)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (9)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/435cb476-f7d7-4eb4-953b-5fe1182326f8)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (10)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/f0efa3f3-6bb9-4c81-a907-cb0243a224df)


## 4. 5-Aspect

![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (11)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/7c1f375d-6491-489e-9219-0271bdb42d8b)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (12)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/54d62234-a30c-418a-9665-ad923615736a)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (13)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/4fa86637-2f71-43ee-9595-87badbfd7c9b)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (14)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/9e79c943-e423-40c1-ae43-ae566f30eb7d)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (15)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/0c4fc87b-4b5c-4ab3-9221-cf1dab243902)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (16)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/29a2582d-a85b-46ac-9eca-38b9764b61b2)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (17)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/5c9a7add-ed46-4c61-9676-b2a31b255781)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (18)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/4faadf33-2573-4d56-88fe-d379b1147a69)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (19)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/25d36ee6-da61-4b15-b5a7-9c9f742b675b)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (20)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/d1ba1533-07bd-4ca7-b7be-0b74a2539661)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (21)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/ce7b0645-07af-4afd-8bc1-efcba565bb9a)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (22)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/58e863c2-613d-4b05-9797-ee2369385a84)


## 5. Result

### Leader Board - 2th

![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (23)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/97348b22-af71-4a78-b551-b97dd7d3b02f)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (24)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/f918e55a-d5af-4be2-8e97-712d1496891d)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (25)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/6dac382b-8ebd-493e-8049-3bec542e5332)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (26)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/5ff1ba0d-e2cb-4615-8727-7d8a2cf68bb4)
![패스트캠퍼스  Upstage AI Lab 1기 IR Final pptx (27)](https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/18365fc1-4499-471a-843b-e764a7fea2d1)


### Presentation
- [Google Project](https://docs.google.com/presentation/d/1mjfruR3dbH1T0Uw-ifn6qpIbkh_5tFiU/edit?usp=sharing&ouid=112740872612879476638&rtpof=true&sd=true)

## etc

### Meeting Log

- 전체적인 내용은 [진행 Notion](https://www.notion.so/Scientific-Knowledge-Question-Answering-cd175584a0e7473b8c205c34ac673683?pvs=4), [간트차트](https://sixth-drum-9ac.notion.site/Final-d590cb0c11044d83a8d2a52459747117?pvs=4)에서 확인하실 수 있습니다.
- 4월 22일 (월) 10:00 ~ 5월 2일 (목) 19:00 : Online Meeting

### Reference

