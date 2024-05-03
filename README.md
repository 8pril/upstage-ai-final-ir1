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
</tr>
<tr>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/76687996/6c21c014-1e77-4ac1-89ac-72b7615c8bf5"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/c4cb11ba-e02f-4776-97c8-9585ae4b9f1d"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/3d913931-5797-4689-aea2-3ef12bc47ef0"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/0f945311-9828-4e50-a60c-fc4db3fa3b9d"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/a4dbcdb5-1d28-4b91-8555-1168abffc1d0"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/HojunJ/conventional-repo/assets/76687996/d2bef206-7699-4028-a744-356b1950c4f1"  width="250"  height="300"/>  </td>
</tr>
<tr>
<td>  <div  align=center>  <a  href="https://github.com/3minka">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/HyeokBro">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/d-yeon">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/Daw-ny">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/devhyuk96">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
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

Dialogue Summarization 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회입니다. 

일상생활에서 대화는 항상 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.

그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다.

이를 돕기 위해, 우리는 이번 대회에서 일상 대화를 바탕으로 요약문을 생성하는 모델을 구축합니다!

![image](https://github.com/HojunJ/conventional-repo/assets/76687996/1ba682aa-f341-4e84-a788-57994fa845ba)

참가자들은 대회에서 제공된 데이터셋을 기반으로 모델을 학습하고, 대화의 요약문을 생성하는데 중점을 둡니다. 이를 위해 다양한 구조의 자연어 모델을 구축할 수 있습니다.

제공되는 데이터셋은 오직 "대화문과 요약문"입니다. 회의, 일상 대화 등 다양한 주제를 가진 대화문과, 이에 대한 요약문을 포함하고 있습니다.

참가자들은 이러한 비정형 텍스트 데이터를 고려하여 모델을 훈련하고, 요약문의 생성 성능을 높이기 위한 최적의 방법을 찾아야 합니다.

경진대회의 목표는 정확하고 일반화된 모델을 개발하여 요약문을 생성하는 것입니다. 나누는 많은 대화에서 핵심적인 부분만 모델이 요약해주니, 업무 효율은 물론이고 관계도 개선될 수 있습니다. 또한, 참가자들은 모델의 성능을 평가하고 대화문과 요약문의 관계를 심층적으로 이해함으로써 자연어 딥러닝 모델링 분야에서의 실전 경험을 쌓을 수 있습니다.

본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.

> input : 249개의 대화문  
> output : 249개의 대화 요약문

## Evaluation Metric

ROUGE는 텍스트 요약, 기계 번역과 같은 태스크를 평가하기 위해 사용되는 대표적인 metric입니다. 모델이 생성한 요약본 혹은 번역본을 사람이 만든 참조 요약본과 비교하여 점수를 계산합니다.

ROUGE-Recall: 참조 요약본을 구성하는 단어들 중 모델 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수입니다.

ROUGE-Precision: 모델 요약본을 구성하는 단어들 중 참조 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수입니다.

ROUGE-N과 ROUGE-L은 비교하는 단어의 단위 개수를 어떻게 정할지에 따라 구분됩니다.

ROUGE-N은 unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표입니다.

ROUGE-1는 모델 요약본과 참조 요약본 간에 겹치는 unigram의 수를 비교합니다.

ROUGE-2는 모델 요약본과 참조 요약본 간에 겹치는 bigram의 수를 비교합니다.

ROUGE-L: LCS 기법을 이용해 최장 길이로 매칭되는 문자열을 측정합니다. n-gram에서 n을 고정하지 않고, 단어의 등장 순서가 동일한 빈도수를 모두 세기 때문에 보다 유연한 성능 비교가 가능합니다.

ROUGE-F1은 ROUGE-Recall과 ROUGE-Precisioin의 조화 평균입니다.

![image](https://github.com/HojunJ/conventional-repo/assets/76687996/2cedb7b5-81be-4d68-986c-3f2c4dfd675b)

## 2. Components

### Directory

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/76687996/17569632-122c-4b30-93d1-3c08717d32e1)

## 3. Strategy

### Dataset overview
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/c6a1a3c8-08af-4dc6-929d-f1c61c6e3534)

### EDA
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/5b85248b-269b-4df0-be33-85d329ebeed0)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/29d08d90-a9e8-494b-ac35-400d056ae1f3)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/e97abadd-c40a-4ba3-b434-31c506c45174)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/30121d6b-8827-4d97-81be-8650d4f604fb)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/3d459173-eef9-4ccb-91f2-691a6dd6b629)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/994e06d7-84c3-4271-b86e-442af6e6e641)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/4163ca34-8384-44f5-ba1a-fc35b78223b5)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/84657fd7-c340-4390-9744-ad85ff67affd)

## 4. 4-Aspect Modeling
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/95d6ba9e-7d3e-4306-b06c-ed928663baa0)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/559c8436-fbea-4732-a9ab-9d3781882f17)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/ecb6f5c2-89d0-47b0-a8eb-c84930570354)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/2e1888d6-b257-4aa8-a8ea-2561569cd0cb)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/b547e854-fcd2-4802-889e-4b441c16ea7e)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/8070f104-efd7-4266-b9f0-74d475ed69e7)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/2e875f42-5856-48fd-802d-5db9c82ae755)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/77cda53a-3ab2-426a-889d-66325b9a63ce)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/2f15018c-7d74-4871-9eea-262dd1e5d074)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/9890e050-7791-423d-905e-1d9a50c38b09)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/0e597060-11be-4d43-8461-99a747d81e03)


## 5. Result

### Leader Board - 6th

![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/bf431440-4cd5-4caa-ae69-8721dc2cfaf2)
![image](https://github.com/Daw-ny/2024_LG_Aimers/assets/76687996/619017bc-240f-459f-8681-57fd2c3b73bc)

### Presentation
- [Google Project](https://docs.google.com/presentation/d/15cafHlTN6UNRAf8-hrg2hwa0m6lNMWFO/edit?usp=sharing&ouid=107968498421720497028&rtpof=true&sd=true)

## etc

### Meeting Log

- 전체적인 내용은 [Notion](https://www.notion.so/1-9ca0f519bc5143d5a541cc547ed278b4), [Notion2](https://www.notion.so/Dialogue-Summarization-43fc1c2d025b4cd09d4babbe8ab7a1c9?pvs=4)에서 확인하실 수 있습니다.
- Mar 8 ~ Mar 20 : Online Meeting

### Reference

1. Beam Search 등 생성전략 정리된 것
    - 생성전략 참고 블로그 : https://littlefoxdiary.tistory.com/46
    - softmax 시 사이즈 제한하는 방법 : https://wikidocs.net/72820
2. T5 에서 데이터 입력되는 순서, 배치사이즈가 자꾸 사이즈 에러가 난다. BART 와는 뭐가 다른지?
    - https://discuss.huggingface.co/t/errors-when-fine-tuning-t5/3527
    - 아직 해결 못했어요. 다음에 할 때 이것도 고려해보아요

3. 형태소 분석기 U tube
    - https://youtube.com/watch?v=Ks3iUZlAqvA&si=fOxaYEeKGJNhYizP

4. **Hugging Face Documentation**  
    - https://huggingface.co/docs

5. Mecab
    - https://openuiz.blogspot.com/2018/12/mecab-ko-dic.html
    - https://sswwd.tistory.com/65
    - https://luminitworld.tistory.com/104

6. Kiwi
    - https://github.com/bab2min/Kiwi
