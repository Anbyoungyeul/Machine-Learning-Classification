import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import codecs
import scipy.sparse
import glob
import os
import shutil
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn import metrics

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

print('검정 셋은 이후 임의로 분리합니다.')
file_source = './valid__benign/'
file_destination = './train__benign/'

get_files = os.listdir(file_source)
print('valid__benign내 파일--->train__benign 으로 이동중...')
for g in get_files:
    shutil.move(file_source + g, file_destination)

file_source2 = './valid__malware/'
file_destination2 = './train__malware/'

get_files2 = os.listdir(file_source2)
print('valid__malware내 파일--->train__malware 으로 이동중...')
for g in get_files2:
    shutil.move(file_source2 + g, file_destination2)

bigram_tokens="00,01,02,03,04,05,06,07,08,09,0a,0b,0c,0d,0e,0f,10,11,12,13,14,15,16,17,18,19,1a,1b,1c,1d,1e,1f,20,21,22,23,24,25,26,27,28,29,2a,\
2b,2c,2d,2e,2f,30,31,32,33,34,35,36,37,38,39,3a,3b,3c,3d,3e,3f,40,41,42,43,44,45,46,47,48,49,4a,4b,4c,4d,4e,4f,50,51,52,53,54,55,56,57,58,\
59,5a,5b,5c,5d,5e,5f,60,61,62,63,64,65,66,67,68,69,6a,6b,6c,6d,6e,6f,70,71,72,73,74,75,76,77,78,79,7a,7b,7c,7d,7e,7f,80,81,82,83,84,85,86,\
87,88,89,8a,8b,8c,8d,8e,8f,90,91,92,93,94,95,96,97,98,99,9a,9b,9c,9d,9e,9f,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,aa,ab,ac,ad,ae,af,b0,b1,b2,b3,b4,b5,\
b6,b7,b8,b9,ba,bb,bc,bd,be,bf,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,ca,cb,cc,cd,ce,cf,d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db,dc,dd,de,df,e0,e1,e2,e3,e4,\
e5,e6,e7,e8,e9,ea,eb,ec,ed,ee,ef,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,fa,fb,fc,fd,fe,ff,??"

bigram_tokens=bigram_tokens.split(",")

# 00과 FF 사이에는 256개의 고유한 값이 있으므로 16진수 값의 각 쌍을 하나의 단어로 간주하여 256개의 고유한 값을 다룬다
# 따라서 아래 함수는 가능한 모든 bigrams_counts 조합을 추출한다.
def calculate_bigram(bigram_tokens):
    sentence=""
    vocabulary_list_for_byte_bigrams=[]
    for i in tqdm(range(len(bigram_tokens))):
        for j in range(len(bigram_tokens)):
            bigram=bigram_tokens[i]+" "+bigram_tokens[j]
            sentence=sentence+bigram+","
            vocabulary_list_for_byte_bigrams.append(bigram)
    return vocabulary_list_for_byte_bigrams

vocabulary_list_for_byte_bigrams = calculate_bigram(bigram_tokens) 

print('train__benign opcode 추출 중...')
# 여기서부터 train__benign 파일을 불러와서 opcode 2-gram 추출 후 txt로 저장하는 코드
opcodes_for_bigram = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']

# 빠른 런타임을 위해 리스트를 딕셔너리 자료구조로 변경
dict_asm_opcodes = dict(zip(opcodes_for_bigram, [1  for i in range(len(opcodes_for_bigram))]))
# opcodes_benign_asm_files라는 디렉토리가 존재하지 않으면 디렉토리 생성. txt를 담기 위한 디렉토리임
if not os.path.isdir("./opcodes_benign_asm_files"):
    os.mkdir('opcodes_benign_asm_files')

# txt로 저장하는 과정
def calculate_sequence_of_opcodes():
    asm_file_names=os.listdir('train__benign') # train__benign 폴더에 존재하는 파일 가져오기
    for this_asm_file in tqdm(asm_file_names): # tqdm 라이브러리를 통한 for in 문 진행상황 시각화
        each_asm_opcode_file = open("./opcodes_benign_asm_files/{}_opcode_asm_bi_grams.txt".format(this_asm_file.split('.')[0]), "w+") # 기존의 asm파일을 쪼개서 each_asm_opcode_file에 쓰기
        sequence_of_opcodes = ""
        with codecs.open('train__benign/' + this_asm_file, encoding='cp1252', errors ='replace') as asm_file: 
            for lines in asm_file:
                
                line = lines.rstrip().split()            
                
                for word in line: # 오른쪽 공백을 제거하고 공백을 기준으로 문자열 나누기
                    if dict_asm_opcodes.get(word)==1: #  dict_asm_opcode가 올바르게 가져와졌으면
                        sequence_of_opcodes += word + ' ' # sequence_of_opcode에 word 추가
        each_asm_opcode_file.write(sequence_of_opcodes + "\n") # each_asm_opcode_file에 sequence_of_opcode 쓰기 
        each_asm_opcode_file.close() #파일 닫기
    
calculate_sequence_of_opcodes()

opcodes_asm__bigram_vocabulary = calculate_bigram(opcodes_for_bigram)

print('csv로 저장중...')
vectorizer_opcode = CountVectorizer(
    tokenizer=lambda x: x.split(),
    lowercase=False,
    ngram_range=(2, 2),
    vocabulary=opcodes_asm__bigram_vocabulary,
)  

file_list_opcode = os.listdir("./opcodes_benign_asm_files")

opcode_features = ["ID"] + vectorizer_opcode.get_feature_names() + ["labels"] # 지도 학습을 위해 labels라는 columns 추가

opcodes_benign_asm_bigram_df = pd.DataFrame(columns=opcode_features)

if not os.path.isdir("./featurization"):
    os.mkdir('featurization')
    
with open(
      "./featurization/opcodes_benign_asm_bigram_df.csv", mode="w"
) as opcodes_benign_asm_bigram_df:

    opcodes_benign_asm_bigram_df.write(",".join(map(str, opcode_features)))

    opcodes_benign_asm_bigram_df.write("\n")

    for _, this_asm_file in tqdm(enumerate(file_list_opcode)):

        this_file_id = this_asm_file.split("_")[0]  # 각 어셈 파일의 ID
        this_file_labels = "0" # 정상 파일의 라벨링 값은 0으로 설정
        this_asm_file = open("opcodes_benign_asm_files/" + this_asm_file)

        corpus_opcodes_from_this_asm_file = [
            this_asm_file.read().replace("\n", " ").lower()
        ]  
           #어셈 파일에 모든 opcodes 를 가질 변수

        bigrams_opcodes_asm = vectorizer_opcode.transform(
            corpus_opcodes_from_this_asm_file
        )  

        # 데이터 프레임의 각 행을 해당 이 어셈파일의 바이그램 수로 업데이듯
        
        # 이 행렬의  밀집한 넘파이 배열 표현을 반환
       
        row = scipy.sparse.csr_matrix(bigrams_opcodes_asm).toarray()

        opcodes_benign_asm_bigram_df.write(
            ",".join(map(str, [this_file_id] + list(row[0]) + [this_file_labels]))
        )  # 이 csv어셈 파일에서 단일행 사용

        opcodes_benign_asm_bigram_df.write("\n")

        this_asm_file.close()


opcodes_benign_asm_bigram_df = pd.read_csv(
    "./featurization/opcodes_benign_asm_bigram_df.csv"
)

opcodes_benign_asm_bigram_df.head()

print('train__malware opcode 추출 중...')
# 여기서부터는 train__malware에 대한 동작인데 위에랑 동작 같으니까 설명
opcodes_for_bigram = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']

# 빠른 런타임을 위해 리스트를 딕셔너리 자료구조로 변경
dict_asm_opcodes = dict(zip(opcodes_for_bigram, [1 for i in range(len(opcodes_for_bigram))]))
if not os.path.isdir("./opcodes_malware_asm_files"):
    os.mkdir('opcodes_malware_asm_files')

def calculate_sequence_of_opcodes():
    asm_file_names=os.listdir('train__malware')
    for this_asm_file in tqdm(asm_file_names):
        each_asm_opcode_file = open("./opcodes_malware_asm_files/{}_opcode_asm_bi_grams.txt".format(this_asm_file.split('.')[0]), "w+")
        sequence_of_opcodes = ""
        with codecs.open('train__malware/' + this_asm_file, encoding='cp1252', errors ='replace') as asm_file:
            for lines in asm_file:
                
                line = lines.rstrip().split()            
                
                for word in line:
                    if dict_asm_opcodes.get(word)==1:
                        sequence_of_opcodes += word + ' '
        each_asm_opcode_file.write(sequence_of_opcodes + "\n")
        each_asm_opcode_file.close()
    
calculate_sequence_of_opcodes()

opcodes_asm__bigram_vocabulary = calculate_bigram(opcodes_for_bigram)

print('csv로 저장중...')
vectorizer_opcode = CountVectorizer(
    tokenizer=lambda x: x.split(),
    lowercase=False,
    ngram_range=(2, 2),
    vocabulary=opcodes_asm__bigram_vocabulary,
)  

file_list_opcode = os.listdir("./opcodes_malware_asm_files")

opcode_features = ["ID"] + vectorizer_opcode.get_feature_names() + ["labels"] 

opcodes_benign_asm_bigram_df = pd.DataFrame(columns=opcode_features)

if not os.path.isdir("./featurization"):
    os.mkdir('featurization')
    
with open(
    "./featurization/opcodes_malware_asm_bigram_df.csv", mode="w"
) as opcodes_benign_asm_bigram_df:

    opcodes_benign_asm_bigram_df.write(",".join(map(str, opcode_features)))

    opcodes_benign_asm_bigram_df.write("\n")

    for _, this_asm_file in tqdm(enumerate(file_list_opcode)):

        this_file_id = this_asm_file.split("_")[0] 
        this_file_labels = "1"
        this_asm_file = open("opcodes_malware_asm_files/" + this_asm_file)

        corpus_opcodes_from_this_asm_file = [
            this_asm_file.read().replace("\n", " ").lower()
        ]  

        bigrams_opcodes_asm = vectorizer_opcode.transform(
            corpus_opcodes_from_this_asm_file
        )  

        row = scipy.sparse.csr_matrix(bigrams_opcodes_asm).toarray()

        opcodes_benign_asm_bigram_df.write(
            ",".join(map(str, [this_file_id] + list(row[0]) + [this_file_labels]))
        )  

        opcodes_benign_asm_bigram_df.write("\n")

        this_asm_file.close()


opcodes_benign_asm_bigram_df = pd.read_csv(
    "./featurization/opcodes_malware_asm_bigram_df.csv"
)

opcodes_benign_asm_bigram_df.head()
print('test_benign opcode 추출 중...')
opcodes_for_bigram = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']

dict_asm_opcodes = dict(zip(opcodes_for_bigram, [1 for i in range(len(opcodes_for_bigram))]))
if not os.path.isdir("./opcodes_test_benign_asm_files"):
    os.mkdir('opcodes_test_benign_asm_files')

def calculate_sequence_of_opcodes():
    asm_file_names=os.listdir('test_benign')
    for this_asm_file in tqdm(asm_file_names):
        each_asm_opcode_file = open("./opcodes_test_benign_asm_files/{}_opcode_asm_bi_grams.txt".format(this_asm_file.split('.')[0]), "w+")
        sequence_of_opcodes = ""
        with codecs.open('test_benign/' + this_asm_file, encoding='cp1252', errors ='replace') as asm_file:
            for lines in asm_file:
                
                line = lines.rstrip().split()            
                
                for word in line:
                    if dict_asm_opcodes.get(word)==1:
                        sequence_of_opcodes += word + ' '
        each_asm_opcode_file.write(sequence_of_opcodes + "\n")
        each_asm_opcode_file.close()
    
calculate_sequence_of_opcodes()

opcodes_asm__bigram_vocabulary = calculate_bigram(opcodes_for_bigram)
print('csv로 저장중...')
vectorizer_opcode = CountVectorizer(
    tokenizer=lambda x: x.split(),
    lowercase=False,
    ngram_range=(2, 2),
    vocabulary=opcodes_asm__bigram_vocabulary,
) 

file_list_opcode = os.listdir("./opcodes_test_benign_asm_files")

opcode_features = ["ID"] + vectorizer_opcode.get_feature_names() + ["labels"]

opcodes_benign_asm_bigram_df = pd.DataFrame(columns=opcode_features)

if not os.path.isdir("./test_set"):
    os.mkdir('test_set')
    
with open(
    "./test_set/opcodes_test_benign_asm_bigram_df.csv", mode="w"
) as opcodes_benign_asm_bigram_df:

    opcodes_benign_asm_bigram_df.write(",".join(map(str, opcode_features)))

    opcodes_benign_asm_bigram_df.write("\n")

    for _, this_asm_file in tqdm(enumerate(file_list_opcode)):

        this_file_id = this_asm_file.split("_")[0]
        this_file_labels = "0"
        this_asm_file = open("opcodes_test_benign_asm_files/" + this_asm_file)

        corpus_opcodes_from_this_asm_file = [
            this_asm_file.read().replace("\n", " ").lower()
        ]

        bigrams_opcodes_asm = vectorizer_opcode.transform(
            corpus_opcodes_from_this_asm_file
        )

        row = scipy.sparse.csr_matrix(bigrams_opcodes_asm).toarray()

        opcodes_benign_asm_bigram_df.write(
            ",".join(map(str, [this_file_id] + list(row[0]) + [this_file_labels]))
        )

        opcodes_benign_asm_bigram_df.write("\n")

        this_asm_file.close()


opcodes_benign_asm_bigram_df = pd.read_csv(
    "./test_set/opcodes_test_benign_asm_bigram_df.csv"
)
print('test_malware opcode 추출 중...')
opcodes_for_bigram = ['jmp', 'mov', 'retf', 'push', 'pop', 'xor', 'retn', 'nop', 'sub', 'inc', 'dec', 'add','imul', 'xchg', 'or', 'shr', 'cmp', 'call', 'shl', 'ror', 'rol', 'jnb','jz','rtn','lea','movzx']

dict_asm_opcodes = dict(zip(opcodes_for_bigram, [1 for i in range(len(opcodes_for_bigram))]))
if not os.path.isdir("./opcodes_test_malware_asm_files"):
    os.mkdir('opcodes_test_malware_asm_files')

def calculate_sequence_of_opcodes():
    asm_file_names=os.listdir('test_malware')
    for this_asm_file in tqdm(asm_file_names):
        each_asm_opcode_file = open("./opcodes_test_malware_asm_files/{}_opcode_asm_bi_grams.txt".format(this_asm_file.split('.')[0]), "w+")
        sequence_of_opcodes = ""
        with codecs.open('test_malware/' + this_asm_file, encoding='cp1252', errors ='replace') as asm_file:
            for lines in asm_file:
                
                line = lines.rstrip().split()            
                
                for word in line:
                    if dict_asm_opcodes.get(word)==1:
                        sequence_of_opcodes += word + ' '
        each_asm_opcode_file.write(sequence_of_opcodes + "\n")
        each_asm_opcode_file.close()
    
calculate_sequence_of_opcodes()

opcodes_asm__bigram_vocabulary = calculate_bigram(opcodes_for_bigram)
print('csv로 저장중...')
vectorizer_opcode = CountVectorizer(
    tokenizer=lambda x: x.split(),
    lowercase=False,
    ngram_range=(2, 2),
    vocabulary=opcodes_asm__bigram_vocabulary,
) 

file_list_opcode = os.listdir("./opcodes_test_malware_asm_files")

opcode_features = ["ID"] + vectorizer_opcode.get_feature_names() + ["labels"]

opcodes_benign_asm_bigram_df = pd.DataFrame(columns=opcode_features)

if not os.path.isdir("./test_set"):
    os.mkdir('test_set')
    
with open(
    "./test_set/opcodes_test_malware_asm_bigram_df.csv", mode="w"
) as opcodes_benign_asm_bigram_df:

    opcodes_benign_asm_bigram_df.write(",".join(map(str, opcode_features)))

    opcodes_benign_asm_bigram_df.write("\n")

    for _, this_asm_file in tqdm(enumerate(file_list_opcode)):

        this_file_id = this_asm_file.split("_")[0]
        this_file_labels = "1"
        this_asm_file = open("opcodes_test_malware_asm_files/" + this_asm_file)

        corpus_opcodes_from_this_asm_file = [
            this_asm_file.read().replace("\n", " ").lower()
        ]

        bigrams_opcodes_asm = vectorizer_opcode.transform(
            corpus_opcodes_from_this_asm_file
        )

        row = scipy.sparse.csr_matrix(bigrams_opcodes_asm).toarray()

        opcodes_benign_asm_bigram_df.write(
            ",".join(map(str, [this_file_id] + list(row[0]) + [this_file_labels]))
        )

        opcodes_benign_asm_bigram_df.write("\n")

        this_asm_file.close()


opcodes_benign_asm_bigram_df = pd.read_csv(
    "./test_set/opcodes_test_malware_asm_bigram_df.csv"
)

# 위에서 저장한 csv 파일 두개를 병합.
print('train_set.csv 병합중...')
input_file = r'featurization' # csv파일들이 있는 디렉토리 위치
output_file = r'featurization\train_set.csv' # 병합하고 저장하려는 파일명

allFile_list = glob.glob(os.path.join(input_file, 'opcodes_*')) # glob함수로 opcodes_로 시작하는 파일들을 모은다
print(allFile_list)
allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다

for file in allFile_list:
    df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
    allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다

dataCombine = pd.concat(allData, axis=0, ignore_index=True) # concat함수를 이용해서 리스트의 내용을 병합
# axis=0은 수직으로 병합함. axis=1은 수평. ignore_index=True는 인덱스 값이 기존 순서를 무시하고 순서대로 정렬되도록 한다.
dataCombine.to_csv(output_file, index=False) # to_csv함수로 저장한다. 인덱스를 빼려면 False로 설정
print('test_set.csv 병합중...')
input_file = r'test_set'
output_file = r'test_set\test_set.csv' # 병합하고 저장하려는 파일명

allFile_list = glob.glob(os.path.join(input_file, 'opcodes_*')) # glob함수로 opcodes_로 시작하는 파일들을 모은다
print(allFile_list)
allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다

for file in allFile_list:
    df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
    allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다

dataCombine = pd.concat(allData, axis=0, ignore_index=True) # concat함수를 이용해서 리스트의 내용을 병합
# axis=0은 수직으로 병합함. axis=1은 수평. ignore_index=True는 인덱스 값이 기존 순서를 무시하고 순서대로 정렬되도록 한다.
dataCombine.to_csv(output_file, index=False) # to_csv함수로 저장한다. 인덱스를 빼려면 False로 설정

data = pd.read_csv('featurization/train_set.csv')
test_data = pd.read_csv('test_set/test_set.csv')
data.head()

data.info()

data.drop(['ID'], axis=1, inplace=True)

data.describe()

sns.countplot(data['labels'])

train, test = train_test_split(data, test_size=0.2, random_state=0) 

x_train = train.drop(['labels'], axis=1)
y_train = train.labels

x_test = test.drop(['labels'], axis=1)
y_test = test.labels

print(len(train), len(test))
print('train_set 학습중...')
# SVM
model = svm.SVC(gamma='scale') # svm 모델은 분류에 사용되는 지도학습 머신러닝 모델
model.fit(x_train, y_train) # 모델 학습

y_pred = model.predict(x_test) # 학습된 모델을 통한 모델 예측

print('SVM: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100)) # SVM 모델을 통한 정확도 평가 점수 출력

print(metrics.classification_report(y_test, y_pred)) # 분류 성능 결과 표 출력


# DecisionTreeClassifier
model = DecisionTreeClassifier() # 결정트리 형태 분류 모델 생성
model.fit(x_train, y_train) # 모델 학습

y_pred = model.predict(x_test) # 학습된 모델을 통한 모델 예측

print('DecisionTreeClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100))  # DecisionTreeClassifier를 통한 정확도 평가 점수 출력


# KNeighborsClassifier
model = KNeighborsClassifier() # K-최근접 이웃 모델 생성
model.fit(x_train, y_train) # 모델 학습

y_pred = model.predict(x_test) # 학습된 모델을 통한 모델 예측

print('KNeighborsClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100)) # KNeighborsClassifier를 통한 정확도 평가 점수 출력


# LogisticRegression
model = LogisticRegression(solver='lbfgs', max_iter=2000) # 로지스틱 회귀 분류 모델 생성
model.fit(x_train, y_train) # 모델 학습

y_pred = model.predict(x_test) # 학습된 모델을 통한 모델 예측

print('LogisticRegression: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100)) # 로지스틱 회기분류를 통한 정확도 평가 점수 출력


# RandomForestClassifier
model = RandomForestClassifier(n_estimators=100) # 랜덤 포레스트 분류 생성 델생
model.fit(x_train, y_train) # 모델 학습

y_pred = model.predict(x_test) # 테스트셋의 X에 대한 예측값 y, 학습된 모델을 통한 모델 예측

print('RandomForestClassifier: %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100)) # RandomForestClassifier를 통한 정확도 평가 점수 출력


features = pd.Series(  
    model.feature_importances_,  # 중요도 순서대로 series값 정렬
    index=x_train.columns #
).sort_values(ascending=False) # 내림차순 정렬

print(features) 

top_5_features = features.keys()[:5] 

print(top_5_features)



model = svm.SVC(gamma='scale') # 1 / (n_features )를 감마 값 사용
model.fit(x_train[top_5_features], y_train) 

y_pred = model.predict(x_test[top_5_features]) #  학습된 모델을 통한 상위 5개 예측

print('SVM(Top 5): %.2f' % (metrics.accuracy_score(y_pred, y_test) * 100)) # SVM 상위 5개 정확도 평가 점수 출력


model = svm.SVC(gamma='scale')

cv = KFold(n_splits=5, shuffle=True, random_state=0) # 5개의 폴드세트를 분리함, 데이터를 분리하기 전에 데이터를 미리 섞을지 결정, 난수 값을 지정하면 여러번 다시 수행해도 동일한 결과가 나오게 해줌

accs, scores = [], []

for train_index, test_index in cv.split(data[top_5_features]): # 
    x_train = data.iloc[train_index][top_5_features]
    y_train = data.iloc[train_index].labels 

    
    x_test = data.iloc[test_index][top_5_features]
    y_test = data.iloc[test_index].labels

    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test) # 학습된 모델을 통한 모델 예측

    accs.append(metrics.accuracy_score(y_test, y_pred))

print(accs)


model = svm.SVC(gamma='scale')

cv = KFold(n_splits=5, shuffle=True, random_state=0)

accs = cross_val_score(model, data[top_5_features], data.labels, cv=cv)

print(accs)

# 모든 모델 테스트
print('train_set 학습 결과')
models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),  # solver: 최적화에 사용될 알고리즘 결정, max_iter:solver가 수렴하는 데 걸리는 최대 반복 횟수
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)  # n_estimators : 결정 tree의 개수
}

cv = KFold(n_splits=5, shuffle=True, random_state=0) # 5개의 폴드세트를 분리함, 데이터를 분리하기 전에 데이터를 미리 섞을지 결정, 난수 값을 지정하면 여러번 다시 수행해도 동일한 결과가 나오게 해줌

for name, model in models.items():
    scores = cross_val_score(model, data[top_5_features], data.labels, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))

print('데이터 정규화')
# 정규화
from sklearn.preprocessing import MinMaxScaler  #  각 feature의 최솟값과 최댓값을 기준으로 0~1 구간 내에 균등하게 값을 배정

scaler = MinMaxScaler(feature_range=(0, 1)) # 훈련 세트의 주어진 범위에 있도록 각 기능을 개별적으로 확장 및 변환
scaled_data = scaler.fit_transform(data[top_5_features]) # 상위 5개 변환

models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(), 
    'KNeighborsClassifier': KNeighborsClassifier(), 
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000), # solver: 최적화에 사용될 알고리즘 결정, max_iter:solver가 수렴하는 데 걸리는 최대 반복 횟수
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100) # n_estimators : 결정 tree의 개수
}

cv = KFold(n_splits=5, shuffle=True, random_state=0) # 5개의 폴드세트를 분리함, 데이터를 분리하기 전에 데이터를 미리 섞을지 결정, 난수 값을 지정하면 여러번 다시 수행해도 동일한 결과가 나오게 해줌

for name, model in models.items():
    scores = cross_val_score(model, scaled_data, data.labels, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100)) # 모델 이름과 정확도

print('test_set 적용 결과')
# test_set 적용 결과
x_test = test_data.drop(['labels'], axis=1)
y_test = test_data.labels

models = {
    'SVM': svm.SVC(gamma='scale'),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),  # solver: 최적화에 사용될 알고리즘 결정, max_iter:solver가 수렴하는 데 걸리는 최대 반복 횟수
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)  # n_estimators : 결정 tree의 개수
}

cv = KFold(n_splits=5, shuffle=True, random_state=0) # 5개의 폴드세트를 분리함, 데이터를 분리하기 전에 데이터를 미리 섞을지 결정, 난수 값을 지정하면 여러번 다시 수행해도 동일한 결과가 나오게 해줌

for name, model in models.items():
    scores = cross_val_score(model, data[top_5_features], data.labels, cv=cv)
    
    print('%s: %.2f%%' % (name, np.mean(scores) * 100))