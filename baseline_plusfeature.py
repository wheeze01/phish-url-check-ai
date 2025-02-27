import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import re  # 정규 표현식 추가
import math # 엔트로피 계산
from collections import Counter  # Counter 추가

import warnings
warnings.filterwarnings(action='ignore')

### Data Load

# 학습/평가 데이터 로드
train_df = pd.read_csv('fake-url-check/open/train.csv')
test_df = pd.read_csv('fake-url-check/open/test.csv')

# '[.]'을 '.'으로 복구
train_df['URL'] = train_df['URL'].str.replace(r'\[\.\]', '.', regex=True)
test_df['URL'] = test_df['URL'].str.replace(r'\[\.\]', '.', regex=True)

### Freature-Engineering (FE)

## 새로운 변수 생성
# URL 길이
train_df['length'] = train_df['URL'].str.len()
test_df['length'] = test_df['URL'].str.len()

# 서브도메인 개수
train_df['subdomain_count'] = train_df['URL'].str.split('.').apply(lambda x: len(x) - 2)
test_df['subdomain_count'] = test_df['URL'].str.split('.').apply(lambda x: len(x) - 2)

# 특수 문자('-', '_', '/') 개수
train_df['special_char_count'] = train_df['URL'].apply(lambda x: sum(1 for c in x if c in '-_/'))
test_df['special_char_count'] = test_df['URL'].apply(lambda x: sum(1 for c in x if c in '-_/'))

# URL 경로의 깊이
train_df['path_depth'] = train_df['URL'].str.count('/')
test_df['path_depth'] = test_df['URL'].str.count('/')

# 연속된 숫자의 길이
train_df['max_numeric_sequence'] = train_df['URL'].apply(lambda x: max([len(seq) for seq in re.findall(r'\d+', x)] or [0]))
test_df['max_numeric_sequence'] = test_df['URL'].apply(lambda x: max([len(seq) for seq in re.findall(r'\d+', x)] or [0]))

# query_length
train_df['query_length'] = train_df['URL'].apply(lambda x: len(x.split('?')[1]) if '?' in x else 0)
test_df['query_length'] = test_df['URL'].apply(lambda x: len(x.split('?')[1]) if '?' in x else 0)

# num_query_params
train_df['num_query_params'] = train_df['URL'].apply(lambda x: len(x.split('?')[1].split('&')) if '?' in x else 0)
test_df['num_query_params'] = test_df['URL'].apply(lambda x: len(x.split('?')[1].split('&')) if '?' in x else 0)

# fragment_length
train_df['fragment_length'] = train_df['URL'].apply(lambda x: len(x.split('#')[1]) if '#' in x else 0)
test_df['fragment_length'] = test_df['URL'].apply(lambda x: len(x.split('#')[1]) if '#' in x else 0)

# path_tokens_count
train_df['path_tokens_count'] = train_df['URL'].apply(lambda x: len(x.split('?')[0].split('/')[1:]))
test_df['path_tokens_count'] = test_df['URL'].apply(lambda x: len(x.split('?')[0].split('/')[1:]))

# has_port
train_df['has_port'] = train_df['URL'].apply(lambda x: 1 if re.search(r':\d+', x) else 0)
test_df['has_port'] = test_df['URL'].apply(lambda x: 1 if re.search(r':\d+', x) else 0)

# num_digits
train_df['num_digits'] = train_df['URL'].apply(lambda x: len(re.findall(r'\d', x)))
test_df['num_digits'] = test_df['URL'].apply(lambda x: len(re.findall(r'\d', x)))

# num_special_chars
special_chars = ['@', '=', ';', '%', '+'] # 예시
train_df['num_special_chars'] = train_df['URL'].apply(lambda x: sum(x.count(c) for c in special_chars))
test_df['num_special_chars'] = test_df['URL'].apply(lambda x: sum(x.count(c) for c in special_chars))

# has_encoded_char
train_df['has_encoded_char'] = train_df['URL'].apply(lambda x: 1 if re.search(r'%[0-9a-fA-F]{2}', x) else 0)
test_df['has_encoded_char'] = test_df['URL'].apply(lambda x: 1 if re.search(r'%[0-9a-fA-F]{2}', x) else 0)

# longest_word_length
def get_longest_word_length(url):
    words = re.findall(r'\w+', url)  # 단어 추출 (알파벳, 숫자, _)
    if not words:
        return 0
    return max(len(word) for word in words)

train_df['longest_word_length'] = train_df['URL'].apply(get_longest_word_length)
test_df['longest_word_length'] = test_df['URL'].apply(get_longest_word_length)

# is_ip_address
train_df['is_ip_address'] = train_df['URL'].apply(lambda x: 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', x.split('/')[0]) else 0)
test_df['is_ip_address'] = test_df['URL'].apply(lambda x: 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', x.split('/')[0]) else 0)

# domain_hyphens
train_df['domain_hyphens'] = train_df['URL'].apply(lambda x: 1 if '-' in x.split('/')[0] else 0)
test_df['domain_hyphens'] = test_df['URL'].apply(lambda x: 1 if '-' in x.split('/')[0] else 0)

# entropy
def calculate_entropy(text):
    if not text:
        return 0
    length = len(text)
    counts = Counter(text)  # 수정된 부분
    entropy = 0
    for count in counts.values():
        p_x = float(count) / length
        if p_x > 0:
            entropy -= p_x * math.log(p_x, 2)
    return entropy

train_df['entropy'] = train_df['URL'].apply(calculate_entropy)
test_df['entropy'] = test_df['URL'].apply(calculate_entropy)



### EDA

## 악성 여부에 따른 분포 확인
# 변수 목록 (새로 추가된 변수 포함)
variables = ['length', 'subdomain_count', 'special_char_count', 'path_depth',
             'max_numeric_sequence', 'query_length', 'num_query_params',
             'fragment_length', 'path_tokens_count', 'has_port', 'num_digits',
             'num_special_chars', 'has_encoded_char', 'longest_word_length',
             'is_ip_address', 'domain_hyphens', 'entropy']


# 박스플롯 (시간 관계상 생략 - 필요하면 주석 해제 후 실행)
# for var in variables:
#     plt.figure(figsize=(8, 5))
#     sns.boxplot(data=train_df, x='label', y=var)
#     plt.title(f"Boxplot of {var} by is_malicious")
#     plt.xlabel("is_malicious")
#     plt.ylabel(var)
#     plt.xticks([0, 1], ['Non-Malicious', 'Malicious'])
#     plt.show()


### 상관 관계 분석 (새로운 변수 포함)

# 상관계수 계산
correlation_matrix = train_df[variables + ['label']].corr()

# 히트맵 시각화 (시간 관계상 생략 - 필요하면 주석 해제 후 실행)
# plt.figure(figsize=(12, 10)) # 크기 좀 더 키움
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm') # cmap 추가
# plt.title("Correlation Matrix")
# plt.show()


### 4. Pre-processing (전처리)

# 학습을 위한 학습 데이터의 피처와 라벨 준비
X = train_df[variables]  # 모든 숫자형 feature 사용
y = train_df['label']

# 추론을 위한 평가 데이터의 피처 준비
X_test = test_df[variables]

### K-Fold Model Training (모델 학습)

# XGBoost 학습 및 모델 저장 (K-Fold)
kf = KFold(n_splits=4, shuffle=True, random_state=42)
models = []  # 모델을 저장할 리스트
auc_scores = []

for idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print('-'*40)
    print(f'Fold {idx + 1} 번째 XGBoost 모델을 학습합니다.')
    print('Epoch|           Train AUC                 |          Validation AUC')

    # XGBoost 모델 학습
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="auc",
    )

    # 학습 및 Validation 성능 모니터링
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True,  # verbose를 True로 설정하여 학습 과정을 출력
        # early_stopping_rounds=5 # 조기 종료 (필요하면 주석 해제)
    )

    models.append(model)  # 모델 저장

    # 검증 데이터 예측 및 ROC-AUC 계산
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"Fold {idx + 1} CV ROC-AUC: {auc:.4f}")
    print('-'*40)
    auc_scores.append(auc)

print(f"K-Fold 평균 ROC-AUC: {np.mean(auc_scores):.4f}")

### 6. K-Fold Ensemble Inference (K-Fold 앙상블 추론)

# 평가 데이터 추론
# 각 Fold 별 모델의 예측 확률 계산
test_probabilities = np.zeros(len(X_test))

for model in models:
    test_probabilities += model.predict_proba(X_test)[:, 1]  # 악성 URL(1)일 확률 합산

# Soft-Voting 앙상블 (Fold 별 모델들의 예측 확률 평균)
test_probabilities /= len(models)
print('Inference Done.')

### 7. Submission (제출 파일 생성)

# 결과 저장
test_df['probability'] = test_probabilities
test_df[['ID', 'probability']].to_csv('./fake-url-check/submission-plusFeature.csv', index=False)
print('Done.')