from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import spy
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def split_data(data, size):
    """
    @param -> data : pandas DataFrame
    @return -> train, test : split into 2 pandas DFs WITH  HEADERS
    """
    column_names = list(data.columns.values)
    column_dict = dict(zip(range(len(column_names)), column_names))
    train, test = train_test_split(data, test_size=size)
    train = pd.DataFrame(train)
    train.rename(columns=column_dict, inplace=True)
    test = pd.DataFrame(test)
    test.rename(columns=column_dict, inplace=True)
    return train, test

data = pd.read_csv('~/Desktop/fletcher_project/review_data/review_data_ids.csv')
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(data['review'])
dense_vectors = doc_vectors.todense()
vectors_df = pd.DataFrame(dense_vectors)
#add vectors
data = pd.concat([data, vectors_df], axis=1)


U1 = data[data['source'] == 'trip_advisor']
U1 = U1.drop(['source', 'review', 'sentiment'], axis=1)
P1 = data[data['source'] == 'expedia']
P1 = P1.drop(['source', 'review', 'sentiment'], axis=1)

# perform SPY step of PU classification
spy = spy.Spy(P1, U1, 20, 1.5)
RN = spy.RN[spy.RN['spy'] == 0]
ids = RN['id']

# create final 'fakes' dataset
fakes = data[data['id'].isin(ids)]
mturk_fakes = data[data['source'] == 'mturk']
fakes = pd.concat([fakes, mturk_fakes], ignore_index=True)
fakes['prediction'] = np.zeros(len(fakes))
# create final 'real' dataset
real = data[data['prediction'] == 1]
# concat
to_classify = pd.concat([fakes, real], ignore_index=True)

# train classifier
train, test = split_data(to_classify, 0.3)
X_train = train.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
y_train = pd.DataFrame(train['prediction'], columns=['prediction'])
X_test = test.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
y_test = pd.DataFrame(test['prediction'], columns=['prediction'])

svc = SVC(kernel='rbf', probability=True)
svc.fit(X_train, y_train)
result = svc.predict(X_test)
result_prob1 = svc.predict_proba(X_test)
y_true = np.array(y_test['prediction'].tolist(), dtype='float32')
y_pred = np.array(result, dtype='float32')
print accuracy_score(y_true, y_pred)
print precision_score(y_true, y_pred)
print recall_score(y_true, y_pred)


# 'fakes' dataset with NO MTURK fakes
fakes1 = data[data['id'].isin(ids)]
fakes1['prediction'] = np.zeros(len(fakes1))
# create final 'real' dataset
real1 = data[data['prediction'] == 1]
# concat
to_classify1 = pd.concat([fakes1, real1], ignore_index=True)

train1, test1 = split_data(to_classify1, 0.3)
X_train1 = train1.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
y_train1 = pd.DataFrame(train1['prediction'], columns=['prediction'])
X_test1 = test1.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
y_test1 = pd.DataFrame(test1['prediction'], columns=['prediction'])

svc1 = SVC(kernel='rbf', probability=True)
svc1.fit(X_train1, y_train1)
result1 = svc.predict(X_test1)
result_prob1 = svc.predict_proba(X_test1)
y_true1 = np.array(y_test1['prediction'].tolist(), dtype='float32')
y_pred1 = np.array(result1, dtype='float32')
print accuracy_score(y_true1, y_pred1)
print precision_score(y_true1, y_pred1)
print recall_score(y_true1, y_pred1)


