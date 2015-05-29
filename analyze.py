from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import spy_final2

data = pd.read_csv('~/Desktop/fletcher_project/review_data/review_data.csv')
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(data['review'])
dense_vectors = doc_vectors.todense()
vectors_df = pd.DataFrame(dense_vectors)
#add vectors
data = pd.concat([data, vectors_df], axis=1)




U = data[data['source'] == 'trip_advisor']
U = U.drop(['source', 'review', 'sentiment'], axis=1)
P = data[data['source'] == 'expedia']
P = P.drop(['source', 'review', 'sentiment'], axis=1)


spy = spy_final2.Spy(P, U, 20, 1.5)

RN = spy.RN[spy.RN['spy'] == 0]
print len(RN)
mturk = data[data['source'] == 'mturk']
print len(mturk)
real_negatives = pd.concat([RN, mturk], ignore_index=True)
print len(real_negatives)

print len(U)
print len(spy.RN)
print len(spy.RN[spy.RN['spy'] == 1])

print .2*len(P)
