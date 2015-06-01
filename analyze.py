from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import spy


def split_data(self, data, size):
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
to_classify = pd.concat([fakes, real], ignore_index=True


# train classifier

