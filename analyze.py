from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import spy
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_roc(test_y, result_prob):
    '''
    @param -> test_y : the original y (actual values for the pred)
              result_prob : classification results with probabilities
    @return -> NULL : only displays a graph
    '''
    false_positive_rate, recall, thresholds = roc_curve(test_y, result_prob[:,1])
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.show()


print 'reading in data....\n'
data = pd.read_csv('~/Desktop/fletcher_project/review_data/review_data_ids.csv')
# init sklearn vectorizer which removes stop words
vectorizer = TfidfVectorizer(stop_words="english")
# vectorize documents
doc_vectors = vectorizer.fit_transform(data['review'])
dense_vectors = doc_vectors.todense()
vectors_df = pd.DataFrame(dense_vectors)
# add vectors to dataframe
data = pd.concat([data, vectors_df], axis=1)

#prep data for SPY step
U1 = data[data['source'] == 'trip_advisor']
U1 = U1.drop(['source', 'review', 'sentiment'], axis=1)
P1 = data[data['source'] == 'expedia']
P1 = P1.drop(['source', 'review', 'sentiment'], axis=1)

# perform SPY step of PU classification
print 'running spy....\n'
spy = spy.Spy(P1, U1, 20, 1.5)
RN = spy.RN[spy.RN['spy'] == 0]
ids = RN['id']

# create final 'fakes' dataset
fakes = data[data['id'].isin(ids)]
# include fakes from mturk
mturk_fakes = data[data['source'] == 'mturk']
fakes = pd.concat([fakes, mturk_fakes], ignore_index=True)
fakes['prediction'] = np.zeros(len(fakes))
# create final 'real' dataset
real = data[data['prediction'] == 1]
# concat
to_classify = pd.concat([fakes, real], ignore_index=True)


# prep data for classifier
train, test = split_data(to_classify, 0.3)
X_train = train.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
y_train = pd.DataFrame(train['prediction'], columns=['prediction'])
X_test = test.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
y_test = pd.DataFrame(test['prediction'], columns=['prediction'])

# train classifier
print 'training classifier...\n'
svc = SVC(kernel='rbf', C=1, gamma=1, probability=True)
svc.fit(X_train, y_train)
result = svc.predict(X_test)
result_prob = svc.predict_proba(X_test)
y_true = np.array(y_test['prediction'].tolist(), dtype='float32')
y_pred = np.array(result, dtype='float32')
# print results
print 'accuracy:', accuracy_score(y_true, y_pred)
print 'precision:', precision_score(y_true, y_pred)
print 'recall:', recall_score(y_true, y_pred)

# uncomment line below to show ROC curve
# plot_roc(y_test, result_prob)


# # 'fakes' dataset with NO MTURK fakes
# fakes1 = data[data['id'].isin(ids)]
# fakes1['prediction'] = np.zeros(len(fakes1))
# # create final 'real' dataset
# real1 = data[data['prediction'] == 1]
# # concat
# to_classify1 = pd.concat([fakes1, real1], ignore_index=True)

# train1, test1 = split_data(to_classify1, 0.3)
# X_train1 = train1.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
# y_train1 = pd.DataFrame(train1['prediction'], columns=['prediction'])
# X_test1 = test1.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1)
# y_test1 = pd.DataFrame(test1['prediction'], columns=['prediction'])

# svc1 = SVC(kernel='rbf', probability=True)
# svc1.fit(X_train1, y_train1)
# result1 = svc.predict(X_test1)
# result_prob1 = svc.predict_proba(X_test1)
# y_true1 = np.array(y_test1['prediction'].tolist(), dtype='float32')
# y_pred1 = np.array(result1, dtype='float32')
# print accuracy_score(y_true1, y_pred1)
# print precision_score(y_true1, y_pred1)
# print recall_score(y_true1, y_pred1)


reviews = [
            'I loved this hotel! It was beatiful in every way I could think of. Breakfast was on point, with good service and a very large selection of food, both at the buffet a la carte. The rooms were well decorated, and although not huge, made a good use of the available space. The staff was curteous and happy to help us with any problems we had. We called the concierge to ask for an extra bed for our son and it arrived promptly and free of charge! Prices were high, but more or less what you would expect from a hotel of this level. Would definitely reccomend.',
            "In my search for the perfect Italian resort, I came across the wonderful Borgo Egnazia Puglia. Everything about my stay exceeded my expectations! At first I was skeptical about taking my vacation their, but after doing some online research, the choice was clear. The amazing weather I experienced during my stay certainly helped my attitude towards Borgo Egnazia Puglia, but that's not the only reason I recommend this resort. /  / Located directly on the water, the views were amazing and the air was crisp and clear. Every breath I took was equally invigorating and refreshing. The decor and style of the resort is very pleasing to the senses, and a great blend of traditional architecture and modern conveniences. /  / The rooms were very clean, and the staff extremely helpful and prompt. The pool was plenty large enough that I never felt crowded by the other guests. I enjoyed my time in the cafe as well where the food and service (once again) exceeded my expectations. My suite had a balcony with an amazing view that I also spent a fair amount of time enjoying. /  / The rooms were very comfortably furnished, and my bed was more comfortable than the bed I have at home! I heard a lot of good things about the on site golf course, but I'm not that avid of a player so I can't speak from experience in that regard. /  / All in all, it was an extremely pleasant visit, and I highly recommend Borgo Egnazia Puglia for your Italian vacation",
            "I recently had the opportunity to stay at Borgo Egnazia Puglia. I would recommend this resort to anyone who is planning on vacationing in Italy. The grounds are magnificent. I had a suite in their luxurious hotel. The decor was unique and beautiful. I felt like royalty as I was made to feel very special. The cuisine was excellent. The resort offers a golf course, for those who choose to partake. The resort has large swimming pools and a beach for swimming or just basking in the sun. All in all this was the best vacation resort I have ever gone too.",
            "Talk about an amazing experience! The Borgo Egnazia Puglia provides all that and a bag of chips, from stunning rooms to an immaculate, peaceful setting and all of the amenities I could every desire, this hotel is truly an immaculate place to stay. My room was huge, filled with neutral colors for a tranquil environment that made it hard to want to leave. I loved the outstanding customer service, the upscale, on-site restaurants, concierge service and   the abundance of activities ranging from golf to tennis. Of course the spa was a really special treat that spoiled me so much. For anyone who wants to stay in absolute luxury, a world within a hotel, this is the can't miss destination.",
            'I stayed at the nomad hotel in new york two weeks ago and I was appalled. The receptionist could not pull up our room reservation on their booking system and it took over 40 minutes for them to track it down. Given the price of the hotel, we were extecting to at least be offered some sort of refreshement during our wait, but nothing. We finally got to our room, and it was cramped and had a horrible view of a back alley. We asked to change, and depite being inconvenienced earlier, they informed us we would have to pay for an upgrade to change rooms as none of the same typology as ours were available. The decor and general atmosophere was actually quite nice, but the service we recieved was definitely sub-par. Would reccomend staying elsewhere in new york, espeically given this very high price point!',
            ]
review_vector = vectorizer.transform(reviews)
dense_review_vector = review_vector.todense()
vector_df = pd.DataFrame(dense_review_vector)
print vector_df
res_review_prob = svc.predict_proba(dense_review_vector)
print res_review_prob, '\n\n'

dana = pd.read_csv('~/Desktop/fletcher_project/review_data/dana.csv')
dana_vector = vectorizer.transform(dana['Review'].apply(lambda x: str(x.decode('latin-1').encode("utf-8"))))
dense_dana_vectors = dana_vector.todense()
dana_df = pd.DataFrame(dense_dana_vectors)
dana_prob = svc.predict_proba(dense_dana_vectors)
dana_rez = svc.predict(dense_dana_vectors)
for i in xrange(len(dana_prob)):
    print dana_prob[i], dana['prediction'][i]

print 'accuracy:', accuracy_score(np.array(dana['prediction'], dtype='float32'), np.array(dana_rez, dtype='float32'))
print 'precision:', recall_score(np.array(dana['prediction'], dtype='float32'), np.array(dana_rez, dtype='float32'))
print 'recall:', precision_score(np.array(dana['prediction'], dtype='float32'), np.array(dana_rez, dtype='float32'))

# print cross_val_score(SVC(kernel='rbf', probability=True), to_classify.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1), to_classify['prediction'], cv=10, verbose=10, scoring='recall')

# print get_best_model_params(SVC(), to_classify.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1), to_classify['prediction'], param_grid_svc, 'accuracy', 3)

# plot_roc(y_true, result_prob)
# plot_roc(y_true1, result_prob1)
