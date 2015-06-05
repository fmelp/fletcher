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
from sklearn.learning_curve import learning_curve
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


def get_best_model_params(clf, train_X, train_y, param_grid, scoring_metric, cv):
    '''
    @param -> train_X : n-feature matrix : train feature data
              train_y : 1-feature matrix : train result data
              clf : sklearn_classifier : simply initiated classifier of choice
              param_grid : list of dictionaries :
                           [{'max_depth':[1,2,3]}] : of parameters to tweak
              scoring_metric : str : accuracy, precision, recall, f1 or others(?)
              cv : int : number of times to run cross cross validation

    @return -> best_estimator_ : sklearn_classifier : classifier tuned w best params
               grid_scores : list : summary of results

    NOTE: Look @ output in console to see runtimes of each of the params

    '''
    grid_search = GridSearchCV(clf, param_grid,
                                   scoring=scoring_metric, cv=cv, verbose=10)
    grid_search.fit(train_X, train_y)
    return grid_search.best_estimator_, grid_search.grid_scores_

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

# param_grid_svc = [{"C": [0.01, 0.1, 1], "kernel": ['rbf'], "gamma":[0.0, 0.01, 0.1, 1]}]
# print get_best_model_params(SVC(), to_classify.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1), to_classify['prediction'], param_grid_svc, 'accuracy', 3)

print cross_val_score(SVC(kernel='rbf', C=1, gamma=1, probability=True), to_classify.drop(['prediction', 'source', 'id', 'review', 'sentiment'], axis=1), to_classify['prediction'], cv=20)

# plot_roc(y_true, result_prob)
# plot_roc(y_true1, result_prob1)
