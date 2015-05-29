import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

class Spy:
    """
    PU algorithm -> Learning from positive[label] and unlabelled data
    useful in this case because I want to classify reviews into real or fake
        no data classified as fake, plenty of data classified as real (sites like expedia)

    STEP 1 of PU:
        - SPY algorithm
            - determines a set of 'real negative data' (RN)
                - from only unlabelled(U) and positive(P) data

    1. RN = null
    2. S = 15% of P , randomly selected
    3. U' = U u S  -> label: 0
    4. P' = P - S  -> label: 1
    5. Run I-EM on (U', P') -> produces NB classifier
    6. classify each document in U' using the NB classifier created
    7. determine probability threshold (th) using S
        - this part is arbitrary
        - need to look at what values of S are classified as
            - determine from there
    8. last step in psuedo-code:
         for each document d-e-U'
            if Pr(1|d) < th then:
                RN = RN u {d}
    """

    def __init__(self, positive_pandas_df, unlabelled_pandas_df, perc_to_select):
        # try:
            assert isinstance(positive_pandas_df, pd.DataFrame), "must be a pandas DF"
            assert isinstance(unlabelled_pandas_df, pd.DataFrame), "must be a pandas DF"
            #init U -> unlabelled data
            self.U = unlabelled_pandas_df
            self.U['spy'] = [0]*len(self.U)
            #init P -> positive data
            self.P = positive_pandas_df
            self.P['spy'] = [0]*len(self.P)
            #init RN -> real negatives as null
            self.RN = None
            self.th = None
            #assert U and P have same structure
                # NOTE: (U will need a NaN y column)
            assert 'prediction' in self.P.columns, 'prediction must be a column in P'
            assert 'prediction' in self.U.columns, 'prediction must be a column in U'
            assert self.P.shape == self.U.shape, 'U and P must have same shape'
            #insert 'spy' Ps into U
            self.Pp, self.S = self.split_data(self.P, (perc_to_select/100.0))
            self.S['spy'] = [1]*len(self.S)
            self.Up = pd.concat([self.U, self.S], ignore_index=True)
            self.Up['prediction'] = [0]*len(self.Up)
            self.find_threshold()
            self.RN = self.Up[self.Up.prob_yes <= self.th]
        # except:
        #     print "Files must be in csv format, and must have same column format"

    def classify(self):
        df_fit = pd.concat([self.Up, self.Pp], ignore_index=True)
        classifier = MultinomialNB()
        fitted = classifier.fit(df_fit.drop(['prediction', 'spy'], axis=1),
                                df_fit['prediction'])
        probs = fitted.predict_proba(self.Up.drop(['prediction', 'spy'], axis=1))
        pred = fitted.predict(self.Up.drop(['prediction', 'spy'], axis=1))
        self.Up['NB_pred'] = pred
        self.Up['prob_yes'] = [x[1] for x in probs]

    def find_threshold(self):
        #add probs to Up
        self.classify()
        neg_class = self.Up[self.Up['NB_pred'] == 0]
        th = np.mean(neg_class['prob_yes'])
        self.th = th

    # def get_RN(self):
    #     self.RN = self.Up[self.Up.prob_yes <= self.th]


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

