import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from xgboost import XGBClassifier

if __name__ == '__main__':
    
    # data
    train_data = pd.read_csv("training_data.csv", ",")
    test_data = pd.read_csv("test_data.csv", ",")
    train_x = pd.read_csv("train_pocessed.csv")
    train_y = train_data[['stars']] # row 2002 * 1
    test_x = pd.read_csv("test_pocessed.csv")



    # tfidf for train
    train_x = np.array(train_x).ravel()
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_x)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # X_train_tfidf is train_x

    # tfidf for test

    test_x = np.array(test_x).ravel()
    X_new_counts = count_vect.transform(test_x)
    X_test_tfidf = tfidf_transformer.transform(X_new_counts)
    # X_test_tfidf is test_x

    # build model

    model = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        seed=27,
        num_class=6)
    model.fit(X_train_tfidf, train_y)
    print(model.score(X_train_tfidf, train_y))
    # predicted
    pred = model.predict(X_test_tfidf)
    test_data = test_data.review_id
    test_data = test_data.values

    # write pred csv
    cou = {"0": test_data,
           "1": pred
           }
    finaldf = pd.DataFrame(cou)
    finaldf.to_csv("cp2.csv", header=False, index=False)
