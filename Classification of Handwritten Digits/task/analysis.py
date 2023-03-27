import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from warnings import simplefilter


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    target_predict = model.predict(features_test)
    score = accuracy_score(target_test, target_predict)
    return model.__class__.__name__, score


if __name__ == '__main__':
    simplefilter(action='ignore', category=FutureWarning)
    random_state = 40
    limit_size = 6000
    (x_train, y_train), (x_test, y_tst) = keras.datasets.mnist.load_data()
    new_form = (x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_train_reshape = np.reshape(x_train, new_form)
    x_cut = x_train_reshape[0:limit_size]
    y_cut = y_train[0:limit_size]
    X_train, X_test, y_train, y_test = train_test_split(x_cut, y_cut, test_size=0.3, random_state=random_state)

    # normalize the data
    X_train_norm = Normalizer().fit_transform(X_train)
    X_test_norm = Normalizer().fit_transform(X_test)

    # a list of models
    models = [KNeighborsClassifier(),
              DecisionTreeClassifier(random_state=random_state),
              LogisticRegression(random_state=random_state),
              RandomForestClassifier(random_state=random_state)]

    # res_norm = [fit_predict_eval(i, X_train_norm, X_test_norm, y_train, y_test) for i in models]
    # res = [fit_predict_eval(i, X_train, X_test, y_train, y_test) for i in models]
    # for i in res_norm:
    #     print(f"Model: {i[0]}\nAccuracy: {i[1]}\n")
    # res_norm.sort(key=lambda x: x[1], reverse=True)
    # sum_accuracy_norm = sum([x[1] for x in res_norm])
    # sum_accuracy = sum([x[1] for x in res])

    # print(f"The answer to the 1st question: {'yes' if sum_accuracy_norm > sum_accuracy else 'no'}\n")
    #
    # print(f"The answer to the 2nd question: "
    #       f"{res_norm[0][0]}-{round(res_norm[0][1], 3)}, "
    #       f"{res_norm[1][0]}-{round(res_norm[1][1], 3)}")

    # stage 5
    # KNeighbours
    knnb_param = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
    clf_kn = GridSearchCV(KNeighborsClassifier(), param_grid=knnb_param, scoring='accuracy', n_jobs=-1)
    clf_kn.fit(X_train_norm, y_train)
    best_est_kn = clf_kn.best_estimator_
    acc_kn = accuracy_score(y_test, clf_kn.predict(X_test_norm))

    # h = fit_predict_eval(clf_kn.best_estimator_, X_train_norm, X_test, y_train, y_test)

    # RandomForest

    rf_param = {'n_estimators': [588, 590], 'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced', 'balanced_subsample']}

    clf_rf = GridSearchCV(RandomForestClassifier(random_state=random_state),
                          param_grid=rf_param, scoring='accuracy', n_jobs=-1)
    clf_rf.fit(X_train_norm, y_train)
    best_est_rf = clf_rf.best_estimator_
    acc_rf = accuracy_score(y_test, clf_rf.predict(X_test_norm))
    h = fit_predict_eval(best_est_rf, X_train_norm, X_test, y_train, y_test)

    # 'criterion': ['gini', 'entropy', 'log_loss']}

    #  total

    print(f"K-nearest neighbours algorithm best estimator: {best_est_kn}\n"
          f"accuracy: {acc_kn}\n")

    print(f"Random forest algorithm best estimator: {best_est_rf}\n"
          f"accuracy: {acc_rf}")

    # print(h)
