import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.datasets import mnist

# Download and reshape data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
x = np.concatenate((x_train, x_test))[:6000]
y = np.concatenate((y_train, y_test))[:6000]

# Split into datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)


# function to work with multiple models
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # fit the model
    model = model.fit(features_train, target_train)
    # make a prediction
    y_pred = model.predict(features_test)
    # calculate accuracy
    score = accuracy_score(target_test, y_pred)
    print(f"Model: {model}\nAccuracy: {score}\n")
    return score


# models
models_results_dict = {KNeighborsClassifier(): 0, DecisionTreeClassifier(random_state=40): 0,
                       LogisticRegression(solver="liblinear", random_state=40): 0,
                       RandomForestClassifier(random_state=40): 0}

# apply function to model
for ml_model in models_results_dict.keys():
    models_results_dict[ml_model] = fit_predict_eval(
        model=ml_model,
        features_train=x_train,
        features_test=x_test,
        target_train=y_train,
        target_test=y_test
    )

# find better model and print results
accurate_model = max(models_results_dict, key=models_results_dict.get)
print(f"The answer to the question: {str(accurate_model)[:str(accurate_model).find('(')]}"
      f" - {round(models_results_dict[accurate_model], 3)}")