# import warnings
# warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

for _ in range(300):
    train_df = pd.read_csv("./data/titanic/train.csv")
    test_df = pd.read_csv("./data/titanic/test.csv")

    train_df = train_df.drop(["Ticket", "Cabin"], axis=1)
    test_df = test_df.drop(["Ticket", "Cabin"], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset["Title"] = dataset.Name.str.extract("([A-Za-z]+)", expand=False)

    for dataset in combine:
        dataset["Title"] = dataset["Title"].replace(["Lady", "Countess", "Capt", "Col", 
                                                    "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
        dataset["Title"] = dataset["Title"].replace(["Mlle", "Ms"], "Miss")
        dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")

    title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
    for dataset in combine:
        dataset["Title"] = dataset["Title"].map(title_mapping)
        dataset["Title"] = dataset["Title"].fillna(0)

    train_df = train_df.drop(["Name", "PassengerId"], axis=1)
    test_df = test_df.drop(["Name"], axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset["Sex"] = dataset["Sex"].map({"female":0, "male":1}).astype(int)

    guess_ages = np.zeros((2,3))
    for dataset in combine:
        for i in range(0,2):
            for j in range(0,3):
                guess_df = dataset[(dataset["Sex"] == i) & \
                                (dataset["Pclass"] == j+1)]["Age"].dropna()
                age_guess = guess_df.median()
                guess_ages[i, j] = int( age_guess/0.5+0.5)*0.5

        for i in range(0,2):
            for j in range(0,3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), "Age"] = guess_ages[i, j]
        dataset["Age"] = dataset["Age"].astype(int)

    train_df["AgeBand"] = pd.cut(train_df["Age"], 5)

    for dataset in combine:
        dataset.loc[dataset["Age"] <= 16, "Age"] = 0
        dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 32), "Age"] = 1
        dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 48), "Age"] = 2
        dataset.loc[(dataset["Age"] > 48) & (dataset["Age"] <= 64), "Age"] = 3
        dataset.loc[dataset["Age"] > 64, "Age"] = 4

    train_df = train_df.drop(["AgeBand"], axis=1)
    combine = [train_df, test_df]
    for dataset in combine:
        dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    for dataset in combine:
        dataset["IsAlone"] = 0
        dataset.loc[dataset["FamilySize"]==1, "IsAlone"] = 1
    train_df = train_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)
    test_df = test_df.drop(["Parch", "SibSp", "FamilySize"], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset["Age*Pclass"] = dataset.Age * dataset.Pclass
        dataset.drop(["Pclass"], axis=1)
        dataset.drop(["Age"], axis=1)
    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
    for dataset in combine:
        dataset["Embarked"] = dataset["Embarked"].map({"S":0, "C":1, "Q":2}).astype(int)
    test_df.fillna({"Fare":test_df["Fare"].dropna().median()}, inplace=True)
    train_df["FareBand"] = pd.qcut(train_df["Fare"], 4)
    for dataset in combine:
        dataset.loc[dataset["Fare"] <= 7.91, "Fare"] = 0
        dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] < 14.454), "Fare"] = 1
        dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] < 31), "Fare"] = 2
        dataset.loc[(dataset["Fare"] > 31), "Fare"] = 3
        dataset["Fare"] = dataset["Fare"].astype(int)

    train_df = train_df.drop(["FareBand"], axis=1)
    combine = [train_df, test_df]

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()

    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

    coeff_df = pd.DataFrame(train_df.columns.delete(0))
    coeff_df.columns = ["Feature"]
    coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
    coeff_df['Absolute Correlation'] = coeff_df['Correlation'].abs()
    sorted_coeff_df = coeff_df.sort_values(by="Absolute Correlation", ascending=False)
    sorted_coeff_df = sorted_coeff_df.drop(columns=["Correlation"])

    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

    gaussian = GaussianNB()
    gaussian.fit(X_train, Y_train)
    Y_pred = gaussian.predict(X_test)
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

    perceptron = Perceptron()
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

    linear_svc = LinearSVC(dual="auto")
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

    models = pd.DataFrame({
        "Model" : ["Support Vector Machines", "KNN", "LogisticRegression",
                "Random Forest", "Naive Bayes", "Perceptron",
                "Stochastic Gradient Decent", "LinearSVC",
                "Decision Tree"],
        "Score": [acc_svc, acc_knn, acc_log,
                acc_random_forest, acc_gaussian, acc_perceptron,
                acc_sgd, acc_linear_svc, acc_decision_tree]})

    print(models)