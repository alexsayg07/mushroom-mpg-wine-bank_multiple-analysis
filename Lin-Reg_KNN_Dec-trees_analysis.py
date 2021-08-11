# Yasmin Gil
# yasmingi@usc.edu
# Linear Regression, Decision Tree and KNN predictors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.metrics
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from yellowbrick.regressor import ResidualsPlot



def WineQualityClassification():
    # load data
    winequality_raw = pd.read_csv('winequality.csv')

    # Normalize everything but quality variable
    # separate quality column
    quality_df = winequality_raw['Quality']
    winequality_raw.drop(['Quality'], axis=1, inplace=True)
    normalize = Normalizer()
    normalize.fit(winequality_raw)
    winequality = pd.DataFrame(normalize.transform(winequality_raw), columns=winequality_raw.columns)
    winequality['Quality'] = quality_df
    print(winequality.head())

    # partition set 60/20/20 train valid test
    # 0.2 test size
    X = winequality.iloc[:, :11]
    y = quality_df
    X_split, X_test, y_split, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2020)
    # 0.6 train size 0.2 valid size
    X_train, X_valid, y_train, y_valid = train_test_split(X_split, y_split, test_size=0.25, stratify=y_split, random_state=2020)
    # build KNN classification model to predict quality & iterate k 1 through 30
    neighbors = np.arange(1, 31)
    train_accuracy = np.empty(30)
    valid_accuracy = np.empty(30)

    for k in neighbors:
        knn1 = KNeighborsClassifier(n_neighbors=k)
        knn2 = KNeighborsClassifier(n_neighbors=k)
        knn1.fit(X_train, y_train)
        knn2.fit(X_valid, y_valid)
        train_accuracy[k-1] = knn1.score(X_train, y_train)
        valid_accuracy[k-1] = knn2.score(X_valid, y_valid)

    # plot accuracies of A train and B train
    fig, ax = plt.subplots(1,1)
    ax.plot(neighbors, train_accuracy, label='A train')
    ax.plot(neighbors, valid_accuracy, label='B train')
    ax.set(xlabel='Number of Neighbors', ylabel='Accuracy', title='Varying accuracy for A train and B train')
    ax.legend()
    plt.grid(alpha=0.4)

    # best accuracies
    print("The best accuracy for A train would be k=1 and starts decreasing significantly after k=3")
    print("The best accuracy for B train would be k=1 and starts decreasing significantly after k=3")
    print("The best accuracy overall would be k=1 before significant loss in accuracy")


    # generate predictions for test with k = 1
    knnA = KNeighborsClassifier(n_neighbors=1)
    knnA.fit(X_train, y_train)
    y_predA = knnA.predict(X_test)
    # plot confusion matrix
    cf = metrics.confusion_matrix(y_test, y_predA)
    display_cf = sklearn.metrics.ConfusionMatrixDisplay(cf, display_labels=quality_df.unique())
    display_cf.plot()

    # print accuracy of model on the test data set & results dataframe
    score = knnA.score(X_test, y_test)
    print("Accuracy:", score)
    actual_pred = {'Quality' : y_test, 'PredictedQuaity' : y_predA}
    actual_pred_df = pd.DataFrame(actual_pred)
    results = pd.concat([X_test, actual_pred_df], axis=1)
    print(results.head())

    plt.show()


def PersonalLoanPredictionTree():
    # load data and print target variable
    loan_data_raw = pd.read_csv('UniversalBank.csv')
    # print(loan_data_raw.head())
    print("1. Target variable is Personal Loan")
    loan_data = loan_data_raw.drop(['Row', 'ZIP Code'], axis=1)
    personal_loan = loan_data['Personal Loan']

    # partition data
    X = loan_data.drop(['Personal Loan'], axis=1)
    y = personal_loan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020, stratify=y)

    # how many cases in training partition accepted loan?
    print("4. Number of people who accepted loan in train dataset:",
          y_train.value_counts()[1])

    # plot tree using entropy
    model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=2020)
    model_tree.fit(X_train, y_train)
    plt.figure(figsize=(17, 8))
    tree.plot_tree(model_tree, fontsize= 7, filled=True, feature_names=X.columns, class_names=['Reject', 'Accept'])

    # how many were classified correctly
    y_pred_train = model_tree.predict(X_train)

    cf = metrics.confusion_matrix(y_train, y_pred_train)
    display_cf = metrics.ConfusionMatrixDisplay(cf, display_labels=['non-acceptors', 'acceptors'])
    display_cf.plot()
    print("6. Acceptors classified as non-acceptors: ", cf[0, 1])
    print("7. Non-Acceptors classified as acceptors: ", cf[1, 0])

    # print accuracies of model
    print('Accuracy on test partition:', model_tree.score(X_test, y_test))
    print('Accuracy on train partition:', model_tree.score(X_train, y_train))

    plt.show()
    # section


def MushroomEdibilityTree():
    # load data
    mushroom_data_raw = pd.read_csv('mushrooms.csv')
    # variable of interest : class: p = poisonous, e=edible
    # drop useless info
    mushroom_data = mushroom_data_raw.drop(['veil-type'], axis=1)

    # change types to category & encode labels & save in a list for later
    mushroom_data = mushroom_data.astype('category')
    label_encoder = LabelEncoder()
    label_encoder_codes = list()
    for category in mushroom_data.columns:
        mushroom_data[category] = label_encoder.fit_transform(mushroom_data[category])
        # print(label_encoder.classes_)
        keys = label_encoder.classes_
        # print(label_encoder.transform(label_encoder.classes_))
        values = label_encoder.transform(label_encoder.classes_)
        dictionary = dict(zip(keys, values))
        label_encoder_codes.append(dictionary)
    # print(label_encoder_codes[0])

    # build tree
    X = mushroom_data.iloc[:, 1:]
    y = mushroom_data['class']
    # print(X.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020, stratify=y)
    model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=2020)
    model_tree.fit(X_train, y_train)

    # plot conf matrix and tree
    disp = plot_confusion_matrix(model_tree, X_test, y_test, display_labels=['Edible', 'Poisonous'])
    plot1 = plt.figure(1)
    plt.title('Confusion Matrix for Mushroom Toxicity')
    plot2 = plt.figure(2, figsize=(11, 6))
    plt.suptitle("Decision Tree for Mushroom Toxicity classification")
    tree.plot_tree(model_tree, fontsize= 7, filled=True, feature_names=X.columns, class_names=['Edible', 'Poisonous'])

    # print conf matrix, and accuracy scores
    print(disp.confusion_matrix)
    accuracy_train = model_tree.score(X_train, y_train)
    accuracy_test = model_tree.score(X_test, y_test)
    print("3. Accuracy for the training data:", accuracy_train)
    print("4. Accuracy for the testing data:", accuracy_test)

    # top three features for determining toxicity
    print("6. gill-color, spore-print-color, and gill-size are the top three features"
          " that are used to determine toxicity")

    # classify the following mushroom
    mushroom_1 = np.array([['x', 's', 'n', 't', 'y', 'f', 'c', 'n', 'k', 'e', 'e',
                  's', 's', 'w', 'w', 'w', 'o', 'p', 'r', 's', 'u']])
    # convert from categorical to codes
    # print(mushroom_1)
    for i in range(0, len(mushroom_1[0])):
        mushroom_1[0][i] = label_encoder_codes[i+1].get(mushroom_1[0][i])
    # print(mushroom_1)
    prediction = model_tree.predict(mushroom_1)

    # convert prediction code to class label
    class_key_list = list(label_encoder_codes[0].keys())
    class_values_list = list(label_encoder_codes[0].values())
    position = class_values_list.index(prediction)
    # print(label_encoder_codes[0])
    print('7. The mushroom was classified as:', class_key_list[position])

    plt.show()


def VehicleMPGLinearReg():
    # load data
    auto_data_raw = pd.read_csv('auto-mpg.csv')
    auto_data = auto_data_raw.drop(['No', 'car_name'], axis=1)
    # print(auto_data.columns)
    print('1. Mean of mpg:', auto_data['mpg'].mean())
    print('2. Median value of mpg:', auto_data['mpg'].median())
    # plot for data distribution
    plt.boxplot(auto_data['mpg'])
    plt.ylabel("MPG Values")
    plt.title("MPG boxplot")
    plt.grid(alpha=0.2)
    print("The mean is slightly higher than the median.")
    print("Looking at the boxplot, the median is very slightly skewed negatively of the mean")
    print("Overall, the data has normal skew")

    # Which variables have most correlation, make pairplot matrix
    sb.pairplot(auto_data)
    # cylinders, acceleration and car year don't show strong correlation, seems like big r2 value
    auto_data.drop(['cylinders', 'acceleration', 'model_year'], axis=1, inplace=True)

    print("5. Displacement or weight seem to be the most strongly correlated to mpg")
    print("6. Model_year seems to be the most weakly correlated")

    # Plot mpg vs displacement
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(auto_data['displacement'], auto_data['mpg'])
    ax[0].set(xlabel='Displacement', ylabel='MPG', title='MPG vs displacement')

    # build linear regression model
    x = auto_data['displacement']
    X = np.expand_dims(x, axis=1)
    y = auto_data['mpg']
    model = LinearRegression()
    model.fit(X, y)

    # predict & plot (8f)
    y_pred = model.predict(X)
    ax[1].scatter(x, y)
    ax[1].plot(x, y_pred)
    ax[1].set(xlabel='Displacement', title='Linear Regression for MPG vs dx')

    print('8a. Intercept value B0:', model.intercept_)
    print('8b. Coefficient value B1:', model.coef_)
    print('8c. MPG =', model.intercept_, '+', model.coef_, '* Displacement')
    print('8d. The predicted value of MPG decreases as displacement increases, hence negative coefficient')
    # 8e.
    print('8e. Given car with displacement = 220, MPG=', model.predict([[220]]))

    # plot residuals
    ridge = Ridge()
    visualizer = ResidualsPlot(ridge)
    visualizer.fit(X, y)
    visualizer.show()
    plt.show()


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    WineQualityClassification()
    PersonalLoanPredictionTree()
    MushroomEdibilityTree()
    VehicleMPGLinearReg()


if __name__ == "__main__":
    main()
