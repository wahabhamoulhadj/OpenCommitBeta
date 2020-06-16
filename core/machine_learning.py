"""
Script for applying the machine learning algorithms.
"""

__author__ = "Wahab Hamou-Lhadj and Mohammed Shehab"
__copyright__ = "Copyright (c) 2019 Wahab Hamou-Lhadj and Mohammed Shehab"

import os

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample

from core.code_similarity import write_text
from core.config import project_name
from downloader.loader import create_nested_dir

# The main features that used for train model
features = ["exp", "rexp", "sexp", "NDEV", "AGE", "NUC", "LA", "LD", "NF", "LT", "entropy", "NS", "ND",
            "commit_time", "fix"]

project_name = project_name


def random_forest(train_features, train_labels):
    rf = RandomForestClassifier(random_state=2018, n_jobs=-1, n_estimators=1000, max_depth=7)
    rf.fit(train_features, train_labels)
    print('Training random forest model complete.')
    return rf


def logistic_regression(train_features, train_labels):
    logreg = LogisticRegression(random_state=2018, solver='liblinear')
    logreg.fit(train_features, train_labels)
    print('Training logistic regression model complete.')
    return logreg


def support_vector(train_features, train_labels):
    clf = SVC(C=0.3, random_state=2018, tol=0.0001, kernel='poly', max_iter=-1)
    clf.fit(train_features, train_labels)
    print('Training support vector model complete.')
    return clf


def train_model(dataset_path: str, model_name: str, data_method: str = 'down',
                test_size: float = 0.50) -> None:
    """
    The main function to train machine learning models
    :param dataset_path: the location for csv file that contains dataset, the default location is in results folder e.g. (./results/druid_sample_features.csv)
    :param model_name: the name of machine learning model (rf-> random_forest, lf-> logistic_regression and svm-> support vectors).
    :param data_method: the method used to solve unbalanced data. (down-> down-sampling and over-> for up-sampling).
    :param test_size: the size for testing phase.
    :return: None
    """
    # Read data from file
    df = pd.read_csv(dataset_path)
    # Balance the data by method name
    df = balance_data(df=df, data_method=data_method)
    # Split data for train and test
    train_features, test_features, train_labels, test_labels = train_test_split(df[features], df['label'],
                                                                                test_size=test_size,
                                                                                random_state=42)

    if model_name.upper() == 'LR':
        trained_model = logistic_regression(train_features=train_features, train_labels=train_labels)
    elif model_name.upper() == 'RF':
        trained_model = random_forest(train_features=train_features, train_labels=train_labels)
    elif model_name.upper() == 'SVM':
        trained_model = support_vector(train_features=train_features, train_labels=train_labels)

    else:
        print(
            "Error : model name is not supported, the support models are: random forest -> RF or logistic "
            "regression -> LR")
        print("please use model name RF for random forest or LR for logistic regression")
        return

    predictions = test_model(model=trained_model, model_name=model_name,
                             test_features=test_features, test_labels=test_labels)
    train_features['label'] = train_labels
    test_features['label'] = test_labels
    export_dataset(data=df,
                   train=train_features,
                   test=test_features,
                   predictions=predictions)


def balance_data(df: object, data_method: str):
    """

    :param df: pandas data frame
    :type data_method: string name of balancing process
    """
    if df is None:
        print("Dataset is not found.")
        return
    if data_method == 'over':
        df_fix = df[df.label == 0]
        df_bug = df[df.label == 1]

        bug_up_sampled = resample(df_bug,
                                  replace=True,  # sample with replacement
                                  n_samples=len(df_fix),  # match number in majority class
                                  random_state=48)  # reproducible results
        df = pd.concat([df_fix, bug_up_sampled])
    else:
        df_fix = df[df.label == 0]
        df_bug = df[df.label == 1]

        bug_up_sampled = resample(df_fix,
                                  replace=False,  # sample with replacement
                                  n_samples=len(df_bug),  # match number in majority class
                                  random_state=48)  # reproducible results
        df = pd.concat([df_bug, bug_up_sampled])

    return df


def test_model(model, model_name, test_features, test_labels):
    print(f"+ Test model {model_name}.")
    predictions = model.predict(test_features)

    labels = ['0', '1']
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    print('Support Vector results:')
    print(classification_report(test_labels, predictions, target_names=labels))
    print("TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('Precision = {} , Recall = {}'.format(precision, recall))

    report_ml = "TP = {}, TN = {}, FP = {}, FN = {}".format(tp, tn, fp, fn)
    report_ml += '\r\n' + 'Precision = {} , Recall = {}'.format(precision, recall)
    write_text(report_ml, './results/' + project_name, project_name + '_ml_results_' + model_name, '.txt')
    print(f"+ Model {model_name} is tested.")
    return predictions


def export_dataset(data, train, test, predictions):
    data_path = './results/' + project_name + '/dataset/'
    if not os.path.exists(data_path):
        create_nested_dir(data_path)

    data.to_csv(data_path + project_name + '_data.csv', header=True, index=False)
    commit_hex = data['commit_hex']
    commit_time = data['commit_time']
    df = pd.DataFrame(columns=["commit_hex"], data=commit_hex)
    df["commit_time"] = commit_time
    train = train.merge(df, on='commit_time', how='left')
    train.to_csv(data_path + project_name + '_train.csv', header=True, index=False)
    test["predictions"] = predictions
    test = test.merge(df, on='commit_time', how='left')
    test.to_csv(data_path + project_name + '_test.csv', header=True, index=False)
