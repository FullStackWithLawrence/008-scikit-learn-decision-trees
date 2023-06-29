# Encoding: utf-8
"""
    written by:     Lawrence McDaniel
                    https://lawrencemcdaniel.com

    date:           jun-2023

    usage:          minimalist implementation of Decision Tree model.
"""
import os
import warnings

# ------------------------------------------------------------------------------
# IMPORTANT: DON'T FORGET TO INSTALL THESE LIBRARIES WITH pip
# ------------------------------------------------------------------------------
# Code to ignore warnings from function usage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the Machine Learning models we require from Scikit-Learn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, recall_score

# module initializations
sns.set()
HERE = os.path.abspath(os.path.dirname(__file__))
warnings.filterwarnings("ignore")


def metrics_score(actual, predicted):
    """
    Create a common function for measuring the
    accuracy of both the train as well as test data.
    """
    print("Metrics Score.")
    print(classification_report(actual, predicted))

    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=["Not Cancelled", "Cancelled"],
        yticklabels=["Not Cancelled", "Cancelled"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()


def prepare_data():
    """
    Raw database transformations:
        - clean the data
        - remove columns that don't contain any information
        - recast data types as necessary
        - convert categorical data into series of dummy columns
        - split dependent / independent variables
        - split training / test data sets
    """
    print("Preparing data sets")
    original_db = pd.read_csv(os.path.join(HERE, "data", "reservations-db.csv"))

    # need to be careful to only work with a **COPY** of the original
    # source data, lest we accidentally permanently modify any of this
    # raw data.
    data = original_db.copy()

    # remove the ID column from the data set, since it contains
    # no predictive information.
    data = data.drop(["Booking_ID"], axis=1)

    # recast dependent variable as boolean
    data["booking_status"] = data["booking_status"].apply(
        lambda x: 1 if x == "Canceled" else 0
    )

    # hive off the dependent variable, "booking_status"
    x = data.drop(["booking_status"], axis=1)
    y = data["booking_status"]

    # encode all categorical features
    x = pd.get_dummies(x, drop_first=True)

    # Split data in train and test sets
    return train_test_split(x, y, test_size=0.30, stratify=y, random_state=1)


def decision_tree():
    """
    - create training and test data sets
    - create a Logistic Regression model
    - train the model
    - generate confusion matrix and f-score for the training set
    - generate confusion matrix and f-score for the test set
    """
    print("Decision Tree")
    x_train, x_test, y_train, y_test = prepare_data()

    print("- training")
    model_dt = DecisionTreeClassifier(class_weight={0: 0.17, 1: 0.83}, random_state=1)
    model_dt.fit(x_train, y_train)

    print("- modeling on training data")
    pred_train_dt = model_dt.predict(x_train)
    metrics_score(y_train, pred_train_dt)

    print("- modeling on test data")
    pred_test_dt = model_dt.predict(x_test)
    metrics_score(y_test, pred_test_dt)

    # Metrics to evaluate the model
    # ---------------------------------

    # Choose the type of classifier.
    estimator = DecisionTreeClassifier(class_weight={0: 0.17, 1: 0.83}, random_state=1)

    # Grid of parameters to choose from
    parameters = {
        "max_depth": np.arange(2, 7, 2),
        "max_leaf_nodes": [50, 75, 150, 250],
        "min_samples_split": [10, 30, 50, 70],
    }
    scorer = metrics.make_scorer(recall_score, pos_label=1)

    # Run the grid search
    grid_obj = GridSearchCV(estimator, parameters, scoring=scorer, cv=10)
    grid_obj = grid_obj.fit(x_train, y_train)

    # Set the clf to the best combination of parameters
    estimator = grid_obj.best_estimator_

    # Fit the best algorithm to the data.
    estimator.fit(x_train, y_train)

    dt_tuned = estimator.predict(x_train)
    metrics_score(y_train, dt_tuned)

    # Checking performance on the training dataset
    print("- remodeling on training data")
    y_pred_tuned = estimator.predict(x_test)
    metrics_score(y_test, y_pred_tuned)

    # visualization of decision tree
    feature_names = list(x_train.columns)
    plt.figure(figsize=(20, 10))
    out = tree.plot_tree(
        estimator,
        max_depth=3,
        feature_names=feature_names,
        filled=True,
        fontsize=9,
        node_ids=False,
        class_names=None,
    )
    # below code will add arrows to the decision tree split if they are missing
    for o in out:
        arrow = o.arrow_patch
        if arrow is not None:
            arrow.set_edgecolor("black")
            arrow.set_linewidth(1)
    plt.show()


if __name__ == "__main__":
    decision_tree()
