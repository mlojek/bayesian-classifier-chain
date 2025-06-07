"""
Experiment to find the best label order in the Bayesian Classifier Chain
by randomly sampling N label orders.
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from xgboost import XGBClassifier

from datasets import load_bibtex_dataset, load_youtube_dataset
from models.bayesian_classifier_chain import BayesianClassifierChain


def find_best_label_order(
    classifier,
    x_train,
    y_train,
    x_test,
    y_test,
    n_samples=100,
):
    """
    TODO docstring
    """
    best_found_permutation = None
    best_hamming_loss = np.inf
    num_labels = np.array(y_train).shape[1]

    for _ in tqdm(range(n_samples), desc="Searching for best label order"):
        label_permutation = np.random.permutation(num_labels).tolist()

        chain = BayesianClassifierChain(
            classifier=classifier, custom_label_order=label_permutation
        )

        chain.fit(x_train, y_train)
        results = chain.evaluate(x_test, y_test)

        if results["hamming_loss"] < best_hamming_loss:
            best_found_permutation = label_permutation
            best_hamming_loss = results["hamming_loss"]
            print(f"Found new best permutation {best_found_permutation}")

    return best_found_permutation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=["youtube", "bibtex"],
        help="Multilabel classification dataset to use.",
    )
    parser.add_argument(
        "classifier",
        choices=["xgboost", "naive_bayes", "logistic_regression"],
        help="Classifier to use in the chain.",
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of random orders to check.",
    )
    args = parser.parse_args()

    # constants
    RANDOM_SEED = 0
    TEST_SIZE = 0.1

    # load the dataset
    match args.dataset:
        case "youtube":
            loaded_dataset = load_youtube_dataset()
        case "bibtex":
            loaded_dataset = load_bibtex_dataset()
        case _:
            raise ValueError(f"Invalid dataset name {args.dataset}!")

    # split the dataset into train and test parts
    X = loaded_dataset.data
    Y = loaded_dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    # create the classifier
    match args.classifier:
        case "xgboost":
            testClassifier = XGBClassifier(
                objective="binary:logistic",
                random_state=RANDOM_SEED,
            )
        case "naive_bayes":
            testClassifier = GaussianNB()
        case "logistic_regression":
            testClassifier = LogisticRegression(random_state=RANDOM_SEED)
        case _:
            raise ValueError(f"Invalid classifier name {args.classifier}!")

    # perform the experiment
    best_found = find_best_label_order(
        testClassifier,
        X_train,
        y_train,
        X_test,
        y_test,
        n_samples=args.num_samples,
    )
    print(best_found)
