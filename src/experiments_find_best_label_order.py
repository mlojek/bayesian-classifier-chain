from itertools import permutations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

from datasets import load_bibtex_dataset, load_youtube_dataset
from models.bayesian_classifier_chain import BayesianClassifierChain


def find_best_label_order(classifier, x_train, y_train, x_test, y_test):
    best_found_permutation = None
    best_hamming_loss = np.inf

    for label_permutation in tqdm(permutations(range(np.array(y_train).shape[1]))):
        chain = BayesianClassifierChain(
            classifier=classifier, custom_label_order=label_permutation
        )

        chain.fit(x_train, y_train)
        results = chain.evaluate(x_test, y_test)

        if results["hamming_loss"] < best_hamming_loss:
            best_found_permutation = label_permutation
            best_hamming_loss = results["hamming_loss"]

            print(f"Found new best permuation {best_found_permutation}")

    return best_found_permutation


data_split_random_seed = 0
test_set_size = 0.1

loaded_dataset = load_youtube_dataset()
X = loaded_dataset.data
Y = loaded_dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=test_set_size,
    random_state=data_split_random_seed,
)

# custom_label_order = list(range((np.array(y_train)).shape[1]))
# temp = custom_label_order[3]
# custom_label_order[3] = custom_label_order[10]
# custom_label_order[10] = temp
# custom_label_order.reverse()

testClassifier = GaussianNB()
find_best_label_order(testClassifier, X_train, y_train, X_test, y_test)
# testChain = BayesianClassifierChain(
#     classifier=testClassifier,
#     custom_label_order=custom_label_order,
# )

# testChain.fit(X_train, y_train)
# results = testChain.evaluate(X_test, y_test)

# print(
#     f"subset_accuracy: {results['subset_accuracy']}, hamming_loss: {results['hamming_loss']}"
# )
# print(
#     f"precision_score: {results['precision_score']}, recall_score: {results['recall_score']}"
# )
