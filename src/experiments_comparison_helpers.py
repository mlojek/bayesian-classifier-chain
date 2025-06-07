import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss
from scipy.stats import mannwhitneyu
from sklearn.base import ClassifierMixin as SklearnClassifier
from sklearn.base import clone
from typing import Dict, Callable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.bayesian_classifier_chain import BayesianClassifierChain


def is_a_stochastically_greater_than_b(a, b) -> tuple[float, float, bool]:
    true_p_value = mannwhitneyu(a, b, alternative="greater")[1]
    if (true_p_value < 0.05):
        return true_p_value, None, True
    else:
        false_p_value = mannwhitneyu(a, b, alternative="less")[1]
        if (false_p_value < 0.05):
            return true_p_value, false_p_value, False
    return true_p_value, false_p_value, None


def compare_series_results(series_results: Dict[str, Dict[str, float]]):
    chain_accs = series_results['chain_results']['accuracies']
    chain_precs = series_results['chain_results']['precisions']
    chain_recalls = series_results['chain_results']['recalls']
    chain_hls = series_results['chain_results']['hamming_losses']
    single_accs = series_results['single_results']['accuracies']
    single_precs = series_results['single_results']['precisions']
    single_recalls = series_results['single_results']['recalls']
    single_hls = series_results['single_results']['hamming_losses']

    is_accuracy_better = "Inconclusive"
    is_precision_better = "Inconclusive"
    is_recall_better = "Inconclusive"
    is_hamming_loss_better = "Inconclusive"
    # is chain accuracy better? (greater?)
    accs_p_value, accs_p_value_negated, res = is_a_stochastically_greater_than_b(
        chain_accs, single_accs)
    if res is not None:
        is_accuracy_better = res
    # is chain precision better? (greater?)
    precs_p_value, precs_p_value_negated, res = is_a_stochastically_greater_than_b(
        chain_precs, single_precs)
    if res is not None:
        is_precision_better = res
    # is chain recall better? (greater?)
    recalls_p_value, recalls_p_value_negated, res = is_a_stochastically_greater_than_b(
        chain_recalls, single_recalls)
    if res is not None:
        is_recall_better = res
    # is chain hamming loss better? (less?)
    hls_p_value, hls_p_value_negated, res = is_a_stochastically_greater_than_b(
        single_hls, chain_hls)
    if res is not None:
        is_hamming_loss_better = res

    print(
        f"P values for: accuracy: {accs_p_value}, precision: {precs_p_value}, recall: {recalls_p_value}, hamming loss: {hls_p_value}")
    print(
        f"'negated' P values for: accuracy: {accs_p_value_negated}, precision: {precs_p_value_negated}, recall: {recalls_p_value_negated}, hamming loss: {hls_p_value_negated} ['None' means the p-value gave conclusive result]")
    print(f"Is accuracy better for chain? {is_accuracy_better}")
    print(f"Is precision better for chain? {is_precision_better}")
    print(f"Is recall better for chain? {is_recall_better}")
    print(f"Is hamming loss better for chain? {is_hamming_loss_better}")


def perform_train_evaluate_chain_vs_single(classifier_chain: BayesianClassifierChain, single_classifier: SklearnClassifier, iterations_count: int, load_dataset_function: Callable, test_set_size: float) -> Dict[str, Dict[str, float]]:
    # how many series of "fitting and evaluating" to perform

    loaded_dataset = load_dataset_function()
    X = loaded_dataset.data
    Y = loaded_dataset.target

    # record prediction evaluations for Mann-Whitney U test later
    chain_accs = []
    chain_precs = []
    chain_recalls = []
    chain_hls = []
    single_accs = []
    single_precs = []
    single_recalls = []
    single_hls = []

    print(
        f"starting comparison: bayesian chain of: {classifier_chain.base_classifier} vs single {single_classifier}")
    for split_seed in tqdm(range(0, iterations_count), desc="Performing series of 'fitting and evaluating' on different dataset splits for comparison between classifier chain vs single"):
        # perform data set split
        data_split_random_seed = split_seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=test_set_size, random_state=data_split_random_seed)

        # Evaluate chain of classifiers
        classifier_chain.fit(X_train, y_train)
        results = classifier_chain.evaluate(X_test, y_test)

        # collect predictions of single classifiers
        predicted_labels_by_single_classifier = np.zeros(
            (X_test.shape[0], y_test.shape[1]))
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        for i in range(y_train.shape[1]):
            current_classifier = clone(single_classifier)

            current_classifier.fit(X_train_dense, y_train[:, i])
            y_predicted = current_classifier.predict(X_test_dense)
            predicted_labels_by_single_classifier[:, i] = y_predicted

        # Evaluate predictions of single classifiers
        single_acc = accuracy_score(
            y_test, predicted_labels_by_single_classifier)
        single_prec = precision_score(
            y_test, predicted_labels_by_single_classifier, average='macro')
        single_recall = recall_score(
            y_test, predicted_labels_by_single_classifier, average='macro')
        single_hl = hamming_loss(y_test, predicted_labels_by_single_classifier)

        results2 = {"subset_accuracy": single_acc, "precision_score": single_prec,
                    "recall_score": single_recall, "hamming_loss": single_hl}

        # printouts commented out for faster execution and cleaner log, results are stored in chain_results and single_results lists in the returned dictionary
        # print(
        #     f"\n(Chain)  subset_accuracy: {results['subset_accuracy']}, hamming_loss: {results['hamming_loss']}")

        # print(
        #     f"(Single) subset_accuracy: {results2['subset_accuracy']}, hamming_loss: {results2['hamming_loss']}")

        chain_accs.append(results['subset_accuracy'])
        chain_precs.append(results['precision_score'])
        chain_recalls.append(results['recall_score'])
        chain_hls.append(results['hamming_loss'])
        single_accs.append(results2['subset_accuracy'])
        single_precs.append(results2['precision_score'])
        single_recalls.append(results2['recall_score'])
        single_hls.append(results2['hamming_loss'])

    chain_results = {"accuracies": chain_accs, "precisions": chain_precs,
                     "recalls": chain_recalls, "hamming_losses": chain_hls}
    single_results = {"accuracies": single_accs, "precisions": single_precs,
                      "recalls": single_recalls, "hamming_losses": single_hls}
    return {"chain_results": chain_results, "single_results": single_results}
