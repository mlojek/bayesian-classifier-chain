from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from datasets import load_bibtex_dataset, load_youtube_dataset
from models.bayesian_classifier_chain import BayesianClassifierChain
from experiments_comparison_helpers import compare_series_results, perform_train_evaluate_chain_vs_single

# Config
# jeśli 'True': Przeprowadź eksperymenty służące znalezieniu najlepszej kolejności etykietowania
# jeśli 'False': Wykonaj jednorazowy trening (na najlepszej kolejności etykietowania znalezionej w toku eksperymentów)
perform_best_label_order_search = False
# przeprowadzone 25 iteracji, w wersji "szybkiej" notatnika ustawione na 2
EXPERIMENT_ITERATIONS_COUNT = 1

#!python src/experiments-xgboost.py
if not perform_best_label_order_search:
    classifier_random_seed = 0
    data_split_random_seed = 0
    test_set_size = 0.1

    testClassifier = XGBClassifier(
        objective="binary:logistic", random_state=classifier_random_seed)

    # TODO: replace with best found label order
    best_found_label_order = None
    bestChainXGBoostBT = BayesianClassifierChain(
        classifier=testClassifier, custom_label_order=best_found_label_order)

    # testClassifier = GaussianNB()
    # bestChainXGBoostBT = BayesianClassifierChain(classifier=testClassifier)

    loaded_dataset = load_youtube_dataset()
    X = loaded_dataset.data
    Y = loaded_dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_set_size, random_state=data_split_random_seed)

    bestChainXGBoostBT.fit(X_train, y_train)
    results = bestChainXGBoostBT.evaluate(X_test, y_test)

    print(
        f"subset_accuracy: {results['subset_accuracy']}, hamming_loss: {results['hamming_loss']}")

# perform comparison experiments
classifier_chain = bestChainXGBoostBT
single_classifier = XGBClassifier(
    objective="binary:logistic", random_state=classifier_random_seed)
# single_classifier = GaussianNB()

iterations_count = EXPERIMENT_ITERATIONS_COUNT
load_dataset_function = load_youtube_dataset
test_set_size = 0.1
series_results_xgb_bt = perform_train_evaluate_chain_vs_single(
    classifier_chain=classifier_chain, single_classifier=single_classifier, iterations_count=iterations_count, load_dataset_function=load_dataset_function, test_set_size=test_set_size)

# analyze results
compare_series_results(series_results=series_results_xgb_bt)
