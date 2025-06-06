from datasets import load_youtube_dataset, load_bibtex_dataset
from models.bayesian_classifier_chain import BayesianClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np

data_split_random_seed = 0
test_set_size = 0.1

loaded_dataset = load_youtube_dataset()
X = loaded_dataset.data
Y = loaded_dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=test_set_size, random_state=data_split_random_seed)

custom_label_order = list(range((np.array(y_train)).shape[1]))
temp = custom_label_order[3]
custom_label_order[3] = custom_label_order[10]
custom_label_order[10] = temp
# custom_label_order.reverse()

testClassifier = GaussianNB()
testChain = BayesianClassifierChain(
    classifier=testClassifier, custom_label_order=custom_label_order)

testChain.fit(X_train, y_train)
results = testChain.evaluate(X_test, y_test)

print(
    f"subset_accuracy: {results['subset_accuracy']}, hamming_loss: {results['hamming_loss']}")
print(
    f"precision_score: {results['precision_score']}, recall_score: {results['recall_score']}")
