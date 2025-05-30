from datasets import load_youtube_dataset, load_bibtex_dataset
from models.bayesian_classifier_chain import BayesianClassifierChain
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

classifier_random_seed = 0
data_split_random_seed = 0
test_set_size = 0.1

testClassifier = XGBClassifier(
    objective="binary:logistic", random_state=classifier_random_seed)

# this could be useful when comparing bayesian chain of xgboost to just running xgboost in multilabel mode
# testClassifier = XGBClassifier(
#     objective="multi:softprob", random_state=randomnessSeed)
testChain = BayesianClassifierChain(classifier=testClassifier)

loaded_dataset = load_youtube_dataset()
X = loaded_dataset.data
Y = loaded_dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=test_set_size, random_state=data_split_random_seed)

testChain.fit(X_train, y_train)
results = testChain.evaluate(X_test, y_test)

print(
    f"subset_accuracy: {results['subset_accuracy']}, hamming_loss: {results['hamming_loss']}")
