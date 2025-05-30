from datasets import load_youtube_dataset, load_bibtex_dataset
from models.bayesian_classifier_chain import BayesianClassifierChain
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

randomnessSeed = 0

testClassifier = XGBClassifier(
    objective="binary:logistic", random_state=randomnessSeed)

# this could be useful when comparing bayesian chain of xgboost to just running xgboost in multilabel mode
# testClassifier = XGBClassifier(
#     objective="multi:softprob", random_state=randomnessSeed)
testChain = BayesianClassifierChain(classifier=testClassifier)

loaded_dataset = load_youtube_dataset()
X = loaded_dataset.data
Y = loaded_dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.8, random_state=0)

testChain.fit(X_train, y_train)
results = testChain.evaluate(X_test, y_test)

print(
    f"subset_accuracy: {results['subset_accuracy']}, hamming_loss: {results['hamming_loss']}")
