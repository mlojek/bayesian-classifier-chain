"""
Bayesian Classifier Chain multi-label classifier.
"""

from typing import Dict, List

import numpy as np
from sklearn.base import ClassifierMixin as SklearnClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, hamming_loss, recall_score, precision_score


class BayesianClassifierChain:
    """
    Bayesian Classifier Chain for multi-label classification.
    It uses previous labels to predict subsequent labels.
    """

    def __init__(self, classifier: SklearnClassifier):
        """
        Initializes the classifier chain.

        Args:
            classifier (SklearnClassifier): A scikit-learn classifier used in the chain.
        """
        self.base_classifier: SklearnClassifier = classifier
        self.classifiers: List[SklearnClassifier] = []

        self.label_order: List[int] = []
        self.n_labels: int = 0

    def fit(self, features: List[List[float]], labels: List[List[int]]) -> None:
        """
        Trains the classifier chain.

        Args:
            features (List[List[float]]): Dataset features.
            labels (List[List[float]]): Dataset labels.
        """
        features = features.toarray()
        labels = np.array(labels)

        self.n_labels = labels.shape[1]
        self.label_order = list(range(self.n_labels))
        self.classifiers = []

        extended_features = features.copy()

        for i in self.label_order:
            current_classifier = clone(self.base_classifier)

            current_classifier.fit(extended_features, labels[:, i])
            self.classifiers.append(current_classifier)

            extended_features = np.hstack([extended_features, labels[:, [i]]])

    def predict(self, features: List[List[float]]) -> List[List[int]]:
        """
        Predicts labels for input data.

        Args:
            features (List[List[float]]): Dataset features.

        Returns:
            List[List[int]]: Predicted dataset labels.
        """
        features = features.toarray()
        predicted_labels = np.zeros((features.shape[0], self.n_labels))

        features_extended = features.copy()

        for i, current_classifier in enumerate(self.classifiers):
            y_predicted = current_classifier.predict(features_extended)
            predicted_labels[:, i] = y_predicted
            features_extended = np.hstack(
                [features_extended, y_predicted.reshape(-1, 1)]
            )

        return predicted_labels

    def evaluate(
        self, features: List[List[float]], labels: List[List[int]]
    ) -> Dict[str, float]:
        """
        Evaluates the model using subset accuracy and hamming loss.

        Args:
            features (List[List[float]]): Dataset features.
            labels (List[List[int]]): Dataset labels.

        Returns:
            Dict[str, float]: Dictionary with metrics.
        """
        predicted_labels = self.predict(features)

        acc = accuracy_score(labels, predicted_labels)
        prec = precision_score(labels, predicted_labels)
        rec = recall_score(labels, predicted_labels)
        hl = hamming_loss(labels, predicted_labels)

        return {"subset_accuracy": acc, "precision_score": prec, "recall_score": rec, "hamming_loss": hl}
