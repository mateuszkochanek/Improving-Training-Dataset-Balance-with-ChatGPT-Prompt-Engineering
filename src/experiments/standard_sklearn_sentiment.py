import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


class ModelTester:
    def __init__(self, experiment_name, model, train_data: pd.DataFrame, test_data: pd.DataFrame, n_splits: int = 5):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.n_splits = n_splits

        self.experiment_name = experiment_name

    def train_and_evaluate(self):
        X_train = self.train_data['text']
        y_train = self.train_data['label']
        X_test = self.test_data['text']
        y_test = self.test_data['label']

        # Create a pipeline with TF-IDF and the chosen model
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', self.model),
        ])

        # Initialize the cross-validation splitter
        skf = StratifiedKFold(n_splits=self.n_splits)

        # Initialize lists to store metrics for each fold
        metrics = []

        # Cross-validation loop
        current_fold = 0
        for train_index, valid_index in skf.split(X_train, y_train):
            X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

            # Train the model on the current fold
            pipeline.fit(X_train_fold, y_train_fold)

            # Predict labels for the validation set
            y_pred = pipeline.predict(X_valid_fold)

            # Calculate and store the metrics
            metrics.append(
                {"experiment_name": self.experiment_name,
                 "fold": current_fold,
                 "step": "valid",
                 'epoch': None,
                 'loss': None,
                 'accuracy': accuracy_score(y_valid_fold, y_pred),
                 'precision': precision_score(y_valid_fold, y_pred),
                 'recall': recall_score(y_valid_fold, y_pred),
                 'f1': f1_score(y_valid_fold, y_pred)}
            )
            y_pred_test = pipeline.predict(X_test)
            metrics.append(
                {"experiment_name": self.experiment_name,
                 "fold": current_fold,
                 "step": "test",
                 'epoch': None,
                 'loss': None,
                 'accuracy': accuracy_score(y_pred_test, y_test),
                 'precision': precision_score(y_pred_test, y_test),
                 'recall': recall_score(y_pred_test, y_test),
                 'f1': f1_score(y_pred_test, y_test)}
            )
            current_fold += 1

        # Train the model on the full training set and evaluate on the test set
        pipeline.fit(X_train, y_train)
        
        return metrics
