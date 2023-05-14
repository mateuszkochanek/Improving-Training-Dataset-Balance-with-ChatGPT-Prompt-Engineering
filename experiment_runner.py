import sys
import os
import pandas as pd
from datasets import load_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

sys.path.append('src')

from data_preparation.bert_data_loader import BERTDataLoaderConstructor
from experiments.standard_BERT_sentiment import StandardBERTExperiment
from experiments.standard_sklearn_sentiment import ModelTester
from utils.helper_functions import check_available_device, prepare_train_data


def run_BERT_experiment(experiment_name, fraction_negative=1.0, k_folds=5, replace_csv=None):
    metrics = []
    loader_constructor = BERTDataLoaderConstructor(replace_csv)
    fold_dataloaders = loader_constructor.construct_kfold_dataloaders(fraction_negative=fraction_negative,
                                                                      k_folds=k_folds)
    test_dataloader = loader_constructor.construct_test_dataloader()
    for current_fold, (train_dataloader, val_dataloader) in enumerate(fold_dataloaders):
        print(f"________________________ Fold {current_fold} ________________________")
        experiment_runner = StandardBERTExperiment(
            experiment_name, train_dataloader, val_dataloader, test_dataloader, DEVICE, lr=2e-5
        )
        metrics.extend(
            experiment_runner.run_train_test(epochs=4, current_fold=current_fold)
        )
        print(f"________________________________________________________")
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join('results', experiment_name + '_metrics.csv'), index=False, sep=";")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join('results', experiment_name + '_metrics.csv'), index=False, sep=";")


def run_sklearn_experiment(experiment_name, model, fraction_negative=1.0, k_folds=5, replace_csv=None):
    train_data = prepare_train_data(fraction_negative, replace_csv=replace_csv)
    dataset_test = load_dataset("imdb", split="test")
    test_data = pd.DataFrame(dataset_test)

    tester = ModelTester(experiment_name, model, train_data, test_data, n_splits=k_folds)
    metrics = tester.train_and_evaluate()

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join('results', experiment_name + '_metrics.csv'), index=False, sep=";")


def run_experiments(model_type, experiment_prefix, fraction_negative=1.0, k_folds=5, replace_csv=None):
    print("############################################################")
    print("Experiment ", experiment_prefix, "_", model_type, " is running")
    print("############################################################")
    if model_type == "BERT":
        run_BERT_experiment(f"{experiment_prefix}_BERT", fraction_negative, k_folds, replace_csv)
    elif model_type == "NaiveBayes":
        run_sklearn_experiment(f"{experiment_prefix}_NaiveBayes", MultinomialNB(), fraction_negative, k_folds,
                               replace_csv)
    elif model_type == "RandomForest":
        run_sklearn_experiment(f"{experiment_prefix}_RandomForest", RandomForestClassifier(), fraction_negative,
                               k_folds, replace_csv)
    else:
        print("Invalid model type")


if __name__ == '__main__':
    DEVICE = check_available_device()
    model_types = ["BERT", "NaiveBayes", "RandomForest"]
    experiment_settings = [
        ("full", 1.0, None),
        ("unbalanced_03", 0.3, None),
        ("unbalanced_02", 0.2, None),
        ("unbalanced_01", 0.1, None),
        ("composite_03", 0.3, "./data/negative_reviews.csv"),
        ("composite_02", 0.2, "./data/negative_reviews.csv"),
        ("composite_01", 0.1, "./data/negative_reviews.csv"),
        ("basic_03", 0.3, "./data/basic_negative_reviews.csv"),
        ("basic_02", 0.2, "./data/basic_negative_reviews.csv"),
        ("basic_01", 0.1, "./data/basic_negative_reviews.csv"),
        ("similar_02", 0.2, "./data/similar_negative_reviews_02.csv"),
        ("similar_01", 0.1, "./data/similar_negative_reviews_01.csv"),
        ("similar_02", 0.2, "./data/similar_negative_reviews_02.csv"),
        ("similar_01", 0.1, "./data/similar_negative_reviews_01.csv"),
    ]

    for model_type in model_types:
        for experiment_suffix, fraction_negative, replace_csv in experiment_settings:
            run_experiments(model_type, experiment_suffix, fraction_negative, k_folds=5, replace_csv=replace_csv)
