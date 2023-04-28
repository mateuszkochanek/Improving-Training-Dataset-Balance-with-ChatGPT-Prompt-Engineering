import sys
import os
import pandas as pd
from datasets import load_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

sys.path.append('src')

from data_preparation.standard_data_loader import StandardDataLoaderConstructor
from experiments.standard_BERT_sentiment import StandardBERTExperiment
from experiments.standard_sklearn_sentiment import ModelTester
from utils.helper_functions import check_available_device, prepare_train_data


def run_BERT_experiment(experiment_name, fraction_negative=1.0, k_folds=5, replace_csv=None):
    metrics = []
    loader_constructor = StandardDataLoaderConstructor(replace_csv)
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


if __name__ == '__main__':
    DEVICE = check_available_device()

    # standard BERT experiment on full dataset
    #run_BERT_experiment("standard_BERT_fulldataset")
    # standard BERT experiment on unbalanced dataset
    #run_BERT_experiment("standard_BERT_unbalanceddataset", fraction_negative=0.05)
    # BERT experiment on dataset with synthetic data
    #run_BERT_experiment("synthetic_BERT_unbalanceddataset", fraction_negative=0.05, replace_csv="./data/negative_review")

    # standard NaiveBayes experiment on full dataset
    run_sklearn_experiment("standard_NaiveBayes_fulldataset", MultinomialNB(), fraction_negative=1.0, k_folds=5, replace_csv=None)
    # standard NaiveBayes experiment on unbalanced dataset
    run_sklearn_experiment("standard_NaiveBayes_unbalanceddataset", MultinomialNB(), fraction_negative=0.05, k_folds=5)
    # NaiveBayes experiment on dataset with synthetic data
    run_sklearn_experiment("synthetic_NaiveBayes_unbalanceddataset", MultinomialNB(), fraction_negative=0.05, k_folds=5, replace_csv="./data/negative_reviews.csv")

    # standard RandomForestClassifier experiment on full dataset
    run_sklearn_experiment("standard_RandomForest_fulldataset", RandomForestClassifier(), fraction_negative=1.0, k_folds=5, replace_csv=None)
    # standard RandomForestClassifier experiment on unbalanced dataset
    run_sklearn_experiment("standard_RandomForest_unbalanceddataset", RandomForestClassifier(), fraction_negative=0.05, k_folds=5)
    # RandomForestClassifier experiment on dataset with synthetic data
    run_sklearn_experiment("synthetic_RandomForest_unbalanceddataset", RandomForestClassifier(), fraction_negative=0.05, k_folds=5, replace_csv="./data/negative_reviews.csv")
