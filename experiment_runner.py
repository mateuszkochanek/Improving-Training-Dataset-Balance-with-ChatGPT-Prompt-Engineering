import sys
import os
import pandas as pd

sys.path.append('src')

from data_preparation.standard_data_loader import StandardDataLoaderConstructor
from experiments.standard_BERT_sentiment import StandardBERTExperiment
from experiments.standard_sklearn_sentiment import StandardBERTExperiment
from utils.helper_functions import check_available_device


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

def run_NaiveBayes_experiment(experiment_name, fraction_negative=1.0, k_folds=5, replace_csv=None):
    train_data = prepare_train_data(fraction_negative, replace_csv='./data/negative_reviews.csv')
    dataset_test = load_dataset("imdb", split="test")
    test_data = pd.DataFrame(dataset_test)

    nb_model = MultinomialNB()
    nb_tester = ModelTester(nb_model, train_data, test_data)
    nb_pipeline = nb_tester.train

if __name__ == '__main__':
    DEVICE = check_available_device()
    run_BERT_experiment("standard_BERT_fulldataset")  # standard BERT experiment on full dataset
    run_BERT_experiment("standard_BERT_unbalanceddataset", fraction_negative=0.05)  # standard BERT experiment on unbalanced dataset
    run_BERT_experiment("synthetic_BERT_unbalanceddataset", fraction_negative=0.05, replace_csv="./data/negative_review")  # standard BERT experiment on dataset with synthetic data

    # loader_constructor = StandardDataLoaderConstructor()
    # train_dataloader, val_dataloader, test_dataloader = loader_constructor.construct_dataloaders(fraction_negative=0.1)
    # experiment = StandardBERTExperiment(train_dataloader, val_dataloader, test_dataloader, "standard_BERT", lr=2e-5)
    # experiment.run_tests(epochs=1)
