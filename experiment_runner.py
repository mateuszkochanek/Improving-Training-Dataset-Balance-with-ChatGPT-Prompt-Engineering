import sys
sys.path.append('src')

from data_preparation.standard_data_loader import StandardDataLoaderConstructor
from experiments.standard_BERT_sentiment import StandardBERTExperiment

if __name__ == '__main__':
    loader_constructor = StandardDataLoaderConstructor()
    train_dataloader, val_dataloader, test_dataloader = loader_constructor.construct_dataloaders(fraction_negative=0.1)
    experiment = StandardBERTExperiment(train_dataloader, val_dataloader, test_dataloader, "standard_BERT", lr=2e-5)
    experiment.run_tests(epochs=1)
