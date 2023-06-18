import torch
import pandas as pd
import numpy as np
from datasets import load_dataset


def check_available_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def prepare_sampled_train_data(fraction_negative: float, replace_csv: str = None) -> pd.DataFrame:
    dataset_train = load_dataset("imdb", split="train")
    train_data = pd.DataFrame(dataset_train)

    if 0 < fraction_negative < 1:
        negative_indices = train_data[train_data['label'] == 0].index
        negative_sample = np.random.choice(negative_indices, int(len(negative_indices) * fraction_negative),
                                           replace=False)
        train_data = pd.concat([train_data[train_data['label'] == 1], train_data.loc[negative_sample]])

        if replace_csv is not None:
            csv_data = pd.read_csv(replace_csv, sep=';', index_col=None)

            num_samples_to_replace = int(len(negative_indices) * (1 - fraction_negative))

            csv_data = csv_data.sample(frac=1).reset_index(drop=True)
            csv_sample = csv_data.iloc[:num_samples_to_replace]

            train_data = pd.concat([train_data, csv_sample])

    return train_data


def prepare_train_data(fraction_negative: float, replace_csv: str = None, original_samples_size: int = None) -> pd.DataFrame:
    dataset_train = load_dataset("imdb", split="train")
    train_data = pd.DataFrame(dataset_train)

    if original_samples_size is not None:
        positive_indices = train_data[train_data['label'] == 1].index
        negative_indices = train_data[train_data['label'] == 0].index

        positive_sample = np.random.choice(positive_indices, original_samples_size, replace=False)
        negative_sample = np.random.choice(negative_indices, original_samples_size, replace=False)

        train_data = pd.concat([train_data.loc[positive_sample], train_data.loc[negative_sample]], ignore_index=True)

    if 0 < fraction_negative < 1:
        negative_indices = train_data[train_data['label'] == 0].index
        negative_sample = negative_indices[:int(len(negative_indices) * fraction_negative)]
        train_data = pd.concat([train_data[train_data['label'] == 1], train_data.loc[negative_sample]])

        if replace_csv is not None:
            csv_data = pd.read_csv(replace_csv, sep=';', index_col=None)

            num_samples_to_replace = int(len(negative_indices) * (1 - fraction_negative))

            csv_data = csv_data.sample(frac=1).reset_index(drop=True)
            csv_sample = csv_data.iloc[:num_samples_to_replace]

            train_data = pd.concat([train_data, csv_sample])

    return train_data
