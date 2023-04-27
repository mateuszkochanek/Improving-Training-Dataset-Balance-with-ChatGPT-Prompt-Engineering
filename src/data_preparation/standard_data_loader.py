import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset
from typing import Tuple, List

from utils.helper_functions import prepare_train_data


class StandardDataLoaderConstructor:

    def __init__(self, replace_csv):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.replace_csv = replace_csv
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = []
        attention_masks = []

        for text in data['text']:
            encoded_text = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_attention_mask=True,
                truncation=True,
                return_tensors='pt'
            )

            input_ids.append(encoded_text['input_ids'])
            attention_masks.append(encoded_text['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(data['label'].values)

        return input_ids, attention_masks, labels

    def construct_dataloaders(self, fraction_negative: float = 1.0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data = prepare_train_data(fraction_negative, replace_csv=self.replace_csv)
        train_df, val_df = train_test_split(train_data, stratify=train_data['label'], test_size=0.1, random_state=666)

        train_input_ids, train_attention_masks, train_labels = self.preprocess_data(train_df)
        val_input_ids, val_attention_masks, val_labels = self.preprocess_data(val_df)

        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

        test_dataloader = self.construct_test_dataloader()

        return train_dataloader, val_dataloader, test_dataloader

    def construct_kfold_dataloaders(self, fraction_negative: float = 1.0, k_folds: int = 5) -> List[Tuple[DataLoader, DataLoader]]:
        train_data = prepare_train_data(fraction_negative, replace_csv=self.replace_csv)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=666)
        fold_dataloaders = []

        for train_index, val_index in tqdm(skf.split(train_data, train_data['label']), desc="Create fold dataloaders:"):
            train_df, val_df = train_data.iloc[train_index], train_data.iloc[val_index]

            train_input_ids, train_attention_masks, train_labels = self.preprocess_data(train_df)
            val_input_ids, val_attention_masks, val_labels = self.preprocess_data(val_df)

            train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

            val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
            val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

            fold_dataloaders.append((train_dataloader, val_dataloader))

        return fold_dataloaders

    def construct_test_dataloader(self) -> DataLoader:
        dataset_test = load_dataset("imdb", split="test")
        test_df = pd.DataFrame(dataset_test)

        test_input_ids, test_attention_masks, test_labels = self.preprocess_data(test_df)

        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

        return test_dataloader
