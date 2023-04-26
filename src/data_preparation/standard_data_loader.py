import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset


class StandardDataLoaderConstructor:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess_data(self, data):
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

    def construct_dataloaders(self, fraction_negative=1.0):
        dataset_train = load_dataset("imdb", split="train")
        dataset_test = load_dataset("imdb", split="test")

        train_data = pd.DataFrame(dataset_train)
        test_df = pd.DataFrame(dataset_test)

        if 0 < fraction_negative < 1:
            negative_indices = train_data[train_data['label'] == 0].index
            negative_sample = np.random.choice(negative_indices, int(len(negative_indices) * fraction_negative),
                                               replace=False)
            train_data = pd.concat([train_data[train_data['label'] == 1], train_data.loc[negative_sample]])

        train_df, val_df = train_test_split(train_data, stratify=train_data['label'], test_size=0.1, random_state=666)

        train_input_ids, train_attention_masks, train_labels = self.preprocess_data(train_df)
        val_input_ids, val_attention_masks, val_labels = self.preprocess_data(val_df)
        test_input_ids, test_attention_masks, test_labels = self.preprocess_data(test_df)

        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    loader_constructor = StandardDataLoaderConstructor()
    train_dataloader, val_dataloader, test_dataloader = loader_constructor.construct_dataloaders(fraction_negative=0.1)
