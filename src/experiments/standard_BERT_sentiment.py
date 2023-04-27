import os
import torch
import pickle
import pandas as  dataset_test = load_dataset("imdb", split="test")
        test_df = pd.DataFrame(dataset_test)pd
from tqdm import tqdm
import torch.optim as optim
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Any, Dict, List


class StandardBERTExperiment:

    def __init__(self,
                 experiment_name: str,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 device: Any,
                 lr: float = 2e-5):

        self.device = device
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.experiment_name = experiment_name

    def _common_step(self,
                     dataloader: DataLoader,
                     desc: str,
                     is_train: bool = False) -> Tuple[float, float, float, float, float]:
        all_preds = []
        all_labels = []
        running_loss = 0

        for batch in tqdm(dataloader, desc=desc):
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            batch_loss = self.loss_fn(logits, labels)
            running_loss += batch_loss.item()

            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            if is_train:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        epoch_loss = running_loss / len(self.val_dataloader)
        return epoch_loss, accuracy, precision, recall, f1

    def run_train_test(self, epochs: int = 5, current_fold: int = None) -> List[Dict]:
        metrics = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss, accuracy, precision, recall, f1 = self._common_step(self.train_dataloader, "Training",
                                                                            is_train=True)
            print(
                f'Training - Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
            metrics.append(
                {"experiment_name": self.experiment_name, "fold": current_fold, "step": "train", 'epoch': epoch,
                 'loss': epoch_loss,
                 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

            self.model.eval()
            with torch.no_grad():
                epoch_loss, accuracy, precision, recall, f1 = self._common_step(self.val_dataloader, "Validation")
                print(
                    f'Validation - Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
                metrics.append(
                    {"experiment_name": self.experiment_name, "fold": current_fold, "step": "validate", 'epoch': epoch,
                     'loss': epoch_loss,
                     'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

        self.model.eval()
        with torch.no_grad():
            test_loss, test_accuracy, test_precision, test_recall, test_f1 = self._common_step(self.test_dataloader,
                                                                                               "Testing")
            print(
                f'Testing - Loss: {test_loss}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}')
            metrics.append(
                {"experiment_name": self.experiment_name, "fold": current_fold, "step": "test", 'epoch': None,
                 'loss': test_loss,
                 'accuracy': test_accuracy, 'precision': test_precision, 'recall': test_recall,
                 'f1': test_f1})
        return metrics
