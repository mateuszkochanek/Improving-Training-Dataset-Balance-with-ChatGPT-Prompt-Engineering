import torch
import pickle
from tqdm import tqdm
import torch.optim as optim
from transformers import BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


class SentimentAnalyzer:

    def __init__(self, train_dataloader, val_dataloader, test_dataloader, experiment_name, lr=2e-5):

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = self.check_available_device()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.experiment_name = experiment_name

    def check_available_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")
        return device

    def _common_step(self, dataloader, desc, is_train=False):
        all_preds = []
        all_labels = []
        running_loss = 0

        for batch in tqdm(dataloader, desc=desc):
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_masks)
            loss = outputs[0]
            running_loss += loss.item()
            _, preds = torch.max(outputs[0], dim=1)

            if is_train:
                loss.backward()
                self.optimizer.step()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        epoch_loss = running_loss / len(self.val_dataloader)
        return epoch_loss, accuracy, precision, recall, f1

    def run_tests(self, epochs=5):
        metrics = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss, accuracy, precision, recall, f1 = _common_step(self.train_dataloader, "Training", is_train=True)
            print(
                f'Training - Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
            metrics.append(
                {"experiment_name": self.experiment_name, "step": "train", 'epoch': epoch, 'loss': epoch_loss,
                 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
            )

            self.model.eval()
            with torch.no_grad():
                epoch_loss, accuracy, precision, recall, f1 = _common_step(self.train_dataloader, "Validation")
                print(
                    f'Validation - Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
                metrics.append(
                    {"experiment_name": self.experiment_name, "step": "validate", 'epoch': epoch, 'loss': epoch_loss,
                     'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

        self.model.eval()
        with torch.no_grad():
            test_loss, accuracy, precision, recall, f1 = _common_step(self.train_dataloader, "Testing")
            print(
                f'Testing - Loss: {test_loss}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}')
            metrics.append({"experiment_name": self.experiment_name, "step": "test", 'epoch': None, 'loss': test_loss,
                            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1})

    self.plot_metrics(val_metrics)

    with open('test_metrics.pkl', 'wb') as f:
        pickle.dump({'accuracy': test_accuracy, 'precision': test_precision, 'recall': test_recall, 'f1': test_f1}, f)

    with open('train_metrics.pkl', 'wb') as f:
        pickle.dump(train_metrics, f)

    with open('val_metrics.pkl', 'wb') as f:
        pickle.dump(val_metrics, f)


def plot_metrics(self, val_metrics):
    epochs = [metric['epoch'] for metric in val_metrics]
    accuracy = [metric['accuracy'] for metric in val_metrics]

    plt.figure()
    sns.lineplot(x=epochs, y=accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.savefig("accuracy_per_epoch.png")

    plt.show()


if name == 'main':
    sentiment_analyzer = SentimentAnalyzer(train_dataloader, val_dataloader, test_dataloader)
    sentiment_analyzer.run_tests(epochs=5)
