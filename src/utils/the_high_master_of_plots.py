import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


class TheHighMasterOfPlots:
    def __init__(self, folder: str):
        self.folder = folder
        self.save_folder = os.path.join("results", "plots")

    def gather_metrics(self) -> pd.DataFrame:
        # Read all CSV files in the folder and store them in a list
        metrics_files = glob.glob(os.path.join(self.folder, "*_metrics.csv"))
        metrics_list = []

        for file in metrics_files:
            df = pd.read_csv(file, sep=';')
            metrics_list.append(df)

        # Concatenate all the dataframes in the list
        metrics_data = pd.concat(metrics_list, axis=0, ignore_index=True)
        print(metrics_data)
        return metrics_data

    def plot_mean_accuracy(self, metrics_data: pd.DataFrame, file_name: str, model_name: str = None):
        # Filter the data to only include the "test" step
        test_data = metrics_data[metrics_data["step"] == "test"]
        
        if model_name is not None:
            test_data = test_data[test_data["experiment_name"].str.contains(model_name)]
    
        # Calculate mean accuracy for each experiment_name
        mean_accuracy = test_data.groupby("experiment_name")["accuracy"].mean().reset_index()
    
        # Create the plot
        plt.figure(figsize=(10, 6))
        bar_plot = sns.barplot(x="experiment_name", y="accuracy", data=mean_accuracy)
        plt.xlabel("Experiment Name")
        plt.ylabel("Mean Accuracy")
        plt.title("Mean Accuracy for Test Step")
        plt.xticks(rotation=45)
    
        # Add numbers above each bar
        for i, bar in enumerate(bar_plot.patches):
            bar_plot.annotate(
                f'{bar.get_height():.3f}',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom',
                fontsize=10, color='black'
            )
    
        # Save the plot to a file
        plt.savefig(os.path.join(self.save_folder, file_name), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Instantiate the MasterPlotter class
    plotter = TheHighMasterOfPlots("results")

    # Gather metrics from the CSV files
    metrics_data = plotter.gather_metrics()

    # Create the mean accuracy plot for the "test" step and save it to a file
    plotter.plot_mean_accuracy(metrics_data, file_name="mean_accuracy_test_step.png")
    plotter.plot_mean_accuracy(metrics_data, model_name="NaiveBayes", file_name="mean_accuracy_test_step_NaiveBayes.png")
    plotter.plot_mean_accuracy(metrics_data, model_name="RandomForest", file_name="mean_accuracy_test_step_RandomForest.png")
    plotter.plot_mean_accuracy(metrics_data, model_name="BERT", file_name="mean_accuracy_test_step_BERT.png")