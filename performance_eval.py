from sklearn.metrics import precision_recall_fscore_support
from Data_processor import Data_processor

def evaluate_performance(gold_path, predicted_path):
    gold_data = Data_processor(gold_path).data
    predicted_data = Data_processor(predicted_path).data

    gold_sequences = [item.split()[1] for sublist in gold_data for item in sublist]
    predicted_sequences = [item.split()[1] for sublist in predicted_data for item in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(gold_sequences, predicted_sequences, average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

evaluate_performance('train/test.txt', 'Output/output.out')
