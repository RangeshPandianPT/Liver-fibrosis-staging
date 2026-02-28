import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

def calculate_metrics():
    # Load predictions
    df = pd.read_csv('d:/ALS/outputs/deit_predictions.csv')
    
    # Get true labels
    y_true = df['true_label']
    
    # Get predicted labels by finding column with max probability
    prob_cols = [c for c in df.columns if c.startswith('deit_') and c.endswith('_prob')]
    # Class mapping from F0-F4 based on column index 0-4
    class_mapping = {0: 'F0', 1: 'F1', 2: 'F2', 3: 'F3', 4: 'F4'}
    
    y_pred_idx = np.argmax(df[prob_cols].values, axis=1)
    y_pred = [class_mapping[i] for i in y_pred_idx]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    calculate_metrics()
