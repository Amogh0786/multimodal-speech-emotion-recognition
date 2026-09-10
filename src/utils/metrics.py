from sklearn.metrics import f1_score, recall_score, confusion_matrix
import numpy as np

def compute_metrics(preds, labels):
    """Computes Macro-F1 and UAR (Unweighted Average Recall)."""
    macro_f1 = f1_score(labels, preds, average='macro')
    uar = recall_score(labels, preds, average='macro') # Unweighted Average Recall is macro recall
    cm = confusion_matrix(labels, preds)
    return {
        "macro_f1": macro_f1,
        "uar": uar,
        "confusion_matrix": cm.tolist()
    }
