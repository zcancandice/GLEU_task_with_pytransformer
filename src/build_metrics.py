from sklearn import metrics
from transformers import EvalPrediction
import numpy as np


def compute_classify_metrics(true_label, pred_label):
    accuracy = metrics.accuracy_score(true_label, pred_label)
    class_report = metrics.classification_report(
        true_label, pred_label, output_dict=True)
    result = {'accuracy': accuracy}
    result.update(class_report)
    return result


def build_compute_classify_metrics_fn():
    def compute_metrics_fn(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return compute_classify_metrics(p.label_ids, preds)
    return compute_metrics_fn


if __name__ == "__main__":
    x = [1, 1, 1, 0, 0, 0]
    y = [1, 0, 1, 0, 0, 0]
    m = compute_classify_metrics(x, y)
    print(m)