##############################################################################
# IMPORTS
##############################################################################

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report


##############################################################################
# FUNCTIONS
##############################################################################

def predictions(ls):
    res = []
    for element in ls:
        if element >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return res


def export_metrics(metrics_binary, data_dir):
    df = pd.DataFrame()
    df['precision_0'] = metrics_binary.get('precision', 0)
    df['recall_0'] = metrics_binary.get('recall', 0)
    df['f1_0'] = metrics_binary.get('f1-score', 0)
    df['support_0'] = metrics_binary.get('support', 0)

    df['precision_1'] = metrics_binary.get('precision', 1)
    df['recall_1'] = metrics_binary.get('recall', 1)
    df['f1_1'] = metrics_binary.get('f1-score', 1)
    df['support_1'] = metrics_binary.get('support', 1)

    df.to_csv(data_dir + 'metrics.csv')
    return


def plot_metrics(metrics, x, plot_dir, class_number):
    plt.figure()
    plt.plot(x, metrics.get('recall', class_number), label='Class ' + str(class_number) + ' recall')

    plt.plot(x, metrics.get('precision', class_number), label='Class ' + str(class_number) + ' precision')

    plt.plot(x, metrics.get('f1-score', class_number), label='Class ' + str(class_number) + ' f1-score')

    plt.xticks(x)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.grid(True)
    plt.title("Recall, Precision and F1-Score on \n pneumonia detection (validation)")
    plt.xlabel("Epoch #")
    plt.legend(loc='best')
    plt.savefig(plot_dir + "metrics_" + str(class_number) + ".png")
    return


def plot_metrics_2(recall, precision, f1, x, plot_dir, class_number):
    plt.figure()
    plt.plot(x, recall, label='Class ' + str(class_number) + ' recall')
    plt.plot(x, precision, label='Class ' + str(class_number) + ' precision')
    plt.plot(x, f1, label='Class ' + str(class_number) + ' f1-score')

    plt.xticks(x)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.grid(True)
    plt.title("Recall, Precision and F1-Score on \n pneumonia detection (validation)")
    plt.xlabel("Epoch #")
    plt.legend(loc='best')
    plt.savefig(plot_dir + "metrics_" + str(class_number) + ".png")
    return


##############################################################################
# CLASS
##############################################################################

class PrecisionRecallF1scoreMetrics(Callback):
    def __init__(self, generator, model):
        self.TARGET_NAMES = ['Normal', 'Pneumonia']
        self.generator = generator
        self.model = model
        self.reports = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.generator)
        y_pred = predictions(y_pred)
        report = classification_report(self.generator.classes, y_pred,
                                       output_dict=True)
        self.reports.append(report)
        return

    def get(self, metrics, of_class):
        return [report[str(of_class)][metrics] for report in self.reports]
