import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def display_confustion_matrix_plot(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    model.train()

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,
                annot=True,
                cbar=False,
                fmt='d',
                square=True,
                vmin=0,
                vmax=0.7*np.max(cm),
                xticklabels=labels,
                yticklabels=labels
    )
    plt.title("Sign Language Classification Heat Map")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
