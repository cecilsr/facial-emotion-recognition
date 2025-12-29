"""Plotting utilities for evaluation and hyperparameter analysis."""


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def betaplotter(best_params, learning_rates, beta1_values, beta2_values, accuracy_grid, outfile_name):
    """
    Plot accuracy heatmaps over Adam beta parameters.

    Generates a colormap of validation accuracy for the slice of the
    grid search accuracies corresponding to the best learning rate and saves
    the figure to disk.

    Parameters
    ----------
    best_params : dict
        Dictionary containing the best hyperparameters, with best learning rate at key `"learning_rate"`.
    learning_rates : array-like
        Array of learning rates used in the grid search.
    beta1_values : array-like
        Values of Adam beta1 used in the grid search.
    beta2_values : array-like
        Values of Adam beta2 used in the grid search.
    accuracy_grid : numpy.ndarray
        Accuracy values with shape
        (len(learning_rates), len(beta1_values), len(beta2_values)).
    outfile_name : str
        Base filename for saving the generated plot.

    Returns
    -------
    None
    """
    
    best_lr = best_params["learning_rate"]
    best_lr_slice_idx = np.where(np.isclose(learning_rates, best_lr))[0][0]
    beta_accuracies = accuracy_grid[:, :, best_lr_slice_idx]


    # plt.figure(figsize=(8, 6))
    ax = plt.gca()
    im = plt.imshow(beta_accuracies, origin="lower", aspect="auto")
    plt.colorbar(im, label="Accuracy [%]")
    plt.xticks(range(len(beta2_values)), beta2_values)
    plt.yticks(range(len(beta1_values)), beta1_values)
    plt.xlabel("$\\beta_2$")
    plt.ylabel("$\\beta_1$")
    # plt.title(f"Adam beta1/beta2 tuning (lr={best_lr})")
            
    # Writing numbers inside each box
    for i in range(beta_accuracies.shape[0]):
            for j in range(beta_accuracies.shape[1]):
                val = beta_accuracies[i, j]
                if np.isfinite(val):
                    text_color = "black" if im.norm(val) > 0.5 else "white"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=11)

    plt.tight_layout()
    plt.savefig(outfile_name)
    plt.show()

def etaplotter(models, learning_rates, beta1_values, beta2_values, outfile_name):
    """
    Plot accuracy as a function of learning rate.

    Generates a plot of validation accuracy for the slice of the
    grid search accuracies corresponding to the best beta1 and beta2, and saves
    the figure to disk.

    Parameters
    ----------
    learning_rates : array-like
        Learning rates evaluated.
    eta_accuracies : array-like
        Validation accuracies from grid search.
    best_beta1 : float
        Best-performing Adam beta1 value.
    best_beta2 : float
        Best-performing Adam beta2 value.
    outfile_name : str
        Output filename for the saved plot.

    Returns
    -------
    None
    """

    for model in models:
        name = model["name"]
        best_params = model["best_params"]
        accuracy_grid = model["accuracy_grid"]

        b1 = best_params["beta1"]
        b2 = best_params["beta2"]

        b1_idx = np.where(np.isclose(beta1_values, b1))[0][0]
        b2_idx = np.where(np.isclose(beta2_values, b2))[0][0]

        eta_acc = accuracy_grid[b1_idx, b2_idx, :]

        plt.plot(learning_rates, eta_acc, marker="o", label=f"{name}")

    plt.xscale("log")
    plt.xlabel("Learning rate ($\\eta$)")
    plt.ylabel("Accuracy [%]")
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig(outfile_name)
    plt.show()

def confusion_plotter(trained_model, test_loader, device, fig_name):
    """Plot and save a confusion matrix for a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained classification model.
    dataloader : DataLoader
        Data loader providing evaluation samples.
    device : torch.device
        Device used for inference.
    outfile_name : str
        Output filename for the saved plot.

    Returns
    -------
    None
    """
    y_true_test = []
    y_pred_test = []

    # Collecting test predictions to make confusion matrix
    trained_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in test_loader:
                data, target = sample["image"], sample["emolabel"]
                data, target = data.to(device), target.to(device)
                outputs = trained_model(data)
                _, predicted = torch.max(outputs.data, 1)
            
                y_true_test.extend(target.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())
            
                total += target.size(0)
                correct += (predicted == target).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Confusion matrix

    fig, ax = plt.subplots(figsize=(7, 6)) 
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true_test, 
        y_pred_test,  
        normalize="true",      
        values_format=".2f",
        cmap="Blues",
        ax=ax
    )

    # Tick labels
    tick_labels = [emostring(i) for i in range(7)]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Predicted label", fontsize = 17)
    ax.set_ylabel("True label", fontsize = 17)

    # Making text inside boxes smaller
    for text in disp.text_.ravel(): 
        text.set_fontsize(13)                

    plt.tight_layout()
    plt.savefig(fig_name, bbox_inches="tight")
    plt.show()

def emostring(emolabel, zeroindex=True):
    """Convert an emotion label to its string representation.

    Parameters
    ----------
    emolabel : int
        Emotion label.
    zeroindex : bool, optional
        If True, labels are assumed to be in [0, 6].
        If False, labels are assumed to be in [1, 7].

    Returns
    -------
    str
        Emotion name corresponding to the label.
    """

    emotions = ("Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral")
    idx = int(emolabel)
    if not zeroindex:
        idx -= 1
    return emotions[idx]