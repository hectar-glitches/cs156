import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def load_data():
    """Load handwritten-digit images and labels.

    NOTE: true MNIST (28x28, 70k images) requires an internet download via
    sklearn.datasets.fetch_openml('mnist_784'), which isn't available in
    this sandboxed environment. This uses sklearn's built-in 'digits' set
    instead -- same idea (labeled images of handwritten 0-9s), just
    smaller (8x8 pixels, ~1,800 images), and it ships with the library so
    no download is needed. Swap in fetch_openml if you have internet
    access and want the real thing.
    """
    digits = load_digits()
    X = digits.data      # shape (n_samples, 64): each row is a flattened 8x8 image
    y = digits.target    # shape (n_samples,): the true digit, 0-9
    return X, y


def split_data(X, y, test_size=0.3, random_state=101):
    """Split features/labels into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_naive_bayes(X_train, y_train):
    """Fit a Gaussian Naive Bayes classifier. Gaussian NB assumes each
    pixel's brightness, within a given digit class, is roughly normally
    distributed -- a simplifying assumption, but a fast, solid baseline
    for image classification."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test):
    """Compute accuracy, F1 score, and a confusion matrix for a fitted
    classifier. Returns a dict of results and prints a readable summary."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')      # unweighted average across classes
    f1_weighted = f1_score(y_test, y_pred, average='weighted')  # weighted by class frequency
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 (macro): {f1_macro:.3f}")
    print(f"F1 (weighted): {f1_weighted:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_pred': y_pred,
    }


def plot_confusion_matrix(cm, class_labels=None):
    """Visualize a confusion matrix. Rows = true digit, columns = predicted
    digit; the diagonal shows correct predictions, off-diagonal cells show
    which digits get mixed up with which."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('Naive Bayes: Confusion Matrix')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = fit_naive_bayes(X_train, y_train)
    results = evaluate_classifier(model, X_test, y_test)

    fig = plot_confusion_matrix(results['confusion_matrix'], class_labels=sorted(set(y)))
    fig.savefig('/mnt/user-data/outputs/naive_bayes_confusion_matrix.png', dpi=150)
    print("Saved plot to naive_bayes_confusion_matrix.png")
