import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


def load_data():
    """Load the Iris data set into a DataFrame with numeric feature columns
    plus a numeric 'species' column (0=setosa, 1=versicolor, 2=virginica)."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df


def fit_linear_regression(df, feature='petal length (cm)', target='sepal length (cm)'):
    """Fit a LinearRegression predicting a continuous target (sepal length)
    from a single continuous feature (petal length)."""
    X = df[[feature]]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, (X_train, X_test, y_train, y_test)


def fit_logistic_regression(df, feature='petal length (cm)', target_species=0):
    """Fit a LogisticRegression predicting a binary label -- whether a flower
    IS a given species (1) or not (0) -- from the same single feature, so it
    can be plotted on the same x-axis as the linear regression."""
    X = df[[feature]]
    y = (df['species'] == target_species).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, (X_train, X_test, y_train, y_test)


def plot_fits(lin_model, lin_data, log_model, log_data, feature='petal length (cm)'):
    """Plot the linear regression fit (a straight line) and the logistic
    regression fit (an S-shaped probability curve) side-by-side, both
    against the same feature on the x-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Linear regression -----------------------------------------
    X_train, X_test, y_train, y_test = lin_data
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    x_range = np.linspace(X_all[feature].min(), X_all[feature].max(), 100).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range, columns=[feature])

    axes[0].scatter(X_all, y_all, alpha=0.5, label='data')
    axes[0].plot(x_range, lin_model.predict(x_range_df), color='red', linewidth=2, label='linear fit')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('sepal length (cm)')
    axes[0].set_title('Linear Regression')
    axes[0].legend()

    # --- Right: Logistic regression ----------------------------------------
    X_train2, X_test2, y_train2, y_test2 = log_data
    X_all2 = pd.concat([X_train2, X_test2])
    y_all2 = pd.concat([y_train2, y_test2])

    probs = log_model.predict_proba(x_range_df)[:, 1]  # P(class = 1)

    axes[1].scatter(X_all2, y_all2, alpha=0.5, label='data (0 = not species, 1 = species)')
    axes[1].plot(x_range, probs, color='red', linewidth=2, label='logistic fit (P = species)')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('P(species = setosa)')
    axes[1].set_title('Logistic Regression')
    axes[1].legend()

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    iris_df = load_data()

    lin_model, lin_data = fit_linear_regression(iris_df)
    log_model, log_data = fit_logistic_regression(iris_df, target_species=0)  # 0 = setosa

    fig = plot_fits(lin_model, lin_data, log_model, log_data)
    fig.savefig('/mnt/user-data/outputs/regression_comparison.png', dpi=150)
    print('Saved plot to regression_comparison.png')
