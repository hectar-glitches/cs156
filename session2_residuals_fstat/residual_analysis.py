import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression


def load_data():
    """Load Iris into a DataFrame with feature columns plus a numeric
    'species' column (0=setosa, 1=versicolor, 2=virginica)."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df


def fit_and_get_residuals(df, feature='petal length (cm)', target='sepal length (cm)'):
    """Fit linear regression on the FULL dataset (not train/test split, since
    we're diagnosing model fit, not testing generalization) and compute the
    residual (actual - predicted) for every point."""
    X = df[[feature]]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    df = df.copy()
    df['predicted'] = model.predict(X)
    df['residual'] = df[target] - df['predicted']  # actual minus predicted

    return model, df


def plot_residuals(df, feature='petal length (cm)'):
    """Two diagnostic views:
    left: residuals vs. fitted values, colored by species -- reveals whether
          errors are random noise or hide a species-based confound.
    right: histogram of residuals -- checks whether errors are roughly
          symmetric/centered at zero (as a good linear fit should be)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    species_colors = {'setosa': 'tab:blue', 'versicolor': 'tab:orange', 'virginica': 'tab:green'}

    for species_name, group in df.groupby('species_name'):
        axes[0].scatter(
            group['predicted'], group['residual'],
            label=species_name, color=species_colors[species_name],
            alpha=0.7, edgecolor='black', linewidth=0.4
        )
    axes[0].axhline(0, color='black', linewidth=1, linestyle='--')
    axes[0].set_xlabel('Predicted sepal length (cm)')
    axes[0].set_ylabel('Residual (actual - predicted)')
    axes[0].set_title('Residuals vs. Fitted Values')
    axes[0].legend(title='Species')

    axes[1].hist(df['residual'], bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    axes[1].axvline(0, color='black', linewidth=1, linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Residuals')

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    iris_df = load_data()
    model, iris_df = fit_and_get_residuals(iris_df)

    print(f"Mean residual: {iris_df['residual'].mean():.4f}  (should be ~0 for a well-fit model)")
    print(f"Residual std dev: {iris_df['residual'].std():.4f}")
    print("\nMean residual by species (reveals the confound):")
    print(iris_df.groupby('species_name')['residual'].mean())

    fig = plot_residuals(iris_df)
    fig.savefig('/mnt/user-data/outputs/residual_plot.png', dpi=150)
    print("\nSaved plot to residual_plot.png")
