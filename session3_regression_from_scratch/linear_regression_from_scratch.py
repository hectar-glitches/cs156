"""
Linear Regression from scratch (gradient descent), applied to:
    predicting sepal length (cm) from petal length (cm), Iris dataset.

THE BIG IDEA -- why this doesn't fit the data "perfectly":
Linear regression commits, in advance, to only ever describing the
relationship as a STRAIGHT LINE: y_hat = w*x + b. That's just 2 numbers
(w and b) trying to summarize 150 real, noisy, biologically-varied
flowers. There is no straight line that passes through all 150 points --
real flowers don't grow in perfectly straight numerical proportion. So
instead of "matching" every point, the model searches for the ONE line
that minimizes the TOTAL squared error across every point simultaneously.
Some points will sit above the line, some below -- that's expected, and
is exactly what the residual plots we made earlier were visualizing.

HOW IT ACTUALLY FITS: gradient descent.
We don't jump straight to the best w and b. Instead we:
  1. Start with a random guess for w and b (here, both start at 0).
  2. Measure how wrong the current line is (the loss: mean squared error).
  3. Compute the *gradient* -- the direction that would make the loss
     WORSE if we moved w and b that way. We step in the OPPOSITE
     direction (downhill on the loss surface).
  4. Repeat thousands of times. Each step, the line gets a little better.
  5. Eventually w and b stop changing much -- we've reached (near) the
     bottom of the loss "bowl", i.e. converged to the best-fit line.
This iterative "walk downhill until you stop improving" process IS what
"fitting" a linear regression model actually means, mechanically.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# --- Load the same data we've been using throughout ------------------------
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
x = df['petal length (cm)'].values
y = df['sepal length (cm)'].values
n = len(x)

# Standardize x for numerically stable gradient descent (large raw values
# can make the loss surface badly scaled and slow/unstable to descend).
# We'll un-standardize the final coefficients at the end so they're
# directly comparable to sklearn's.
x_mean, x_std = x.mean(), x.std()
x_scaled = (x - x_mean) / x_std

# --- Initialize parameters --------------------------------------------------
w = 0.0   # slope, starts with "no relationship assumed"
b = 0.0   # intercept, starts at zero
learning_rate = 0.1   # how big a step to take each iteration
n_iterations = 1000

loss_history = []

# --- Gradient descent loop ---------------------------------------------------
for i in range(n_iterations):
    # Step 1: current model's predictions with today's w, b
    y_pred = w * x_scaled + b

    # Step 2: measure the error -- Mean Squared Error is the LOSS FUNCTION
    # this model is trying to minimize. This single number is what
    # "badness of fit" means to a linear regression model.
    errors = y_pred - y
    loss = np.mean(errors ** 2)
    loss_history.append(loss)

    # Step 3: compute the gradient -- the partial derivative of the loss
    # with respect to w and with respect to b. This tells us exactly
    # which direction (and how strongly) to nudge each parameter to
    # reduce the loss.
    grad_w = (2 / n) * np.sum(errors * x_scaled)
    grad_b = (2 / n) * np.sum(errors)

    # Step 4: take a small step DOWNHILL (opposite the gradient direction)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

# --- Convert back to original (unscaled) petal-length units -----------------
# Since we fit on standardized x, we need to translate w, b back to the
# scale of the real petal-length numbers to compare against sklearn.
w_original = w / x_std
b_original = b - (w * x_mean / x_std)

print("=== Linear Regression from scratch ===")
print(f"Final loss (MSE): {loss_history[-1]:.5f}")
print(f"Learned slope (w):     {w_original:.5f}")
print(f"Learned intercept (b): {b_original:.5f}")

# --- Sanity check against sklearn's closed-form solution --------------------
from sklearn.linear_model import LinearRegression
sklearn_model = LinearRegression().fit(x.reshape(-1, 1), y)
print("\n=== sklearn's LinearRegression (for comparison) ===")
print(f"sklearn slope:     {sklearn_model.coef_[0]:.5f}")
print(f"sklearn intercept: {sklearn_model.intercept_:.5f}")
