"""
Logistic Regression from scratch (gradient descent), applied to:
    predicting "is this flower a setosa?" (yes/no) from petal length (cm).

THE BIG IDEA -- why this doesn't fit the data "perfectly":
Logistic regression commits, in advance, to describing the relationship
as a single S-SHAPED CURVE (a sigmoid) squashed between 0 and 1:
    p_hat = sigmoid(w*x + b)
It is NOT trying to output exactly 0 or exactly 1 for every point --
it's trying to output a well-calibrated PROBABILITY. Even when classes
are almost perfectly separable (as setosa is here), the model is still
just fitting the two numbers w and b that shape one particular S-curve
-- it can shift the curve left/right and control how steeply it rises,
but it can't bend, wiggle, or contort into an arbitrary shape. If the
true boundary were messier or the classes overlapped more (as with
versicolor vs. virginica), this same rigid S-curve would visibly not
"match the data perfectly" -- which is the point: it's a simple,
generalizable rule, not a memorized lookup table.

HOW IT ACTUALLY FITS: gradient descent on a different loss function.
Instead of Mean Squared Error, classification uses LOG LOSS
(binary cross-entropy):
    loss = -mean( y*log(p_hat) + (1-y)*log(1-p_hat) )
This loss punishes confident WRONG predictions especially harshly (e.g.
predicting p_hat=0.99 for something that's actually class 0 is
penalized far more than predicting p_hat=0.6). We use the exact same
"walk downhill" gradient descent procedure as linear regression, just
with a different loss function and a sigmoid squashing step -- showing
these two models share the same fitting *mechanism*, they just differ
in what loss they minimize and what shape they're allowed to take.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# --- Load data, build the binary target -------------------------------------
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

x = df['petal length (cm)'].values
y = (df['species'] == 0).astype(float).values  # 1 = setosa, 0 = not setosa
n = len(x)

# Standardize x, same reasoning as the linear regression script: keeps
# gradient descent numerically well-behaved.
x_mean, x_std = x.mean(), x.std()
x_scaled = (x - x_mean) / x_std


def sigmoid(z):
    """Squashes any real number into the range (0, 1) -- this is what
    turns a plain linear combination (w*x + b) into something we can
    interpret as a probability."""
    return 1 / (1 + np.exp(-z))


# --- Initialize parameters --------------------------------------------------
w = 0.0
b = 0.0
learning_rate = 0.5
n_iterations = 2000

loss_history = []

# --- Gradient descent loop ---------------------------------------------------
for i in range(n_iterations):
    # Step 1: linear combination, then squash through the sigmoid to get
    # a predicted probability for each flower being setosa.
    z = w * x_scaled + b
    p_hat = sigmoid(z)

    # Step 2: measure badness of fit using LOG LOSS (cross-entropy), not
    # mean squared error -- this is the loss function logistic regression
    # is actually trying to minimize. A tiny epsilon avoids log(0).
    eps = 1e-9
    loss = -np.mean(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))
    loss_history.append(loss)

    # Step 3: gradient of the log loss w.r.t. w and b. (It has a
    # deceptively simple form -- the same "error * x" pattern as linear
    # regression's gradient, just with p_hat instead of a raw linear
    # prediction.)
    errors = p_hat - y
    grad_w = (1 / n) * np.sum(errors * x_scaled)
    grad_b = (1 / n) * np.sum(errors)

    # Step 4: step downhill.
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

# --- Convert back to original (unscaled) petal-length units -----------------
w_original = w / x_std
b_original = b - (w * x_mean / x_std)

print("=== Logistic Regression from scratch ===")
print(f"Final loss (log loss): {loss_history[-1]:.5f}")
print(f"Learned w: {w_original:.5f}")
print(f"Learned b: {b_original:.5f}")

# Accuracy check on the training data
final_preds = (sigmoid(w * x_scaled + b) >= 0.5).astype(float)
accuracy = np.mean(final_preds == y)
print(f"Training accuracy: {accuracy:.3f}")

# --- Sanity check against sklearn ------------------------------------------
from sklearn.linear_model import LogisticRegression
sklearn_model = LogisticRegression().fit(x.reshape(-1, 1), y)
print("\n=== sklearn's LogisticRegression (for comparison) ===")
print(f"sklearn w: {sklearn_model.coef_[0][0]:.5f}")
print(f"sklearn b: {sklearn_model.intercept_[0]:.5f}")
print(f"sklearn training accuracy: {sklearn_model.score(x.reshape(-1, 1), y):.3f}")

# --- IMPORTANT: why our coefficients don't match sklearn's, even though
# both get 100% accuracy -----------------------------------------------
# Setosa is PERFECTLY separable from the other species using petal length
# alone (there's a clean gap, no overlap). When classes are perfectly
# separable, plain unregularized logistic regression has no finite best
# answer: making w and b larger and larger keeps pushing predicted
# probabilities closer to exactly 0 or 1, which keeps *reducing* log
# loss forever, with no minimum to converge to. Our gradient descent
# genuinely never converges here -- it just keeps growing w and b the
# longer you let it run (try increasing n_iterations and watch the
# printed w, b keep climbing).
#
# sklearn's LogisticRegression avoids this by applying L2 regularization
# by default (a penalty term added to the loss that discourages
# excessively large coefficients), which keeps its answer finite and
# well-behaved even on perfectly separable data. That regularization
# term is *why* sklearn's w and b differ from ours -- not because either
# implementation is "wrong," but because we're minimizing two subtly
# different loss functions (plain log loss vs. log loss + penalty).
print("\nNote: coefficients differ from sklearn because setosa is perfectly")
print("separable by petal length -- unregularized logistic regression has no")
print("finite optimum here (w, b grow without bound the longer you train),")
print("while sklearn's default L2 regularization keeps its coefficients finite.")
