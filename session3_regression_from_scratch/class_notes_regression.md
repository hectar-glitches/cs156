# Class Notes: Tensors, Linear Regression, and Logistic Regression

## The big picture

A **tensor** here just means a structured numerical container for a physical
property, think of a single number, a list of numbers, or a grid of numbers,
each representing some measurable real-world quantity. In this class:

- **Iris dataset**: each flower is a tensor of 4 numbers (sepal length, sepal
  width, petal length, petal width, all in cm), representing physical flower
  anatomy.
- **Digits dataset**: each handwritten digit is a tensor of pixel brightness
  values (an 8x8 or 28x28 grid, flattened into a row), representing an image.

The whole point of the class: take these tensors, and use two classic models,
**linear regression** and **logistic regression**, to predict something
useful from them. Linear regression predicts a continuous number.
Logistic regression predicts a category (specifically, a probability of
class membership).

---

## Linear Regression

**What it predicts:** a continuous numeric value (e.g., sepal length from
petal length).

**The model's shape:** a straight line, fixed in advance.

$$\hat{y} = \text{intercept} + \text{slope} \times x$$

It only ever has 2 parameters (with one input feature): the intercept and
the slope. It cannot bend or curve, no matter how the data is shaped.

**The loss function it minimizes:** Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$$

**How it actually "fits":** either a closed-form solution (the normal
equations, what `sklearn.linear_model.LinearRegression` uses internally), or
iteratively via **gradient descent**:
1. Start with a guess for the parameters (e.g., both = 0).
2. Measure the current loss (MSE).
3. Compute the gradient (which direction makes the loss worse).
4. Step in the *opposite* direction (downhill).
5. Repeat until the parameters stop changing much (convergence).

Both approaches land on the same answer for plain linear regression, we
verified this ourselves: our from-scratch gradient descent implementation
matched sklearn's closed-form solution to 5 decimal places.

**Why it never fits the data perfectly:** real data has noise, and the model
is restricted to one straight line for the *entire* dataset. If there's a
confound (something else affecting the outcome that isn't in the model), the
line becomes a compromise that doesn't perfectly serve any subgroup.
Example: fitting sepal length from petal length across all 3 iris species
ignores that each species may have a slightly different relationship. This
showed up directly in our residual analysis (see below).

### Diagnosing a bad linear regression fit: residuals

A **residual** is: $\text{residual} = y_{\text{actual}} - y_{\text{predicted}}$

Plot residuals against fitted values. A **good** fit shows residuals
scattered randomly around zero, no visible shape. **Bad signs:**
- Curved pattern -> relationship isn't actually linear
- Funnel/fan shape -> error variance isn't constant (heteroscedasticity)
- Clusters or bands -> a missing variable is creating structure

**What we actually found:** coloring the residual plot by species revealed
distinct bands, setosa's errors leaned positive, versicolor's leaned
negative. The *overall* mean residual was ~0 (looked fine in aggregate!) but
broken out by species, a clear systematic pattern emerged. This is the kind
of thing a single summary number (like MSE alone) will never show you, you
need the plot.

### R² (coefficient of determination)

$$R^2 = \frac{\text{variation in } y \text{ explained by the model}}{\text{total variation in } y}$$

Equivalently: $R^2 = \dfrac{SS(\text{mean}) - SS(\text{fit})}{SS(\text{mean})}$

- $SS(\text{mean})$: total squared error if you just guessed the average $y$
  for everyone (the flat line).
- $SS(\text{fit})$: total squared error using your actual regression line.

**Important direction check:** R² tells you how much of the variation in
**y** (the outcome/thing being predicted) is explained by **x** (the
predictor). It is not symmetric in general, don't flip this around. For
simple regression with one predictor, $R^2 = r^2$ (squared correlation
coefficient), which is where the symmetry confusion often comes from.

**Our example:** petal length explained 76% of the variation in sepal length
($R^2 = 0.76$).

### The F-statistic (is the model's improvement real, or just noise?)

$$F = \frac{\big(SS(\text{mean}) - SS(\text{fit})\big) / (p_{\text{fit}} - p_{\text{mean}})}{SS(\text{fit}) / (n - p_{\text{fit}})}$$

- $p_{\text{mean}} = 1$ (just the average, one parameter)
- $p_{\text{fit}} = 2$ (intercept + slope, for one predictor)
- Numerator: improvement in fit, per extra parameter added
- Denominator: leftover unexplained error, per remaining degree of freedom

Large $F$ -> the added parameter (the slope) is doing real work. $F$ near 1
-> the slope isn't helping much over just guessing the mean.

The **F-distribution** is the theoretical distribution of $F$ values you'd
see purely from random noise if there were truly no relationship. You
compare your observed $F$ against this distribution to get a **p-value**:
the probability of seeing an $F$ this large (or larger) by pure chance.

**Equivalent test using correlation directly:**

$$t = r\sqrt{\frac{n-2}{1-r^2}}$$

compared against a t-distribution with $n-2$ degrees of freedom. For simple
regression with one predictor, $t^2 = F$ exactly, same test, same p-value,
different formula.

**Our example:** $F = 468.55$, $p \approx 10^{-47}$, essentially impossible
by chance.

---

## Logistic Regression

**What it predicts:** a probability that a case belongs to a particular
category (e.g., is this flower a setosa, yes or no).

**The model's shape:** an S-shaped curve (sigmoid), fixed in advance.

$$\hat{p} = \sigma(w x + b) = \frac{1}{1 + e^{-(wx+b)}}$$

It cannot output raw numbers outside 0 to 1, and it cannot bend into
anything other than that one S-curve shape (shifted/stretched by $w$ and
$b$).

**The loss function it minimizes:** Log Loss / binary cross-entropy

$$\text{loss} = -\frac{1}{n}\sum_i \Big[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\Big]$$

This penalizes **confident wrong** predictions much more harshly than a
mildly wrong prediction, unlike MSE, which treats all errors symmetrically.

**How it fits:** gradient descent, same mechanical process as linear
regression (walk downhill on the loss surface), just with a different loss
function and a sigmoid squashing step.

**Why it never "matches" the data perfectly:** it's still constrained to
one fixed S-curve shape. Even when a class is almost perfectly separable
(like setosa by petal length), the model is just finding the best-fit
version of that one curve, not memorizing individual points.

### The perfect-separability gotcha (important!)

If the classes are **perfectly separable** (like setosa vs. not-setosa using
petal length), **unregularized** logistic regression has **no finite
optimal answer**. Making $w$ and $b$ larger and larger keeps pushing
predicted probabilities closer to exactly 0 or 1, which keeps *reducing*
the loss forever, no true minimum exists.

**We verified this ourselves**: running our from-scratch gradient descent
longer and longer made the coefficients keep growing without bound
(`-4.4 -> -7.7 -> -12.0 -> -17.4...`), never settling down. `sklearn`'s
`LogisticRegression` avoids this by applying **L2 regularization** by
default (a penalty discouraging large coefficients), which is *why* our raw
coefficients didn't match sklearn's, even though both achieved 100%
accuracy.

### Diagnosing a bad logistic regression fit: sensitivity & specificity

At a chosen probability threshold (commonly 0.5), build a confusion matrix:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

- **Sensitivity** (recall, true positive rate): $\dfrac{TP}{TP+FN}$, of all
  actual positives, how many did we catch?
- **Specificity** (true negative rate): $\dfrac{TN}{TN+FP}$, of all actual
  negatives, how many did we correctly rule out?
- **Accuracy**: $\dfrac{TP+TN}{\text{total}}$, can be misleading alone,
  especially with imbalanced classes (a model that always predicts "no"
  can have high accuracy and specificity but zero sensitivity).
- **ROC curve**: plots sensitivity vs. (1 - specificity) across *all*
  thresholds. **AUC** (area under it) summarizes fit quality independent of
  any one threshold choice.

---

## Loss Function vs. Metric (don't mix these up)

| | Loss Function | Metric |
|---|---|---|
| **Purpose** | What the model directly minimizes during training | How a human evaluates performance afterward |
| **Measured on** | Training data, during fitting | Often held-out test data |
| **Linear regression example** | MSE | R², MAE, RMSE |
| **Logistic regression example** | Log loss | Accuracy, F1, sensitivity, AUC |

A model doesn't necessarily directly optimize the metric you care about,
that's why both concepts matter separately.

---

## Key vocabulary, quick reference

| Term | Definition |
|---|---|
| **Residual** | $y_{\text{actual}} - y_{\text{predicted}}$, for one data point |
| **Accuracy** | proportion of predictions that were exactly correct |
| **Sensitivity (recall)** | $TP / (TP + FN)$, catches actual positives |
| **Specificity** | $TN / (TN + FP)$, catches actual negatives |
| **Loss function** | quantity minimized during training (MSE, log loss) |
| **Metric** | quantity used to evaluate performance (accuracy, F1, R², AUC) |
| **Gradient descent** | iterative "walk downhill" process for finding best-fit parameters |
| **Regularization** | a penalty added to the loss that discourages extreme parameter values (keeps unregularized/separable-data blowups from happening) |
| **R²** | fraction of variation in y explained by the model |
| **F-statistic** | ratio testing whether an added parameter meaningfully improves fit vs. noise |
| **p-value** | probability of seeing a result this extreme (or more) by pure chance, if the null hypothesis were true |

---

## Big-picture takeaway for this class

Both models are deliberately simple, restricted to one fixed mathematical
shape (a line, or an S-curve), and neither is trying to touch every data
point exactly. Instead, they search (via gradient descent, or a closed-form
shortcut) for the *single best version* of that fixed shape, the one that
minimizes total error across the whole dataset. "Badness of fit" doesn't
show up as one clean failure, it shows up as *systematic patterns* left
over in the residuals (linear regression) or as *values that never
converge* under specific conditions like perfect separability (logistic
regression). A single aggregate summary number (mean residual approx 0, or
accuracy = 100%) can look perfectly healthy while still hiding exactly this
kind of structure, which is why every metric in this class got paired with
an actual plot or a deeper check, not just a single number in isolation.
