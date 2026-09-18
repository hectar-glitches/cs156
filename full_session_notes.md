# Full Session Notes: Linear Algebra, Vector Spaces, and Multivariate Regression

## How the readings connect to everything we did

- **Murphy 3.1 (Joint distributions)**: covariance and correlation. Two
  variables that move together have nonzero covariance; correlation
  normalizes that into a bounded, unit-free number. This is the
  "co-linearity" the study guide mentions, NOT the same thing as the
  "collinearity" problem in regression (where predictors are correlated
  with each other), though the two ideas rhyme.
- **Murphy 7.1-7.2 (matrices, matrix multiplication)**: this is the
  span/basis/linear-map material. A matrix is a machine that transforms
  vectors; multiplying a vector by a matrix is a specific case of a
  linear function; a matrix's columns fully describe what it does to
  every input, because every input is just a combination of basis
  vectors (see "The big proof," below).
- **Murphy 7.3.1**: system of linear equations, the direct bridge to
  solving `X*beta = Y`.
- **3Blue1Brown series**: the geometric intuition for everything above,
  span, basis, linear transformations, determinant, matrix
  multiplication as composition. Watch this if any of the algebra below
  feels like symbol-pushing without a picture in your head.
- **StatQuest (multiple regression)**: the applied, intuitive version of
  the OLS derivation below, less notation, same underlying math.

---

## Part A: Vector space foundations

### What a vector space is

A collection of vectors where adding two of them, or scaling one by a
number, always produces another vector still inside the same space. Our
data points (each Titanic passenger's `[Pclass, Age, SibSp]`) are
vectors living in R^3.

### Span

The span of a set of vectors is every vector reachable by combining
them:
```
span({v1, ..., vn}) = { v : v = a1*v1 + a2*v2 + ... + an*vn,  a_i in R }
```

**Key theorem**: if you have exactly `n` linearly independent vectors,
each living in R^n, their span is ALL of R^n. Fewer independent vectors
than dimensions -> you only span a smaller slice (a line or plane
inside the bigger space). This single theorem explains almost every
"weird result" in this session (see Part C).

### Linear independence

A set of vectors is independent if the only way to combine them into
the zero vector is using all-zero coefficients. Practically: check
`det(X) != 0` for a square matrix, or `np.linalg.matrix_rank(X)` equals
the number of columns.

### Range (a.k.a. column space, a.k.a. image)

For a matrix `A`, `range(A) = span(columns of A)`. This is the set of
every output `y = Ax` could ever produce, no matter what `x` you feed
it. This is NOT automatically the whole codomain, if `A` has more rows
than columns, the range is a lower-dimensional slice sitting inside a
bigger space (this is exactly the OLS situation: n rows, p+1 columns,
n >> p+1).

---

## Part B: Matrix operations, quick reference

- **Matrix-vector multiply** `y = Ax`: each output = one row of `A`
  dotted with `x`. Equivalently, `y` is a combination of `A`'s columns,
  weighted by `x`'s entries.
- **Matrix-matrix multiply** `C = AB`: entry `C_ij` = row `i` of `A`
  dotted with column `j` of `B`.
- **Transpose** `A^T`: flips rows into columns.
- **Inverse** `A^-1`: undoes what `A` does; only exists for square,
  full-rank (independent-column) matrices. `A^-1 * A = I`.
- **Trace** `tr(A)`: sum of diagonal entries; equals the sum of
  eigenvalues; `tr(AB) = tr(BA)` even when `AB != BA`.
- **Norm** `||A||`: a generalized "size" measure obeying non-negativity,
  definiteness, absolute homogeneity, and the triangle inequality.
  Frobenius norm treats the matrix as one long vector; the induced norm
  is the largest possible stretching factor `A` can apply to any input.
- **Condition number** `kappa(A) = ||A|| * ||A^-1||`: ratio of the most
  to least a matrix stretches different directions. Large condition
  number = small input errors get massively amplified when solving
  `Ax=b`.

---

## Part C: The full OLS derivation (matrix notation)

### The model

For `n` passengers and `k=3` predictors (we chose Pclass, Age, SibSp):

```
Y = X*beta + epsilon
```

- `Y` is n x 1 (Fare values)
- `X` is n x 4 (leading column of 1s for the intercept, then the 3
  predictor columns)
- `beta` is 4 x 1: `[beta_0, beta_1, beta_2, beta_3]`
- `epsilon` is n x 1 (error terms)

### The loss function (Ordinary Least Squares)

```
L(beta) = (Y - X*beta)^T * (Y - X*beta) = sum_i (y_i - x_i^T*beta)^2
```

### Expand it

Using `(A-B)^T(A-B) = A^T*A - A^T*B - B^T*A + B^T*B`, and the fact that
a scalar equals its own transpose (`(X*beta)^T*Y = Y^T*X*beta`):

```
L(beta) = Y^T*Y - 2*beta^T*X^T*Y + beta^T*X^T*X*beta
```

### First derivative

Using the matrix-calculus identities `d(a^T*b)/db = a` and
`d(b^T*A*b)/db = 2*A*b` (valid since `X^T*X` is always symmetric):

```
dL/dbeta = -2*X^T*Y + 2*X^T*X*beta
```

**This is exactly the "matrix of partial derivatives" the assignment
asks about.** It's a 4x1 vector (one partial derivative per parameter:
`dL/d(beta_0)`, `dL/d(beta_1)`, `dL/d(beta_2)`, `dL/d(beta_3)`), called
the **gradient**. The matrix-calculus reference paper linked in the
assignment is the general cheat sheet for rules like these, this
specific derivation only needed two identities from it.

### Set to zero, solve (the Normal Equation)

```
-2*X^T*Y + 2*X^T*X*beta = 0
X^T*X*beta = X^T*Y
beta = (X^T*X)^-1 * X^T*Y
```

### Verified against real data

Fitting on all 714 Titanic passengers with complete Age data:

```
intercept (beta_0) = 121.8853
Pclass   (beta_1)  = -37.3874
Age      (beta_2)  = -0.2725
SibSp    (beta_3)  =  8.8276
```

(Small differences from earlier runs in this session are expected,
they came from fitting on a 70% train split rather than the full
dataset. Always double check whether a reported beta came from a
train subset or the whole dataset before comparing numbers across
sessions, this is the single most common source of "the numbers don't
match!" confusion.)

---

## Part D: The 3x3 exact-solve example, and why it's weird

### The dimension error to catch

Asked for a "1x3 target Y" -- **wrong**. With 3 data points, each
contributing ONE fare value, Y must be **3x1** (one row per passenger,
matching the rows of X), not 1x3 (which would mean one passenger with
three different fares).

### The matrices (passengers 1, 2, 7; no intercept column this time)

```
X = [ 3  22  1 ]        Y = [ 7.25   ]
    [ 1  38  1 ]            [ 71.28  ]
    [ 1  54  0 ]            [ 51.86  ]
```

### Solving it

Since X is square, solve directly: `beta = X^-1 * Y`

```
det(X) = -124  (nonzero -> invertible -> columns independent -> span all of R^3)
beta = [-21.19, 1.35, 41.07]
X @ beta reproduces Y EXACTLY (residuals ~ 0, loss = 0)
```

### Why this is NOT a real "least squares fit"

With exactly 3 unknowns and exactly 3 equations, there's no error left
to minimize, the system is forced to pass through all 3 points exactly.
Compare the coefficients (`Pclass: 1.35`) against the REAL 714-point
model (`Pclass: -37.39`), wildly different, even opposite sign. **This
is expected, not a bug.** Real OLS only becomes a genuine compromise
(nonzero residuals, balancing error across many points) when `n > p`
(more data points than parameters). With `n = p`, you just get exact
interpolation through arbitrary points, which generalizes terribly and
proves nothing.

### The vector-space explanation

1. The 3 columns of X are 3 vectors in R^3.
2. Since they're independent (det != 0), by the span theorem they
   span ALL of R^3.
3. So ANY target Y is guaranteed reachable, exactly. Zero residual
   isn't evidence of a good model, it's a mathematical guarantee that
   was true before you even looked at the data.
4. Contrast with the real 714-passenger case: X has 714 rows but only
   4 columns, so its column space is at best a 4-dimensional slice
   inside a 714-dimensional space. Y essentially never lands exactly
   in that thin slice. What OLS actually does is find the
   **orthogonal projection** of Y onto that column space, the closest
   reachable point, leaving a nonzero residual vector perpendicular to
   the column space.

---

## Part E: Measuring fit as a vector-space quantity

### Sum of Squared Residuals (SSR)

```
SSR = sum_i (y_i - y_hat_i)^2 = ||residual vector||_2^2
```

**Spatial meaning**: this is the squared Euclidean (L2) length of the
residual vector, the straight-line "distance" between the actual Y
vector and the predicted Y-hat vector, sitting in n-dimensional space.
OLS minimizing SSR = OLS finding the point in the column space of X
that is literally CLOSEST (in ordinary straight-line distance) to Y.
That's the projection from Part D, item 4, made precise.

### Sum of Absolute Residuals (SAR)

```
SAR = sum_i |y_i - y_hat_i| = ||residual vector||_1
```

**Spatial meaning**: the L1 ("taxicab") length of the same residual
vector, adding up distances along each axis separately rather than
going diagonally. Minimizing SAR (instead of SSR) is a DIFFERENT
optimization problem (median regression / LAD regression), less
sensitive to big outliers than SSR, since SSR squares errors and SAR
doesn't.

### Worked by hand: SSR and SAR for our 3 chosen passengers

Using the FULL-dataset model (`beta_0=121.89, Pclass=-37.39,
Age=-0.27, SibSp=8.83`) applied to passengers 1, 2, and 7:

| Passenger | Actual Fare | Predicted Fare | Residual |
|---|---|---|---|
| 1 | 7.2500 | 12.5567 | -5.3067 |
| 2 | 71.2833 | 82.9723 | -11.6890 |
| 7 | 51.8625 | 69.7854 | -17.9229 |

```
SSR = (-5.3067)^2 + (-11.6890)^2 + (-17.9229)^2
    = 28.161 + 136.633 + 321.230
    = 486.02

SAR = |-5.3067| + |-11.6890| + |-17.9229|
    = 5.3067 + 11.6890 + 17.9229
    = 34.92
```

**Important distinction from Part D**: these residuals are NONZERO,
because we're now using the coefficients fit on all 714 passengers
(the real, compromise-based OLS fit), not the exact 3x3 local solve.
This is the correct way to interpret "calculate this measure for your
three data points", use the real model's predictions, not the toy
exact system's.

### Code for the full dataset

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

titanic = pd.read_csv("titanic.csv")
model_df = titanic[["Fare", "Pclass", "Age", "SibSp"]].dropna(subset=["Age"])

X = model_df[["Pclass", "Age", "SibSp"]]
y = model_df["Fare"]

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
residuals = y.values - y_pred

ssr = np.sum(residuals**2)
sar = np.sum(np.abs(residuals))

print(f"SSR (full dataset, n={len(y)}): {ssr:.4f}")
print(f"SAR (full dataset, n={len(y)}): {sar:.4f}")
```

Actual output on the real 714-passenger dataset:
```
SSR (full dataset, n=714): 1312882.8386
SAR (full dataset, n=714): 15535.8215
```

(SSR is on a totally different numeric scale than SAR, squares of
fares in the hundreds produce numbers in the hundred-thousands. Don't
be alarmed by the size, compare SSR to SSR and SAR to SAR, never
directly to each other.)

---

## Part F: Robot Interview, what to actually watch for

When you paste your own interview transcript and annotate it, these
are the specific failure patterns worth hunting for, based on real
mistakes that showed up or were narrowly avoided in this session:

1. **Silent dimension errors.** Watch for "1xN" vs "Nx1" mixups,
   exactly like the Y-vector error in Part D. A model may state a
   dimension casually without checking it against the actual data
   shape.
2. **Confusing the exact 3x3 solve with genuine OLS.** If an answer
   presents the toy exact-solve coefficients as if they "confirm" or
   are comparable to the full-dataset OLS coefficients, that's a real
   conceptual error, they answer different questions.
3. **Sign and transpose slips in the calculus.** The derivative
   `-2*X^T*Y + 2*X^T*X*beta` is easy to get subtly wrong (dropping a
   factor of 2, transposing the wrong matrix). Always verify
   numerically (as we did, matching sklearn's output to 5 decimal
   places) rather than trusting symbolic algebra alone.
4. **Train/test split confusion.** Coefficients fit on a 70% split vs.
   the full dataset will differ slightly. An answer that doesn't
   specify which one it used, and then gets "surprised" the numbers
   don't match a previous answer, is a red flag worth annotating.
5. **Treating correlation and regression slope as the same thing.**
   From the Murphy 3.1 reading: `corr(X,Y)` and the regression slope
   `Cov(X,Y)/Var(X)` are NOT the same formula and are only numerically
   equal when Var(X) = Var(Y). A model conflating these is a real
   mathematical error.
6. **Overclaiming R-squared/fit quality from a small or convenient
   subset.** E.g., presenting a 100% in-sample accuracy or a
   zero-residual toy example as evidence of a "good model" without
   flagging that it's guaranteed by the math (Part D) rather than
   earned by genuine predictive power.

---

## Quick-reference formula sheet

| Concept | Formula |
|---|---|
| Linear model | `Y = X*beta + epsilon` |
| OLS loss | `L(beta) = (Y-X*beta)^T(Y-X*beta)` |
| Gradient of loss | `dL/dbeta = -2*X^T*Y + 2*X^T*X*beta` |
| Normal equation | `beta = (X^T*X)^-1 * X^T*Y` |
| SSR | `sum(residual_i^2) = \|\|residual\|\|_2^2` |
| SAR | `sum(\|residual_i\|) = \|\|residual\|\|_1` |
| Span | `{v : v = sum(a_i * v_i)}` |
| Range of a matrix | `span(columns of A)` |
| Trace | `sum of diagonal entries = sum of eigenvalues` |
| Condition number | `kappa(A) = \|\|A\|\| * \|\|A^-1\|\|` |

