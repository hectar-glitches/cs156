# Decision Trees: A Quick-Reference Study Guide

## 1. What a Decision Tree Actually Is

- A decision tree is a **flowchart-like model**: start at the top (the
  root), answer a yes/no question about one feature at each step, and
  follow the matching branch down until you hit a leaf.
- **Structure**: root node -> internal (decision) nodes, each holding
  a rule like `Age < 30?` -> leaf nodes, each holding a final
  prediction.
- **Making a prediction**: drop a new data point in at the root, let it
  flow down through whichever branches its feature values satisfy,
  and read off the prediction at whatever leaf it lands in.
- **Classification trees**: each leaf predicts the majority class of
  the training points that ended up there (or a class probability
  distribution).
- **Regression trees**: each leaf predicts the average of the target
  values that ended up there, instead of a class.

## 2. How a Tree "Learns" (Building It From Data)

- Trees are built **top-down, greedily, and recursively**, this
  process is called recursive binary splitting.
- At every node: try every feature, and every possible split point
  within that feature, and score each candidate split using an
  **impurity measure** (see below).
- Pick whichever single split gives the **best improvement** in
  purity, apply it, then repeat the entire process independently on
  each of the two resulting child nodes.
- "Greedy" means the tree never looks ahead, it always takes the best
  split available *right now*, with no guarantee this leads to the
  globally best tree overall.

## 3. Measuring "How Good" a Split Is: Impurity Measures

A node is **pure** if every point inside it belongs to the same class
(easy, confident prediction). A node is **impure** if it's a mix of
classes (harder to predict confidently). Splits are chosen specifically
to *reduce* impurity as much as possible.

### Entropy (information-theory based)
```
H = -sum_i( p_i * log2(p_i) )
```
- `p_i` = proportion of class `i` in the node.
- H = 0 when the node is perfectly pure (one class, p=1).
- H is at its MAXIMUM when classes are perfectly balanced (for 2
  classes, max H = 1.0, at p = 0.5/0.5).

### Gini Impurity (the focus of this guide)
```
Gini = 1 - sum_i( p_i^2 )
```
- Interpretation: **the probability of misclassifying a randomly
  picked point from this node**, if you labeled it randomly according
  to the node's own class proportions.
- Same core properties as entropy: Gini = 0 when pure; Gini is at its
  maximum when classes are balanced (for 2 classes, max Gini = 0.5, at
  p = 0.5/0.5; more generally, max Gini = 1 - 1/k for k balanced
  classes).
- **Key practical difference from entropy**: no logarithms, so it's
  computationally cheaper, this matters when a tree has to evaluate
  thousands of candidate splits. In practice, Gini and entropy usually
  pick very similar splits; entropy penalizes impurity slightly more
  aggressively due to the logarithmic curve.
- **Historical note**: Gini is the default in CART (used by
  scikit-learn's `DecisionTreeClassifier`); entropy/information gain
  was used in the earlier ID3 and C4.5 algorithms.

### Worked example: real numbers, same split, both metrics

Using the Titanic dataset, splitting on `Sex` to predict `Survived`:

| | n | Entropy | Gini |
|---|---|---|---|
| Parent (no split) | 891 | 0.9607 | 0.4730 |
| Female | 314 | 0.8237 | 0.3828 |
| Male | 577 | 0.6992 | 0.3064 |
| **Weighted children** | | **0.7430** | **0.3334** |
| **Gain from splitting** | | **0.2177** | **0.1396** |

Both metrics agree: splitting on `Sex` meaningfully reduces impurity
(a real, historically-grounded result, women were prioritized for
lifeboats). Note Gini's absolute scale is smaller than entropy's (max
0.5 vs max 1.0 for two classes), so **never compare a raw Gini number
directly to a raw entropy number**, only compare Gini-to-Gini or
entropy-to-entropy.

## 4. Choosing the Best Split, Mechanically

- For a **continuous feature** (like Age): sort the unique values,
  try a threshold between every consecutive pair, evaluate each one.
- For a **categorical feature** (like Embarked): try splitting each
  category out individually (or subsets of categories) versus the rest.
- For every candidate split: compute the **weighted impurity** of the
  two resulting children (weighted by how many points land in each),
  then compute **gain** = parent impurity - weighted child impurity.
- The split with the highest gain, across every feature and every
  threshold checked, wins at that node.

## 5. Controlling Growth (Avoiding Overfitting)

Left unchecked, a tree will keep splitting until every leaf is
perfectly pure, often just one data point per leaf, which memorizes
the training data and generalizes terribly. Common controls:

- `max_depth`: hard cap on how many splits deep the tree can go.
- `min_samples_split`: a node needs at least this many points before
  it's even allowed to split further.
- `min_samples_leaf`: every leaf must contain at least this many points.
- `max_leaf_nodes`: hard cap on total number of leaves.
- `min_impurity_decrease`: a split must improve impurity by at least
  this much to be worth making.
- **Pruning**: an alternative to stopping early, grow the full tree
  first, then cut back branches afterward that don't meaningfully
  improve validation performance (cost-complexity pruning is the
  standard technique here).

## 6. Strengths and Weaknesses

**Strengths**
- Highly interpretable, you can literally read the decision logic.
- No need to scale or normalize features (splits are just thresholds).
- Naturally captures nonlinear relationships and interactions between
  features, with zero extra feature engineering.
- Handles a mix of numeric and categorical data without preprocessing.

**Weaknesses**
- Prone to overfitting (high variance): small changes in training data
  can produce a very different tree.
- Greedy construction means no guarantee of a globally optimal tree.
- Can be biased toward features with many possible split points.
- Individually, trees are often outperformed by ensembles built from
  many trees (Random Forests, Gradient Boosted Trees), which trade
  some interpretability for much better accuracy and stability.

## Quick Formula Reference

| Concept | Formula |
|---|---|
| Entropy | `H = -sum(p_i * log2(p_i))` |
| Gini Impurity | `Gini = 1 - sum(p_i^2)` |
| Information Gain | `parent_H - weighted_avg(child_H)` |
| Gini Gain | `parent_Gini - weighted_avg(child_Gini)` |
| Max Gini (2 classes) | `0.5`, at p=0.5/0.5 |
| Max Entropy (2 classes) | `1.0`, at p=0.5/0.5 |
