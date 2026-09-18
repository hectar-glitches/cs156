import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Your original functions, unchanged
# ---------------------------------------------------------------------------
def leaf_gini_impurity(leaf_ys):
    if len(leaf_ys) == 0:
        return 0
    else:
        return 1 - (sum(leaf_ys == 0)/len(leaf_ys))**2 - (sum(leaf_ys == 1)/len(leaf_ys))**2

def weighted_gini_impurity(left_ys, right_ys):
    total = len(left_ys) + len(right_ys)
    left_leaf = leaf_gini_impurity(left_ys) * len(left_ys)/total
    right_leaf = leaf_gini_impurity(right_ys) * len(right_ys)/total
    return left_leaf + right_leaf

def gini_impurity(x_data, y_data, axis, threshold):
    left_mask = x_data[:, axis] <= threshold
    right_mask = x_data[:, axis] > threshold
    return weighted_gini_impurity(y_data[left_mask], y_data[right_mask])


# ---------------------------------------------------------------------------
# NEW: find the best split by searching REAL candidate thresholds for each
# axis, instead of a fixed [0.0, 0.1, ..., 0.9] list. Candidate thresholds
# are the midpoints between consecutive sorted unique values, this is the
# standard approach (also mentioned in the MLU-Explain reading) and it
# automatically adapts to whatever range/scale the data actually has.
# ---------------------------------------------------------------------------
def find_best_split(x_data, y_data):
    best_gini = float('inf')
    best_axis = None
    best_threshold = None

    n_features = x_data.shape[1]
    for axis in range(n_features):
        values = np.sort(np.unique(x_data[:, axis]))
        # candidate thresholds: midpoints between consecutive unique values
        candidate_thresholds = (values[:-1] + values[1:]) / 2

        for threshold in candidate_thresholds:
            g = gini_impurity(x_data, y_data, axis, threshold)
            if g < best_gini:
                best_gini = g
                best_axis = axis
                best_threshold = threshold

    return best_gini, best_axis, best_threshold


# ---------------------------------------------------------------------------
# NEW: the actual branching / recursion.
#
# A tree node is represented as a dict:
#   Leaf node:     {'leaf': True, 'prediction': <majority class>}
#   Decision node: {'leaf': False, 'axis': <0 or 1>, 'threshold': <float>,
#                    'left': <child node>, 'right': <child node>}
#
# The recursion mirrors the ID3 steps from the reading:
#   1. If this node is already pure, or we've hit a stopping condition,
#      make it a LEAF (majority class).
#   2. Otherwise, find the best split (using find_best_split above).
#   3. Partition the data using that split.
#   4. Recurse separately on the left and right subsets to build the
#      left and right children.
# ---------------------------------------------------------------------------
def build_tree(x_data, y_data, depth=0, max_depth=4, min_samples_split=5):
    n_samples = len(y_data)
    current_gini = leaf_gini_impurity(y_data)

    def make_leaf():
        # majority class in this node becomes the prediction
        prediction = 1 if np.sum(y_data == 1) >= np.sum(y_data == 0) else 0
        return {'leaf': True, 'prediction': prediction, 'n_samples': n_samples, 'gini': current_gini}

    # --- Stopping conditions ---
    if current_gini == 0:                       # already pure
        return make_leaf()
    if depth >= max_depth:                       # hit depth limit
        return make_leaf()
    if n_samples < min_samples_split:             # too few points to split further
        return make_leaf()

    # --- Find the best split among both axes ---
    best_gini, best_axis, best_threshold = find_best_split(x_data, y_data)

    # If splitting doesn't actually improve purity, stop here too
    if best_gini >= current_gini or best_axis is None:
        return make_leaf()

    left_mask = x_data[:, best_axis] <= best_threshold
    right_mask = ~left_mask

    # --- Recurse: build the left and right subtrees ---
    left_subtree = build_tree(x_data[left_mask], y_data[left_mask], depth + 1, max_depth, min_samples_split)
    right_subtree = build_tree(x_data[right_mask], y_data[right_mask], depth + 1, max_depth, min_samples_split)

    return {
        'leaf': False,
        'axis': best_axis,
        'threshold': best_threshold,
        'left': left_subtree,
        'right': right_subtree,
        'n_samples': n_samples,
        'gini': current_gini,
    }


# ---------------------------------------------------------------------------
# NEW: walk a single point down the tree to get a prediction, exactly the
# "compare feature d_i to threshold t_i, go left or right" mechanism from
# earlier in this conversation.
# ---------------------------------------------------------------------------
def predict_one(tree, x):
    node = tree
    while not node['leaf']:
        if x[node['axis']] <= node['threshold']:
            node = node['left']
        else:
            node = node['right']
    return node['prediction']

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


# ---------------------------------------------------------------------------
# Try it on make_moons data
# ---------------------------------------------------------------------------
x_data, y_data = make_moons(n_samples=200, noise=0.2, random_state=42)

my_tree = build_tree(x_data, y_data, max_depth=4, min_samples_split=5)
my_predictions = predict(my_tree, x_data)
my_accuracy = np.mean(my_predictions == y_data)
print(f"Custom tree training accuracy: {my_accuracy:.4f}")

# Sanity check against sklearn's own implementation, same max_depth
sk_tree = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=5, random_state=42)
sk_tree.fit(x_data, y_data)
sk_accuracy = sk_tree.score(x_data, y_data)
print(f"sklearn tree training accuracy:  {sk_accuracy:.4f}")


# ---------------------------------------------------------------------------
# Visualize both decision boundaries side by side
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
xx, yy = np.meshgrid(
    np.linspace(x_data[:, 0].min() - 0.3, x_data[:, 0].max() + 0.3, 300),
    np.linspace(x_data[:, 1].min() - 0.3, x_data[:, 1].max() + 0.3, 300)
)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

Z_custom = predict(my_tree, grid_points).reshape(xx.shape)
axes[0].contourf(xx, yy, Z_custom, alpha=0.5, cmap='RdBu')
axes[0].scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='RdBu', edgecolor='black', s=20)
axes[0].set_title(f'Custom recursive tree (acc={my_accuracy:.3f})')

Z_sklearn = sk_tree.predict(grid_points).reshape(xx.shape)
axes[1].contourf(xx, yy, Z_sklearn, alpha=0.5, cmap='RdBu')
axes[1].scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='RdBu', edgecolor='black', s=20)
axes[1].set_title(f'sklearn DecisionTreeClassifier (acc={sk_accuracy:.3f})')

fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/custom_tree_vs_sklearn.png', dpi=150)
print("Saved comparison plot.")
