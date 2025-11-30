import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):

    if (np.all(feature_vector == feature_vector[0]) or
            len(feature_vector) < 2):
        return np.array([]), np.array([]), None, None

    sort_idx = np.argsort(feature_vector)
    sorted_features = feature_vector[sort_idx]
    sorted_targets = target_vector[sort_idx]

    diff_positions = np.where(sorted_features[1:] != sorted_features[:-1])[0] + 1

    if len(diff_positions) == 0:
        return np.array([]), np.array([]), None, None

    thresholds = (sorted_features[diff_positions - 1] + sorted_features[diff_positions]) / 2.0

    cumsum = np.cumsum(sorted_targets)
    total_ones = cumsum[-1]

    left_counts = diff_positions
    left_ones = cumsum[diff_positions - 1]

    right_counts = len(feature_vector) - left_counts
    right_ones = total_ones - left_ones

    p1_left = left_ones / left_counts
    p0_left = 1 - p1_left

    p1_right = right_ones / right_counts
    p0_right = 1 - p1_right

    H_left = 1 - p1_left ** 2 - p0_left ** 2
    H_right = 1 - p1_right ** 2 - p0_right ** 2

    ginis = - (left_counts / len(feature_vector)) * H_left - (right_counts / len(feature_vector)) * H_right

    if len(ginis) == 0:
        return np.array([]), np.array([]), None, None

    best_idx = np.argmax(ginis)

    max_gini = ginis[best_idx]
    same_gini = np.where(ginis == max_gini)[0]
    if len(same_gini) > 1:
        best_idx = same_gini[np.argmin(thresholds[same_gini])]

    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if (self._max_depth is not None and depth >= self._max_depth or
                len(sub_y) < self._min_samples_split or
                np.all(sub_y == sub_y[0])):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split_best = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count

                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None:
                continue

            if feature_type == "real":
                current_split = feature_vector < threshold
            elif feature_type == "categorical":
                threshold_categories = [cat for cat, idx in categories_map.items() if idx < threshold]
                current_split = np.array([x in threshold_categories for x in sub_X[:, feature]])

            left_count = np.sum(current_split)
            right_count = np.sum(~current_split)
            if (left_count < self._min_samples_leaf or
                    right_count < self._min_samples_leaf):
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold
                split_best = current_split

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":

            counts = Counter(sub_X[:, feature_best])
            clicks = Counter(sub_X[sub_y == 1, feature_best])
            ratio = {}
            for key, current_count in counts.items():
                if key in clicks:
                    current_click = clicks[key]
                else:
                    current_click = 0
                ratio[key] = current_click / current_count

            sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
            categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}

            node["categories_split"] = [cat for cat, idx in categories_map.items() if idx < threshold_best]

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"], depth + 1)

    def _predict_node(self, x, node):

        if node["type"] == "terminal":
            return node["class"]

        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)