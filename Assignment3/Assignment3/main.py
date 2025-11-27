import numpy as np
import pandas as pd
from collections import defaultdict

def load_zoo(path):
    df = pd.read_csv(path, header=None)
    name_col = 0
    class_col = df.shape[1] - 1
    feature_cols = [c for c in df.columns if c not in [name_col, class_col]]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[class_col].to_numpy(dtype=int)
    return X, y

def load_weather(path):
    df = pd.read_csv(path)
    cols = list(df.columns)
    label_col = cols[-1]
    feature_cols = cols[:-1]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)
    return X, y

def _stratified_split_indices(y, test_ratio=0.2, ensure_min_one_per_class=True):
    y = np.asarray(y)
    n_samples = len(y)
    class_to_indices = {}
    for i, c in enumerate(y):
        class_to_indices.setdefault(c, []).append(i)

    test_indices = []
    for c, idx_list in class_to_indices.items():
        idx_list = np.array(idx_list)
        np.random.shuffle(idx_list)
        n_c = len(idx_list)
        n_test_c = int(round(test_ratio * n_c))
        if ensure_min_one_per_class:
            n_test_c = max(1, n_test_c)
        else:
            n_test_c = min(n_c, max(1, n_test_c))
        test_indices.append(idx_list[:n_test_c])

    test_indices = np.concatenate(test_indices)
    all_idx = np.arange(n_samples)
    mask = np.ones(n_samples, dtype=bool)
    mask[test_indices] = False
    train_indices = all_idx[mask]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

def make_repeated_splits(X, Y, n_repeats=10, test_ratio=0.2, ensure_all_classes_in_test=False):
    X = np.asarray(X)
    Y = np.asarray(Y)
    splits = []
    for _ in range(n_repeats):
        train_idx, test_idx = _stratified_split_indices(
            Y,
            test_ratio=test_ratio,
            ensure_min_one_per_class=ensure_all_classes_in_test,
        )
        splits.append((X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]))
    return splits

class ExactBayesClassifier:
    def __init__(self):
        self.class_priors = None
        self.joint_probs = None
        self.classes_ = None

    def _vector_to_key(self, x_row):
        return tuple(x_row.tolist())

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        class_counts = defaultdict(int)
        joint_counts = defaultdict(lambda: defaultdict(int))
        for i in range(n_samples):
            c = y[i]
            key = self._vector_to_key(X[i])
            class_counts[c] += 1
            joint_counts[key][c] += 1
        self.classes_ = np.array(sorted(class_counts.keys()))
        total_samples = float(n_samples)
        self.class_priors = {c: class_counts[c] / total_samples for c in self.classes_}
        self.joint_probs = {}
        for key, cdict in joint_counts.items():
            self.joint_probs[key] = {}
            for c in self.classes_:
                if class_counts[c] > 0:
                    self.joint_probs[key][c] = cdict.get(c, 0) / class_counts[c]
                else:
                    self.joint_probs[key][c] = 0.0

    def _predict_one(self, x_row):
        key = self._vector_to_key(x_row)
        scores = []
        for c in self.classes_:
            prior = self.class_priors[c]
            if key in self.joint_probs:
                cond = self.joint_probs[key].get(c, 0.0)
            else:
                cond = 0.0
            scores.append(prior * cond)
        scores = np.array(scores)
        if np.all(scores == 0):
            priors = np.array([self.class_priors[c] for c in self.classes_])
            return self.classes_[np.argmax(priors)]
        return self.classes_[np.argmax(scores)]

    def predict(self, X):
        X = np.asarray(X)
        preds = [self._predict_one(X[i]) for i in range(X.shape[0])]
        return np.array(preds, dtype=self.classes_.dtype)

class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = None
        self.feature_value_counts = None
        self.feature_value_domains = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.array(sorted(np.unique(y)))
        self.class_counts = {c: 0 for c in self.classes_}
        self.feature_value_counts = {c: [defaultdict(int) for _ in range(n_features)] for c in self.classes_}
        self.feature_value_domains = [set() for _ in range(n_features)]
        for i in range(n_samples):
            c = y[i]
            self.class_counts[c] += 1
            for j in range(n_features):
                v = int(X[i, j])
                self.feature_value_counts[c][j][v] += 1
                self.feature_value_domains[j].add(v)

    def _predict_one(self, x_row):
        x_row = x_row.astype(int)
        total_samples = sum(self.class_counts.values())
        best_c = None
        best_log_prob = -np.inf
        for c in self.classes_:
            if self.class_counts[c] == 0:
                continue
            log_p = np.log(self.class_counts[c] / total_samples)
            for j, v in enumerate(x_row):
                domain = self.feature_value_domains[j]
                k = len(domain)
                count_v = self.feature_value_counts[c][j].get(v, 0)
                log_p += np.log((count_v + 1) / (self.class_counts[c] + k))
            if log_p > best_log_prob:
                best_log_prob = log_p
                best_c = c
        return best_c

    def predict(self, X):
        X = np.asarray(X)
        preds = [self._predict_one(X[i]) for i in range(X.shape[0])]
        return np.array(preds, dtype=self.classes_.dtype)

class LinearRegressionClassifier:
    def __init__(self):
        self.w = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array(sorted(np.unique(y)))
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)
        self.w = XtX_inv @ X.T @ y

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds_real = X @ self.w
        preds = []
        for r in preds_real:
            diffs = np.abs(self.classes_ - r)
            preds.append(self.classes_[np.argmin(diffs)])
        return np.array(preds, dtype=self.classes_.dtype)

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    zoo_X, zoo_y = load_zoo("zoo/zoo.data")
    weather_X, weather_y = load_weather("WeatherAndPlayData.txt")

    zoo_splits = make_repeated_splits(zoo_X, zoo_y, n_repeats=10, test_ratio=0.2, ensure_all_classes_in_test=True)
    weather_splits = make_repeated_splits(weather_X, weather_y, n_repeats=10, test_ratio=0.2, ensure_all_classes_in_test=True)

    models = {
        "ExactBayes": ExactBayesClassifier,
        "NaiveBayes": NaiveBayesClassifier,
        "LinearRegression": LinearRegressionClassifier,
    }

    for dataset_name, splits in [("Zoo", zoo_splits), ("Weather", weather_splits)]:
        for model_name, ModelClass in models.items():
            accs = []
            for X_train, y_train, X_test, y_test in splits:
                model = ModelClass()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accs.append(accuracy_score(y_test, y_pred))
            accs = np.array(accs)
            print(dataset_name, model_name, "mean_acc", accs.mean(), "std", accs.std())
