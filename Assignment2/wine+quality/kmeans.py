import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_wine(path):
    df = pd.read_csv(path, sep=";")
    y = df["quality"].values
    X = df.drop(columns=["quality"]).values.astype(float)
    return X, y, df.columns.tolist()

def zscore(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return (X - mean) / std

def stra1_center(X, k):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    centers = np.random.uniform(mins, maxs, size=(k, X.shape[1]))
    return centers

def stra2_center(X, k): #quartile mean 4, but you said five quartiles
    quartile1 = np.percentile(X, 25, axis=0)
    quartile3 = np.percentile(X, 75, axis=0)
    centers = np.random.uniform(quartile1, quartile3, size=(k, X.shape[1]))
    return centers

def pairwise_distances_squared(A, B):
    A2 = (A**2).sum(axis=1, keepdims=True)
    B2 = (B**2).sum(axis=1, keepdims=True).T
    cross = A @ B.T
    return A2 + B2 - 2*cross

def kmeans(X, k, initialize_func, max_iter=300, tol=1e-6):
    centers = initialize_func(X, k).copy()
    prev_inertia = None

    for it in range(max_iter):
        d2 = pairwise_distances_squared(X, centers)
        labels = d2.argmin(axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = (labels == j)
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
            else:
                # pick random reinit
                new_centers[j] = X[np.random.randint(0, X.shape[0])]
        centers = new_centers

        inertia = float(np.sum((X - centers[labels])**2))

        if prev_inertia is not None and abs(prev_inertia - inertia) <= tol * (prev_inertia + 1e-12):
            break

        prev_inertia = inertia

    return centers, labels, inertia

def sse_calculate(X, ks, initialize_func, repeats=3): 
    sse_stds = []

    for k in ks:
        sses = []
        for _ in range(repeats):
            centers, labels, inertia = kmeans(X, k, initialize_func)
            sses.append(inertia)
        sse_means.append(np.mean(sses))
        sse_stds.append(np.std(sses))

    return np.array(sse_means), np.array(sse_stds)

def purity_score(y_true, y_pred):
    N = len(y_true)
    total = 0
    for c in np.unique(y_pred):
        mask = (y_pred == c)
        if mask.any():
            _, counts = np.unique(y_true[mask], return_counts=True)
            total += counts.max()
    return total / N

def pca_svd(X, n_components=2):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    Xproj = Xc @ comps.T
    return Xproj, comps, S

def plot_elbow(ks, means, stds, title, outpath):
    plt.figure()
    plt.errorbar(ks, means, yerr=stds, fmt='-o')
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_pca2d(Xproj, labels, title, outpath):
    plt.figure()
    for c in np.unique(labels):
        mask = (labels == c)
        plt.scatter(Xproj[mask,0], Xproj[mask,1], s=8, label=f"Cluster {c}", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def run_pipeline(X, y, dataset_name, strategy, ks, best_k=None):
    if strategy == 1:
        initialize_func = stra1_center
    else:
        initialize_func = stra2_center


    sse_mean, sse_std = sse_calculate(X, ks, initialize_func)
    elbow_path = f"elbow_{dataset_name}_strategy{strategy}.png"
    plot_elbow(ks, sse_mean, sse_std, f"Elbow - {dataset_name} (Strategy {strategy})", elbow_path)

    if best_k is None:
        rel_drops = []
        for i in range(1, len(ks)):
            drop = sse_mean[i-1] - sse_mean[i]
            rel = drop / sse_mean[i-1] if sse_mean[i-1] > 0 else 0
            rel_drops.append(rel)

        chosen = ks[-1]
        for i, rel in enumerate(rel_drops, start=1):
            if rel < 0.10:
                chosen = ks[i]
                break
        best_k = chosen

    centers, labels, inertia = kmeans(X, best_k, initialize_func)
    purity = purity_score(y, labels)

    Xproj, comps, S = pca_svd(X)
    pca_path = f"pca2d_{dataset_name}_strategy{strategy}_k{best_k}.png"
    plot_pca2d(Xproj, labels, f"PCA 2D - {dataset_name} (Strategy {strategy}, k={best_k})", pca_path)

    return {
        "dataset": dataset_name,
        "strategy": strategy,
        "best_k": best_k,
        "inertia": float(inertia),
        "purity": float(purity),
        "elbow_plot": elbow_path,
        "pca2d_plot": pca_path
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    RED_PATH = os.path.join(script_dir, "winequality-red.csv")
    WHITE_PATH = os.path.join(script_dir, "winequality-white.csv")

    X_red, y_red, _ = load_wine(RED_PATH)
    X_white, y_white, _ = load_wine(WHITE_PATH)

    Xred_z = zscore(X_red)
    Xwhite_z = zscore(X_white)

    ks = list(range(2, 13))
    results = []

    for dataset_name, X, y in [("red_wine", Xred_z, y_red), ("white_wine", Xwhite_z, y_white)]:
        for strategy in (1, 2):
            res = run_pipeline(X, y, dataset_name, strategy, ks)
            results.append(res)

    pd.DataFrame(results).to_csv("results_summary.csv", index=False)
    print(pd.DataFrame(results))

main()
