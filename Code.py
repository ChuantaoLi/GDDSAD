import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import lil_matrix, diags
import time
import warnings
import os

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_keel_dat(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('@')]
    data = [line.split(',') for line in lines if line]
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values
    y = df.iloc[:, -1].astype('category').cat.codes.values
    if len(np.unique(y)) != 2:
        raise ValueError("NOT BINARY CLASSIFICATION")
    return X, y


def calculate_gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sens * spec)


class OverSample:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def _determine_clusters(self, X):
        min_clusters = 2
        max_clusters = min(10, len(X) - 1)
        best_bic = np.inf
        best_n = 2
        for n in range(min_clusters, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        return best_n

    def _generate_samples(self, X_min, clusters, n_samples):
        synthetic = []
        rng = np.random.default_rng(self.random_state)
        for cluster_id in np.unique(clusters):
            cluster_data = X_min[clusters == cluster_id]
            if len(cluster_data) < 2:
                continue
            cluster_ratio = len(cluster_data) / len(X_min)
            n_cluster_samples = int(n_samples * cluster_ratio)
            for _ in range(n_cluster_samples):
                i1, i2 = rng.choice(len(cluster_data), 2, replace=False)
                alpha = rng.uniform()
                new_point = cluster_data[i1] + alpha * (cluster_data[i2] - cluster_data[i1])
                synthetic.append(new_point)
        if len(synthetic) == 0:
            return np.empty((0, X_min.shape[1]))
        return np.array(synthetic)[:n_samples]

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y)
        classes = np.unique(y)
        majority_class = max(classes, key=lambda c: np.sum(y == c))
        minority_class = min(classes, key=lambda c: np.sum(y == c))
        X_min, X_maj = X[y == minority_class], X[y == majority_class]
        n_to_generate = len(X_maj) - len(X_min)
        if n_to_generate <= 0:
            return X, y
        n_clusters = self._determine_clusters(X_min)
        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        clusters = gmm.fit_predict(X_min)
        synthetic_X = self._generate_samples(X_min, clusters, n_to_generate)
        if len(synthetic_X) == 0:
            return X, y
        synthetic_y = np.full(len(synthetic_X), minority_class)
        return np.vstack([X, synthetic_X]), np.concatenate([y, synthetic_y])


def construct_graph(X, y, k=5, rho=1.0):
    n = len(X)
    G = lil_matrix((n, n))

    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        X_class = X[idx]

        if len(idx) > 1:
            knn_graph = kneighbors_graph(
                X_class, n_neighbors=min(k, len(idx) - 1),
                mode='distance', include_self=False
            )
            knn_graph.data = np.exp(-(knn_graph.data ** 2) / rho)
            sym_graph = knn_graph.maximum(knn_graph.T)

            rows, cols = sym_graph.nonzero()
            G[idx[rows], idx[cols]] = sym_graph.data

    G = G.tocsr()
    D = diags(G.sum(axis=1).A.ravel(), format='csr')
    L = D - G
    return G, L


def compute_lda_matrices(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    Sb = np.zeros((n_features, n_features))
    Sw = np.zeros((n_features, n_features))

    overall_mean = np.mean(X, axis=0)

    for c in classes:
        X_c = X[y == c]
        n_c = X_c.shape[0]
        mean_c = np.mean(X_c, axis=0)

        Sb += n_c * np.outer((mean_c - overall_mean), (mean_c - overall_mean))

        Sw += (X_c - mean_c).T @ (X_c - mean_c)

    return Sb, Sw


def lda_enhanced_projection(X, y, L, alpha=0.5):
    Sb, Sw = compute_lda_matrices(X, y)

    if L.shape[0] < 1000:
        XLX = X.T @ L.toarray() @ X
    else:
        XLX = X.T @ (L @ X)

    A = alpha * XLX + (1 - alpha) * Sw
    B = alpha * np.eye(X.shape[1]) + (1 - alpha) * Sb

    eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(A) @ B)

    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    max_dim = len(np.unique(y)) - 1
    return eigvecs[:, :max_dim]


def stump_classify(data_matrix, dim, thresh, inequal):
    ret = np.ones(data_matrix.shape[0])
    if inequal == 'lt':
        ret[data_matrix[:, dim] <= thresh] = -1.0
    else:
        ret[data_matrix[:, dim] > thresh] = -1.0
    return ret


def build_stump(X, y, D):
    m, n = X.shape
    best_stump = {}
    best_pred = np.zeros(m)
    min_err = np.inf

    for dim in range(n):
        for thresh in np.unique(X[:, dim]):
            for inequal in ('lt', 'gt'):
                pred = stump_classify(X, dim, thresh, inequal)
                err = (pred != y).astype(float)
                weighted_err = np.dot(D, err)
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_stump = {'dim': dim, 'thresh': thresh, 'ineq': inequal}
                    best_pred = pred.copy()
    return best_stump, min_err, best_pred


def compute_group_weights(H, iteration, num_iter, eta=1):
    mu_H = np.mean(H)
    sigma_H = np.std(H)

    lower_bound = max(0, mu_H - sigma_H)
    upper_bound = min(1, mu_H + sigma_H)

    bins = [0, lower_bound, upper_bound, 1.0]
    groups = np.digitize(H, bins) - 1
    groups = np.clip(groups, 0, 2)

    C_bar = np.zeros(3)
    for j in range(3):
        mask = (groups == j)
        if np.any(mask):
            C_bar[j] = H[mask].mean()

    i, n = iteration, num_iter
    alpha = np.tan((i - 1) * np.pi / (2 * n))
    beta = 1 / (1 + np.exp(-eta * ((2 * i / n) - 1)))

    delta = np.zeros(3)
    for j in range(3):
        if (C_bar[j] < lower_bound) or (C_bar[j] > upper_bound):
            delta[j] = alpha
        else:
            delta[j] = beta * alpha

    w = 1.0 / (C_bar + delta + 1e-8)
    return groups, w / w.sum()


def ada_boost_train_dynamic(X, y, num_iter=30, random_state=42, k=5):
    classes = np.unique(y)
    pos, neg = classes[0], classes[1]
    y_enc = np.where(y == pos, 1, -1)
    label_map = {1: pos, -1: neg}
    m = X.shape[0]
    agg_est = np.zeros(m)
    classifiers, betas = [], []

    minority_class = classes[np.argmin(np.bincount(y))]
    majority_class = classes[np.argmax(np.bincount(y))]
    N_minority = sum(y == minority_class)
    N_majority = sum(y == majority_class)
    imbalance_ratio = N_majority / N_minority

    for i in range(1, num_iter + 1):
        p = 1 / (1 + np.exp(-agg_est))
        P_correct = np.where(y_enc == 1, p, 1 - p)
        P_wrong = 1 - P_correct
        H = 1 - (P_correct - P_wrong)
        H = H / 2.0

        groups, w_norm = compute_group_weights(H, i, num_iter)

        Ntarget = int(N_minority * 1.2) if imbalance_ratio > 4 else N_minority

        maj_idx = np.where(y_enc == -1)[0]
        min_idx = np.where(y_enc == 1)[0]
        maj_groups = groups[maj_idx]
        sampled_maj = []

        for j in range(3):
            group_idx = maj_idx[maj_groups == j]
            Nj = int(Ntarget * w_norm[j])
            if Nj > 0 and len(group_idx) > 0:
                sampled = np.random.choice(
                    group_idx, Nj,
                    replace=len(group_idx) < Nj
                )
                sampled_maj.extend(sampled)

        if len(sampled_maj) < Ntarget:
            remaining = np.setdiff1d(maj_idx, sampled_maj)
            sampled_maj.extend(np.random.choice(
                remaining, Ntarget - len(sampled_maj),
                replace=True
            ))

        X_group = np.vstack([X[sampled_maj], X[min_idx]])
        y_group = np.concatenate([y_enc[sampled_maj], y_enc[min_idx]])

        overSampler = OverSample(random_state=random_state)
        try:
            X_res, y_res = overSampler.fit_resample(X_group, y_group)
        except ValueError:
            X_res, y_res = X_group, y_group

        G, L = construct_graph(X_res, y_res, k=k, rho=1)
        P = lda_enhanced_projection(X_res, y_res, L, alpha=0.5)
        X_proj = X_res @ P
        X_full_proj = X @ P

        D_sub = np.ones(len(y_res)) / len(y_res)
        stump, error, _ = build_stump(X_proj, y_res, D_sub)
        error = max(error, 1e-16)
        beta = 0.5 * np.log((1 - error) / error)

        classifiers.append({**stump, 'beta': beta, 'P': P})
        betas.append(beta)

        pred = stump_classify(X_full_proj, stump['dim'], stump['thresh'], stump['ineq'])
        agg_est += beta * pred

    return classifiers, betas, label_map


def ada_classify(X, classifiers, betas, label_map):
    agg = np.zeros(X.shape[0])
    for clf in classifiers:
        P = clf['P']
        X_proj = X @ P
        pred = stump_classify(X_proj, clf['dim'], clf['thresh'], clf['ineq'])
        agg += clf['beta'] * pred
    pred_enc = np.sign(agg)
    return np.where(pred_enc == 1, label_map[1], label_map[-1])


def run_repeated_holdout(X, y, random_state=42, repeat=5, test_size=0.3, k=5):
    metrics = {'gmean': [], 'auc': []}

    for i in range(repeat):
        rs = random_state + i
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=rs, stratify=y
        )
        classifiers, betas, label_map = ada_boost_train_dynamic(
            X_train, y_train, random_state=rs, k=k)
        preds = ada_classify(X_test, classifiers, betas, label_map)

        metrics['gmean'].append(calculate_gmean(y_test, preds))
        metrics['auc'].append(roc_auc_score(y_test, preds))

        print(f"[Run {i + 1}] GMean={metrics['gmean'][-1]:.4f}, AUC={metrics['auc'][-1]:.4f}")

    print("\n=== Final Results ===")
    results = {
        k: (np.mean(v), np.std(v))
        for k, v in metrics.items()
    }
    for k in metrics:
        print(f"{k.title()}: {results[k][0]:.3f}±{results[k][1]:.3f}")
    return results


if __name__ == '__main__':
    '''Experiment1'''
    data = pd.read_csv('Experiment1/7Ydata.csv').dropna()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    start_time = time.time()
    results = run_repeated_holdout(X, y, k=3)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.4f} seconds")

    '''Experiment2'''
    datasets = [
        {'name': 'ecoli-0_vs_1'}, {'name': 'ecoli1'}, {'name': 'ecoli2'},
        {'name': 'ecoli3'}, {'name': 'haberman'}, {'name': 'yeast3'}
    ]

    all_results = {}


    def format_result(values):
        return f"{np.mean(values):.3f}±{np.std(values):.3f}"


    k = 5

    for dataset in datasets:
        print(f'\n==== Processing Dataset: {dataset["name"]} ====')
        data_folder = f'Experiment2/{dataset["name"]}-5-fold'
        ggmean, aauc = [], []

        for fold in range(1, 6):
            try:
                train_file = os.path.join(data_folder, f'{dataset["name"]}-5-{fold}tra.dat')
                test_file = os.path.join(data_folder, f'{dataset["name"]}-5-{fold}tst.dat')

                X_train, y_train = load_keel_dat(train_file)
                X_test, y_test = load_keel_dat(test_file)

                classifiers, betas, label_map = ada_boost_train_dynamic(X_train, y_train, k=k)
                preds = ada_classify(X_test, classifiers, betas, label_map)

                gmean = calculate_gmean(y_test, preds)
                auc = roc_auc_score(y_test, preds)

                ggmean.append(gmean)
                aauc.append(auc)
                print(f'Fold {fold}: G-Mean={gmean:.4f}, AUC={auc:.4f}')

            except Exception as e:
                print(f"Error processing {dataset['name']} fold {fold}: {str(e)}")
                ggmean.append(0)
                aauc.append(0)

        all_results[dataset["name"]] = {
            'G-Mean': ggmean,
            'AUC': aauc
        }

    print('\n' + '=' * 50)
    print('Final Results (Mean±Std, 3 decimal places)')
    print('=' * 50)

    results_table = []
    for name, res in all_results.items():
        results_table.append({
            'Dataset': name,
            'G-Mean': format_result(res['G-Mean']),
            'AUC': format_result(res['AUC'])
        })

    df = pd.DataFrame(results_table)
    print(df.to_string(index=False))
