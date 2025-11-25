"""
cluster_utils.py
Reusable clustering utilities for process mining trace clustering.

This module provides:
- extraction of trace sequences from event logs
- text vectorization (TF-IDF, Doc2Vec)
- clustering methods (KMeans, SOM, HDBSCAN)
- estimation of optimal number of clusters
- cluster mining/summary computation
- silhouette trend visualization

All functions are designed to be modular and reusable across notebooks.
"""

# Standard library
import os
import random
from pathlib import Path

# Numerical / Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from IPython.display import display

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from minisom import MiniSom
from hdbscan import HDBSCAN


# ======================================================================
# Global configuration
# ======================================================================

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


# ======================================================================
# 1) Extract trace sequences from event logs
# ======================================================================
"""
Converts event-level logs into trace-level representations.
This is required because clustering operates at the trace level,
not the event level.

Output structure:
- One row per case
- A 'trace_str' column containing strings such as "A B C D"
- An optional 'variant' column for grouping infrequent variants
"""
def extract_trace_sequences(
    df,
    case_col="case:concept:name",
    activity_col="concept:name",
    timestamp_col="time:timestamp",
    min_variant_freq=1
):
    """
    Convert an event log DataFrame into a standardized sequences DataFrame
    containing:
        - case_id column (case:concept:name)
        - trace_str  (space-separated activity sequence)
        - variant    (filtered sequence or '__OTHER__')

    This function produces the same structure as the previous `filtered`
    DataFrame used in the process-mining pipeline.
    """

    # Ensure chronological order
    df = df.sort_values([case_col, timestamp_col])

    # Build trace strings
    seqs = (
        df.groupby(case_col)[activity_col]
        .apply(lambda s: " ".join(s.astype(str).tolist()))
        .rename("trace_str")
        .to_frame()
    )

    # Convert index → column (needed for PM4Py cluster evaluation)
    seqs = seqs.reset_index()

    # Variant frequency computation
    variant_counts = seqs["trace_str"].value_counts()

    if min_variant_freq > 1:
        common = set(variant_counts[variant_counts >= min_variant_freq].index)
        seqs["variant"] = seqs["trace_str"].apply(
            lambda x: x if x in common else "__OTHER__"
        )
    else:
        seqs["variant"] = seqs["trace_str"]

    return seqs




# ======================================================================
# 2) Vectorization
# ======================================================================
"""
Converts textual trace strings into numerical feature vectors.
Supports:
- TF-IDF with SVD (dimensionality reduction)
- Doc2Vec embeddings
- both at once

The resulting vectors can be used by clustering algorithms.
"""
def vectorize(sequences, method="both"):
    """
    Convert trace strings into TF-IDF/SVD or Doc2Vec embeddings.

    Returns:
    encoders  : dict mapping embedding name → numpy array
    artifacts : fitted vectorizers/models for reuse
    """
    encoders = {}
    artifacts = {}

    # ----- TF-IDF + SVD -----
    if method in ("both", "tfidf"):
        tfidf = TfidfVectorizer(
            token_pattern=r"[^ ]+",
            lowercase=False,
            ngram_range=(1, 3)
        )
        X_tfidf = tfidf.fit_transform(sequences)

        svd_components = max(min(10, X_tfidf.shape[1] - 1), 2)
        svd = TruncatedSVD(n_components=svd_components, random_state=GLOBAL_SEED)
        X_tfidf_svd = svd.fit_transform(X_tfidf)

        X_tfidf_std = StandardScaler().fit_transform(X_tfidf_svd)

        encoders["TFIDF_SVD"] = X_tfidf_std
        artifacts["tfidf"] = tfidf
        artifacts["svd"] = svd

    # ----- Doc2Vec -----
    if method in ("both", "doc2vec"):

        tagged_docs = [
            TaggedDocument(words=trace.split(), tags=[str(i)])
            for i, trace in enumerate(sequences)
        ]

        model = Doc2Vec(
            vector_size=100,
            min_count=1,
            epochs=40,
            seed=GLOBAL_SEED,
            workers=1
        )
        model.build_vocab(tagged_docs)
        model.train(
            tagged_docs,
            total_examples=model.corpus_count,
            epochs=model.epochs
        )

        X_doc2vec = np.array([
            model.infer_vector(trace.split(), epochs=40, alpha=0.025)
            for trace in sequences
        ])
        X_doc2vec_std = StandardScaler().fit_transform(X_doc2vec)

        encoders["DOC2VEC"] = X_doc2vec_std
        artifacts["doc2vec_model"] = model

    return encoders, artifacts



# ======================================================================
# 3) Clustering evaluation metrics
# ======================================================================
"""
Provides standard internal clustering quality metrics.
Used for comparing different clusterings and selecting the best K.
"""
def evaluate_clustering(X, labels):
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) == len(labels):
        return {"silhouette": np.nan, "ch": np.nan, "db": np.nan}

    return {
        "silhouette": silhouette_score(X, labels),
        "ch": calinski_harabasz_score(X, labels),
        "db": davies_bouldin_score(X, labels)
    }



# ======================================================================
# 4) Sweeps over K (KMeans and SOM)
# ======================================================================
"""
Sweeps across K values (e.g., 2–15) and identifies the best
number of clusters based on silhouette score.

This separates:
- running algorithms
- evaluating algorithms
- selecting the best K
"""
def run_kmeans_sweep(X, k_values):
    best = None
    results = []

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=20, random_state=GLOBAL_SEED)
        labels = km.fit_predict(X)
        metrics = evaluate_clustering(X, labels)

        result = {"k": k, "model": km, "labels": labels, **metrics}
        results.append(result)

        if best is None or metrics["silhouette"] > best["silhouette"]:
            best = result

    return best, results



def run_som_no_bmu(X, n_clusters, xdim=5, ydim=5, sigma=1.0, lr=0.5, iters=5000):
    """
    Trains a Self-Organizing Map (SOM) and clusters its neuron weights.

    Steps:
    1) normalize input
    2) train SOM grid
    3) extract neuron weights
    4) cluster neurons with KMeans
    5) assign each trace to its nearest neuron
    """
    X_scaled = MinMaxScaler().fit_transform(X)

    som = MiniSom(
        x=xdim, y=ydim,
        input_len=X_scaled.shape[1],
        sigma=sigma,
        learning_rate=lr,
        random_seed=GLOBAL_SEED
    )
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, iters)

    weights = som.get_weights().reshape(-1, X_scaled.shape[1])

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=GLOBAL_SEED)
    neuron_labels = km.fit_predict(weights)

    dists = np.linalg.norm(X_scaled[:, None, :] - weights[None, :, :], axis=2)
    closest = dists.argmin(axis=1)

    return neuron_labels[closest]



def run_som_sweep(X, k_values):
    """Same purpose as the KMeans sweep but using SOM-based clustering."""
    best = None
    results = []

    for k in k_values:
        labels = run_som_no_bmu(X, k)
        metrics = evaluate_clustering(X, labels)

        result = {"k": k, "labels": labels, **metrics}
        results.append(result)

        if best is None or metrics["silhouette"] > best["silhouette"]:
            best = result

    return best, results



# ======================================================================
# 5) HDBSCAN
# ======================================================================
"""
Density-based clustering that automatically determines
the number of clusters.

num_clusters is ignored because HDBSCAN does not use K.
"""
def run_hdbscan(X, min_cluster_size=5):
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom"
    )

    labels = hdb.fit_predict(X)
    metrics = evaluate_clustering(X, labels)

    return {
        "model": hdb,
        "labels": labels,
        "k": len(np.unique(labels[labels >= 0])),
        **metrics
    }



# ======================================================================
# 6) Unified clustering interface
# ======================================================================
"""
Provides a simple, consistent API for:
- choosing the number of clusters
- performing clustering
- working with multiple algorithms in the same workflow
"""
def find_num_clusters(X, k_values, cluster_algo="kmeans"):
    """Unified interface for selecting K."""
    if cluster_algo == "kmeans":
        return run_kmeans_sweep(X, k_values)
    elif cluster_algo == "som":
        return run_som_sweep(X, k_values)
    else:
        raise ValueError(f"Unsupported algorithm: {cluster_algo}")



def cluster_traces(X, num_clusters=None, cluster_algo="kmeans"):
    """
    Unified interface for performing clustering.

    Behavior:
    - KMeans and SOM require num_clusters
    - HDBSCAN ignores num_clusters
    """
    if cluster_algo == "kmeans":
        if num_clusters is None:
            raise ValueError("num_clusters is required for KMeans.")
        km = KMeans(n_clusters=num_clusters, n_init=20, random_state=GLOBAL_SEED)
        labels = km.fit_predict(X)
        return labels, km

    elif cluster_algo == "som":
        if num_clusters is None:
            raise ValueError("num_clusters is required for SOM.")
        labels = run_som_no_bmu(X, num_clusters)
        return labels, None

    elif cluster_algo == "hdbscan":
        if num_clusters is not None:
            print("⚠️ Warning: num_clusters is ignored for HDBSCAN.")
        hdb = HDBSCAN(min_cluster_size=5, min_samples=5)
        labels = hdb.fit_predict(X)
        return labels, hdb

    else:
        raise ValueError(f"Unsupported algorithm: {cluster_algo}")



# ======================================================================
# 7) Cluster mining / summarization
# ======================================================================
"""
Computes descriptive statistics for each cluster:
- cluster size
- percentage of total traces
- trace length distributions
- variant entropy (behavioral variability)
- number of unique variants

Useful for understanding the behavior of each cluster.
"""
def mine_from_clusters(labels, num_clusters=None, sequences_df=None):
    """
    Compute cluster-level statistics from labels and the sequences dataframe.

    Parameters
    ----------
    labels : array-like
        Cluster labels for each trace.
    num_clusters : int, optional
        Unused, but kept for API consistency.
    sequences_df : pd.DataFrame
        Must contain:
        - 'trace_str'
        - 'variant'

    Returns
    -------
    pd.DataFrame
        Per-cluster summary statistics.
    """

    if sequences_df is None:
        raise ValueError("sequences_df must be provided.")

    tmp = sequences_df.copy()
    tmp["_cluster_tmp"] = labels

    rows = []
    for c, g in tmp.groupby("_cluster_tmp"):
        n = len(g)
        lengths = g["trace_str"].str.split().map(len)

        rows.append({
            "cluster": int(c),
            "num_traces": n,                          # renamed from 'size'
            "pct": round(100 * n / len(tmp), 2),
            "avg_len": float(lengths.mean()),
            "median_len": float(lengths.median()),
            "unique_variants": g["variant"].nunique()
        })

    return pd.DataFrame(rows).sort_values("cluster", ascending=True)





# ======================================================================
# 8) Silhouette trend plotting
# ======================================================================
"""
Plots silhouette score vs. K.
This helps visualize and justify the choice of the number of clusters.
"""
def plot_silhouette_trend(results, title):
    k_vals = [r["k"] for r in results if not np.isnan(r["silhouette"])]
    sil_vals = [r["silhouette"] for r in results if not np.isnan(r["silhouette"])]

    if len(k_vals) == 0:
        print(f"No valid silhouette scores for {title}.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, sil_vals, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.show()
