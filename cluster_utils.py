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
        .apply(lambda s: " ".join(s.apply(lambda a: a.replace(" ", "_"))))
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

    # PM4Py downstream requires case IDs as index
    seqs = seqs.set_index("case:concept:name")

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
    if method in ("both", "tfidf_svd"):
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

        encoders["tfidf_svd"] = X_tfidf_std
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

        encoders["doc2vec"] = X_doc2vec_std
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

# ======================================================================
# 9) Global Process Discovery (PM4Py)
# ======================================================================
"""
This section provides a full global process discovery pipeline:
- Convert event log → PM4Py EventLog
- Discover model using Inductive / Heuristics / Alpha miner
- Visualize Petri nets, BPMN, or Heuristics Nets
- Evaluate fitness + precision + F-score
- Compute variant variability
"""

# -----------------------
# Required Imports
# -----------------------
# PM4Py evaluation
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness.algorithm import Variants as FitnessVariants
from pm4py.algo.evaluation.precision.algorithm import Variants as PrecisionVariants


# Log conversion utilities
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

# Fitness (token-based replay) and precision
from pm4py.algo.evaluation.replay_fitness.variants import token_replay as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.evaluation.replay_fitness.variants import token_replay as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

# This section MUST be ran before Process discovery
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.heuristics_net import converter as hn_converter

from pm4py.algo.evaluation.replay_fitness.algorithm import Variants as FitnessVariants
from pm4py.algo.evaluation.precision.algorithm import Variants as PrecisionVariants

from pm4py.visualization.petri_net import visualizer as pn_visualizer

from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

# -----------------------
# Settings
# -----------------------
MINER_TYPE = "inductive"   # "inductive", "alpha", "heuristics"
RANDOM_STATE = 42
EVAL_SAMPLE_SIZE = 2000     # Sample size for conformance evaluation
FITNESS_VARIANT = FitnessVariants.TOKEN_BASED
PRECISION_VARIANT = PrecisionVariants.ETCONFORMANCE_TOKEN


# ======================================================================
# 9.1) Miner Selection Wrapper
# ======================================================================
def discover_model_for_miner(log):
    """
    Return: net, im, fm, heuristics_net, process_tree
    """
    if MINER_TYPE == "inductive":
        process_tree = inductive_miner.apply(log)
        net, im, fm = pt_converter.apply(process_tree)
        return net, im, fm, None, process_tree

    elif MINER_TYPE == "alpha":
        net, im, fm = alpha_miner.apply(log)
        return net, im, fm, None, None

    elif MINER_TYPE == "heuristics":
        heu_net = heuristics_miner.apply_heu(log)
        net, im, fm = hn_converter.apply(heu_net)
        return net, im, fm, heu_net, None

    else:
        raise ValueError(f"❌ Unknown MINER_TYPE: {MINER_TYPE}")


# ======================================================================
# 9.2) Sampling Utility
# ======================================================================
def maybe_sample_log(event_log, max_traces):
    """Return sampled log if too large."""
    if max_traces is None or len(event_log) <= max_traces:
        return event_log
    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(event_log), size=max_traces, replace=False
    )
    return EventLog([event_log[i] for i in sorted(idx)])


# ======================================================================
# 9.3) Variability Measure
# ======================================================================
def compute_variability_ratio(log):
    """Unique variants / total traces."""
    if len(log) == 0:
        return 0.0
    variant_set = set()
    for trace in log:
        seq = tuple(e["concept:name"] for e in trace)
        variant_set.add(seq)
    return len(variant_set) / len(log)


# ======================================================================
# 9.4) Global Model Discovery + Evaluation
# ======================================================================
def evaluate_global_model(df, metrics_df=None):
    """
    Full global model discovery → visualization → evaluation.
    """

    # --- Convert Pandas → PM4Py EventLog ---
    params = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY:
              "case:concept:name"}
    global_log = log_converter.apply(
        df, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params
    )

    # --- Discovery ---
    net, im, fm, heu_net, process_tree = discover_model_for_miner(global_log)

    # --- Visualization ---
    if MINER_TYPE == "inductive":
        print("Rendering BPMN...")
        bpmn_graph = process_tree_converter.apply(
            process_tree,
            variant=process_tree_converter.Variants.TO_BPMN
        )
        gviz = bpmn_visualizer.apply(bpmn_graph)
        bpmn_visualizer.view(gviz)

    elif MINER_TYPE == "heuristics":
        print("Rendering Heuristics Net...")
        gviz = hn_visualizer.apply(heu_net)
        hn_visualizer.view(gviz)

    # Alpha miner uses PN visualizer automatically
    else:
        print("Rendering Petri Net...")
        gviz = pn_visualizer.apply(net, im, fm)
        pn_visualizer.view(gviz)

    # --- Conformance Checking ---
    eval_log = maybe_sample_log(global_log, EVAL_SAMPLE_SIZE)
    fit_res = fitness_evaluator.apply(eval_log, net, im, fm,
                                      variant=FITNESS_VARIANT)
    fitness = fit_res.get(
        "average_trace_fitness", fit_res.get("perc_fit_traces", np.nan)
    )
    precision = precision_evaluator.apply(eval_log, net, im, fm,
                                          variant=PRECISION_VARIANT)
    fscore = 2 * (precision * fitness) / (precision + fitness) if (
        precision + fitness) > 0 else 0

    # Variability
    global_variability = compute_variability_ratio(global_log)

    # --- Build Metrics Row ---
    row = {
        "Method": "Global",
        "Cluster": "Global",
        "NumTraces": len(global_log),
        "Precision": float(precision),
        "Fitness": float(fitness),
        "FScore": float(fscore),
        "VariabilityRatio": float(global_variability)
    }

    # Add row to metrics table
    if metrics_df is None or len(metrics_df) == 0:
        metrics_df = pd.DataFrame([row])
    else:
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)

    print(
        f"🌐 Global Model ({MINER_TYPE}) → "
        f"Precision: {precision:.3f}, Fitness: {fitness:.3f}, "
        f"F-score: {fscore:.3f}, Variability Ratio: {global_variability:.3f}"
    )

    return metrics_df


# =============================================================
# Per‑Cluster Process Discovery & Evaluation (Annotated Version)
# =============================================================
# This script mirrors the global process discovery workflow but applied
# separately to each cluster. For every cluster, we:
# 1. Extract all traces belonging to that cluster
# 2. Convert them to an EventLog
# 3. Discover a process model (IMf version of Inductive Miner)
# 4. Optionally visualize the model (BPMN)
# 5. Compute conformance (precision, fitness, F-score)
# 6. Compute variability ratio
# 7. Append results to a cluster‑level metrics table


# =============================================================
# 0) Discovery Function — IMf Variant
# =============================================================
# IMf = Inductive Miner (infrequent) — a *more flexible* configuration.
# It captures more behavioral detail (higher fitness) at the cost of more
# complex / less generalizable models.

# Needed for DFG visualization
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer
# Needed for process tree visualization 
from pm4py.visualization.process_tree import visualizer as pt_visualizer


# ======================================================================
# 10.1) IMf Miner Configurations (Balanced / Strict / Flexible)
# ======================================================================
def discover_model_for_miner_imf_balanced(log):
    """Balanced IMf miner."""
    tree = inductive_miner.apply(
        log,
        variant=inductive_miner.Variants.IMf,
        parameters={"noise_threshold": 0.2}
    )
    net, im, fm = pt_converter.apply(tree)
    return net, im, fm, tree


def discover_model_for_miner_imf_strict(log):
    """Strict IMf miner."""
    tree = inductive_miner.apply(
        log,
        variant=inductive_miner.Variants.IMf,
        parameters={
            "noise_threshold": 0.4,
            "min_dfg_occurrences": 2
        }
    )
    net, im, fm = pt_converter.apply(tree)
    return net, im, fm, tree


def discover_model_for_miner_imf_flexible(log):
    """Flexible IMf miner."""
    tree = inductive_miner.apply(
        log,
        variant=inductive_miner.Variants.IMf,
        parameters={
            "noise_threshold": 0.1,
            "min_dfg_occurrences": 1
        }
    )
    net, im, fm = pt_converter.apply(tree)
    return net, im, fm, tree


# ======================================================================
# 10.2) Per-Cluster Discovery + Evaluation
# ======================================================================
def discover_and_evaluate_per_cluster(
    df: pd.DataFrame,
    filtered: pd.DataFrame,
    cluster_col: str,
    method_name: str = None,
    cluster_metrics_df: pd.DataFrame = None,
    skip_noise: bool = True,
    noise_label: int = -1,
    visualize: bool = False,
    model_type: str = "bpmn",  # "bpmn", "pn", "dfg", "tree"
    miner_variant: str = "balanced",  # NEW: choose IMf mode
):

    method_name = method_name or cluster_col

    # Ensure filtered has case IDs as index
    if filtered.index.name != "case:concept:name":
        if "case:concept:name" in filtered.columns:
            filtered = filtered.set_index("case:concept:name")
        else:
            raise ValueError("`filtered` must have case ids on the index.")

    if cluster_col not in filtered.columns:
        raise ValueError(f"{cluster_col} missing in filtered: {filtered.columns}")

    # Cluster iteration
    for c, case_ids in filtered.groupby(cluster_col).groups.items():

        if skip_noise and c == noise_label:
            continue

        cluster_df = df[df["case:concept:name"].isin(case_ids)].copy()
        if cluster_df.empty:
            continue

        # Standard timestamp normalization
        cluster_df = dataframe_utils.convert_timestamp_columns_in_df(cluster_df)
        cluster_df = cluster_df.sort_values(
            ["case:concept:name", "time:timestamp"],
            ignore_index=True
        )

        params = {
            "case_id":       "case:concept:name",
            "activity_key":  "concept:name",
            "timestamp_key": "time:timestamp",
        }

        log = log_converter.apply(
            cluster_df,
            variant=log_converter.Variants.TO_EVENT_LOG,
            parameters=params
        )

        # -------------------------------
        # IMf Model Discovery
        # -------------------------------
        if miner_variant == "balanced":
            net, im, fm, tree = discover_model_for_miner_imf_balanced(log)
        elif miner_variant == "strict":
            net, im, fm, tree = discover_model_for_miner_imf_strict(log)
        elif miner_variant == "flexible":
            net, im, fm, tree = discover_model_for_miner_imf_flexible(log)
        else:
            raise ValueError("Invalid miner_variant: choose balanced/strict/flexible.")

        eval_log = log

        # -------------------------------
        # Conformance Evaluation (same API as global)
        # -------------------------------
        fit_res = fitness_evaluator.apply(eval_log, net, im, fm, variant=FitnessVariants.TOKEN_BASED)
        fitness = fit_res.get("average_trace_fitness", fit_res.get("log_fitness", np.nan))

        precision = precision_evaluator.apply(eval_log, net, im, fm, variant=PrecisionVariants.ETCONFORMANCE_TOKEN)

        if precision is not None and fitness is not None and (precision + fitness) > 0:
            fscore = 2 * precision * fitness / (precision + fitness)
        else:
            fscore = 0.0

        variability = compute_variability_ratio(log)

        # -------------------------------
        # Append metrics
        # -------------------------------
        row = {
            "Method":           method_name,
            "Cluster":          f"Cluster {c}",
            "NumTraces":        len(log),
            "Precision":        float(precision),
            "Fitness":          float(fitness),
            "FScore":           float(fscore),
            "VariabilityRatio": float(variability),
            "Miner":            f"IMf-{miner_variant}",
        }

        if cluster_metrics_df is None:
            cluster_metrics_df = pd.DataFrame([row])
        else:
            cluster_metrics_df = pd.concat([cluster_metrics_df, pd.DataFrame([row])], ignore_index=True)

        print(
            f"🔹 {method_name} – Cluster {c}: "
            f"P={precision:.3f}, F={fitness:.3f}, F1={fscore:.3f}, Var={variability:.3f}, "
            f"Traces={len(log)}"
        )

        # -------------------------------
        # Visualization
        # -------------------------------
        if visualize:
            mt = model_type.lower()

            if mt == "bpmn":
                graph = process_tree_converter.apply(tree, variant=process_tree_converter.Variants.TO_BPMN)
                gviz = bpmn_visualizer.apply(graph)
                bpmn_visualizer.view(gviz)

            elif mt == "pn":
                gviz = pn_visualizer.apply(net, im, fm)
                pn_visualizer.view(gviz)

            elif mt == "dfg":
                dfg = dfg_discovery.apply(log)
                gviz = dfg_visualizer.apply(dfg, log=log)
                dfg_visualizer.view(gviz)

            elif mt == "tree":
                gviz = pt_visualizer.apply(tree)
                pt_visualizer.view(gviz)

            else:
                raise ValueError("model_type must be: bpmn, pn, dfg, tree.")

    return cluster_metrics_df
