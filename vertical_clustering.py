# -*- coding: utf-8 -*-
####--------------SECTION 0 - ENVIRONMENTAL SETUP-------------------------
import os # Used to create a structured output directory per abstraction level "k"
from dataclasses import dataclass   # Defines confirguration objects
from typing import Dict, List, Tuple # Helps in distinguising between event classes, macro activites, and abstraction levels(k)

import numpy as np # Supports efficient matrix operation for event-class corellation matrix
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram # Used to visualize event-class ehirarchy(dendogram)


# ========= pm4py Imports ==========
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as process_tree_converter
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer

# ========= Clustering ==========
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import MinMaxScaler

# ========= Semantic Naming (optional) ==========
from sklearn.feature_extraction.text import TfidfVectorizer

'''
These constants define:
- how cases are identified,
- how event order is determined,
- and which column represents the event class.

This is crucial because:
- Günther’s correlation computation assumes ordered traces,
- and clustering is performed over event classes, not arbitrary attributes.
'''
# ========= Constants ==========
CASE_COL = "case:concept:name"
TS_COL = "time:timestamp"
ACTIVITY_COL = "activity"
CONCEPT_COL = "concept:name"

####--------------SECTION 1 - DATA LOADING AND CONFIGURATION-------------------------
def load_ircc_csv(path: str) -> pd.DataFrame:
    """Load and sort an IRCC CSV dataset by case then timestamp.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Ordered event log.
    """
    df = pd.read_csv(path)
    df[TS_COL] = pd.to_datetime(df[TS_COL])   # Convert timestamp to datemime
    return df.sort_values([CASE_COL, TS_COL]).reset_index(drop=True)  # Traces are sorted by case number and timestamp

def ensure_output_dirs(base_dir: str):
    """
    --------------------Parameters--------------------
    base_dir: str
    The root output directory for a single abstraction level (e.g. level_k_12/).
    --------------------------------------------------

    Create output folder structure.
    """
    for sub in ["bpmn_global", "bpmn_macro", "bpmn_activity_lifecycle"]: # Create a fixed foler structure under base_dir
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

# ============================================================
# Parameters that control abstraction behavior
# ============================================================
@dataclass
class SegmentationConfig:
    """
    window_size: look-back window size (Paper)
    attenuation: Events that occur close together contribute more to correlation than those far apart
    linkage_method: SciPy linkage method (paper used complete-linkage conceptually;
                    original code used average; keep average by default)
    """
    window_size: int = 6
    attenuation: float = 0.5
    linkage_method: str = "complete"  

####--------------SECTION 2 - EVENT CLASS CORRELATION (GLOBAL SCAN)-------------------------
'''
Computes global event-class proximity based on co-occurrence in ordered traces; captures behavioral closeness but 
deliberately leaves control-flow direction to the discovery phase.
'''
def compute_event_class_correlation(
    df: pd.DataFrame,
    label_col: str,
    cfg: SegmentationConfig
) -> Tuple[np.ndarray, List[str]]:
    """
    --------------------Parameters----------------------------------------
    df: pd.DataFrame
    The raw event log, before any abstraction. This is scanned in the global scanning pass
    
    label_col: str
    The column defining the event-class alphabet used for correlation. Correlations are computer between event classes, not 
    event instances, changing this column changes what is being clustered
    
    cfg: SegmentationConfig
    A configuration object bundling correlation parameters: window_sizem attenuation, and linkage_method
    ----------------------------------------------------------------------------------------------------
    Compute co-occurrence correlations between event classes using:
    - look-back window
    - exponential attenuation
    
    weight = attenuation^distance
    
    Returns (corr_matrix_normalized, classes)
    
    This function provides behavioral evidence used for abstraction
    """

    # ---------Guards for parameter validity (Option A)---------
    if cfg.window_size < 0:
        raise ValueError("window_size must be >= 0")
    if not (0 < cfg.attenuation <= 1):
        raise ValueError("attenuation must be in (0,1]")
    # ----------------------------------------------------------

    classes = sorted(df[label_col].dropna().unique().tolist())  # Extract event classes (activity labels)

    # ---------Guard for empty / all-NaN label column ----------
    # If there are no valid event classes, return an empty correlation structure cleanly.
    if len(classes) == 0:
        return np.zeros((0, 0), dtype=float), []
    # ----------------------------------------------------------

    idx = {c: i for i, c in enumerate(classes)}  # Stable index mapping for the correlation matrix
    
    n = len(classes)  # Defines dimensionality of the correlation matrix
    corr = np.zeros((n, n), dtype=float)  # Initializes the event-class correlation matrix

    w = cfg.window_size  # sliding window
    a = cfg.attenuation  # attenuation

    # -------A Global scan across all traces (Paper: scanning pass)--------
    for _, group in df.groupby(CASE_COL):
        # Orders events within each trace, then filters out NaN labels so idx[...] never sees NaN
        events = group.sort_values(TS_COL)[label_col].tolist()
        events = [e for e in events if pd.notna(e)]  # <-- FIX for KeyError: nan

        L = len(events)
        if L == 0:
            continue  # Trace has no valid labels after filtering; skip it safely
        
        for pos_y in range(L):
            y = events[pos_y]
            iy = idx[y]

            start = max(0, pos_y - w)
            for pos_x in range(start, pos_y):
                x = events[pos_x]
                ix = idx[x]

                dist = pos_y - pos_x - 1
                weight = (a ** dist)
                corr[ix, iy] += weight
                corr[iy, ix] += weight

    #---------Normalization----------
    if corr.size > 0 and corr.max() > 0:  # corr.size guard is defensive for safety
        corr = corr / corr.max()
        corr = np.clip(corr, 0, 1)

    return corr, classes

'''
This is the first substantive methodological step.

Everything that follows, like hierarchy, abstraction levels, segmentation, models all depends entirely on this output.
'''

####--------------SECTION 3 - EVENT CLASS CORRELATION (GLOBAL SCAN)-------------------------

'''
This step converts the global event-class correlation matrix into a distance matrix and builds a single hierarchical 
clustering. The resulting hierarchy encodes all abstraction levels and is built once and reused, enabling consistent 
selection of different macro-level abstractions by cutting the same hierarchy at different heights.
'''

def build_event_class_hierarchy(
    corr: np.ndarray,
    cfg: SegmentationConfig
) -> np.ndarray:
    """
    ----------Parameters------------------------------
    corr: np.ndarray
    The normalized event-class correlation matrix produced earlier.
    
    cfg: SegmentationConfig
    Provides the linkage method used for hierarchical clustering. Controls how distance between cluster are computed
    --------------------------------------------------------------------------------------------------------------
    Build the hierarchical clustering tree (linkage matrix Z).
    Keep Z so we can choose any abstraction level later (adaptive GTS).
    """

    # -------------------Guards / Defensive checks-------------------
    if corr is None:
        raise ValueError("corr must not be None")

    if not isinstance(corr, np.ndarray):
        corr = np.asarray(corr)

    if corr.ndim != 2:
        raise ValueError("corr must be a 2D array")

    n_rows, n_cols = corr.shape
    if n_rows != n_cols:
        raise ValueError("corr must be square (n x n)")

    if n_rows < 2:
        raise ValueError("corr must be at least 2x2 to build a hierarchy")

    if not np.isfinite(corr).all():
        raise ValueError("corr contains NaN or inf")

    # This function assumes normalized correlations in [0,1]
    cmin = float(corr.min())
    cmax = float(corr.max())
    if cmin < 0.0 or cmax > 1.0:
        raise ValueError("corr values must be within [0,1]")
    # ---------------------------------------------------------------

    dist = 1.0 - corr  # Correlation is converted into distance because distance tells the clustering algorithm
                       # how far apart an event class is from another

    iu = np.triu_indices_from(dist, 1)  # Hierarchical clustering expects a condensed distance vector,
                                        # so only the upper triangle is required (symmetric matrix)

    Z = linkage(dist[iu], method=cfg.linkage_method)  # Where the hierarchy is built.
                                                      # - Each event class starts as its own leaf
                                                      # - Clusters are merged bottom-up based on distance
                                                      # - The result is a single dendrogram over event classes

    return Z


'''
This is the transition point from behavioral evidence (correlation) -> abstraction structure (hierarchy)

Everything after this is projection, not clustering.
'''

def plot_event_class_dendrogram(
    Z,
    labels,
    max_d=None,
    title="Event Class Hierarchy (Dendrogram)",
    figsize=(14, 6)
):
    """
    Visualize the event-class dendrogram.
    ---------Parameters---------
    Args:
        Z: linkage matrix
        labels: list of event class names (leaf labels)
        max_d: optional distance threshold to draw a horizontal cut line
        title: plot title
    ------------------------------------
    """
    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=max_d
    )

    if max_d is not None:  # Horizontal line that represent an abstraction level cut
        plt.axhline(y=max_d, c="red", lw=2, linestyle="--")

    plt.title(title)
    plt.xlabel("Event Classes")
    plt.ylabel("Distance (1 - Correlation)")
    plt.tight_layout()
    plt.show()

def get_cut_height_for_k(Z, k):
    """
    Approximate dendrogram height corresponding to k clusters In the paper, PROM functionality allowed authors to slide
    This line up and down which selects the number of clusters. To immitate PROM functionality, the line is estimated
    based on the abstraction level (i.e. number of macro clusters chosen).
    """
    n = Z.shape[0] + 1 # The number of original event classes "n" (i.e. leaves)
    if k <= 1 or k >= n:  # A boundary check to avoid trivial abstraction levels, which are not useful to visualize or analyze
                           # 1. No abstraction at all
                           # 2. Collapse all event classes into a single macro activity 
        return None
    
    return Z[-(k - 1), 2]  # The dendogram height where the heirarchy would result in "k" clusters

# ------------------------------------------------------------
# ADAPTIVE GLOBAL TRACE SEGMENTATION (Figure 3 in Günther et al.)
#
# Applies a globally-defined abstraction mapping to all traces.
#
# Steps:
# 1. Replace each activity with its macro activity.
# 2. Collapse consecutive repetitions of the same macro activity.
#
# Segmentation in this context means deterministic trace rewriting
# induced by a global abstraction level, not local boundary detection.
# ------------------------------------------------------------


def apply_cluster_rewrite(df: pd.DataFrame, mapping: Dict[str, int], src_col: str, dst_col: str) -> pd.DataFrame:
    """
    ---------Parameters-----------------------------
    df: the current event log table (raw log)

    mapping: global mapping from event class → cluster id
    (derived earlier from the hierarchy cut at level k)

    src_col: the column containing the original activity label
    (usually "activity")

    dst_col: the column to store the macro label
    (usually "macro_activity_id")
    ----------------------------------------------------

    This is projection in Günther terms: applying an abstraction mapping to the log.
    """
    out = df.copy() # Preserve the raw log, re-writting will create a derived log called "out"
    out[dst_col] = out[src_col].map(lambda v: mapping.get(v, -1)) # Pure projection (i.e. applying a global mapping to relabel 
                                                                   # events)
                                                                   # 1. Every event instance with label v gets replaced by its 
                                                                   #    macro cluster id
                                                                   # 2. No ordering changes
                                                                   # 3. No segmentation decisions
                                                                   # 4. Mapping is fixed globally
    #".map" -> for each value in this column, apply a function and replace the value with the result
    #"lambda" -> takes on input v(event label) and returns one ouput(macro cluster ID)
    #".get" -> Look up cluster ID for this event class
    #".get(v, -1)" -> if v exists in mapping, return Cluster ID, if not return -1
    # "mapping" is not learned during segmentation — it’s computed once from the global hierarchy and then treated as fixed.
    
    return out # Returns a log that is structurally the same as the input log, but now has macro cluster IDs

def collapse_consecutive_in_trace(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    ---------Parameters-------------------
    df: the rewritten log (already has macro labels)

    label_col: the label column to collapse on
    (here: "macro_activity_id")
    ------------------------------------
    
    The function collapses uninterrupted repetitions of the same cluster label.
    This is the key missing part for Fig. 3-like simplification.
    """
    kept_rows = [] # Stores the first event of each run 
    
    for case_id, group in df.groupby(CASE_COL): # Trace-level pass to apply the same collapse to each trace
        group = group.sort_values(TS_COL) # Gurantees events collapse based on consecutive occurrences
        prev = object() # Makes it so that the first event always differs from the previous event
        
        #-------Collapse logic-----------
        for _, row in group.iterrows(): # iterate over(scans) the events in each trace, in order, after abstraction
            curr = row[label_col] # Extracts the macro activity label(cluster ID) of the current event
                                   # At this point, each event belongs to a macro activity because global abstraction mapping 
                                   # has already been applied
            if curr != prev: # Checks if the current macro activity is different from the immedeatly preceeding one in the trace
                              # It's a segment boundary definition, a new segments starts only when the abstract activity label 
                              # changes 
                kept_rows.append(row) # Keeps the first event of a run identical to macro activities  
                                       # (i.e. one macro-level segment in the trace) 
            prev = curr # Update the memory of the previous macro activity ensuring that the next event will be compared 
                         # only to its immediate predecessor, enforcing locality. (e.g.) A A B B A collapses into A B A
        #-----------------------------------
    return pd.DataFrame(kept_rows).reset_index(drop=True)

# Concepts
# 1. What is meant by "pure projection"
#                   Abstraction = defining a higher-level activity space
#                   Projection = applying that abstraction to rewrite traces
# They are related, but not the same operation.


#do_collapse = True


def adaptive_global_trace_segmentation(
    df: pd.DataFrame,
    mapping: Dict[str, int],
    src_col: str = ACTIVITY_COL,
    out_col: str = "macro_activity_id",
    do_collapse: bool = True
) -> pd.DataFrame:
    """
    ---------Parameters---------
    mapping: fixed global mapping derived from hierarchy cut.

    src_col: raw event-class label (e.g., activity).

    out_col: new column to hold macro labels.

    do_collapse: enables the Figure 3 collapse step.
    -------------------------------------
    
    Implements the paper's adaptive global trace segmentation for a chosen abtraction level.
    - rewrite events using mapping (event class -> cluster)
    - collapse consecutive identical cluster labels per trace
    """
    rewritten = apply_cluster_rewrite(df, mapping, src_col=src_col, dst_col=out_col) # Step 1: Rewrite
                                                                        # -Each event label is replaced by its macro acvitiy
                                                                        # -This is a pure projection
                                                                        # -No ordering is changed
    if do_collapse:   # Step 2: Collapse - A key simplification step
        rewritten = collapse_consecutive_in_trace(rewritten, out_col) # -Only consecutive repetitions are collapsed
                                                                      # -Order is perserved
                                                                      # -yields segments at the macro level
        # (e.g.) A A B B A collapses into A B A
        #        A B A stays as A B A
    return rewritten # Return macro-level log (same structure as original df, but with macro cluster ids and potentially fewer
                      # rows due to collapse)



'''
“Once the hierarchy cut gives a fixed mapping from activities to macro activities, apply that mapping to every event in 
every trace. That’s the rewrite step. Then apply the Figure 3 collapse step: consecutive repetitions of the same 
macro activity are merged. This creates macro-level segments without any local boundary detection or per-trace
optimization it’s deterministic rewriting driven by the chosen abstraction level.
'''

'''
Phase A selects abstraction levels by cutting the global event-class hierarchy and then projects those abstractions onto 
the log using deterministic trace rewriting and collapse, exactly as shown in Günther’s Figure 3.

Key takeaways (simple and defensible)
- Hierarchy built once
- Abstraction levels selected by cuts
- Projection and global
- Segmentation is deterministic
- No trace-level optimization
'''

def get_mapping_for_k_clusters(Z: np.ndarray, classes: List[str], k: int) -> Dict[str, int]:
    """
    ---------Parameters---------
    Z: the global event-class hierarchy (linkage matrix)

    classes: ordered list of original event classes (leaf labels)

    k: desired number of macro activities (abstraction level)
    -----------------------------------
    
    This function selects an abstraction level from the global event-class hierarchy    
    """
    
    # ---------Guards: abstraction level validity (paper Step: choose k)---------
    n = len(classes)
    if not isinstance(k, int):
        raise ValueError("k must be an integer")
    if n == 0:
        raise ValueError("classes must not be empty")
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and n_classes={n}")
    # -------------------------------------------------------------------------

    labels = fcluster(Z, k, criterion="maxclust") - 1  # 1. fcluster cuts the heirarchy so that k clusters remain 
                                                        # 2. Each event class gets a cluster ID
                                                        # 3. IDs are shifted to 0-based indexing for consistency with "-1"
                                                        # 4. maxclust -> cut the heirarchy so that the result has the most
                                                        #    clusters
    
    return {cls: int(cid) for cls, cid in zip(classes, labels)} # Result is dictionary for activity -> macro activity
                                                                 # No trace-level information is used here
    
    # classes -> one leave in event-class heirarchy, one original acitivity label in the log
    # labels -> NumPy array of cluster ID's
    # zip -> takes two sequences (a,b) and returns pairs of corresponding elements
    # cls -> one activity label
    # cid -> corresponding cluster ID for cls
    
    # The code returns a constructed global abstraction mapping by pairing each original activity label with the 
    #  macro cluster ID assigned to it when cutting the hierarchy

def build_abstracted_logs(
    df: pd.DataFrame,
    Z: np.ndarray,
    classes: List[str],
    k_levels: List[int],
    do_collapse: bool = True
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, Dict[str, int]]]:
    """
    ---------Parameters-------------------
    df: raw event log

    Z: global hierarchy (built once)

    classes: event-class labels

    k_levels: list of abstraction levels to explore

    do_collapse: whether to apply Figure 3 collapse
    -----------------------------------
    Paper-core phase:
    - select abstraction levels by cutting the hierarchy
    - project abstraction onto traces (Figure 3 behavior)

    Returns:
        abstracted_logs: k -> DataFrame with macro_activity_id
        mappings:        k -> (activity -> cluster_id)
    """
    abstracted_logs = {}  # The macro-level log at abstraction level "k"
    mappings = {}          # Abstraction definition used for that log 
                           # Allows for evaluation across abstraction levels

    for k in k_levels: # Iterate over all abstraction levels (i.e. adaptive part of the method)
        mapping = get_mapping_for_k_clusters(Z, classes, k) # Select abstraction level from the global event class heirarchy
                                                             # Again, NO trace data is used, purely heirarchy based
        #-----Figure 3 in code----------
        df_macro_ids = adaptive_global_trace_segmentation(
            df,
            mapping=mapping,
            src_col=ACTIVITY_COL,
            out_col="macro_activity_id",
            do_collapse=True
        )
        #--------------------------------

        abstracted_logs[k] = df_macro_ids  # save the abstracted logs for process discovery and evaluation metrics
        mappings[k] = mapping     # save the mappings at each abstraction level for subprocess generation and lifecycle analysis

    return abstracted_logs, mappings  # One abstracted log per abstraction level

# ============================================================
# PHASE B — SEMANTIC NAMING 
# Everything up to Phase A defines abstraction behavior; Phase B only improves readability
# ============================================================

def tfidf_cluster_label(activities: List[str], 
                        top_k: int = 2) -> str:
    '''
    ---------Parameters---------
    activities: List[str]
    A list of original activity labels that belong to the same macro cluster. These strings represent the raw activities
    that were grouped together by abstraction
    
    top_k: int = 2
    The number of most representative terms to extract using TF-IDF (i.e. how many keywords should summarize the macro label?)
    -----------------------------------
    The function takes the names of activities in a cluster and returns a short, human-readable label using TF-IDF; it operates 
    purely on text and has no effect on abstraction.”
    '''
    vec = TfidfVectorizer(stop_words="english") # Remove common words that don't help distinguish clusters
    X = vec.fit_transform(activities) # Fits TF-IDF on activity names within a cluster
    scores = X.sum(axis=0).A1 # Aggregates importance scores across all activities
    vocab = np.array(vec.get_feature_names_out()) # Retrieves vocabulary terms
    top_terms = vocab[np.argsort(scores)[::-1][:top_k]]  # Selects the top k most representative terms
    return " ".join(top_terms) # a concise label string

def get_semantic_cluster_mapping_tfidf(mapping: Dict[str, int], 
                                       top_k_terms: int = 2) -> Dict[int, str]:
    '''
    ---------Parameters-------------------
    mapping:
    The global abstraction definition
    (original activity → macro cluster ID)

    top_k_terms:
    How many keywords to use when generating a human-readable name
    ------------------------------------------------------------
    
    Builds readable names for each macro cluster and is not part of the Günther methodology. Inspired by the paper
    Granular Computing in Process Mining
    '''
    inv: Dict[int, List[str]] = {} # Creates an inverse mapping dictionary that groups original activities by macro cluster
    
    for act, cid in mapping.items(): # Iterate over every original activity and its assigned macro cluster ID
        inv.setdefault(cid, []).append(act) # Groups activity labels by cluster to label each cluster using its macro labels

    sem_map: Dict[int, str] = {} # Output mapping: turning cluster_id into human-readable labels
    
    for cid, acts in inv.items(): # Iterate over each macro cluster and their respective activites
        label = tfidf_cluster_label(acts, top_k=top_k_terms) # Generate labels that are human readable
        sem_map[cid] = beautify_label(label) if label.strip() else f"Cluster {cid}" # Use TF-IDF generated label, otherwise use
                                                                                    # generic macro cluster label like Cluster_1
    return sem_map # returns semantic naming for presentation

def beautify_label(s: str) -> str:
    '''
    ---------Parameters-------------------
    s: str
    A label string, produced by tfidf_cluster_label function.
    --------------------------------------
    
    The function is just a cosmetic helper that formats cluster labels for BPMN readability; it has 
    no methodological significance.”

    This improves BPMN readability.
    '''
    # replaces separators with spaces, applies title casing, trim white space (i.e. string formatting only)
    return s.replace("_", " ").replace("-", " ").title().strip()

def apply_semantic_labels(df: pd.DataFrame, id_col: str, sem_map: Dict[int, str], out_col: str) -> pd.DataFrame:
    '''
    ---------Parameters-------------------
    df: macro-level log with numeric cluster IDs

    id_col: column containing macro cluster IDs

    sem_map: cluster ID → semantic label

    out_col: name of the new readable column
    ----------------------------------------------
    
    Adds a human-readable macro label column derived from cluster IDs.
    '''
    out = df.copy() # Prevents mutation of upstream dataframe across abstraction levels
    out[out_col] = out[id_col].map(lambda cid: sem_map.get(int(cid), f"Cluster_{cid}")) # Maps numeric cluster IDs to readable
                                                                                         # Labels
    # Like before this is a pandas series mapping. 
    # lambda function takes one input "cid"(Macro Cluster ID) and returns a human readable string label
    return out

def apply_semantic_naming_to_logs(
    abstracted_logs: Dict[int, pd.DataFrame],
    mappings: Dict[int, Dict[str, int]],
    top_k_terms: int = 2) -> Tuple[Dict[int, pd.DataFrame], Dict[int, Dict[int, str]]]:
    """
    ---------Parameters---------
    abstracted_logs: output of Phase A (k → macro log)

    mappings: abstraction definitions (k → activity → cluster)

    top_k_terms: naming granularity
    -----------------------------------
    
    Optional interpretability layer:
    - assigns human-readable names to macro clusters
    - does NOT affect abstraction or projection
    """
    named_logs = {}   # log with readable macro labels
    semantic_labels = {}  # label dictionary for abstraction level "k"

    for k, df_macro in abstracted_logs.items():  # Semantic naming is applied independently per abstraction level
        mapping = mappings[k]   # Retrieves the level used to create a macro log

        # -----------Build semantic labels for abstraction level--------
        sem_labels = get_semantic_cluster_mapping_tfidf(
            mapping,
            top_k_terms=top_k_terms
        )
        #----------------------------------------------------------
        
        #-----Add a readable new column--------

        df_named = apply_semantic_labels(
            df_macro,
            id_col="macro_activity_id",
            sem_map=sem_labels,
            out_col="macro_activity"
        )
        #-----------------------------------

        named_logs[k] = df_named # Store the named log for discovery and visualization
        semantic_labels[k] = sem_labels # Store the semantic mapping for reference

    return named_logs, semantic_labels 

# ============================================================
# PHASE C — BPMN DISCOVERY & VISUALIZATION
# "Alphabet" = vocabulary of event labels
# ============================================================

def dataframe_to_event_log(df: pd.DataFrame, label_col: str) -> EventLog:
    """
    --------------------Parameters--------------------
    df: an event log in tabular (DataFrame) form
    (raw log, macro log, or lifecycle log)

    label_col: the column that defines the event label at this level
        -"macro_activity" → global macro BPMN
        -"activity" → subprocess BPMNs
        -"concept:name" → lifecycle BPMNs
    --------------------------------------------------
    
    Convert a DataFrame to a PM4Py EventLog object.
    """
    log = EventLog()
    for case_id, group in df.groupby(CASE_COL):  # Iterate over abstracted traces, no segmentation is done here
        trace = Trace()  # Create pm4py trace object
        
        for _, row in group.sort_values(TS_COL).iterrows(): # Ensures events are added in order of execution
            e = Event()  # Create pm4py event instance
            e[CASE_COL] = str(case_id)  # Add a case identifier to event instance
            
            #---Where abstraction level determines the process alphabet------
            e["concept:name"] = str(row[label_col]) 
            # At Macro level -> macro activites
            # At subprocess level -> original activities
            # At lifecycle level -> lifecycel labels
            # This ensures that model discovery and conformance always operates on a consistent event alphabet
            #-------------------------------------------------------
            e[TS_COL] = row[TS_COL] # Preserves timestamps for ordering
            trace.append(e)  # Add event to the trace
        log.append(trace)  # Add the completed trace to the event log
    return log  

def discover_and_export_bpmn(df: pd.DataFrame, label_col: str, out_path_png: str):
    '''
    ----------Parameters----------
    df: log at the current abstraction level

    label_col: event label defining this view

    out_path_png: where to save the BPMN image
    ------------------------------
    '''
    #------Defensive check-------
    if df.empty:
        print(f"[WARN] Skipping empty BPMN ({out_path_png})")
        return
    # Avoids discovery on empty logs
    # Prevents meaningless models
    #----------------------------
    log = dataframe_to_event_log(df, label_col) # Convert dataframe to event log using the correct event alphabet
    
    #-------Where directionality enters the pipeline------------
    # Up to this point, abstraction only captured behavioral proximity. Here, the inductive 
     #  miner infers sequence, concurrency, and branching.
    tree = inductive_miner.apply(log, variant=inductive_miner.Variants.IMf) 
    #-----------------------------------------------------
    # Convert the discovered process tree into a BPMN
    bpmn_graph = process_tree_converter.apply(tree, variant=process_tree_converter.Variants.TO_BPMN)
    gviz = bpmn_visualizer.apply(bpmn_graph)
    bpmn_visualizer.save(gviz, out_path_png, parameters={"format": "png"}) # Export bpmn as a png
    print(f"[OK] BPMN PNG -> {out_path_png}")


def generate_global_bpmn(df_macro: pd.DataFrame, macro_label_col: str, base_dir: str):
    '''
    ---------Parameters---------
    df_macro
    The macro-level event log produced in Phase A (after abstraction + collapse).

    macro_label_col
    The column defining the macro-level alphabet
    (typically "macro_activity").

    base_dir
    Output directory for this abstraction level
    (e.g., level_k_12/).
    ----------------------------
    
    Generate the top-level process view using macro activities. No abstraction here, just visualization
    '''
    out_png = os.path.join(base_dir, "bpmn_global", "global_macro_bpmn.png") # Define output location
    discover_and_export_bpmn(df_macro, macro_label_col, out_png)  # Discovery happens on macro log using macro labels


def generate_macro_subprocess_bpmns(df_macro: pd.DataFrame, macro_label_col: str, base_dir: str):
    '''
    ---------Parameters---------
    df_macro
    The macro-level log (same as above).

    macro_label_col
    Column used to group events by macro activity.

    base_dir
    Output directory for this abstraction level.
    ---------------------------
    
    For each macro activity, provide drill-down views for each macro activity using the original activity alphabet.
    '''
    for macro, subdf in df_macro.groupby(macro_label_col):   # groups traces by macro activity
        sanitized = str(macro).replace(" ", "_")   # file safe naming
        out_png = os.path.join(base_dir, "bpmn_macro", f"macro_{sanitized}.png") # One BPMN per macro activity
        print(f"[INFO] Macro subprocess BPMN -> {macro}")
        discover_and_export_bpmn(subdf, ACTIVITY_COL, out_png) # Subprocess BPMNs use original activity labels, NOT macro labels


def generate_activity_lifecycle_bpmns(
    df: pd.DataFrame,
    mapping: Dict[str, int],
    sem_labels: Dict[int, str],
    base_dir: str
):
    """
    --------Parameters----------------
    df
    The raw event log, before abstraction.

    mapping
    Global abstraction mapping
    (activity → macro cluster ID).

    sem_labels
    Human-readable names for macro clusters
    (from Phase B).

    base_dir
    Output directory for this abstraction level.
    ---------------------------------------------
    
    Show fine-grained lifecycle behavior within macro clusters.
    
    Uses CONCEPT_COL as lifecycle label.
    
    Generates activity lifecycle BPMNs, grouped by macro cluster, using lifecycle labels as the alphabet.
    This is NOT recursive abstraction its a behavioral zoom-in
    """
    for act, subdf in df.groupby(ACTIVITY_COL): # Iterate over individual activities
        #---Assosciate each activity with its respective macro cluster------
        cid = mapping.get(act, -1)
        cluster_name = sem_labels.get(cid, "Unknown Cluster")
        #-----------------------------------------------------------------
        sanitized_cluster = cluster_name.replace(" ", "_") 
        
        # Group lifecycle models by macro cluster
        cluster_folder = os.path.join(base_dir, "bpmn_activity_lifecycle", sanitized_cluster) 
        os.makedirs(cluster_folder, exist_ok=True)

        sanitized_act = str(act).replace(" ", "_")
        out_png = os.path.join(cluster_folder, f"activity_lifecycle_{sanitized_act}.png")

        print(f"[INFO] Lifecycle BPMN -> {act} (Cluster '{cluster_name}')")
        discover_and_export_bpmn(subdf, CONCEPT_COL, out_png) # Uses lifecycle labels (i.e. concept:name column)


def generate_all_bpmns_for_level(
    df_raw: pd.DataFrame,
    df_macro: pd.DataFrame,
    mapping: Dict[str, int],
    sem_labels: Dict[int, str],
    base_dir: str
):
    """
    ---------Parameters---------

    df_raw
    Raw event log (needed for lifecycle models).

    df_macro
    Macro-level log (needed for global + subprocess models).

    mapping
    Abstraction definition for this level.

    sem_labels
    Semantic names for macro clusters.

    base_dir
    Output directory for this abstraction level.
    ------------------------------------------------------------
    This function is purely an analysis orchestrator for all BPMN views at a single abstraction level.
    
    Analysis layer:
    - global macro BPMN
    - macro subprocess BPMNs
    - activity lifecycle BPMNs
    """
    print("[INFO] Discovering Global BPMN...")
    generate_global_bpmn(df_macro, "macro_activity", base_dir)

    print("[INFO] Discovering Macro Subprocess BPMNs...")
    generate_macro_subprocess_bpmns(df_macro, "macro_activity", base_dir)

    print("[INFO] Discovering Activity Lifecycle BPMNs...")
    generate_activity_lifecycle_bpmns(df_raw, mapping, sem_labels, base_dir)


# ============================================================
# STEP 6 — EVALUATION METRICS (Paper-faithful comparison)
# Does global trace segmentation produce a log that leads to simpler, more understandable 
#  process models?
# ============================================================
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_eval
from pm4py.algo.evaluation.precision import algorithm as precision_eval
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from variant_stats import get_variants_stats, get_variant_ratio, get_variant_coverage, filter_traces_on_variants

import matplotlib.pyplot as plt
plt.ioff()


def compute_macro_conformance(df: pd.DataFrame, label_col: str) -> dict:
    """
    ----------Parameters----------

    df: pd.DataFrame
    The event log at a single abstraction level
    (either raw log or macro log at abstraction level k).

    label_col: str
    The column defining the alphabet for this evaluation:

    raw log → "activity"

    macro log → "macro_activity"
    ----------------------------------------
    The log and model always share the same alphabet (i.e. vocabulary of event labels)
    
    Discover a model from the log, then replay the same log on that model to compute conformance metrics for a macro-level 
    log against a model discovered from the same macro log.
    """

    # --- Convert to PM4Py log ---
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df)
    log = log_converter.apply(df_pm)

    # --- Discover model ---
    tree = inductive_miner.apply(log)
    net, im, fm = pt_converter.apply(
        tree, variant=pt_converter.Variants.TO_PETRI_NET
    )

    # --- Fitness (token-based is usually sufficient & fast) ---
    fitness = fitness_eval.apply(
        log,
        net,
        im,
        fm,
        variant=fitness_eval.Variants.TOKEN_BASED
    )

    # --- Precision ---
    precision = precision_eval.apply(
        log,
        net,
        im,
        fm,
        variant=precision_eval.Variants.ETCONFORMANCE_TOKEN
    )

    return {
        "Fitness": round(fitness["average_trace_fitness"], 3),
        "Precision": round(precision, 3)
    }

import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

def build_conformance_table(
    raw_df: pd.DataFrame,
    clustered_dfs: dict,
    label_col_raw: str = "activity",
    label_col_macro: str = "macro_activity") -> pd.DataFrame:
    """
    ----------Parameters--------------------

    raw_df
    Original log (no abstraction)

    clustered_dfs
    Dictionary:
    k → macro log at level k
    
    label_col_raw / label_col_macro
    Explicitly define the alphabet for each comparison
    ------------------------------------------------------------
    Compute conformance metrics for raw and macro abstraction levels.
    """
    rows = []

    # --- Raw log conformance ---
    raw_metrics = compute_macro_conformance(raw_df, label_col_raw)
    rows.append({"Abstraction": "Raw log", **raw_metrics})

    # --- Macro levels ---
    for k, df_macro in clustered_dfs.items(): # For each abstraction level produce by global trace segmentation
        metrics = compute_macro_conformance(df_macro, label_col_macro)
        rows.append({"Abstraction": f"k = {k}", **metrics})

    return pd.DataFrame(rows)


from variant_stats import get_variants_stats

def compute_log_complexity_metrics(df: pd.DataFrame, label_col: str) -> dict:
    """
    ----------Parameters----------

    df
    Log at one abstraction level

    label_col
    Defines the alphabet used for:
    - variants
    - statistics
    - DFG
    ----------------------------------------
    Compute paper-faithful log complexity metrics for the global (macro) view.
    """

    # --- Prepare PM4Py-compatible dataframe ---
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df) # ensure all timestamp columns are in the format pm4py expects

    # --- Align alphabet for variant statistics ---
    df_for_variants = df_pm.copy()
    df_for_variants["concept:name"] = df_for_variants[label_col] # ensures that metrics are calculated at the same 
                                                                  # abstraction level

    # --- Variant statistics (external utility) ---
    vars_stats = get_variants_stats(df_for_variants)
    trace_variants = len(vars_stats)

    # --- Basic log statistics ---
    num_traces = df_for_variants[CASE_COL].nunique()
    variability_ratio = round(trace_variants / num_traces, 3)

    n_event_classes = df_for_variants["concept:name"].nunique()
    avg_trace_length = df_for_variants.groupby(CASE_COL).size().mean()

    # --- Directly-follows graph complexity ---
    log = log_converter.apply(df_pm)
    dfg = dfg_discovery.apply(log)
    n_dfg_edges = len(dfg)

    return {
        "# Event Classes": n_event_classes,     # Abstraction effect
        "# Traces": num_traces,                 
        "Avg Trace Length": round(avg_trace_length, 2),  # Trace simplification
        "# Trace Variants": trace_variants,   # Behavioral diversity
        "Variability Ratio": variability_ratio,
        "# DFG Edges": n_dfg_edges,
    }



def build_log_complexity_table(
    raw_df: pd.DataFrame,
    clustered_dfs: dict,
    label_col_raw: str = "activity",
    label_col_macro: str = "macro_activity") -> pd.DataFrame:
    """
    --------------------Parameters --------------------

    raw_df: pd.DataFrame
    The original, unabstracted event log.
    Used to compute baseline log complexity metrics at the raw activity level.

    clustered_dfs: dict
    A dictionary mapping abstraction levels to abstracted logs:
    k → macro-level DataFrame
    Each DataFrame represents the log after global trace segmentation at level k.
    
    label_col_raw: str
    The column defining the alphabet of the raw log
    (default: "activity").

    label_col_macro: str
    The column defining the alphabet of macro-level logs
    (default: "macro_activity"). 
    ----------------------------------------------------------------------
    Simply applies the above metrics consistently to:
        - raw log
        - each macro log
    
    This function compares how log-level complexity changes as we move from raw activities to macro activities.
    """
    rows = []

    # Raw log
    raw_metrics = compute_log_complexity_metrics(raw_df, label_col_raw)
    rows.append({"Abstraction": "Raw log", **raw_metrics})

    # Clustered logs
    for k, df_macro in clustered_dfs.items():
        metrics = compute_log_complexity_metrics(df_macro, label_col_macro)
        rows.append({"Abstraction": f"k = {k}", **metrics})

    return pd.DataFrame(rows)


from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.objects.bpmn.obj import BPMN


def compute_bpmn_complexity(df: pd.DataFrame, label_col: str) -> dict:
    """
    ----------Parameters----------

    df: pd.DataFrame
    The event log at a single abstraction level:
    - raw log → raw BPMN
    - macro log → macro BPMN

    label_col: str
    The column defining the alphabet used for model discovery.
    ------------------------------------------------------------

    This is critical because:
    -the BPMN structure depends directly on the alphabet
    -different abstraction levels yield different BPMNs
    
    Measures the Structural complexity of the discovered model, not the log.
    """
    # Convert to log
    log = log_converter.apply(
        dataframe_utils.convert_timestamp_columns_in_df(df)
    )

    # Discover process tree → BPMN
    tree = inductive_miner.apply(log)
    bpmn = pt_converter.apply(tree, variant=pt_converter.Variants.TO_BPMN)

    # Count BPMN elements
    nodes = len(bpmn.get_nodes())
    flows = len(bpmn.get_flows())

    # Count tasks only
    tasks = sum(
        1 for n in bpmn.get_nodes()
        if isinstance(n, BPMN.Task)
    )

    return {
        "# BPMN Nodes": nodes,
        "# BPMN Flows": flows,
        "# BPMN Tasks": tasks,
    }

def build_bpmn_complexity_table(
    raw_df: pd.DataFrame,
    clustered_dfs: dict,
    label_col_raw: str = "activity",
    label_col_macro: str = "macro_activity"
) -> pd.DataFrame:
    """
    ----------Parameters----------

    raw_df: pd.DataFrame
    The original log, used to discover and evaluate M_raw.

    clustered_dfs: dict
    Mapping: k → macro-level log
    Used to discover and evaluate M_k for each abstraction level.
    
    label_col_raw: str
    Alphabet for discovering the raw BPMN.

    label_col_macro: str
    Alphabet for discovering macro BPMNs.
    ----------------------------------------

    
    BPMN complexity table across abstraction levels
    """
    rows = []

    # Raw log BPMN
    raw_metrics = compute_bpmn_complexity(raw_df, label_col_raw)
    rows.append({"Abstraction": "Raw log", **raw_metrics})

    # Macro BPMNs
    for k, df_macro in clustered_dfs.items():
        metrics = compute_bpmn_complexity(df_macro, label_col_macro)
        rows.append({"Abstraction": f"k = {k}", **metrics})

    return pd.DataFrame(rows)