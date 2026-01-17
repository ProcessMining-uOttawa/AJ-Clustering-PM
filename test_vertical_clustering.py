import numpy as np
import pandas as pd
import pytest

from vertical_clustering import (
    CASE_COL, TS_COL, ACTIVITY_COL,
    SegmentationConfig,
    compute_event_class_correlation,
    build_event_class_hierarchy,
    get_mapping_for_k_clusters,
    apply_cluster_rewrite,
    collapse_consecutive_in_trace,
    adaptive_global_trace_segmentation,
    build_abstracted_logs,
)

# ============================================================
# Helpers
# ============================================================

def _df(rows):
    return pd.DataFrame(rows)

def _ts(i: int) -> pd.Timestamp:
    return pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=i)

def _make_case(case_id: str, activities: list[str], start_ts: int = 0):
    return [
        {CASE_COL: case_id, TS_COL: _ts(start_ts + i), ACTIVITY_COL: act}
        for i, act in enumerate(activities)
    ]


# ============================================================
# Fixtures (toy logs)
# ============================================================

@pytest.fixture
def toy_log_simple():
    rows = []
    rows += _make_case("1", ["A", "B", "A", "C"], start_ts=0)
    return _df(rows)

@pytest.fixture
def toy_log_independent_cases():
    # Case1 only A, Case2 only B => must have corr[A,B] = 0 (no cross-case leakage)
    rows = []
    rows += _make_case("1", ["A", "A", "A"], start_ts=0)
    rows += _make_case("2", ["B", "B", "B"], start_ts=100)
    return _df(rows)

@pytest.fixture
def toy_log_all_nan_labels():
    return _df([
        {CASE_COL: "1", TS_COL: _ts(0), ACTIVITY_COL: np.nan},
        {CASE_COL: "1", TS_COL: _ts(1), ACTIVITY_COL: np.nan},
    ])

@pytest.fixture
def toy_log_empty():
    return _df(columns=[CASE_COL, TS_COL, ACTIVITY_COL])

@pytest.fixture
def toy_log_single_class():
    rows = []
    rows += _make_case("1", ["A", "A", "A"], start_ts=0)
    rows += _make_case("2", ["A", "A"], start_ts=100)
    return _df(rows)

@pytest.fixture
def toy_log_duplicate_timestamps():
    # Ambiguous ordering: two events share same timestamp in same case
    return _df([
        {CASE_COL: "1", TS_COL: _ts(0), ACTIVITY_COL: "A"},
        {CASE_COL: "1", TS_COL: _ts(0), ACTIVITY_COL: "B"},
        {CASE_COL: "1", TS_COL: _ts(1), ACTIVITY_COL: "C"},
    ])


# ============================================================
# Layer 1 — Unit tests (innermost functions)
# ============================================================

# -------------------------
# 1A) compute_event_class_correlation
# -------------------------

def test_corr_missing_required_columns_fails_loudly():
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)

    # Missing CASE_COL
    df1 = _df([{TS_COL: _ts(0), ACTIVITY_COL: "A"}])
    with pytest.raises((KeyError, ValueError)):
        compute_event_class_correlation(df1, ACTIVITY_COL, cfg)

    # Missing TS_COL
    df2 = _df([{CASE_COL: "1", ACTIVITY_COL: "A"}])
    with pytest.raises((KeyError, ValueError)):
        compute_event_class_correlation(df2, ACTIVITY_COL, cfg)

    # Missing label_col
    df3 = _df([{CASE_COL: "1", TS_COL: _ts(0), "other": "A"}])
    with pytest.raises((KeyError, ValueError)):
        compute_event_class_correlation(df3, ACTIVITY_COL, cfg)

@pytest.mark.parametrize(
    "window_size, attenuation",
    [
        (-1, 0.6),   # window invalid
        (3, 0.0),    # attenuation invalid
        (3, -0.2),   # attenuation invalid
        (3, 1.1),    # attenuation invalid
    ],
)
def test_corr_invalid_config_raises(window_size, attenuation, toy_log_simple):
    cfg = SegmentationConfig(window_size=window_size, attenuation=attenuation)
    with pytest.raises(ValueError):
        compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)

def test_corr_empty_df_handling(toy_log_empty):
    """
    Desired behavior: empty input should NOT crash with numpy max() error.
    Either:
      - return (0x0 matrix, empty classes), OR
      - raise a clean ValueError("empty log") with clear msg.
    Pick one policy; this test enforces non-silent behavior.
    """
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)

    # Accept either clean ValueError OR graceful empty return
    try:
        corr, classes = compute_event_class_correlation(toy_log_empty, ACTIVITY_COL, cfg)
        assert corr.shape == (0, 0)
        assert classes == []
    except ValueError:
        pass

def test_corr_all_nan_labels_handling(toy_log_all_nan_labels):
    """
    Same expectation as empty: must not blow up with opaque numpy error.
    """
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)
    try:
        corr, classes = compute_event_class_correlation(toy_log_all_nan_labels, ACTIVITY_COL, cfg)
        assert corr.shape == (0, 0)
        assert classes == []
    except ValueError:
        pass

def test_corr_single_activity_class_invariants(toy_log_single_class):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)
    corr, classes = compute_event_class_correlation(toy_log_single_class, ACTIVITY_COL, cfg)
    assert classes == ["A"]
    assert corr.shape == (1, 1)
    assert np.isfinite(corr).all()
    assert np.min(corr) >= 0.0
    assert np.max(corr) <= 1.0
    assert np.allclose(corr, corr.T, atol=1e-12)

def test_corr_duplicate_timestamps_does_not_crash_and_keeps_invariants(toy_log_duplicate_timestamps):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)
    corr, _ = compute_event_class_correlation(toy_log_duplicate_timestamps, ACTIVITY_COL, cfg)
    assert np.allclose(corr, corr.T, atol=1e-12)
    assert np.isfinite(corr).all()
    assert np.min(corr) >= 0.0
    assert np.max(corr) <= 1.0

def test_corr_symmetry_and_bounds(toy_log_simple):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)
    corr, _ = compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)
    assert np.allclose(corr, corr.T, atol=1e-12)
    assert np.isfinite(corr).all()
    assert np.min(corr) >= 0.0
    assert np.max(corr) <= 1.0

def test_corr_no_cross_case_leakage(toy_log_independent_cases):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)
    corr, classes = compute_event_class_correlation(toy_log_independent_cases, ACTIVITY_COL, cfg)
    idx = {c: i for i, c in enumerate(classes)}
    assert corr[idx["A"], idx["B"]] == 0.0
    assert corr[idx["B"], idx["A"]] == 0.0


# -------------------------
# 1B) build_event_class_hierarchy
# -------------------------

def test_hierarchy_non_square_matrix_raises():
    cfg = SegmentationConfig(linkage_method="complete")
    corr = np.zeros((2, 3), dtype=float)
    with pytest.raises(ValueError):
        build_event_class_hierarchy(corr, cfg)

def test_hierarchy_size_lt_2_raises_cleanly():
    cfg = SegmentationConfig(linkage_method="complete")

    with pytest.raises(ValueError):
        build_event_class_hierarchy(np.zeros((0, 0)), cfg)

    with pytest.raises(ValueError):
        build_event_class_hierarchy(np.zeros((1, 1)), cfg)

def test_hierarchy_nan_in_corr_raises():
    cfg = SegmentationConfig(linkage_method="complete")
    corr = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValueError):
        build_event_class_hierarchy(corr, cfg)

def test_hierarchy_out_of_range_corr_raises():
    """
    Desired behavior: corr should be within [0,1].
    If you add guards, enforce them here.
    """
    cfg = SegmentationConfig(linkage_method="complete")
    corr = np.array([[0.0, 2.0], [2.0, 0.0]])
    with pytest.raises(ValueError):
        build_event_class_hierarchy(corr, cfg)

def test_hierarchy_Z_shape_for_n_ge_2(toy_log_simple):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6, linkage_method="complete")
    corr, classes = compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)
    if len(classes) < 2:
        pytest.skip("toy_log_simple unexpectedly has <2 classes")
    Z = build_event_class_hierarchy(corr, cfg)
    assert Z.shape == (len(classes) - 1, 4)


# -------------------------
# 1C) Projection functions
# -------------------------

def test_apply_cluster_rewrite_missing_src_col_raises(toy_log_simple):
    mapping = {"A": 0}
    with pytest.raises(KeyError):
        apply_cluster_rewrite(toy_log_simple, mapping, src_col="missing_col", dst_col="macro_activity_id")

def test_apply_cluster_rewrite_unmapped_goes_to_minus_one(toy_log_simple):
    mapping = {"A": 0}  # B,C unmapped
    out = apply_cluster_rewrite(toy_log_simple, mapping, src_col=ACTIVITY_COL, dst_col="macro_activity_id")
    assert "macro_activity_id" in out.columns
    assert set(out.loc[out[ACTIVITY_COL].isin(["B", "C"]), "macro_activity_id"]) == {-1}

def test_collapse_consecutive_in_trace_no_adjacent_repeats_per_case():
    df = _df([
        {CASE_COL: "1", TS_COL: 1, "macro_activity_id": 0},
        {CASE_COL: "1", TS_COL: 2, "macro_activity_id": 0},
        {CASE_COL: "1", TS_COL: 3, "macro_activity_id": 1},
        {CASE_COL: "1", TS_COL: 4, "macro_activity_id": 1},
        {CASE_COL: "1", TS_COL: 5, "macro_activity_id": 0},
        {CASE_COL: "2", TS_COL: 1, "macro_activity_id": 9},
        {CASE_COL: "2", TS_COL: 2, "macro_activity_id": 9},
    ])
    out = collapse_consecutive_in_trace(df, "macro_activity_id")

    # Ensure no adjacent repeats within each case
    for _, g in out.sort_values([CASE_COL, TS_COL]).groupby(CASE_COL):
        seq = g["macro_activity_id"].tolist()
        assert all(seq[i] != seq[i+1] for i in range(len(seq)-1))

def test_collapse_preserves_case_boundaries():
    """
    Ensure collapse does NOT merge across cases.
    """
    df = _df([
        {CASE_COL: "1", TS_COL: 1, "macro_activity_id": 0},
        {CASE_COL: "1", TS_COL: 2, "macro_activity_id": 0},
        {CASE_COL: "2", TS_COL: 1, "macro_activity_id": 0},
        {CASE_COL: "2", TS_COL: 2, "macro_activity_id": 0},
    ])
    out = collapse_consecutive_in_trace(df, "macro_activity_id")
    # Should keep 1 row per case (not 1 row total)
    assert out[CASE_COL].nunique() == 2
    assert out.shape[0] == 2

def test_collapse_preserves_relative_order_of_retained_events():
    df = _df([
        {CASE_COL: "1", TS_COL: 1, "macro_activity_id": 0, "payload": "first0"},
        {CASE_COL: "1", TS_COL: 2, "macro_activity_id": 0, "payload": "second0"},
        {CASE_COL: "1", TS_COL: 3, "macro_activity_id": 1, "payload": "first1"},
        {CASE_COL: "1", TS_COL: 4, "macro_activity_id": 1, "payload": "second1"},
        {CASE_COL: "1", TS_COL: 5, "macro_activity_id": 0, "payload": "third0"},
    ])
    out = collapse_consecutive_in_trace(df, "macro_activity_id")
    assert out["payload"].tolist() == ["first0", "first1", "third0"]


# ============================================================
# Layer 2 — Integration tests (chains)
# ============================================================

def test_integration_chain_determinism(toy_log_simple):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6, linkage_method="complete")
    corr1, classes1 = compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)
    Z1 = build_event_class_hierarchy(corr1, cfg)
    mapping1 = get_mapping_for_k_clusters(Z1, classes1, k=2)

    corr2, classes2 = compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)
    Z2 = build_event_class_hierarchy(corr2, cfg)
    mapping2 = get_mapping_for_k_clusters(Z2, classes2, k=2)

    assert classes1 == classes2
    assert mapping1 == mapping2

def test_integration_invalid_k_runtime_fails(toy_log_simple):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6, linkage_method="complete")
    corr, classes = compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)
    if len(classes) < 2:
        pytest.skip("Need >=2 classes for k tests")
    Z = build_event_class_hierarchy(corr, cfg)

    # Desired behavior: clean ValueError for invalid k
    with pytest.raises(ValueError):
        get_mapping_for_k_clusters(Z, classes, k=0)

    with pytest.raises(ValueError):
        get_mapping_for_k_clusters(Z, classes, k=len(classes) + 1)

def test_integration_mapping_range_and_coverage(toy_log_simple):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6, linkage_method="complete")
    corr, classes = compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)
    if len(classes) < 2:
        pytest.skip("Need >=2 classes")
    Z = build_event_class_hierarchy(corr, cfg)
    k = 2
    mapping = get_mapping_for_k_clusters(Z, classes, k=k)

    assert set(mapping.keys()) == set(classes)
    assert all(isinstance(v, int) for v in mapping.values())
    assert min(mapping.values()) >= 0
    assert max(mapping.values()) <= (k - 1)

def test_integration_projection_preserves_order_and_cases(toy_log_simple):
    cfg = SegmentationConfig(window_size=3, attenuation=0.6, linkage_method="complete")
    corr, classes = compute_event_class_correlation(toy_log_simple, ACTIVITY_COL, cfg)
    if len(classes) < 2:
        pytest.skip("Need >=2 classes")
    Z = build_event_class_hierarchy(corr, cfg)
    mapping = get_mapping_for_k_clusters(Z, classes, k=2)

    out = adaptive_global_trace_segmentation(
        toy_log_simple,
        mapping=mapping,
        src_col=ACTIVITY_COL,
        out_col="macro_activity_id",
        do_collapse=True,
    )

    # case boundaries preserved
    assert set(out[CASE_COL].unique()) == set(toy_log_simple[CASE_COL].unique())

    # order preserved within case (timestamps non-decreasing for retained events)
    for _, g in out.sort_values([CASE_COL, TS_COL]).groupby(CASE_COL):
        ts = g[TS_COL].tolist()
        assert all(ts[i] <= ts[i+1] for i in range(len(ts)-1))

    # collapse only removes consecutive repeats
    for _, g in out.sort_values([CASE_COL, TS_COL]).groupby(CASE_COL):
        seq = g["macro_activity_id"].tolist()
        assert all(seq[i] != seq[i+1] for i in range(len(seq)-1))

def test_integration_golden_toy_log_cluster_AB_together_more_than_C():
    """
    Construct toy log:
    - Many cases contain A,B adjacent (high corr)
    - Some cases contain only C
    Expect at k=2: A and B in same cluster, C separate.
    """
    rows = []
    # Strong AB co-occurrence
    for i in range(20):
        rows += _make_case(f"ab_{i}", ["A", "B", "A", "B"], start_ts=i * 100)
    # C separate
    for i in range(10):
        rows += _make_case(f"c_{i}", ["C", "C", "C"], start_ts=5000 + i * 100)

    df = _df(rows)
    cfg = SegmentationConfig(window_size=3, attenuation=0.9, linkage_method="complete")
    corr, classes = compute_event_class_correlation(df, ACTIVITY_COL, cfg)
    if set(classes) != {"A", "B", "C"}:
        pytest.skip("Unexpected classes; test assumes A,B,C only")
    Z = build_event_class_hierarchy(corr, cfg)
    mapping = get_mapping_for_k_clusters(Z, classes, k=2)

    assert mapping["A"] == mapping["B"]
    assert mapping["C"] != mapping["A"]


# ============================================================
# Layer 3 — Application tests (end-to-end-ish, but avoid BPMN)
# ============================================================

def test_app_smoke_build_abstracted_logs_runs_and_outputs_have_keys():
    # Tiny synthetic dataset
    rows = []
    rows += _make_case("1", ["A", "B", "C", "A"], start_ts=0)
    rows += _make_case("2", ["A", "C", "C", "B"], start_ts=100)
    rows += _make_case("3", ["B", "A", "B"], start_ts=200)
    df = _df(rows)

    cfg = SegmentationConfig(window_size=3, attenuation=0.7, linkage_method="complete")
    corr, classes = compute_event_class_correlation(df, ACTIVITY_COL, cfg)
    if len(classes) < 2:
        pytest.skip("Need >=2 classes for abstraction levels")
    Z = build_event_class_hierarchy(corr, cfg)

    k_levels = [2, min(3, len(classes))]  # safe
    abstracted_logs, mappings = build_abstracted_logs(df, Z, classes, k_levels, do_collapse=True)

    # Assert outputs exist for each k
    for k in k_levels:
        assert k in abstracted_logs
        assert k in mappings
        assert "macro_activity_id" in abstracted_logs[k].columns

        # No NaNs in macro ids (should be ints or -1)
        assert abstracted_logs[k]["macro_activity_id"].notna().all()

def test_app_poisoned_missing_timestamp_fails():
    df = _df([
        {CASE_COL: "1", ACTIVITY_COL: "A"},
        {CASE_COL: "1", ACTIVITY_COL: "B"},
    ])
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)
    with pytest.raises((KeyError, ValueError)):
        compute_event_class_correlation(df, ACTIVITY_COL, cfg)

def test_app_poisoned_wrong_label_column_fails():
    df = _df([
        {CASE_COL: "1", TS_COL: _ts(0), "wrong": "A"},
    ])
    cfg = SegmentationConfig(window_size=3, attenuation=0.6)
    with pytest.raises((KeyError, ValueError)):
        compute_event_class_correlation(df, ACTIVITY_COL, cfg)

def test_app_poisoned_invalid_k_levels_should_fail_cleanly():
    rows = []
    rows += _make_case("1", ["A", "B", "C"], start_ts=0)
    df = _df(rows)

    cfg = SegmentationConfig(window_size=3, attenuation=0.6, linkage_method="complete")
    corr, classes = compute_event_class_correlation(df, ACTIVITY_COL, cfg)
    if len(classes) < 2:
        pytest.skip("Need >=2 classes")
    Z = build_event_class_hierarchy(corr, cfg)

    # Desired behavior: build_abstracted_logs should fail if k is invalid
    with pytest.raises(ValueError):
        build_abstracted_logs(df, Z, classes, k_levels=[0], do_collapse=True)

    with pytest.raises(ValueError):
        build_abstracted_logs(df, Z, classes, k_levels=[len(classes) + 1], do_collapse=True)
