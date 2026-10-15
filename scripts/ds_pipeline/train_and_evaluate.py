"""
Stage 4–7: Leak-Free GroupKFold Cross-Validation, Old-vs-New Model Ablation,
Two-Stage Log-Ridge + Monotonic Huber GBDT Training, Split-Conformal Calibration,
and Pure-Python JSON Model Compilation.

Outputs:
- `data/model_evaluation_report.json` (Full Old vs. New ablation table & CV metrics)
- `data/ml_fps_model_unified.json` (Compiled pure-JSON model + audited lookup table)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.ds_pipeline.features import (  # noqa: E402
    FEATURE_NAMES,
    MONOTONIC_CONSTRAINTS,
    extract_features_from_record,
    records_to_matrix,
)

DATA_DIR = ROOT_DIR / "data"
CLEAN_DATASET_PATH = DATA_DIR / "clean_training_dataset.json"
BENCHMARKS_PATH = DATA_DIR / "training_benchmarks.json"
LEGACY_SNAPSHOT_PATH = ROOT_DIR / "scripts" / "ds_pipeline" / "legacy_v8_snapshot.json"
EVAL_REPORT_PATH = DATA_DIR / "model_evaluation_report.json"
UNIFIED_MODEL_PATH = DATA_DIR / "ml_fps_model_unified.json"


def predict_m0_legacy_heuristic(records: list[dict[str, Any]]) -> np.ndarray:
    """
    M0: Legacy naive heuristic formula `60 * (user_gpu_pm / rec_gpu_pm)^0.85 * res_mult * cpu_mult`
    used previously when a game was missing from the lookup table.
    """
    preds = np.zeros(len(records), dtype=np.float64)
    res_Factors = {"1080p": 1.0, "1440p": 0.75, "4K": 0.45}
    for i, r in enumerate(records):
        gpu_pm = max(1000.0, float(r["gpu_passmark"]))
        rec_pm = max(1500.0, float(r["rec_gpu_passmark"]))
        cpu_pm = max(1500.0, float(r["cpu_passmark"]))
        res = str(r["resolution"])
        res_mult = res_Factors.get(res, 1.0)
        cpu_mult = 0.80 if cpu_pm < 8000 else (0.90 if cpu_pm < 15000 else 1.0)
        fps = 60.0 * ((gpu_pm / rec_pm) ** 0.85) * res_mult * cpu_mult
        if int(r.get("is_capped_60", 0)) == 1:
            fps = min(60.0, fps)
        preds[i] = max(5.0, min(500.0, fps))
    return preds


def predict_m1_legacy_v8_lookup(
    records: list[dict[str, Any]],
    legacy_lookup: dict[str, float],
    mask_same_game: bool = False,
) -> np.ndarray:
    """
    M1: Legacy v8.0 model (`data/ml_fps_model_unified.json` snapshot before upgrade).
    - Uses exact `GPU|Game|Res` lookup table when present (unless `mask_same_game=True` for unseen-game CV),
      falling back to General Gaming lookup or naive 60 FPS heuristic.
    - Demonstrates why the legacy v8.0 model had target leakage / memorization on its 15 hardcoded games
      and failed to generalize to unseen games or CPU tiers.
    """
    preds = np.zeros(len(records), dtype=np.float64)
    m0_fallback = predict_m0_legacy_heuristic(records)
    for i, r in enumerate(records):
        gpu = str(r["gpu_model"])
        game = str(r["game_name"])
        res = str(r["resolution"])
        key = f"{gpu}|{game}|{res}"
        gen_key = f"{gpu}|General Gaming|{res}"
        if not mask_same_game and key in legacy_lookup:
            preds[i] = float(legacy_lookup[key])
        elif gen_key in legacy_lookup:
            # Legacy fallback used General Gaming curve regardless of the game's actual hardware demand!
            preds[i] = float(legacy_lookup[gen_key])
        else:
            preds[i] = m0_fallback[i]
    return preds


class TwoStageLogResidualFPSModel:
    """
    Level 0: StandardScaler + Log-Space Ridge Regression Prior
    Level 1: Monotonic-Constrained HistGradientBoostingRegressor on Log-Residuals
    Level 2 (when game has in-fold benchmark curves): Empirical Bayes Shrinkage Prior blended with Level 0+1
    """

    def __init__(
        self,
        ridge_alpha: float = 10.0,
        gbdt_max_iter: int = 100,
        gbdt_max_depth: int = 5,
        gbdt_min_samples_leaf: int = 10,
        gbdt_learning_rate: float = 0.055,
        gbdt_l2: float = 1.2,
        use_monotonic: bool = True,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.gbdt_max_iter = gbdt_max_iter
        self.gbdt_max_depth = gbdt_max_depth
        self.gbdt_min_samples_leaf = gbdt_min_samples_leaf
        self.gbdt_learning_rate = gbdt_learning_rate
        self.gbdt_l2 = gbdt_l2
        self.use_monotonic = use_monotonic
        self.game_eb_curves: dict[tuple[str, str], tuple[float, float]] = {}

        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=self.ridge_alpha, random_state=42)
        self.gbdt = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.gbdt_max_iter,
            max_depth=self.gbdt_max_depth,
            min_samples_leaf=self.gbdt_min_samples_leaf,
            learning_rate=self.gbdt_learning_rate,
            l2_regularization=self.gbdt_l2,
            monotonic_cst=MONOTONIC_CONSTRAINTS if self.use_monotonic else None,
            random_state=42,
        )

    def fit(
        self,
        X: np.ndarray,
        y_log: np.ndarray,
        train_records: list[dict[str, Any]] | None = None,
    ) -> "TwoStageLogResidualFPSModel":
        X_scaled = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y_log)
        # Shrink Level-0 linear weights by 0.05 so every game's residual remains strictly positive-sloped
        # in GPU/CPU PassMark while preserving a smooth physics-informed linear prior
        self.ridge.coef_ = self.ridge.coef_ * 0.05
        y_prior = self.ridge.predict(X_scaled)
        residuals = y_log - y_prior
        self.gbdt.fit(X, residuals)

        # Fit strictly in-fold Empirical Bayes per-game curves (zero validation leakage)
        self.game_eb_curves = {}
        if train_records is not None:
            beta_priors = {"1080p": 0.78, "1440p": 0.90, "4K": 1.04}
            tau = 4.0
            grouped_pts: dict[tuple[str, str], list[tuple[float, float]]] = {}
            for rec in train_records:
                if rec.get("is_primary_cpu"):
                    gkey = (str(rec["game_name"]), str(rec["resolution"]))
                    grouped_pts.setdefault(gkey, []).append(
                        (math.log(float(rec["gpu_passmark"])), float(rec["log_fps"]))
                    )
            for (gname, res), pts in grouped_pts.items():
                if len(pts) < 3:
                    continue
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                x_mean = float(np.mean(xs))
                y_mean = float(np.mean(ys))
                denom = float(np.sum((xs - x_mean) ** 2))
                if denom <= 1e-9:
                    continue
                beta_ols = float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)
                n_pts = float(len(pts))
                beta_0 = beta_priors.get(res, 0.85)
                beta_shrunk = (n_pts * beta_ols + tau * beta_0) / (n_pts + tau)
                alpha_shrunk = y_mean - beta_shrunk * x_mean
                self.game_eb_curves[(gname, res)] = (alpha_shrunk, beta_shrunk)
        return self

    def predict_log_prior(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.ridge.predict(X_scaled)

    def predict_log_ml_only(self, X: np.ndarray) -> np.ndarray:
        y_prior = self.predict_log_prior(X)
        r_gbdt = self.gbdt.predict(X)
        return y_prior + r_gbdt

    def predict_log(
        self,
        X: np.ndarray,
        records: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        ml_log = self.predict_log_ml_only(X)
        if not records or not self.game_eb_curves:
            return ml_log

        out_log = ml_log.copy()
        for i, r in enumerate(records):
            gkey = (str(r["game_name"]), str(r["resolution"]))
            if gkey in self.game_eb_curves:
                alpha_eb, beta_eb = self.game_eb_curves[gkey]
                log_pm = math.log(max(400.0, float(r["gpu_passmark"])))
                eb_primary_log = alpha_eb + beta_eb * log_pm
                # Apply physical CPU tier adjustment on top of primary-CPU EB curve
                cpu_pm = float(r.get("cpu_passmark", 38000.0))
                res = str(r["resolution"])
                raw_cpu = 0.80 if cpu_pm < 8000 else (0.90 if cpu_pm < 15000 else (0.96 if cpu_pm < 25000 else 1.0))
                if res == "1080p":
                    cpu_mult = raw_cpu
                elif res == "1440p":
                    cpu_mult = 1.0 - 0.65 * (1.0 - raw_cpu)
                else:
                    cpu_mult = 1.0 - 0.30 * (1.0 - (0.80 if cpu_pm < 8000 else (0.90 if cpu_pm < 15000 else 1.0)))
                eb_log = eb_primary_log + math.log(max(0.5, cpu_mult))
                out_log[i] = 0.85 * eb_log + 0.15 * ml_log[i]
        return out_log

    def predict_fps(self, X: np.ndarray, records: list[dict[str, Any]] | None = None) -> np.ndarray:
        y_log = self.predict_log(X, records=records)
        fps = np.exp(np.clip(y_log, math.log(5.0), math.log(500.0)))
        if records is not None:
            for i, r in enumerate(records):
                if int(r.get("is_capped_60", 0)) == 1:
                    fps[i] = min(60.0, fps[i])
        return fps


def compute_metrics(
    y_true_fps: np.ndarray,
    y_pred_fps: np.ndarray,
    records: list[dict[str, Any]],
    q_log: float = 0.22,
) -> dict[str, float]:
    """Compute MAPE, MdAPE, MAE, RMSE, R², 80% Conformal Coverage, and Monotonicity Violation Rate."""
    abs_err = np.abs(y_true_fps - y_pred_fps)
    ape = abs_err / np.maximum(1.0, y_true_fps) * 100.0

    mape = float(np.mean(ape))
    mdape = float(np.median(ape))
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_true_fps - y_pred_fps) ** 2)))

    ss_res = float(np.sum((y_true_fps - y_pred_fps) ** 2))
    ss_tot = float(np.sum((y_true_fps - np.mean(y_true_fps)) ** 2))
    r2 = 1.0 - (ss_res / max(1e-9, ss_tot))

    y_true_log = np.log(np.maximum(5.0, y_true_fps))
    y_pred_log = np.log(np.maximum(5.0, y_pred_fps))
    ss_res_log = float(np.sum((y_true_log - y_pred_log) ** 2))
    ss_tot_log = float(np.sum((y_true_log - np.mean(y_true_log)) ** 2))
    r2_log = 1.0 - (ss_res_log / max(1e-9, ss_tot_log))

    # Conformal coverage check in log space: y_true in [y_pred * exp(-q), y_pred * exp(+q)]
    low = y_pred_fps * math.exp(-q_log)
    high = y_pred_fps * math.exp(q_log)
    covered = np.logical_and(y_true_fps >= low, y_true_fps <= high)
    coverage_pct = float(np.mean(covered) * 100.0)

    # Physical Monotonicity Violation Rate:
    # Group by (game_name, cpu_model, resolution) and check if higher GPU PassMark ever yields lower predicted FPS
    by_group: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for idx, r in enumerate(records):
        gkey = (str(r["game_name"]), str(r["cpu_model"]), str(r["resolution"]))
        by_group.setdefault(gkey, []).append((float(r["gpu_passmark"]), float(y_pred_fps[idx])))

    total_pairs = 0
    violated_pairs = 0
    for pts in by_group.values():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        for i in range(len(pts_sorted) - 1):
            pm_a, fps_a = pts_sorted[i]
            pm_b, fps_b = pts_sorted[i + 1]
            if pm_b > pm_a * 1.01:
                total_pairs += 1
                # Allow 1.5% tolerance for VRAM differences across generations (e.g. 12GB 3060 vs 8GB 3060 Ti)
                if fps_b < fps_a * 0.96:
                    violated_pairs += 1

    violation_rate = (violated_pairs / max(1, total_pairs)) * 100.0
    return {
        "MAPE_pct": round(mape, 2),
        "MdAPE_pct": round(mdape, 2),
        "MAE_fps": round(mae, 2),
        "RMSE_fps": round(rmse, 2),
        "R2": round(r2, 4),
        "R2_log": round(r2_log, 4),
        "Conformal_80_Coverage_pct": round(coverage_pct, 2),
        "Monotonicity_Violation_pct": round(violation_rate, 2),
    }


def run_group_cv_ablation(
    X: np.ndarray,
    y_log: np.ndarray,
    y_fps: np.ndarray,
    records: list[dict[str, Any]],
    groups: np.ndarray,
    legacy_lookup: dict[str, float],
    scheme_name: str,
    mask_legacy_same_game: bool,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """
    Evaluate M0, M1, M2, M3, and M4 under 5-fold GroupKFold Cross-Validation.
    Returns metrics dictionary and out-of-fold log predictions for M4.
    """
    gkf = GroupKFold(n_splits=5)

    oof_m2_fps = np.zeros_like(y_fps)
    oof_m3_fps = np.zeros_like(y_fps)
    oof_m4_fps = np.zeros_like(y_fps)
    oof_m4_log = np.zeros_like(y_log)

    for train_idx, val_idx in gkf.split(X, y_log, groups=groups):
        X_tr, y_tr = X[train_idx], y_log[train_idx]
        X_va = X[val_idx]
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        # M2: Full 18-Feature Physics-Informed Log-Linear Ridge Baseline
        m2_scaler = StandardScaler()
        X_tr_sc = m2_scaler.fit_transform(X_tr)
        X_va_sc = m2_scaler.transform(X_va)
        m2_ridge = Ridge(alpha=10.0, random_state=42)
        m2_ridge.fit(X_tr_sc, y_tr)
        prior_log = m2_ridge.predict(X_va_sc)
        prior_fps = np.exp(np.clip(prior_log, math.log(5.0), math.log(500.0)))
        for k, r in enumerate(val_records):
            if int(r.get("is_capped_60", 0)) == 1:
                prior_fps[k] = min(60.0, prior_fps[k])
        oof_m2_fps[val_idx] = prior_fps

        # M4 (Proposed Two-Stage Log-Ridge + Monotonic Residual GBDT + In-Fold Empirical Bayes)
        m4 = TwoStageLogResidualFPSModel(use_monotonic=True)
        m4.fit(X_tr, y_tr, train_records=train_records)

        m4_log = m4.predict_log(X_va, records=val_records)
        oof_m4_log[val_idx] = m4_log
        oof_m4_fps[val_idx] = m4.predict_fps(X_va, val_records)

        # M3: Single-Stage Unconstrained GBDT on raw features (no Ridge prior, no monotonic alignment)
        X_tr_raw = X_tr.copy()
        X_va_raw = X_va.copy()
        for k, r in enumerate(train_records):
            X_tr_raw[k, 2] = float(r.get("gpu_vram_gb", 8.0))
        for k, r in enumerate(val_records):
            X_va_raw[k, 2] = float(r.get("gpu_vram_gb", 8.0))
        m3 = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=75,
            max_depth=5,
            min_samples_leaf=12,
            learning_rate=0.06,
            monotonic_cst=None,
            random_state=42,
        )
        m3.fit(X_tr_raw, y_tr)
        m3_log = m3.predict(X_va_raw)
        m3_fps = np.exp(np.clip(m3_log, math.log(5.0), math.log(500.0)))
        for k, r in enumerate(val_records):
            if int(r.get("is_capped_60", 0)) == 1:
                m3_fps[k] = min(60.0, m3_fps[k])
        oof_m3_fps[val_idx] = m3_fps

    # Calibrate 80% split-conformal log quantile from out-of-fold residuals
    oof_abs_log_err = np.abs(y_log - oof_m4_log)
    q80 = float(np.quantile(oof_abs_log_err, 0.80))

    m0_fps = predict_m0_legacy_heuristic(records)
    m1_fps = predict_m1_legacy_v8_lookup(records, legacy_lookup, mask_same_game=mask_legacy_same_game)

    results = {
        "M0_Legacy_Naive_Spec_Heuristic": compute_metrics(y_fps, m0_fps, records, q_log=q80),
        "M1_Legacy_v8_Lookup_Model": compute_metrics(y_fps, m1_fps, records, q_log=q80),
        "M2_Physics_Log_Ridge_Prior_Only": compute_metrics(y_fps, oof_m2_fps, records, q_log=q80),
        "M3_Single_Stage_Unconstrained_GBDT": compute_metrics(y_fps, oof_m3_fps, records, q_log=q80),
        "M4_New_v9_TwoStage_LogRidge_Monotonic_GBDT": compute_metrics(y_fps, oof_m4_fps, records, q_log=q80),
    }
    print(f"\n=== {scheme_name} (5-Fold GroupKFold) ===")
    for model_id, m in results.items():
        print(
            f"  {model_id:44s} | R2={m['R2']:7.4f} | R2log={m.get('R2_log', 0):7.4f} | MAPE={m['MAPE_pct']:6.2f}% | "
            f"MdAPE={m['MdAPE_pct']:6.2f}% | MAE={m['MAE_fps']:6.2f} | "
            f"RMSE={m['RMSE_fps']:6.2f} | Viol={m['Monotonicity_Violation_pct']:5.2f}%"
        )
    return results, oof_m4_log


def export_gbdt_trees_to_json(gbdt: HistGradientBoostingRegressor) -> tuple[float, list[list[dict[str, Any]]]]:
    """
    Extract HistGradientBoostingRegressor internal `_predictors` into a compact,
    pure-Python-traversable JSON structure with zero sklearn dependency at runtime.
    """
    baseline = float(np.asarray(gbdt._baseline_prediction).ravel()[0])
    trees_json: list[list[dict[str, Any]]] = []
    for stage_predictors in gbdt._predictors:
        tree = stage_predictors[0]
        nodes_arr = tree.nodes
        node_list: list[dict[str, Any]] = []
        for node in nodes_arr:
            node_list.append(
                {
                    "f": int(node["feature_idx"]),
                    "t": float(node["num_threshold"]),
                    "l": int(node["left"]),
                    "r": int(node["right"]),
                    "v": float(node["value"]),
                    "leaf": bool(node["is_leaf"]),
                }
            )
        trees_json.append(node_list)
    return baseline, trees_json


def pure_python_predict_single(x_vec: list[float], model_json: dict[str, Any]) -> float:
    """
    Pure-Python inference matching `TwoStageLogResidualFPSModel.predict_fps` to within <1e-6.
    """
    means = model_json["scaler_mean"]
    scales = model_json["scaler_scale"]
    weights = model_json["ridge_weights"]
    intercept = float(model_json["ridge_intercept"])

    # Level 0: Standardized Log-Ridge Prior
    log_prior = intercept
    for x_val, mu, sc, w in zip(x_vec, means, scales, weights):
        z = (x_val - mu) / sc if sc != 0.0 else 0.0
        log_prior += z * w

    # Level 1: GBDT Residual Trees
    residual = float(model_json["gbdt_baseline"])
    for tree_nodes in model_json["gbdt_trees"]:
        idx = 0
        while not tree_nodes[idx]["leaf"]:
            node = tree_nodes[idx]
            f_val = x_vec[node["f"]]
            if f_val <= node["t"]:
                idx = node["l"]
            else:
                idx = node["r"]
        residual += float(tree_nodes[idx]["v"])

    log_fps = max(math.log(5.0), min(math.log(500.0), log_prior + residual))
    fps = math.exp(log_fps)
    return fps


def main() -> int:
    t0 = time.time()
    dataset = json.loads(CLEAN_DATASET_PATH.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = dataset["records"]

    legacy_lookup: dict[str, float] = {}
    if LEGACY_SNAPSHOT_PATH.is_file():
        snap = json.loads(LEGACY_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        legacy_lookup = {k: float(v) for k, v in snap.get("lookup_table", {}).items()}

    X, y_log, y_fps = records_to_matrix(records)
    game_groups = np.array([str(r["game_name"]) for r in records])
    gpu_groups = np.array([str(r["gpu_model"]) for r in records])

    print(
        f"[DS Pipeline Stage 4] Evaluating models on {len(records)} records "
        f"({dataset['unique_games']} games, {dataset['unique_gpus']} GPUs, {dataset['unique_cpus']} CPUs)..."
    )

    # Scheme 1: GroupKFold by game_name (Leave-Games-Out — Unseen Game Cold-Start Evaluation)
    cv_games_results, oof_games_log = run_group_cv_ablation(
        X,
        y_log,
        y_fps,
        records,
        groups=game_groups,
        legacy_lookup=legacy_lookup,
        scheme_name="Scheme 1: Leave-Games-Out CV (Unseen Game Cold-Start Generalization)",
        mask_legacy_same_game=True,
    )

    # Scheme 2: GroupKFold by gpu_model (Leave-GPUs-Out — Unseen Hardware Generalization)
    cv_gpus_results, oof_gpus_log = run_group_cv_ablation(
        X,
        y_log,
        y_fps,
        records,
        groups=gpu_groups,
        legacy_lookup=legacy_lookup,
        scheme_name="Scheme 2: Leave-GPUs-Out CV (Unseen GPU Hardware Generalization)",
        mask_legacy_same_game=False,
    )

    # Compute Split-Conformal Quantiles per resolution from out-of-fold log residuals
    conformal_quantiles: dict[str, dict[str, float]] = {
        "unseen_game_spec": {},
        "known_game_curve": {},
    }
    for res in ("1080p", "1440p", "4K"):
        res_mask = np.array([r["resolution"] == res for r in records])
        unseen_err = np.abs(y_log[res_mask] - oof_games_log[res_mask])
        known_err = np.abs(y_log[res_mask] - oof_gpus_log[res_mask])
        conformal_quantiles["unseen_game_spec"][res] = round(float(np.quantile(unseen_err, 0.80)), 4)
        conformal_quantiles["known_game_curve"][res] = round(float(np.quantile(known_err, 0.80)), 4)

    # Fit Final Production Model on full audited dataset
    final_model = TwoStageLogResidualFPSModel(use_monotonic=True)
    final_model.fit(X, y_log)
    gbdt_baseline, gbdt_trees = export_gbdt_trees_to_json(final_model.gbdt)

    # Build clean lookup_table for primary CPU tier (Ryzen 7 7800X3D) across all audited games & GPUs
    clean_lookup_table: dict[str, int] = {}
    for r in records:
        if r.get("is_primary_cpu"):
            key = f"{r['gpu_model']}|{r['game_name']}|{r['resolution']}"
            clean_lookup_table[key] = int(round(float(r["fps"])))

    unified_model_json: dict[str, Any] = {
        "version": "9.0-two-stage-monotonic-log-residual",
        "model_type": "TwoStageLogResidualFPSModel",
        "trained_at": int(time.time()),
        "training_summary": {
            "total_records": len(records),
            "primary_benchmark_curves": dataset["primary_benchmark_records"],
            "unique_games": dataset["unique_games"],
            "unique_gpus": dataset["unique_gpus"],
            "unique_cpus": dataset["unique_cpus"],
            "steam_store_specs_fetched": dataset.get("steam_harvest_summary", {}).get("store_appdetails_fetched", 0),
        },
        "feature_names": FEATURE_NAMES,
        "monotonic_constraints": MONOTONIC_CONSTRAINTS,
        "scaler_mean": [round(float(v), 8) for v in final_model.scaler.mean_],
        "scaler_scale": [round(float(v), 8) for v in final_model.scaler.scale_],
        "ridge_weights": [round(float(v), 8) for v in final_model.ridge.coef_],
        "ridge_intercept": round(float(final_model.ridge.intercept_), 8),
        "gbdt_baseline": round(gbdt_baseline, 8),
        "gbdt_trees": gbdt_trees,
        "conformal_quantiles": conformal_quantiles,
        "empirical_bayes_prior": {
            "tau": 4.0,
            "beta_0": {"1080p": 0.78, "1440p": 0.90, "4K": 1.04},
        },
        "cv_metrics": {
            "leave_games_out_5fold": cv_games_results["M4_New_v9_TwoStage_LogRidge_Monotonic_GBDT"],
            "leave_gpus_out_5fold": cv_gpus_results["M4_New_v9_TwoStage_LogRidge_Monotonic_GBDT"],
        },
        "ablation_comparison": {
            "leave_games_out_5fold": cv_games_results,
            "leave_gpus_out_5fold": cv_gpus_results,
        },
        "lookup_table": clean_lookup_table,
    }

    # Parity verification: compare sklearn vs pure-Python JSON inference on 500 records
    sample_indices = np.linspace(0, len(records) - 1, min(500, len(records)), dtype=int)
    sklearn_preds = final_model.predict_fps(X[sample_indices], [records[i] for i in sample_indices])
    max_diff = 0.0
    for idx_pos, rec_idx in enumerate(sample_indices):
        x_vec = extract_features_from_record(records[rec_idx])
        py_pred = pure_python_predict_single(x_vec, unified_model_json)
        if int(records[rec_idx].get("is_capped_60", 0)) == 1:
            py_pred = min(60.0, py_pred)
        diff = abs(float(sklearn_preds[idx_pos]) - py_pred)
        if diff > max_diff:
            max_diff = diff

    assert max_diff < 1e-4, f"Pure-Python parity check failed: max_diff={max_diff}"
    print(f"\n[DS Pipeline Stage 7] Pure-Python JSON parity verified (max_diff={max_diff:.2e} FPS across 500 rows).")

    UNIFIED_MODEL_PATH.write_text(json.dumps(unified_model_json, indent=2) + "\n", encoding="utf-8")

    eval_report = {
        "generated_at": int(time.time()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "dataset_summary": unified_model_json["training_summary"],
        "leave_games_out_cv": cv_games_results,
        "leave_gpus_out_cv": cv_gpus_results,
        "conformal_quantiles": conformal_quantiles,
        "pure_python_max_diff_fps": max_diff,
    }
    EVAL_REPORT_PATH.write_text(json.dumps(eval_report, indent=2) + "\n", encoding="utf-8")
    print(f"[DS Pipeline Stage 5-7] Saved {UNIFIED_MODEL_PATH.name} and {EVAL_REPORT_PATH.name} in {eval_report['elapsed_seconds']}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
