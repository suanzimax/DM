"""模型训练、约束处理、候选点生成和 BO 推荐。"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from config import (
    N_ACTUATORS,
    VOLTAGE_BOUNDS,
    BASELINE_FILE,
    MAX_DELTA_FROM_BASELINE,
    FROZEN_START_IDX,
    OPTIMIZED_DIM_INDICES,
    MIN_TRUST_RADIUS,
    RF_MIN_SAMPLES_TO_TRAIN,
    RF_TEST_SIZE,
    RF_RANDOM_STATE,
)


def train_surrogate(X, y, mode_params, sample_weights=None):
    """训练 surrogate 集成模型并返回 holdout RMSE。"""
    if len(y) < RF_MIN_SAMPLES_TO_TRAIN:
        return None, None
    X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
        X,
        y,
        sample_weights if sample_weights is not None else np.ones(len(y)),
        test_size=RF_TEST_SIZE,
        random_state=RF_RANDOM_STATE,
    )
    rf = RandomForestRegressor(
        n_estimators=mode_params["rf"]["n_trees"],
        max_depth=mode_params["rf"]["max_depth"],
        min_samples_split=mode_params["rf"]["min_samples_split"],
        min_samples_leaf=mode_params["rf"]["min_samples_leaf"],
        max_features=mode_params["rf"]["max_features"],
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    et = ExtraTreesRegressor(
        n_estimators=max(120, mode_params["rf"]["n_trees"] // 2),
        max_depth=mode_params["rf"]["max_depth"],
        min_samples_split=mode_params["rf"]["min_samples_split"],
        min_samples_leaf=mode_params["rf"]["min_samples_leaf"],
        max_features=mode_params["rf"]["max_features"],
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)
    et.fit(X_train, y_train, sample_weight=w_train)
    y_pred = 0.5 * (rf.predict(X_val) + et.predict(X_val))
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return {"rf": rf, "et": et}, rmse


def load_baseline_vector(filepath):
    """读取基准 DM 配置第一行电压数据。"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"{filepath} 为空")

    baseline = np.array(list(map(int, lines[0].strip().split('\t'))), dtype=int)
    if len(baseline) != N_ACTUATORS:
        raise ValueError(
            f"{filepath} 维度错误，期望 {N_ACTUATORS} 维，实际 {len(baseline)} 维"
        )
    return baseline


def build_constrained_bounds(baseline):
    """构建相对基准面型的安全搜索范围。"""
    lower = np.maximum(VOLTAGE_BOUNDS[0], baseline - MAX_DELTA_FROM_BASELINE)
    upper = np.minimum(VOLTAGE_BOUNDS[1], baseline + MAX_DELTA_FROM_BASELINE)
    lower[FROZEN_START_IDX:] = baseline[FROZEN_START_IDX:]
    upper[FROZEN_START_IDX:] = baseline[FROZEN_START_IDX:]
    return lower.astype(int), upper.astype(int)


def enforce_hard_constraints(vector, baseline):
    """应用所有硬性约束。"""
    lower_bounds, upper_bounds = build_constrained_bounds(baseline)
    constrained = np.clip(np.asarray(vector, dtype=float), lower_bounds, upper_bounds)
    constrained[FROZEN_START_IDX:] = baseline[FROZEN_START_IDX:]
    return constrained.astype(int)


def compute_feature_importance(models):
    """集成模型的平均特征重要性，用于加权探索。"""
    importance = np.zeros(len(OPTIMIZED_DIM_INDICES), dtype=float)
    model_count = 0
    for model in models.values():
        if hasattr(model, "feature_importances_"):
            importance += model.feature_importances_
            model_count += 1
    if model_count == 0:
        return np.ones(len(OPTIMIZED_DIM_INDICES), dtype=float) / len(OPTIMIZED_DIM_INDICES)
    importance = importance / model_count
    importance = np.maximum(importance, 1e-6)
    return importance / np.sum(importance)


def compute_trust_radius(y_obs):
    """根据近期改进幅度动态收缩/放宽搜索半径。"""
    if len(y_obs) < 8:
        return MAX_DELTA_FROM_BASELINE
    window = min(12, len(y_obs))
    recent = np.asarray(y_obs[-window:], dtype=float)
    best_recent = np.max(recent)
    prev_best = np.max(y_obs[:-window]) if len(y_obs) > window else np.max(y_obs)
    improvement_ratio = (best_recent - prev_best) / max(abs(prev_best), 1.0)
    if improvement_ratio > 0.08:
        return MAX_DELTA_FROM_BASELINE
    if improvement_ratio > 0.03:
        return max(MIN_TRUST_RADIUS, int(MAX_DELTA_FROM_BASELINE * 0.75))
    if improvement_ratio > 0.01:
        return max(MIN_TRUST_RADIUS, int(MAX_DELTA_FROM_BASELINE * 0.55))
    return max(MIN_TRUST_RADIUS, int(MAX_DELTA_FROM_BASELINE * 0.35))


def build_candidate_pool(X_obs, y_obs, baseline, mode_params, importance_weights, trust_radius):
    """全局随机 + 局部扰动，生成下一轮待筛选候选点。"""
    lower_bounds, upper_bounds = build_constrained_bounds(baseline)
    dim = len(OPTIMIZED_DIM_INDICES)
    adaptive_lower = lower_bounds.copy()
    adaptive_upper = upper_bounds.copy()

    if len(y_obs) > 0:
        best_idx = int(np.argmax(y_obs))
        center = np.asarray(X_obs[best_idx], dtype=int)
    else:
        center = baseline[:dim]

    adaptive_lower[:dim] = np.maximum(adaptive_lower[:dim], center - trust_radius)
    adaptive_upper[:dim] = np.minimum(adaptive_upper[:dim], center + trust_radius)
    rng = np.random.default_rng(mode_params["bo"]["random_state"])
    n_candidates = mode_params["bo"]["n_candidates"]
    global_count = max(2000, n_candidates // 2)
    global_points = np.empty((global_count, dim), dtype=int)
    for idx in range(dim):
        global_points[:, idx] = rng.integers(
            adaptive_lower[idx],
            adaptive_upper[idx] + 1,
            size=global_count,
        )

    local_candidates = []
    if len(y_obs) > 0:
        seed_count = min(mode_params["bo"]["n_local_seeds"], len(y_obs))
        top_indices = np.argsort(y_obs)[-seed_count:]
        scales = mode_params["bo"]["local_noise_scales"]
        per_seed = max(80, n_candidates // max(1, seed_count * len(scales)))
        for idx in top_indices:
            seed = X_obs[idx]
            for scale in scales:
                weighted_scale = scale * (0.5 + 2.0 * importance_weights)
                noise = rng.normal(0, weighted_scale, size=(per_seed, dim))
                perturbed = np.rint(seed + noise).astype(int)
                perturbed = np.clip(perturbed, adaptive_lower[:dim], adaptive_upper[:dim])
                local_candidates.append(perturbed)

    baseline_seed = np.repeat(
        center[None, :],
        repeats=max(128, n_candidates // 10),
        axis=0,
    )
    baseline_noise = rng.normal(
        0,
        np.maximum(4.0, trust_radius * (0.08 + importance_weights)),
        size=baseline_seed.shape,
    )
    baseline_points = np.rint(baseline_seed + baseline_noise).astype(int)
    baseline_points = np.clip(baseline_points, lower_bounds[:dim], upper_bounds[:dim])

    all_blocks = [global_points, baseline_points]
    if local_candidates:
        all_blocks.append(np.vstack(local_candidates))
    candidates = np.vstack(all_blocks)

    existing = {tuple(map(int, row)) for row in X_obs}
    filtered = []
    seen = set()
    for row in candidates:
        key = tuple(map(int, row))
        if key in existing or key in seen:
            continue
        seen.add(key)
        filtered.append(row)
    if not filtered:
        filtered.append(np.asarray(center, dtype=int))
    return np.asarray(filtered, dtype=int), int(trust_radius)


def predict_with_uncertainty(models, X):
    """用集成树模型的树间分散度近似预测不确定性。"""
    tree_preds = []
    for model in models.values():
        tree_preds.extend(est.predict(X) for est in model.estimators_)
    tree_matrix = np.vstack(tree_preds)
    mean_pred = np.mean(tree_matrix, axis=0)
    std_pred = np.std(tree_matrix, axis=0)
    return mean_pred, std_pred


def propose_next(models, baseline, X_obs, y_obs, mode_params):
    """利用候选搜索 + UCB 推荐下一个点。"""
    importance_weights = compute_feature_importance(models)
    trust_radius = compute_trust_radius(y_obs)
    candidates, trust_radius = build_candidate_pool(
        X_obs, y_obs, baseline, mode_params, importance_weights, trust_radius
    )
    mean_pred, std_pred = predict_with_uncertainty(models, candidates)
    beta = mode_params["bo"]["exploration_beta"]
    ucb = mean_pred + beta * std_pred
    best_idx = int(np.argmax(ucb))

    x_next = baseline.astype(float).copy()
    x_next[OPTIMIZED_DIM_INDICES] = candidates[best_idx]
    x_next = enforce_hard_constraints(x_next, baseline)
    ranked_dims = np.argsort(importance_weights)[::-1]
    top_dims = [
        {
            "dimension": f"a{int(idx)}",
            "importance": float(importance_weights[idx]),
        }
        for idx in ranked_dims[: min(5, len(ranked_dims))]
    ]
    return (
        x_next,
        float(mean_pred[best_idx]),
        float(std_pred[best_idx]),
        int(len(candidates)),
        int(trust_radius),
        top_dims,
    )
