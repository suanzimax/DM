"""模型训练、约束处理、候选点生成和 BO 推荐。"""

import numpy as np
import warnings
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    N_ACTUATORS,
    VOLTAGE_BOUNDS,
    MAX_DELTA_FROM_BASELINE,
    OPTIMIZED_DIM_INDICES,
    MIN_TRUST_RADIUS,
    GPR_MIN_SAMPLES_TO_TRAIN,
    GPR_NOISE_FLOOR,
    RF_MIN_SAMPLES_TO_TRAIN,
    RF_TEST_SIZE,
    RF_RANDOM_STATE,
)


def train_surrogate(X, y_mean, y_std, mode_params, sample_weights=None):
    """训练两个 GPR：目标均值模型 + shot 波动模型。"""
    if len(y_mean) < GPR_MIN_SAMPLES_TO_TRAIN:
        return None, None

    X = np.asarray(X, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_std = np.asarray(y_std, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dim = X_scaled.shape[1]

    objective_kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e3))
    )
    noise_kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e3))
    )

    if sample_weights is not None:
        alpha = 1.0 / np.maximum(np.asarray(sample_weights, dtype=float), GPR_NOISE_FLOOR)
        alpha = alpha / np.mean(alpha) * max(float(np.var(y_mean)) * 0.02, GPR_NOISE_FLOOR)
    else:
        alpha = np.full(len(y_mean), max(float(np.var(y_mean)) * 0.02, GPR_NOISE_FLOOR))

    objective_gpr = GaussianProcessRegressor(
        kernel=objective_kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=mode_params["bo"]["random_state"],
    )
    noise_gpr = GaussianProcessRegressor(
        kernel=noise_kernel,
        alpha=GPR_NOISE_FLOOR,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=mode_params["bo"]["random_state"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        objective_gpr.fit(X_scaled, y_mean)
        noise_gpr.fit(X_scaled, np.log(np.square(np.maximum(y_std, GPR_NOISE_FLOOR))))

    if len(y_mean) >= max(8, min(RF_MIN_SAMPLES_TO_TRAIN, len(y_mean))):
        X_train, X_val, y_train, y_val, alpha_train, _ = train_test_split(
            X_scaled,
            y_mean,
            alpha,
            test_size=RF_TEST_SIZE,
            random_state=RF_RANDOM_STATE,
        )
        val_model = GaussianProcessRegressor(
            kernel=objective_kernel,
            alpha=alpha_train,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=mode_params["bo"]["random_state"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            val_model.fit(X_train, y_train)
            val_pred = val_model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    else:
        rmse = float(np.sqrt(mean_squared_error(y_mean, objective_gpr.predict(X_scaled))))

    return {
        "objective": objective_gpr,
        "noise": noise_gpr,
        "scaler": scaler,
        "target_mean": float(np.mean(y_mean)),
        "target_std": float(np.std(y_mean)),
        "noise_log_floor": float(np.log(GPR_NOISE_FLOOR)),
    }, rmse


def load_baseline_vector(filepath):
    """读取基准 DM 配置第一行电压数据。"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"{filepath} 为空")

    baseline = np.array(list(map(int, lines[0].strip().split("\t"))), dtype=int)
    if len(baseline) != N_ACTUATORS:
        raise ValueError(
            f"{filepath} 维度错误，期望 {N_ACTUATORS} 维，实际 {len(baseline)} 维"
        )
    return baseline


def build_constrained_bounds(baseline):
    """构建相对基准面型的安全搜索范围。"""
    lower = np.maximum(VOLTAGE_BOUNDS[0], baseline - MAX_DELTA_FROM_BASELINE)
    upper = np.minimum(VOLTAGE_BOUNDS[1], baseline + MAX_DELTA_FROM_BASELINE)

    frozen_mask = np.ones(N_ACTUATORS, dtype=bool)
    frozen_mask[OPTIMIZED_DIM_INDICES] = False
    lower[frozen_mask] = baseline[frozen_mask]
    upper[frozen_mask] = baseline[frozen_mask]
    return lower.astype(int), upper.astype(int)


def enforce_hard_constraints(vector, baseline):
    """应用所有硬性约束。"""
    lower_bounds, upper_bounds = build_constrained_bounds(baseline)
    constrained = np.clip(np.asarray(vector, dtype=float), lower_bounds, upper_bounds)

    frozen_mask = np.ones(N_ACTUATORS, dtype=bool)
    frozen_mask[OPTIMIZED_DIM_INDICES] = False
    constrained[frozen_mask] = baseline[frozen_mask]
    return constrained.astype(int)


def compute_feature_importance(models):
    """用 GPR 的 RBF length scale 估计敏感维度；失败时返回均匀权重。"""
    if not models or "objective" not in models:
        return np.ones(len(OPTIMIZED_DIM_INDICES), dtype=float) / len(OPTIMIZED_DIM_INDICES)

    try:
        length_scale = np.asarray(models["objective"].kernel_.k1.k2.length_scale, dtype=float)
        importance = 1.0 / np.maximum(length_scale, 1e-6)
    except Exception:
        importance = np.ones(len(OPTIMIZED_DIM_INDICES), dtype=float)

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
    lower_opt = lower_bounds[OPTIMIZED_DIM_INDICES]
    upper_opt = upper_bounds[OPTIMIZED_DIM_INDICES]

    if len(y_obs) > 0:
        best_idx = int(np.argmax(y_obs))
        center = np.asarray(X_obs[best_idx], dtype=int)
    else:
        center = baseline[OPTIMIZED_DIM_INDICES]

    adaptive_lower = np.maximum(lower_opt, center - trust_radius)
    adaptive_upper = np.minimum(upper_opt, center + trust_radius)

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
                perturbed = np.clip(perturbed, adaptive_lower, adaptive_upper)
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
    baseline_points = np.clip(baseline_points, lower_opt, upper_opt)

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
    """用目标 GPR 预测均值和模型不确定性。"""
    X_scaled = models["scaler"].transform(np.asarray(X, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_pred, std_pred = models["objective"].predict(X_scaled, return_std=True)
    mean_pred = np.nan_to_num(
        mean_pred,
        nan=models["target_mean"],
        posinf=models["target_mean"] + 5.0 * max(models["target_std"], 1.0),
        neginf=models["target_mean"] - 5.0 * max(models["target_std"], 1.0),
    )
    std_pred = np.nan_to_num(
        std_pred,
        nan=max(models["target_std"], 1.0),
        posinf=max(models["target_std"], 1.0),
        neginf=1.0,
    )
    return mean_pred, np.maximum(std_pred, 1e-12)


def predict_noise_std(models, X):
    """用噪声 GPR 预测该参数点的 shot 波动标准差。"""
    X_scaled = models["scaler"].transform(np.asarray(X, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_var = models["noise"].predict(X_scaled)
    log_var = np.nan_to_num(
        log_var,
        nan=models["noise_log_floor"],
        posinf=np.log(max(models["target_std"] ** 2, GPR_NOISE_FLOOR)),
        neginf=models["noise_log_floor"],
    )
    return np.sqrt(np.exp(log_var))


def expected_improvement(mean_pred, std_pred, best_y, xi=0.01):
    """标准 Expected Improvement。"""
    std_pred = np.maximum(std_pred, 1e-12)
    improvement = mean_pred - best_y - xi
    z = improvement / std_pred
    ei = improvement * norm.cdf(z) + std_pred * norm.pdf(z)
    return np.maximum(ei, 0.0)


def augmented_acquisition(mean_pred, model_std, noise_std, best_y, xi=0.01):
    """增强采集函数：EI * eta，惩罚高 shot 波动区域。"""
    ei = expected_improvement(mean_pred, model_std, best_y, xi=xi)
    eta = 1.0 - noise_std / np.sqrt(model_std**2 + noise_std**2 + 1e-12)
    return ei * np.clip(eta, 0.0, 1.0)


def propose_next(models, baseline, X_obs, y_obs, mode_params):
    """利用候选搜索 + 增强 EI 推荐下一个点。"""
    importance_weights = compute_feature_importance(models)
    trust_radius = compute_trust_radius(y_obs)
    candidates, trust_radius = build_candidate_pool(
        X_obs, y_obs, baseline, mode_params, importance_weights, trust_radius
    )

    mean_pred, std_pred = predict_with_uncertainty(models, candidates)
    noise_std = predict_noise_std(models, candidates)
    acq = augmented_acquisition(
        mean_pred,
        std_pred,
        noise_std,
        float(np.max(y_obs)),
        xi=mode_params["bo"].get("xi", 0.01),
    )
    acq = np.nan_to_num(acq, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
    best_idx = int(np.argmax(acq))

    x_next = baseline.astype(float).copy()
    x_next[OPTIMIZED_DIM_INDICES] = candidates[best_idx]
    x_next = enforce_hard_constraints(x_next, baseline)

    ranked_dims = np.argsort(importance_weights)[::-1]
    top_dims = [
        {
            "dimension": f"a{int(OPTIMIZED_DIM_INDICES[idx])}",
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
