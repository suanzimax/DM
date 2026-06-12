"""通用工具：日志、格式化、训练效果评估、样本权重等。"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from config import (
    PROCESS_LOG_DIR,
    PROCESS_LOG_FILE,
    TRAINING_METRICS_FILE,
    LATEST_TRAINING_SUMMARY_FILE,
    LATEST_RECOMMENDATION_SIGNAL_FILE,
    RF_MIN_SAMPLES_TO_TRAIN,
    RF_RANDOM_STATE,
)


def ensure_runtime_log_dir():
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)


def format_sci(value):
    """统一用科学计数法展示 GUI 数值。"""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return str(value)


def append_operation_log(action, status, details=None):
    """将关键操作写入结构化运行日志。"""
    ensure_runtime_log_dir()
    payload = json.dumps(details or {}, ensure_ascii=False)
    df_log = pd.DataFrame(
        [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, status, payload]],
        columns=["timestamp", "action", "status", "details"],
    )
    if not os.path.exists(PROCESS_LOG_FILE):
        df_log.to_csv(PROCESS_LOG_FILE, index=False)
    else:
        df_log.to_csv(PROCESS_LOG_FILE, mode="a", header=False, index=False)


def persist_training_summary(summary):
    """保存训练效果快照，便于追踪收敛过程。"""
    ensure_runtime_log_dir()
    summary_with_ts = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **summary,
    }
    pd.DataFrame([summary_with_ts]).to_csv(
        TRAINING_METRICS_FILE,
        mode="a" if os.path.exists(TRAINING_METRICS_FILE) else "w",
        header=not os.path.exists(TRAINING_METRICS_FILE),
        index=False,
    )
    with open(LATEST_TRAINING_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_with_ts, f, indent=2, ensure_ascii=False)


def persist_recommendation_signal(signal):
    """保存当前推荐点的关键信号，供外部助手程序读取。"""
    ensure_runtime_log_dir()
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **signal,
    }
    with open(LATEST_RECOMMENDATION_SIGNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def estimate_remaining_shots(relative_rmse, r2):
    """用启发式规则估计还需多少发实验才能进入局部收敛。"""
    if relative_rmse <= 0.45 and r2 >= 0.60:
        return "8-15 发", "接近收敛"
    if relative_rmse <= 0.65 and r2 >= 0.35:
        return "15-25 发", "中后期"
    if relative_rmse <= 0.85 and r2 >= 0.15:
        return "25-40 发", "中期"
    return "40-60 发", "早期"


def estimate_convergence_dataset_size(sample_count, relative_rmse, r2):
    """估计达到相对稳定收敛大约需要的总样本量。"""
    if relative_rmse is None or r2 is None:
        return ">= 50", "至少 50"
    if relative_rmse <= 0.45 and r2 >= 0.60:
        total = max(sample_count + 8, 120)
        return f"{total}-{total + 10}", f"{sample_count + 8}-{sample_count + 15}"
    if relative_rmse <= 0.65 and r2 >= 0.35:
        total = max(sample_count + 15, 130)
        return f"{total}-{total + 12}", f"{sample_count + 15}-{sample_count + 25}"
    if relative_rmse <= 0.85 and r2 >= 0.15:
        total = max(sample_count + 25, 140)
        return f"{total}-{total + 15}", f"{sample_count + 25}-{sample_count + 40}"
    total = max(sample_count + 40, 145)
    return f"{total}-{total + 20}", f"{sample_count + 40}-{sample_count + 60}"


def assess_training_effect(X, y, mode_params):
    """基于当前 lhs_data.csv 评估 surrogate 泛化效果，并估计剩余发次。"""
    if len(y) < RF_MIN_SAMPLES_TO_TRAIN:
        return {
            "sample_count": int(len(y)),
            "cv_rmse": None,
            "cv_mae": None,
            "cv_r2": None,
            "target_std": float(np.std(y)) if len(y) else None,
            "relative_rmse": None,
            "estimated_remaining_shots": "至少再补到 50 发样本",
            "convergence_stage": "数据不足",
            "estimated_total_dataset_size": ">= 50",
        }

    n_splits = min(5, len(y))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RF_RANDOM_STATE)
    rf = RandomForestRegressor(
        n_estimators=mode_params["rf"]["n_trees"],
        max_depth=mode_params["rf"]["max_depth"],
        min_samples_split=mode_params["rf"]["min_samples_split"],
        min_samples_leaf=mode_params["rf"]["min_samples_leaf"],
        max_features=mode_params["rf"]["max_features"],
        random_state=RF_RANDOM_STATE,
        n_jobs=1,
    )
    y_pred = cross_val_predict(rf, X, y, cv=cv, n_jobs=1)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))
    target_std = float(np.std(y))
    relative_rmse = rmse / target_std if target_std > 0 else None
    remaining_shots, stage = estimate_remaining_shots(relative_rmse, r2)
    total_dataset_size, remaining_by_total = estimate_convergence_dataset_size(
        len(y), relative_rmse, r2
    )
    return {
        "sample_count": int(len(y)),
        "cv_rmse": rmse,
        "cv_mae": mae,
        "cv_r2": r2,
        "target_std": target_std,
        "relative_rmse": relative_rmse,
        "estimated_remaining_shots": remaining_shots,
        "estimated_remaining_shots_by_total": remaining_by_total,
        "estimated_total_dataset_size": total_dataset_size,
        "convergence_stage": stage,
    }


def ensure_metadata_columns(df):
    """兼容旧数据集，补齐重复 shot 元数据列。"""
    df = df.copy()
    if "shot_mean" not in df.columns:
        df["shot_mean"] = df["energy"]
    if "shot_std" not in df.columns:
        df["shot_std"] = 0.0
    if "shot_var" not in df.columns:
        df["shot_var"] = np.square(df["shot_std"].astype(float))
    if "repeat_count" not in df.columns:
        df["repeat_count"] = 1
    if "repeat_values" not in df.columns:
        df["repeat_values"] = ""
    return df


def compute_sample_weights(shot_std, repeat_count):
    """按 shot 方差和重复次数生成训练权重。"""
    shot_std = np.asarray(shot_std, dtype=float)
    repeat_count = np.asarray(repeat_count, dtype=float)
    noise_floor = max(np.median(np.abs(shot_std)), 1.0)
    variance = np.square(np.maximum(shot_std, noise_floor * 0.25))
    weights = repeat_count / variance
    weights = np.clip(weights, np.percentile(weights, 5), np.percentile(weights, 95))
    weights = weights / np.mean(weights)
    return weights


def evaluate_repeat_validation_need(pred_mean, pred_std, best_energy, training_effect):
    """判断当前候选点是否值得做 3 发重复验证。"""
    if best_energy is None or best_energy <= 0 or training_effect is None:
        return False, "暂无足够依据判断是否需要重复验证"
    target_std = training_effect.get("target_std") or 0.0
    close_to_best = pred_mean >= 0.95 * best_energy
    low_uncertainty = pred_std <= max(0.08 * best_energy, 0.35 * target_std, 1.0)
    if close_to_best and low_uncertainty:
        return True, "建议做 3 发重复验证"
    if close_to_best:
        return False, "接近当前最优，但预测不确定性仍偏大"
    return False, "当前候选点与当前最优仍有明显差距"
