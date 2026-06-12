"""全局配置参数。"""

import os

# ========== 基础实验参数 ==========
N_ACTUATORS = 52
VOLTAGE_BOUNDS = (-17000, 17000)
DATA_FILE = "lhs_data.csv"
EXPERIMENT_LOG_FILE = "bo_experiment_log.csv"
BASELINE_FILE = "00102.txt"
# 安全阈值：所有推荐电压都必须满足
# baseline[i] - MAX_DELTA_FROM_BASELINE <= a_i <= baseline[i] + MAX_DELTA_FROM_BASELINE
# 你可以直接修改这个值，例如 100、250、500。
MAX_DELTA_FROM_BASELINE = 250

# 允许优化的维度。可以改成任意多个维度，例如：
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# list(range(20))
# [0, 2, 4, 6, 8, 10, 12]
OPTIMIZED_DIM_INDICES = list(range(10))
FROZEN_START_IDX = max(OPTIMIZED_DIM_INDICES) + 1 if OPTIMIZED_DIM_INDICES else 0
MIN_TRUST_RADIUS = 20
DEFAULT_REPEAT_SHOTS = 3

# ========== GPR 增强贝叶斯优化参数 ==========
GPR_MIN_SAMPLES_TO_TRAIN = 12
GPR_NOISE_FLOOR = 1e-6

# ========== 运行日志 ==========
PROCESS_LOG_DIR = os.path.join("suanfa", "runtime_logs")
PROCESS_LOG_FILE = os.path.join(PROCESS_LOG_DIR, "operation_log.csv")
TRAINING_METRICS_FILE = os.path.join(PROCESS_LOG_DIR, "training_metrics.csv")
LATEST_TRAINING_SUMMARY_FILE = os.path.join(PROCESS_LOG_DIR, "latest_training_summary.json")
LATEST_RECOMMENDATION_SIGNAL_FILE = os.path.join(PROCESS_LOG_DIR, "latest_recommendation_signal.json")

# ========== Windows 共享文件夹 ==========
# 你的 Windows 文件夹 C:\BO0612 在 Mac 上挂载后一般对应 /Volumes/BO0612
WINDOWS_SHARE_DIR = "/Volumes/BO0612"
COPY_TXT_TO_WINDOWS_SHARE = True

# 本地 txt 配置保存目录
LOCAL_CONFIG_DIR = "config"

# ========== 代理模型评估参数 ==========
RF_MIN_SAMPLES_TO_TRAIN = 50
RF_TEST_SIZE = 0.2
RF_RANDOM_STATE = 0

FAST_MODE_PARAMS = {
    "rf": {
        "n_trees": 180,
        "max_depth": 14,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "bo": {
        "n_candidates": 6000,
        "n_local_seeds": 8,
        "local_noise_scales": [12, 24, 40],
        "exploration_beta": 1.2,
        "xi": 0.01,
        "random_state": 0,
    },
}

PRECISE_MODE_PARAMS = {
    "rf": {
        "n_trees": 500,
        "max_depth": 18,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "bo": {
        "n_candidates": 18000,
        "n_local_seeds": 12,
        "local_noise_scales": [8, 16, 32, 48],
        "exploration_beta": 1.6,
        "xi": 0.01,
        "random_state": 0,
    },
}
