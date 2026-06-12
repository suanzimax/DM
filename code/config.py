"""全局配置参数。"""

import os

# ========== 基础实验参数 ==========
N_ACTUATORS = 52
VOLTAGE_BOUNDS = (-17000, 17000)
DATA_FILE = "lhs_data.csv"
EXPERIMENT_LOG_FILE = "bo_experiment_log.csv"
BASELINE_FILE = "00102.txt"
MAX_DELTA_FROM_BASELINE = 100
FROZEN_START_IDX = 10
OPTIMIZED_DIM_INDICES = list(range(FROZEN_START_IDX))
MIN_TRUST_RADIUS = 20
DEFAULT_REPEAT_SHOTS = 3

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

# ========== 随机森林代理模型参数 ==========
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
        "random_state": 0,
    },
}
