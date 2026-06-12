"""文件读写：生成 DM txt、复制到 Windows 共享文件夹、保存数据 CSV。"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    N_ACTUATORS,
    DATA_FILE,
    EXPERIMENT_LOG_FILE,
    LOCAL_CONFIG_DIR,
    WINDOWS_SHARE_DIR,
    COPY_TXT_TO_WINDOWS_SHARE,
)
from utils import append_operation_log, ensure_metadata_columns


def copy_file_to_windows_share(local_filepath):
    """
    将本地文件复制到 Windows 共享文件夹。

    前提：Mac 已经通过 Finder 挂载 smb://192.168.10.2/BO0612，
    并且本机路径存在 /Volumes/BO0612。
    """
    src = Path(local_filepath)
    share_dir = Path(WINDOWS_SHARE_DIR)

    if not share_dir.exists():
        append_operation_log(
            "copy_to_windows_share",
            "error",
            {
                "reason": "windows_share_not_mounted",
                "share_dir": str(share_dir),
                "local_file": str(src),
            },
        )
        return None

    dst = share_dir / src.name
    try:
        shutil.copy2(src, dst)
        append_operation_log(
            "copy_to_windows_share",
            "success",
            {"local_file": str(src), "windows_file": str(dst)},
        )
        return str(dst)
    except Exception as exc:
        append_operation_log(
            "copy_to_windows_share",
            "error",
            {
                "reason": str(exc),
                "local_file": str(src),
                "share_dir": str(share_dir),
            },
        )
        return None


def save_dm_txt(vector, shot_id):
    """
    保存推荐的 DM 配置。

    返回：
        local_path, windows_path
    其中 windows_path 可能为 None，表示共享文件夹没有挂载或复制失败。
    """
    os.makedirs(LOCAL_CONFIG_DIR, exist_ok=True)

    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    filename = f"{time_str}_{shot_id}.txt"
    local_path = Path(LOCAL_CONFIG_DIR) / filename

    line1 = "\t".join(str(int(v)) for v in vector)
    line2 = "Set-up\t" + datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(line1 + "\n" + line2 + "\n")

    append_operation_log(
        "save_dm_txt",
        "success",
        {"local_file": str(local_path), "shot_id": int(shot_id)},
    )

    windows_path = None
    if COPY_TXT_TO_WINDOWS_SHARE:
        windows_path = copy_file_to_windows_share(local_path)

    return str(local_path), windows_path


def test_windows_share_write():
    """单独测试 Mac 是否能写入 Windows 共享文件夹。"""
    share_dir = Path(WINDOWS_SHARE_DIR)
    if not share_dir.exists():
        raise FileNotFoundError(
            f"没有找到 {WINDOWS_SHARE_DIR}，请先挂载 smb://192.168.10.2/BO0612"
        )

    test_file = share_dir / f"mac_python_share_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是 Mac Python 写入 Windows 共享文件夹的测试文件。\n")
        f.write(f"写入时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    return str(test_file)


def save_data(x, energy_mean, shot_std=0.0, repeat_count=1):
    """保存实验数据到 CSV，同时保留重复 shot 统计信息。"""
    row = {f"a{i}": int(x[i]) for i in range(N_ACTUATORS)}
    row["energy"] = float(energy_mean)
    row["shot_mean"] = float(energy_mean)
    row["shot_std"] = float(shot_std)
    row["repeat_count"] = int(repeat_count)
    df_new = pd.DataFrame([row])
    if not os.path.exists(DATA_FILE):
        df_new.to_csv(index=False, path_or_buf=DATA_FILE)
    else:
        existing = ensure_metadata_columns(pd.read_csv(DATA_FILE))
        updated = pd.concat([existing, df_new], ignore_index=True)
        updated.to_csv(DATA_FILE, index=False)


def save_experiment_log(surface_vector, energy):
    """保存三列日志：面型数据、实验能量、时间戳。"""
    surface_text = "\t".join(str(int(v)) for v in np.asarray(surface_vector, dtype=int))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_log = pd.DataFrame(
        [[surface_text, float(energy), timestamp]],
        columns=["surface_profile", "energy", "timestamp"],
    )
    if not os.path.exists(EXPERIMENT_LOG_FILE):
        df_log.to_csv(EXPERIMENT_LOG_FILE, index=False)
    else:
        existing = pd.read_csv(EXPERIMENT_LOG_FILE)
        if "timestamp" not in existing.columns:
            existing["timestamp"] = ""
            existing.to_csv(EXPERIMENT_LOG_FILE, index=False)
        df_log.to_csv(EXPERIMENT_LOG_FILE, mode="a", header=False, index=False)
