"""GUI 主界面。"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    BASELINE_FILE,
    DATA_FILE,
    DEFAULT_REPEAT_SHOTS,
    FAST_MODE_PARAMS,
    FROZEN_START_IDX,
    MAX_DELTA_FROM_BASELINE,
    N_ACTUATORS,
    OPTIMIZED_DIM_INDICES,
    PRECISE_MODE_PARAMS,
)
from config_viewer import ConfigViewer
from file_io import save_data, save_dm_txt
from model import (
    build_constrained_bounds,
    enforce_hard_constraints,
    load_baseline_vector,
    propose_next,
    train_surrogate,
)
from utils import (
    append_operation_log,
    assess_training_effect,
    compute_sample_weights,
    ensure_metadata_columns,
    ensure_runtime_log_dir,
    evaluate_repeat_validation_need,
    format_sci,
    persist_recommendation_signal,
    persist_training_summary,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "STHeiti",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class BO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("4.BO 阶段 - DM 多步贝叶斯优化")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 760)
        self.mode_var = tk.StringVar(value="fast")
        ensure_runtime_log_dir()

        if not os.path.exists(DATA_FILE):
            messagebox.showerror("错误", f"找不到 {DATA_FILE}，请先完成 LHS 阶段")
            root.destroy()
            return

        df = ensure_metadata_columns(pd.read_csv(DATA_FILE))
        df.to_csv(DATA_FILE, index=False)
        self.X_opt = df[[f"a{i}" for i in OPTIMIZED_DIM_INDICES]].values
        self.y = df["energy"].values
        self.shot_std = df["shot_std"].to_numpy(dtype=float)
        self.repeat_count = df["repeat_count"].to_numpy(dtype=float)
        self.sample_weights = compute_sample_weights(self.shot_std, self.repeat_count)

        if not os.path.exists(BASELINE_FILE):
            messagebox.showerror("错误", f"找不到基准面型文件 {BASELINE_FILE}")
            root.destroy()
            return
        self.baseline = load_baseline_vector(BASELINE_FILE)
        self.lower_bounds, self.upper_bounds = build_constrained_bounds(self.baseline)

        self.shot_id = len(self.y) + 1
        self.energy_history = []
        self.rmse_history = []
        self.best_energy = -1
        self.current = None
        self.training_effect = None
        self.current_shot_energies = []
        self.current_shot_target = DEFAULT_REPEAT_SHOTS
        self.latest_recommendation_signal = None

        self.setup_ui()
        self.refresh_training_effect(log_action=True)
        self.update_plot()
        self.update_surface_display()
        append_operation_log(
            "app_initialized",
            "success",
            {
                "data_file": DATA_FILE,
                "baseline_file": BASELINE_FILE,
                "sample_count": int(len(self.y)),
            },
        )

    def setup_ui(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_outer = tk.Frame(paned, width=420)
        left_outer.pack_propagate(False)
        paned.add(left_outer, minsize=380, width=420, stretch="never")

        right_frame = tk.Frame(paned)
        paned.add(right_frame, minsize=700, stretch="always")

        left_canvas = tk.Canvas(left_outer, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_frame = tk.Frame(left_canvas)

        left_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
        )
        left_canvas.bind(
            "<Configure>",
            lambda e: left_canvas.itemconfigure("left_frame_window", width=e.width),
        )
        left_canvas.create_window((0, 0), window=left_frame, anchor="nw", tags="left_frame_window")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(left_frame, text="贝叶斯优化控制", font=("Arial", 14, "bold")).pack(pady=10)

        self.label_status = tk.Label(
            left_frame,
            text="点击 '推荐新配置' 开始优化",
            font=("Arial", 10),
            wraplength=260,
            justify=tk.LEFT,
        )
        self.label_status.pack(pady=10)

        self.btn_propose = tk.Button(
            left_frame, text="推荐新配置", command=self.propose_config, font=("Arial", 12), bg="lightblue"
        )
        self.btn_propose.pack(pady=5, fill=tk.X)

        mode_frame = tk.LabelFrame(left_frame, text="优化模式", font=("Arial", 10))
        mode_frame.pack(pady=8, fill=tk.X)

        tk.Radiobutton(mode_frame, text="快模式", variable=self.mode_var, value="fast", font=("Arial", 10)).pack(
            anchor="w", padx=8, pady=2
        )
        tk.Radiobutton(
            mode_frame, text="精细模式", variable=self.mode_var, value="precise", font=("Arial", 10)
        ).pack(anchor="w", padx=8, pady=2)

        self.btn_view_config = tk.Button(
            left_frame, text="查看配置文件", command=self.view_config, font=("Arial", 12), bg="lightyellow"
        )
        self.btn_view_config.pack(pady=5, fill=tk.X)

        tk.Label(left_frame, text="输入实验能量:", font=("Arial", 10)).pack(pady=(20, 5))

        self.entry_energy = tk.Entry(left_frame, width=15, font=("Arial", 12))
        self.entry_energy.pack(pady=5)
        self.entry_energy.bind("<Return>", lambda e: self.add_shot_energy())

        repeat_frame = tk.Frame(left_frame)
        repeat_frame.pack(fill=tk.X, pady=(2, 4))

        tk.Label(repeat_frame, text="每点重复 shots:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.entry_repeat_count = tk.Entry(repeat_frame, width=6, font=("Arial", 11))
        self.entry_repeat_count.pack(side=tk.LEFT, padx=5)
        self.entry_repeat_count.insert(0, str(DEFAULT_REPEAT_SHOTS))

        self.btn_add_shot = tk.Button(
            left_frame, text="记录当前 Shot", command=self.add_shot_energy, font=("Arial", 12), bg="lightgreen"
        )
        self.btn_add_shot.pack(pady=3, fill=tk.X)

        self.btn_submit = tk.Button(
            left_frame, text="按均值提交当前点", command=self.submit_energy, font=("Arial", 12), bg="lightgreen"
        )
        self.btn_submit.pack(pady=5, fill=tk.X)

        self.label_shot_buffer = tk.Label(
            left_frame,
            text="当前点重复 shot: 0 / 3\n均值: N/A",
            font=("Arial", 10),
            justify=tk.LEFT,
            anchor="w",
        )
        self.label_shot_buffer.pack(pady=4, fill=tk.X)

        tk.Label(left_frame, text="优化统计", font=("Arial", 12, "bold")).pack(pady=(30, 10))

        self.label_stats = tk.Label(left_frame, text="", font=("Arial", 10), justify=tk.LEFT, anchor="w")
        self.label_stats.pack(pady=5, fill=tk.X)

        surface_frame = tk.LabelFrame(left_frame, text="当前推荐面形", font=("Arial", 10))
        surface_frame.pack(pady=(15, 5), fill=tk.BOTH, expand=True)

        columns = ("执行器", "基准值", "推荐值", "变化量")
        self.surface_tree = ttk.Treeview(surface_frame, columns=columns, show="headings", height=10)
        for col in columns:
            self.surface_tree.heading(col, text=col)
            self.surface_tree.column(col, width=74, anchor="center")

        surface_scrollbar = ttk.Scrollbar(surface_frame, orient="vertical", command=self.surface_tree.yview)
        self.surface_tree.configure(yscrollcommand=surface_scrollbar.set)
        self.surface_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        surface_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(left_frame, text="完整 52 维向量", font=("Arial", 10)).pack(pady=(8, 4))
        self.vector_text = tk.Text(left_frame, height=4, width=28, font=("Courier", 9))
        self.vector_text.pack(fill=tk.X)
        self.vector_text.insert("1.0", "尚未生成推荐面形")
        self.vector_text.config(state=tk.DISABLED)

        self.label_baseline_info = tk.Label(
            left_frame,
            text=f"基准面形来源: {BASELINE_FILE} 第一行 52 维电压",
            font=("Arial", 9),
            fg="gray",
            wraplength=260,
            justify=tk.LEFT,
        )
        self.label_baseline_info.pack(pady=(4, 0), fill=tk.X)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def view_config(self):
        ConfigViewer(self.root)

    def get_mode_params(self):
        return FAST_MODE_PARAMS if self.mode_var.get() == "fast" else PRECISE_MODE_PARAMS

    def refresh_training_effect(self, log_action=False):
        mode_params = self.get_mode_params()
        summary = assess_training_effect(self.X_opt, self.y, mode_params)
        self.training_effect = summary
        persist_training_summary(summary)
        if log_action:
            append_operation_log("training_effect_assessed", "success", summary)

    def update_shot_buffer_display(self):
        mean_text = format_sci(np.mean(self.current_shot_energies)) if self.current_shot_energies else "N/A"
        self.label_shot_buffer.config(
            text=(
                f"当前点重复 shot: {len(self.current_shot_energies)} / {self.current_shot_target}\n"
                f"均值: {mean_text}"
            )
        )

    def update_surface_display(self):
        for item in self.surface_tree.get_children():
            self.surface_tree.delete(item)

        if self.current is None:
            self.vector_text.config(state=tk.NORMAL)
            self.vector_text.delete("1.0", tk.END)
            self.vector_text.insert("1.0", "尚未生成推荐面形")
            self.vector_text.config(state=tk.DISABLED)
            return

        for idx in range(N_ACTUATORS):
            baseline_val = int(self.baseline[idx])
            current_val = int(self.current[idx])
            delta = current_val - baseline_val
            tag = "changed" if delta != 0 else ""
            self.surface_tree.insert(
                "",
                "end",
                values=(f"A{idx}", str(baseline_val), str(current_val), f"{delta:+d}"),
                tags=(tag,),
            )
        self.surface_tree.tag_configure("changed", background="lightyellow")

        self.vector_text.config(state=tk.NORMAL)
        self.vector_text.delete("1.0", tk.END)
        self.vector_text.insert("1.0", "\t".join(str(int(v)) for v in self.current))
        self.vector_text.config(state=tk.DISABLED)

    def update_plot(self):
        self.ax1.clear()
        self.ax2.clear()

        if self.energy_history:
            shots = list(range(1, len(self.energy_history) + 1))
            self.ax1.plot(shots, self.energy_history, marker="o", linewidth=2, markersize=6)
            self.ax1.axhline(
                y=max(self.energy_history),
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"最大值: {max(self.energy_history):.3f} MeV",
            )
            self.ax1.legend()

        self.ax1.set_title("实验能量优化曲线", fontsize=14)
        self.ax1.set_xlabel("Shot ID")
        self.ax1.set_ylabel("能量 (MeV)")
        self.ax1.grid(True, alpha=0.3)

        if self.rmse_history:
            iterations = list(range(1, len(self.rmse_history) + 1))
            self.ax2.plot(iterations, self.rmse_history, marker="s", color="orange", linewidth=2, markersize=6)

        self.ax2.set_title("Surrogate模型RMSE曲线", fontsize=14)
        self.ax2.set_xlabel("迭代次数")
        self.ax2.set_ylabel("RMSE")
        self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

        stats_text = f"总实验次数: {len(self.energy_history)}\n"
        stats_text += f"训练样本数: {len(self.y)}\n"
        if self.energy_history:
            stats_text += f"当前最优: {format_sci(max(self.energy_history))} MeV\n"
            stats_text += f"平均能量: {format_sci(np.mean(self.energy_history))} MeV\n"
        if self.rmse_history:
            stats_text += f"本轮Holdout RMSE: {format_sci(self.rmse_history[-1])}\n"
        if self.training_effect:
            cv_rmse = self.training_effect["cv_rmse"]
            cv_r2 = self.training_effect["cv_r2"]
            relative_rmse = self.training_effect["relative_rmse"]
            stage = self.training_effect["convergence_stage"]
            remain = self.training_effect["estimated_remaining_shots"]
            total_samples = self.training_effect.get("estimated_total_dataset_size", ">= 50")
            if cv_rmse is not None:
                stats_text += f"CV RMSE: {format_sci(cv_rmse)}\n"
                stats_text += f"CV R²: {format_sci(cv_r2)}\n"
                stats_text += f"相对RMSE: {format_sci(relative_rmse)}\n"
            stats_text += f"收敛阶段: {stage}\n"
            stats_text += f"预计总样本: {total_samples}\n"
            stats_text += f"预计还需: {remain}"

        self.label_stats.config(text=stats_text)

    def get_repeat_target(self):
        raw = self.entry_repeat_count.get().strip()
        try:
            repeat_target = int(raw)
            if repeat_target <= 0:
                raise ValueError
            self.current_shot_target = repeat_target
            return repeat_target
        except Exception:
            append_operation_log("repeat_count_parse", "error", {"raw_value": raw})
            messagebox.showerror("错误", "重复 shots 数必须是正整数")
            return None

    def add_shot_energy(self):
        repeat_target = self.get_repeat_target()
        if repeat_target is None:
            return

        val = self.entry_energy.get().strip()
        try:
            energy = float(val)
        except Exception:
            append_operation_log(
                "add_shot_energy", "error", {"reason": "invalid_numeric_input", "raw_value": val}
            )
            messagebox.showerror("错误", "请输入数字能量")
            return

        if self.current is None:
            append_operation_log("add_shot_energy", "blocked", {"reason": "no_current_configuration"})
            messagebox.showwarning("提示", "请先推荐新配置")
            return

        self.current_shot_energies.append(energy)
        self.entry_energy.delete(0, tk.END)
        self.update_shot_buffer_display()
        append_operation_log(
            "add_shot_energy",
            "success",
            {
                "shot_energy": float(energy),
                "buffer_count": int(len(self.current_shot_energies)),
                "buffer_target": int(repeat_target),
                "buffer_mean": float(np.mean(self.current_shot_energies)),
            },
        )

        if len(self.current_shot_energies) >= repeat_target:
            self.submit_energy()

    def propose_config(self):
        mode_params = self.get_mode_params()
        self.refresh_training_effect(log_action=True)
        rf, rmse = train_surrogate(self.X_opt, self.y, mode_params, self.sample_weights)
        if rf is None:
            append_operation_log(
                "propose_config", "blocked", {"reason": "insufficient_data", "sample_count": int(len(self.y))}
            )
            messagebox.showwarning("提示", "数据不足，请先积累更多点")
            return

        self.rmse_history.append(rmse)

        x_next, pred, pred_std, candidate_count, trust_radius, top_dims = propose_next(
            rf, self.baseline, self.X_opt, self.y, mode_params
        )
        self.current = enforce_hard_constraints(x_next, self.baseline)
        self.current_shot_energies = []
        self.get_repeat_target()
        self.update_shot_buffer_display()
        current_best = float(np.max(self.y)) if len(self.y) else None
        suggest_repeat, suggest_message = evaluate_repeat_validation_need(
            pred, pred_std, current_best, self.training_effect
        )
        if suggest_repeat:
            self.current_shot_target = 3
            self.entry_repeat_count.delete(0, tk.END)
            self.entry_repeat_count.insert(0, "3")
            self.update_shot_buffer_display()

        local_filename, windows_filename = save_dm_txt(self.current, self.shot_id)
        filename_display = local_filename
        if windows_filename:
            filename_display += f"\nWindows共享: {windows_filename}"
        else:
            filename_display += "\nWindows共享: 未复制，请检查 /Volumes/BO0612 是否已挂载"

        remain = self.training_effect["estimated_remaining_shots"] if self.training_effect else "未知"
        top_dims_text = "、".join(f"{item['dimension']}({format_sci(item['importance'])})" for item in top_dims)
        self.label_status.config(
            text=(
                f"推荐配置已保存:\n{filename_display}\n\n"
                f"预测能量均值: {format_sci(pred)} MeV\n"
                f"预测不确定性: ±{format_sci(pred_std)}\n"
                f"本轮筛选候选数: {candidate_count}\n"
                f"当前信赖域半径: ±{trust_radius}\n"
                f"高敏感维度: {top_dims_text}\n"
                f"重复验证判断: {suggest_message}\n"
                f"约束: 0-9 维相对 {BASELINE_FILE} 不超过 ±{MAX_DELTA_FROM_BASELINE}\n"
                f"约束: 10-51 维与 {BASELINE_FILE} 完全一致\n\n"
                f"基于当前 lhs_data.csv 的估计，还需约 {remain} 才可能进入局部收敛\n\n"
                f"当前模式: {'快模式' if self.mode_var.get() == 'fast' else '精细模式'}\n\n"
                "请实验员加载后逐发记录，再按均值提交当前点"
            )
        )
        self.update_surface_display()
        self.latest_recommendation_signal = {
            "shot_id": int(self.shot_id),
            "predicted_energy_mean": float(pred),
            "predicted_energy_std": float(pred_std),
            "candidate_count": int(candidate_count),
            "trust_radius": int(trust_radius),
            "top_sensitive_dimensions": top_dims,
            "best_observed_energy": current_best,
            "repeat_validation_recommended": bool(suggest_repeat),
            "repeat_validation_message": suggest_message,
            "recommended_repeat_count": 3 if suggest_repeat else int(self.current_shot_target),
            "config_file": local_filename,
            "windows_shared_config_file": windows_filename,
            "convergence_stage": self.training_effect["convergence_stage"] if self.training_effect else None,
            "estimated_remaining_shots": remain,
        }
        persist_recommendation_signal(self.latest_recommendation_signal)
        append_operation_log(
            "propose_config",
            "success",
            {
                "shot_id": int(self.shot_id),
                "predicted_energy_mean": float(pred),
                "predicted_energy_std": float(pred_std),
                "candidate_count": int(candidate_count),
                "trust_radius": int(trust_radius),
                "top_sensitive_dimensions": top_dims,
                "holdout_rmse": float(rmse),
                "estimated_remaining_shots": remain,
                "config_file": local_filename,
                "windows_shared_config_file": windows_filename,
                "sample_weight_min": float(np.min(self.sample_weights)),
                "sample_weight_max": float(np.max(self.sample_weights)),
                "repeat_validation_recommended": bool(suggest_repeat),
                "repeat_validation_message": suggest_message,
            },
        )

        self.update_plot()

    def submit_energy(self):
        if not self.current_shot_energies:
            append_operation_log("submit_energy", "blocked", {"reason": "empty_shot_buffer"})
            messagebox.showwarning("提示", "请先记录当前点的 shot 能量")
            return

        if self.current is None:
            append_operation_log("submit_energy", "blocked", {"reason": "no_current_configuration"})
            messagebox.showwarning("提示", "请先推荐新配置")
            return

        energy = float(np.mean(self.current_shot_energies))
        shot_std = float(np.std(self.current_shot_energies, ddof=1)) if len(self.current_shot_energies) > 1 else 0.0
        repeat_count = len(self.current_shot_energies)
        save_data(self.current, energy, shot_std=shot_std, repeat_count=repeat_count)
        self.X_opt = np.vstack([self.X_opt, self.current[OPTIMIZED_DIM_INDICES]])
        self.y = np.append(self.y, energy)
        self.shot_std = np.append(self.shot_std, shot_std)
        self.repeat_count = np.append(self.repeat_count, repeat_count)
        self.sample_weights = compute_sample_weights(self.shot_std, self.repeat_count)
        self.refresh_training_effect(log_action=True)

        self.shot_id += 1
        self.energy_history.append(energy)
        self.best_energy = max(self.best_energy, energy)

        self.label_status.config(
            text=(
                f"已按 {len(self.current_shot_energies)} 发均值提交当前点:\n"
                f"{format_sci(energy)} MeV\n\n"
                f"当前最优能量:\n{format_sci(self.best_energy)} MeV\n\n"
                "可以推荐下一个配置"
            )
        )
        submitted_shots = list(self.current_shot_energies)
        self.current_shot_energies = []
        self.update_shot_buffer_display()
        append_operation_log(
            "submit_energy",
            "success",
            {
                "energy_mean": float(energy),
                "shot_std": float(shot_std),
                "repeat_shot_count": int(len(submitted_shots)),
                "repeat_shot_values": submitted_shots,
                "best_energy": float(self.best_energy),
                "sample_count": int(len(self.y)),
                "estimated_remaining_shots": self.training_effect["estimated_remaining_shots"],
            },
        )

        self.update_plot()
