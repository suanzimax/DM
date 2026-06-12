"""GUI 主界面。"""

import os
import tkinter as tk
from tkinter import messagebox, ttk

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (
    BASELINE_FILE,
    DATA_FILE,
    DEFAULT_REPEAT_SHOTS,
    FAST_MODE_PARAMS,
    MAX_DELTA_FROM_BASELINE,
    N_ACTUATORS,
    OPTIMIZED_DIM_INDICES,
    PRECISE_MODE_PARAMS,
    WINDOWS_SHARE_DIR,
)
from config_viewer import ConfigViewer
from file_io import copy_file_to_windows_share, save_data, save_dm_txt
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
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    }
)


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
        self.initial_sample_count = len(self.y)
        self.energy_history = []
        self.rmse_history = []
        self.holdout_sample_counts = []
        self.cv_rmse_history = []
        self.relative_rmse_history = []
        self.cv_r2_history = []
        self.metric_sample_counts = []
        self.best_energy = float(np.max(self.y)) if len(self.y) else -1
        self.current = None
        self.training_effect = None
        self.current_shot_energies = []
        self.current_shot_target = DEFAULT_REPEAT_SHOTS
        self.latest_recommendation_signal = None
        self.optimized_dim_set = set(OPTIMIZED_DIM_INDICES)
        self.latest_config_file = None
        self.latest_windows_config_file = None

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

        self.btn_resend_config = tk.Button(
            left_frame,
            text="重新发送配置到 Windows",
            command=self.resend_latest_config,
            font=("Arial", 12),
            bg="#ffdca8",
            state=tk.DISABLED,
        )
        self.btn_resend_config.pack(pady=5, fill=tk.X)

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

        columns = ("执行器", "基准值", "推荐值", "变化量", "安全占比", "状态")
        self.surface_tree = ttk.Treeview(surface_frame, columns=columns, show="headings", height=10)
        for col in columns:
            self.surface_tree.heading(col, text=col)
            width = 62 if col in ("执行器", "状态") else 74
            self.surface_tree.column(col, width=width, anchor="center")

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
            text=(
                f"基准面形来源: {BASELINE_FILE} 第一行 52 维电压\n"
                f"安全阈值: 每个优化维度相对基准不超过 ±{MAX_DELTA_FROM_BASELINE}\n"
                f"优化维度: {OPTIMIZED_DIM_INDICES}"
            ),
            font=("Arial", 9),
            fg="gray",
            wraplength=260,
            justify=tk.LEFT,
        )
        self.label_baseline_info.pack(pady=(4, 0), fill=tk.X)

        self.fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
        self.ax1, self.ax2, self.ax3, self.ax4 = axes.ravel()
        self.fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.12, hspace=0.16)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def view_config(self):
        ConfigViewer(self.root)

    def resend_latest_config(self):
        """将最近一次生成的推荐 txt 重新复制到 Windows 共享目录。"""
        if not self.latest_config_file:
            append_operation_log("resend_config_to_windows", "blocked", {"reason": "no_latest_config"})
            messagebox.showwarning("提示", "还没有可重新发送的配置文件，请先推荐新配置")
            return

        if not os.path.exists(self.latest_config_file):
            append_operation_log(
                "resend_config_to_windows",
                "error",
                {"reason": "local_config_missing", "local_file": self.latest_config_file},
            )
            messagebox.showerror("错误", f"本地配置文件不存在:\n{self.latest_config_file}")
            return

        windows_filename = copy_file_to_windows_share(self.latest_config_file)
        if windows_filename:
            self.latest_windows_config_file = windows_filename
            if self.latest_recommendation_signal is not None:
                self.latest_recommendation_signal["windows_shared_config_file"] = windows_filename
                persist_recommendation_signal(self.latest_recommendation_signal)
            append_operation_log(
                "resend_config_to_windows",
                "success",
                {
                    "local_file": self.latest_config_file,
                    "windows_shared_config_file": windows_filename,
                },
            )
            self.label_status.config(
                text=(
                    f"配置文件已重新发送到 Windows 共享:\n{windows_filename}\n\n"
                    f"本地文件:\n{self.latest_config_file}\n\n"
                    "可以继续加载该配置或记录实验 shot"
                )
            )
            messagebox.showinfo("成功", f"已重新发送到 Windows 共享:\n{windows_filename}")
        else:
            append_operation_log(
                "resend_config_to_windows",
                "error",
                {
                    "reason": "copy_failed",
                    "local_file": self.latest_config_file,
                    "windows_share_dir": WINDOWS_SHARE_DIR,
                },
            )
            messagebox.showerror(
                "发送失败",
                f"没有复制成功，请检查 Windows 共享是否已挂载:\n{WINDOWS_SHARE_DIR}",
            )

    def get_mode_params(self):
        return FAST_MODE_PARAMS if self.mode_var.get() == "fast" else PRECISE_MODE_PARAMS

    def refresh_training_effect(self, log_action=False):
        mode_params = self.get_mode_params()
        summary = assess_training_effect(self.X_opt, self.y, mode_params)
        self.training_effect = summary
        if (
            summary.get("cv_rmse") is not None
            and (
                not self.metric_sample_counts
                or self.metric_sample_counts[-1] != int(summary["sample_count"])
            )
        ):
            self.metric_sample_counts.append(int(summary["sample_count"]))
            self.cv_rmse_history.append(float(summary["cv_rmse"]))
            self.relative_rmse_history.append(float(summary["relative_rmse"]))
            self.cv_r2_history.append(float(summary["cv_r2"]))
        persist_training_summary(summary)
        if log_action:
            append_operation_log("training_effect_assessed", "success", summary)

    def update_shot_buffer_display(self):
        mean_text = format_sci(np.mean(self.current_shot_energies)) if self.current_shot_energies else "N/A"
        std_text = (
            format_sci(np.std(self.current_shot_energies, ddof=1))
            if len(self.current_shot_energies) > 1
            else "N/A"
        )
        self.label_shot_buffer.config(
            text=(
                f"当前点重复 shot: {len(self.current_shot_energies)} / {self.current_shot_target}\n"
                f"均值: {mean_text}\n"
                f"标准差: {std_text}"
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
            usage = abs(delta) / max(MAX_DELTA_FROM_BASELINE, 1)
            usage_text = f"{usage * 100:.1f}%"
            if idx not in self.optimized_dim_set:
                status = "固定"
                tag = "frozen"
            elif usage >= 0.9:
                status = "接近阈值"
                tag = "near_limit"
            elif delta != 0:
                status = "优化"
                tag = "changed"
            else:
                status = "优化"
                tag = "optimized"
            self.surface_tree.insert(
                "",
                "end",
                values=(
                    f"A{idx}",
                    str(baseline_val),
                    str(current_val),
                    f"{delta:+d}",
                    usage_text,
                    status,
                ),
                tags=(tag,),
            )
        self.surface_tree.tag_configure("changed", background="lightyellow")
        self.surface_tree.tag_configure("optimized", background="#eef6ff")
        self.surface_tree.tag_configure("near_limit", background="#ffd6d6")
        self.surface_tree.tag_configure("frozen", foreground="gray")

        self.vector_text.config(state=tk.NORMAL)
        self.vector_text.delete("1.0", tk.END)
        self.vector_text.insert("1.0", "\t".join(str(int(v)) for v in self.current))
        self.vector_text.config(state=tk.DISABLED)

    def update_plot(self):
        for extra_axis in list(self.fig.axes[4:]):
            extra_axis.remove()

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        sample_ids = np.arange(1, len(self.y) + 1)
        if len(self.y):
            yerr = np.nan_to_num(self.shot_std, nan=0.0, posinf=0.0, neginf=0.0)
            has_error = np.any(yerr > 0)
            if has_error:
                self.ax1.errorbar(
                    sample_ids,
                    self.y,
                    yerr=yerr,
                    fmt="o-",
                    linewidth=1.1,
                    markersize=3,
                    capsize=1.8,
                    color="tab:blue",
                    ecolor="lightsteelblue",
                    label="mean ± std",
                )
            else:
                self.ax1.plot(sample_ids, self.y, "o-", linewidth=1.1, markersize=3, label="mean")
            if len(self.y) > self.initial_sample_count:
                self.ax1.axvline(
                    self.initial_sample_count + 0.5,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                )
            self.ax1.legend(loc="best")

        self.ax1.set_title("BO 迭代效果")
        self.ax1.set_xlabel("样本序号")
        self.ax1.set_ylabel("mean energy")
        self.ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        self.ax1.grid(True, alpha=0.3)

        if len(self.y):
            self.ax2.plot(
                sample_ids,
                self.shot_std,
                marker="o",
                linewidth=1.1,
                markersize=3,
                color="tab:orange",
                label="shot_std",
            )
            ax2_repeat = self.ax2.twinx()
            ax2_repeat.bar(
                sample_ids,
                self.repeat_count,
                color="lightgray",
                alpha=0.45,
                width=0.8,
                label="repeat_count",
            )
            ax2_repeat.set_ylabel("repeat_count")
            ax2_repeat.tick_params(axis="y", labelsize=7)
            lines, labels = self.ax2.get_legend_handles_labels()
            bars, bar_labels = ax2_repeat.get_legend_handles_labels()
            self.ax2.legend(lines + bars, labels + bar_labels, loc="best")

        self.ax2.set_title("Shot 稳定性")
        self.ax2.set_xlabel("样本序号")
        self.ax2.set_ylabel("shot_std")
        self.ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        self.ax2.grid(True, alpha=0.3)

        metric_x = self.metric_sample_counts if self.metric_sample_counts else list(range(1, len(self.cv_rmse_history) + 1))
        if self.cv_rmse_history:
            self.ax3.plot(
                metric_x,
                self.cv_rmse_history,
                marker="o",
                linewidth=1.1,
                color="tab:purple",
                label="CV RMSE",
            )
        if self.rmse_history:
            self.ax3.plot(
                self.holdout_sample_counts,
                self.rmse_history,
                marker="s",
                linestyle="--",
                linewidth=1.0,
                color="tab:brown",
                label="Holdout RMSE",
            )
        ax3_quality = self.ax3.twinx()
        if self.relative_rmse_history:
            ax3_quality.plot(
                metric_x,
                self.relative_rmse_history,
                marker="^",
                linewidth=1.0,
                color="tab:green",
                label="relative RMSE",
            )
        if self.cv_r2_history:
            ax3_quality.plot(
                metric_x,
                self.cv_r2_history,
                marker="x",
                linewidth=1.0,
                color="tab:red",
                label="CV R²",
            )
        self.ax3.set_title("模型统计")
        self.ax3.set_xlabel("样本数")
        self.ax3.set_ylabel("RMSE")
        self.ax3.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        ax3_quality.set_ylabel("rel. RMSE / R²")
        ax3_quality.tick_params(axis="y", labelsize=7)
        self.ax3.grid(True, alpha=0.3)
        lines, labels = self.ax3.get_legend_handles_labels()
        quality_lines, quality_labels = ax3_quality.get_legend_handles_labels()
        if lines or quality_lines:
            self.ax3.legend(lines + quality_lines, labels + quality_labels, loc="best")

        dims = np.arange(N_ACTUATORS)
        if self.current is not None:
            deltas = self.current.astype(float) - self.baseline.astype(float)
            colors = ["tab:blue" if idx in self.optimized_dim_set else "lightgray" for idx in dims]
            self.ax4.bar(dims, deltas, color=colors, width=0.85)
        else:
            self.ax4.bar(dims, np.zeros(N_ACTUATORS), color="lightgray", width=0.85)
            self.ax4.text(
                0.5,
                0.5,
                "尚未生成推荐面形",
                transform=self.ax4.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=9,
            )
        self.ax4.axhline(MAX_DELTA_FROM_BASELINE, color="red", linestyle="--", linewidth=1)
        self.ax4.axhline(-MAX_DELTA_FROM_BASELINE, color="red", linestyle="--", linewidth=1)
        self.ax4.set_ylim(-MAX_DELTA_FROM_BASELINE * 1.2, MAX_DELTA_FROM_BASELINE * 1.2)
        self.ax4.set_title("面形安全")
        self.ax4.set_xlabel("执行器维度")
        self.ax4.set_ylabel("推荐 - 基准")
        self.ax4.grid(True, axis="y", alpha=0.3)
        for axis in (self.ax1, self.ax2, self.ax3, self.ax4):
            axis.tick_params(axis="both", labelsize=7)
            axis.yaxis.get_offset_text().set_fontsize(7)
            axis.xaxis.get_offset_text().set_fontsize(7)
        self.canvas.draw()

        stats_text = f"训练样本数: {len(self.y)}\n"
        stats_text += f"实验起点: {DATA_FILE} 前 {self.initial_sample_count} 条\n"
        stats_text += f"下一推荐编号: {self.shot_id}\n"
        stats_text += f"本次启动后提交点数: {len(self.energy_history)}\n"
        if len(self.y):
            stats_text += f"当前最优: {format_sci(np.max(self.y))}\n"
            stats_text += f"平均能量: {format_sci(np.mean(self.y))}\n"
            stats_text += f"平均shot_std: {format_sci(np.mean(self.shot_std))}\n"
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
        models, rmse = train_surrogate(
            self.X_opt,
            self.y,
            self.shot_std,
            mode_params,
            self.sample_weights,
        )
        if models is None:
            append_operation_log(
                "propose_config", "blocked", {"reason": "insufficient_data", "sample_count": int(len(self.y))}
            )
            messagebox.showwarning("提示", "GPR 数据不足，请先积累更多点")
            return

        self.rmse_history.append(rmse)
        self.holdout_sample_counts.append(int(len(self.y)))

        x_next, pred, pred_std, candidate_count, trust_radius, top_dims = propose_next(
            models, self.baseline, self.X_opt, self.y, mode_params
        )
        self.current = enforce_hard_constraints(x_next, self.baseline)
        self.current_shot_energies = []
        self.get_repeat_target()
        self.update_shot_buffer_display()
        current_best = float(np.max(self.y)) if len(self.y) else None
        suggest_repeat, suggest_message = evaluate_repeat_validation_need(
            pred, pred_std, current_best, self.training_effect
        )

        local_filename, windows_filename = save_dm_txt(self.current, self.shot_id)
        self.latest_config_file = local_filename
        self.latest_windows_config_file = windows_filename
        self.btn_resend_config.config(state=tk.NORMAL)
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
                f"优化维度: {OPTIMIZED_DIM_INDICES}\n"
                f"安全约束: 优化维度相对 {BASELINE_FILE} 不超过 ±{MAX_DELTA_FROM_BASELINE}\n"
                "非优化维度保持初始面形不变\n\n"
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
            "recommended_repeat_count": int(self.current_shot_target),
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
        save_data(
            self.current,
            energy,
            shot_std=shot_std,
            repeat_count=repeat_count,
            repeat_values=self.current_shot_energies,
        )
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
