"""配置文件验证窗口。"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import VOLTAGE_BOUNDS


class ConfigViewer:
    def __init__(self, parent, config_file=None):
        self.parent = parent
        self.config_data = None
        self.baseline_data = None

        self.window = tk.Toplevel(parent)
        self.window.title("配置文件验证器")
        self.window.geometry("800x700")
        self.window.grab_set()

        self.setup_ui()

        if config_file:
            self.load_config_file(config_file)

    def setup_ui(self):
        file_frame = tk.Frame(self.window)
        file_frame.pack(pady=10, fill="x", padx=10)

        btn_select_config = tk.Button(
            file_frame, text="选择配置文件 (.txt)", command=self.select_config_file
        )
        btn_select_config.pack(side="left", padx=5)

        btn_select_baseline = tk.Button(
            file_frame, text="选择基准文件 (.txt)", command=self.select_baseline_file
        )
        btn_select_baseline.pack(side="left", padx=5)

        self.file_status = tk.Label(file_frame, text="请选择文件", fg="gray")
        self.file_status.pack(side="left", padx=10)

        self.status_label = tk.Label(
            self.window, text="请选择配置文件和基准文件", fg="blue", font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        table_frame = tk.Frame(self.window)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)

        columns = ("执行器", "基准电压", "配置电压", "电压变化")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_frame = tk.Frame(self.window)
        btn_frame.pack(fill="x", padx=10, pady=10)

        btn_close = tk.Button(btn_frame, text="关闭", command=self.window.destroy, font=("Arial", 12))
        btn_close.pack(side="right", padx=5)

        info_label = tk.Label(
            self.window,
            text="配置文件验证器v1.0 - 检查电压值是否在安全范围内",
            fg="gray",
            font=("Arial", 10),
        )
        info_label.pack(pady=5)

    def select_config_file(self):
        filepath = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filepath:
            self.load_config_file(filepath)

    def select_baseline_file(self):
        filepath = filedialog.askopenfilename(
            title="选择基准文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if filepath:
            self.load_baseline_file(filepath)

    def load_config_file(self, filepath):
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            voltage_line = lines[0].strip()
            self.config_data = list(map(int, voltage_line.split("\t")))

            self.file_status.config(text=f"配置文件已加载: {os.path.basename(filepath)}")
            messagebox.showinfo("成功", f"已加载配置文件: {filepath}")

            self.update_display()

        except Exception as e:
            messagebox.showerror("错误", f"加载配置文件失败: {str(e)}")

    def load_baseline_file(self, filepath):
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            voltage_line = lines[0].strip()
            self.baseline_data = list(map(int, voltage_line.split("\t")))

            messagebox.showinfo("成功", f"已加载基准文件: {filepath}")

            self.update_display()

        except Exception as e:
            messagebox.showerror("错误", f"加载基准文件失败: {str(e)}")

    def update_display(self):
        if self.config_data is None:
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.baseline_data and len(self.config_data) != len(self.baseline_data):
            self.status_label.config(
                text=(
                    f"错误：数据长度不匹配！配置文件{len(self.config_data)}个，"
                    f"基准文件{len(self.baseline_data)}个"
                ),
                fg="red",
            )
            return

        out_of_bounds = []
        for i, voltage in enumerate(self.config_data):
            if voltage < VOLTAGE_BOUNDS[0] or voltage > VOLTAGE_BOUNDS[1]:
                out_of_bounds.append(i)

        for i, config_voltage in enumerate(self.config_data):
            if self.baseline_data and i < len(self.baseline_data):
                baseline_voltage = self.baseline_data[i]
                voltage_change = config_voltage - baseline_voltage
                values = (f"A{i}", baseline_voltage, config_voltage, f"{voltage_change:+d}")
            else:
                values = (f"A{i}", "N/A", config_voltage, "N/A")

            if i in out_of_bounds:
                self.tree.insert("", "end", values=values, tags=("warning",))
            else:
                self.tree.insert("", "end", values=values)

        self.tree.tag_configure("warning", background="lightcoral")

        if out_of_bounds:
            self.status_label.config(
                text=f"警告：检测到 {len(out_of_bounds)} 个电压超限！不能加载到DM中",
                fg="red",
            )
        else:
            self.status_label.config(text="所有电压值在安全范围内", fg="green")
