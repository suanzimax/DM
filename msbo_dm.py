import numpy as np
import pandas as pd
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real

# ========== 中文字体配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 参数 ==========
N_ACTUATORS = 52
VOLTAGE_BOUNDS = (-17000, 17000)
DATA_FILE = "lhs_data.csv"
N_TREES = 300

# ========== 配置文件验证窗口 ==========
class ConfigViewer:
    def __init__(self, parent, config_file=None):
        self.parent = parent
        self.config_data = None
        self.baseline_data = None
        
        self.window = tk.Toplevel(parent)
        self.window.title("配置文件验证器")
        self.window.geometry('800x700')
        self.window.grab_set()  # 模态窗口
        
        self.setup_ui()
        
        if config_file:
            self.load_config_file(config_file)

    def setup_ui(self):
        # 文件选择区域
        file_frame = tk.Frame(self.window)
        file_frame.pack(pady=10, fill='x', padx=10)

        btn_select_config = tk.Button(file_frame, text="选择配置文件 (.txt)", 
                                    command=self.select_config_file)
        btn_select_config.pack(side='left', padx=5)

        btn_select_baseline = tk.Button(file_frame, text="选择基准文件 (.txt)", 
                                      command=self.select_baseline_file)
        btn_select_baseline.pack(side='left', padx=5)

        self.file_status = tk.Label(file_frame, text="请选择文件", fg="gray")
        self.file_status.pack(side='left', padx=10)

        # 状态显示
        self.status_label = tk.Label(self.window, text="请选择配置文件和基准文件", 
                                   fg="blue", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # 数据表格
        table_frame = tk.Frame(self.window)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)

        columns = ("执行器", "基准电压", "配置电压", "电压变化")
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)

        # 设置列标题和宽度
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor='center')

        # 滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # 按钮区域
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(fill='x', padx=10, pady=10)

        btn_close = tk.Button(btn_frame, text="关闭", command=self.window.destroy,
                            font=("Arial", 12))
        btn_close.pack(side='right', padx=5)

        # 说明文本
        info_label = tk.Label(self.window, 
                            text="配置文件验证器v1.0 - 检查电压值是否在安全范围内",
                            fg="gray", font=("Arial", 10))
        info_label.pack(pady=5)

    def select_config_file(self):
        filepath = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.load_config_file(filepath)

    def select_baseline_file(self):
        filepath = filedialog.askopenfilename(
            title="选择基准文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.load_baseline_file(filepath)

    def load_config_file(self, filepath):
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
            
            # 读取第一行的电压数据
            voltage_line = lines[0].strip()
            self.config_data = list(map(int, voltage_line.split('\t')))
            
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
            self.baseline_data = list(map(int, voltage_line.split('\t')))
            
            messagebox.showinfo("成功", f"已加载基准文件: {filepath}")
            
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载基准文件失败: {str(e)}")

    def update_display(self):
        if self.config_data is None:
            return
            
        # 清空表格
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 检查数据长度
        if self.baseline_data and len(self.config_data) != len(self.baseline_data):
            self.status_label.config(
                text=f"错误：数据长度不匹配！配置文件{len(self.config_data)}个，基准文件{len(self.baseline_data)}个",
                fg="red"
            )
            return

        # 检查电压阈值
        out_of_bounds = []
        for i, voltage in enumerate(self.config_data):
            if voltage < VOLTAGE_BOUNDS[0] or voltage > VOLTAGE_BOUNDS[1]:
                out_of_bounds.append(i)

        # 填充表格数据
        for i, config_voltage in enumerate(self.config_data):
            if self.baseline_data and i < len(self.baseline_data):
                baseline_voltage = self.baseline_data[i]
                voltage_change = config_voltage - baseline_voltage
                values = (f"A{i}", baseline_voltage, config_voltage, f"{voltage_change:+d}")
            else:
                values = (f"A{i}", "N/A", config_voltage, "N/A")

            # 如果电压超出阈值，用红色标记
            if i in out_of_bounds:
                self.tree.insert("", "end", values=values, tags=("warning",))
            else:
                self.tree.insert("", "end", values=values)

        # 配置警告标签样式
        self.tree.tag_configure("warning", background="lightcoral")

        # 更新状态和按钮
        if out_of_bounds:
            self.status_label.config(
                text=f"警告：检测到 {len(out_of_bounds)} 个电压超限！不能加载到DM中",
                fg="red"
            )
        else:
            self.status_label.config(text="所有电压值在安全范围内", fg="green")
# ========== 原有的函数保持不变 ==========
def train_surrogate(X, y):
    """训练 surrogate (随机森林) 并返回 RMSE"""
    if len(y) < 10:
        return None, None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=N_TREES, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rf, rmse

def propose_next(rf):
    """利用 BO 推荐下一个点"""
    space = [Real(*VOLTAGE_BOUNDS) for _ in range(N_ACTUATORS)]
    def obj(x):
        return -rf.predict([x])[0]
    res = gp_minimize(obj, space, n_calls=30, n_random_starts=5,
                      acq_func="LCB", random_state=0)
    return np.array(res.x), -res.fun

def save_dm_txt(vector, shot_id):
    """保存推荐的 DM 配置"""
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)

    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    filename = f"{time_str}_{shot_id}.txt"
    filepath = os.path.join(config_dir, filename)
    
    line1 = "\t".join(str(int(v)) for v in vector)
    line2 = "Set-up\t" + datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    with open(filepath, "w") as f:
        f.write(line1 + "\n" + line2 + "\n")
    return filepath

def save_data(x, y):
    """保存实验数据到 CSV"""
    df_new = pd.DataFrame([np.concatenate([x, [y]])],
                          columns=[f"a{i}" for i in range(N_ACTUATORS)] + ["energy"])
    if not os.path.exists(DATA_FILE):
        df_new.to_csv(DATA_FILE, index=False)
    else:
        df_new.to_csv(DATA_FILE, mode="a", header=False, index=False)

# ========== 修改后的主GUI类 ==========
class BO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("4.BO 阶段 - DM 多步贝叶斯优化")
        self.root.geometry('1000x700')

        # 读取 LHS 阶段数据
        if not os.path.exists(DATA_FILE):
            messagebox.showerror("错误", f"找不到 {DATA_FILE}，请先完成 LHS 阶段")
            root.destroy()
            return
        df = pd.read_csv(DATA_FILE)
        self.X = df[[f"a{i}" for i in range(N_ACTUATORS)]].values
        self.y = df["energy"].values

        self.shot_id = len(self.y) + 1
        self.energy_history = []
        self.rmse_history = []
        self.best_energy = -1
        self.current = None

        self.setup_ui()
        self.update_plot()

    def setup_ui(self):
        # 创建左右布局
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        tk.Label(left_frame, text="贝叶斯优化控制", font=("Arial", 14, "bold")).pack(pady=10)

        self.label_status = tk.Label(left_frame, text="点击 '推荐新配置' 开始优化", 
                                   font=("Arial", 10), wraplength=200, justify=tk.LEFT)
        self.label_status.pack(pady=10)

        self.btn_propose = tk.Button(left_frame, text="推荐新配置", command=self.propose_config,
                                   font=("Arial", 12), bg="lightblue")
        self.btn_propose.pack(pady=5, fill=tk.X)

        # 新增：查看配置按钮
        self.btn_view_config = tk.Button(left_frame, text="查看配置文件", command=self.view_config,
                                       font=("Arial", 12), bg="lightyellow")
        self.btn_view_config.pack(pady=5, fill=tk.X)

        tk.Label(left_frame, text="输入实验能量:", font=("Arial", 10)).pack(pady=(20,5))
        
        self.entry_energy = tk.Entry(left_frame, width=15, font=("Arial", 12))
        self.entry_energy.pack(pady=5)
        self.entry_energy.bind('<Return>', lambda e: self.submit_energy())

        self.btn_submit = tk.Button(left_frame, text="提交实验能量", command=self.submit_energy,
                                  font=("Arial", 12), bg="lightgreen")
        self.btn_submit.pack(pady=5, fill=tk.X)

        # 统计信息
        tk.Label(left_frame, text="优化统计", font=("Arial", 12, "bold")).pack(pady=(30,10))
        
        self.label_stats = tk.Label(left_frame, text="", font=("Arial", 10), 
                                  justify=tk.LEFT, anchor="w")
        self.label_stats.pack(pady=5, fill=tk.X)

        # 右侧实时图表
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def view_config(self):
        """打开配置文件查看器"""
        ConfigViewer(self.root)

    # ========== 其余方法保持不变 ==========
    def update_plot(self):
        """实时更新图表"""
        self.ax1.clear()
        self.ax2.clear()

        # 能量优化曲线
        if self.energy_history:
            shots = list(range(1, len(self.energy_history) + 1))
            self.ax1.plot(shots, self.energy_history, marker="o", linewidth=2, markersize=6)
            self.ax1.axhline(y=max(self.energy_history), color='red', linestyle='--', alpha=0.7, 
                           label=f'最大值: {max(self.energy_history):.3f} MeV')
            self.ax1.legend()
        
        self.ax1.set_title("实验能量优化曲线", fontsize=14)
        self.ax1.set_xlabel("Shot ID")
        self.ax1.set_ylabel("能量 (MeV)")
        self.ax1.grid(True, alpha=0.3)

        # RMSE 曲线
        if self.rmse_history:
            iterations = list(range(1, len(self.rmse_history) + 1))
            self.ax2.plot(iterations, self.rmse_history, marker="s", color="orange", 
                         linewidth=2, markersize=6)
        
        self.ax2.set_title("Surrogate模型RMSE曲线", fontsize=14)
        self.ax2.set_xlabel("迭代次数")
        self.ax2.set_ylabel("RMSE")
        self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()

        # 更新统计信息
        stats_text = f"总实验次数: {len(self.energy_history)}\n"
        if self.energy_history:
            stats_text += f"当前最优: {max(self.energy_history):.3f} MeV\n"
            stats_text += f"平均能量: {np.mean(self.energy_history):.3f} MeV\n"
        if self.rmse_history:
            stats_text += f"当前RMSE: {self.rmse_history[-1]:.3f}"
        
        self.label_stats.config(text=stats_text)

    def propose_config(self):
        rf, rmse = train_surrogate(self.X, self.y)
        if rf is None:
            messagebox.showwarning("提示", "数据不足，请先积累更多点")
            return
        
        self.rmse_history.append(rmse)

        x_next, pred = propose_next(rf)
        self.current = np.clip(x_next.astype(int), *VOLTAGE_BOUNDS)

        filename = save_dm_txt(self.current, self.shot_id)
        self.label_status.config(
            text=f"推荐配置已保存:\n{filename}\n\n预测能量: {pred:.3f} MeV\n\n请实验员加载并输入实际能量"
        )
        
        self.update_plot()

    def submit_energy(self):
        val = self.entry_energy.get().strip()
        try:
            energy = float(val)
        except:
            messagebox.showerror("错误", "请输入数字能量")
            return

        if self.current is None:
            messagebox.showwarning("提示", "请先推荐新配置")
            return

        save_data(self.current, energy)
        self.X = np.vstack([self.X, self.current])
        self.y = np.append(self.y, energy)

        self.shot_id += 1
        self.energy_history.append(energy)
        self.best_energy = max(self.best_energy, energy)

        self.label_status.config(
            text=f"实验能量已记录:\n{energy:.3f} MeV\n\n当前最优能量:\n{self.best_energy:.3f} MeV\n\n可以推荐下一个配置"
        )
        self.entry_energy.delete(0, tk.END)
        
        self.update_plot()

# ========== 主程序 ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = BO_GUI(root)
    root.mainloop()