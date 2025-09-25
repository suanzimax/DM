import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox
# ========== 中文字体配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ---------- 参数 ----------
N_ACTUATORS = 52
VOLTAGE_BOUNDS = (-17000, 17000)
EXCEL_FILE = "LHS_data.xlsx"
DATA_FILE = "lhs_data.csv"# 记录实验数据的CSV文件


def read_dm_txt(filename):
    """读取 DM txt 文件（取第一行电压向量）"""
    with open(filename, "r") as f:
        lines = f.readlines()
    voltages = [int(v) for v in lines[0].strip().split()]
    return np.array(voltages)

def calc_rmse(X, y):
    """计算随机森林 RMSE"""
    if len(y) < 10:
        return None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# ---------- GUI 类 ----------
class LHS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3.LHS 阶段稳定性检测")

        # 加载已有数据
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            self.X = df[[f"a{i}" for i in range(N_ACTUATORS)]].values
            self.y = df["energy"].values
        else:
            self.X, self.y = np.empty((0, N_ACTUATORS)), np.array([])

        self.rmse_list = []
        self.current_x = None
        self.current_file = None

        # UI 布局
        self.setup_ui()
        self.update_plot()

    def setup_ui(self):
        frame_top = tk.Frame(self.root)
        frame_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        frame_bottom = tk.Frame(self.root)
        frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # Matplotlib 图形区
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_top)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 文件选择按钮
        btn_load = tk.Button(frame_bottom, text="选择 LHS txt 文件", command=self.load_txt)
        btn_load.pack(side=tk.LEFT, padx=5)

        # 显示文件名
        self.label_file = tk.Label(frame_bottom, text="未加载文件")
        self.label_file.pack(side=tk.LEFT, padx=5)

        # 能量输入
        tk.Label(frame_bottom, text="输入实验测得能量 (MeV):").pack(side=tk.LEFT, padx=5)
        self.entry_energy = tk.Entry(frame_bottom, width=10)
        self.entry_energy.pack(side=tk.LEFT, padx=5)

        # 提交按钮
        btn_submit = tk.Button(frame_bottom, text="提交", command=self.submit_energy)
        btn_submit.pack(side=tk.LEFT, padx=5)

        # 退出按钮
        btn_quit = tk.Button(frame_bottom, text="退出", command=self.root.quit)
        btn_quit.pack(side=tk.RIGHT, padx=5)

    def load_txt(self):
        txt_path = filedialog.askopenfilename(title="选择 LHS txt 文件", filetypes=[("Text Files", "*.txt")])
        if not txt_path:
            return
        x = read_dm_txt(txt_path)
        x = np.clip(x, *VOLTAGE_BOUNDS)
        self.current_x = x
        self.current_file = os.path.basename(txt_path)
        self.label_file.config(text=f"已加载: {self.current_file}")

    def submit_energy(self):
        if self.current_x is None:
            messagebox.showerror("错误", "请先选择一个 txt 文件！")
            return
        val = self.entry_energy.get().strip()
        try:
            energy = float(val)  #  输入的是实际实验测得的能量
        except:
            messagebox.showerror("错误", "请输入一个有效数字！")
            return

        # 保存
        df_new = pd.DataFrame([np.concatenate([self.current_x, [energy]])],
                              columns=[f"a{j}" for j in range(N_ACTUATORS)] + ["energy"])
        if not os.path.exists(DATA_FILE):
            df_new.to_csv(DATA_FILE, index=False)
        else:
            df_new.to_csv(DATA_FILE, mode="a", header=False, index=False)

        # 更新数据
        df = pd.read_csv(DATA_FILE)
        self.X = df[[f"a{j}" for j in range(N_ACTUATORS)]].values
        self.y = df["energy"].values

        # 更新 RMSE
        rmse = calc_rmse(self.X, self.y)
        if rmse:
            self.rmse_list.append(rmse)
            if len(self.rmse_list) >= 6:
                recent = self.rmse_list[-6:]
                if (max(recent) - min(recent)) / max(recent) < 0.05:
                    messagebox.showinfo("提示", " LHS 足够，可以进入 BO 阶段")

        # 清空输入
        self.entry_energy.delete(0, tk.END)

        # 刷新曲线
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        if len(self.rmse_list) > 0:
            self.ax.plot(range(1, len(self.rmse_list)+1), self.rmse_list, marker="o")
        self.ax.set_title("RMSE 学习曲线")
        self.ax.set_xlabel("检测次数")
        self.ax.set_ylabel("RMSE")
        self.fig.tight_layout()
        self.canvas.draw()

# ---------- 主入口 ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = LHS_GUI(root)
    root.mainloop()
