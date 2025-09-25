import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import os

# LHS算法实现
def lhs_sampling(n_samples, dimensions, low, high):
    intervals = np.linspace(0, 1, n_samples + 1)
    points = np.random.uniform(size=(n_samples, dimensions))
    for i in range(dimensions):
        np.random.shuffle(points[:, i])
    samples = low + (points * (high - low))
    return samples.astype(int)

# 生成数据并保存为Excel
def generate_and_save():
    filename = 'LHS_data.xlsx'

    # 检查是否存在已有文件
    if os.path.exists(filename):
        overwrite = messagebox.askyesno("文件已存在", f"{filename} 已存在。是否覆盖？")
        if not overwrite:
            messagebox.showinfo("提示", "保留原有文件，未进行新的生成。")
            return

    try:
        N = int(entry_N.get())
        n_dim = int(entry_dim.get())
        if N <= 0 or n_dim <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("错误", "请输入有效的正整数！")
        return

    data = lhs_sampling(N, n_dim, -17000, 17000)
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    messagebox.showinfo("成功", f"已生成并保存为 {filename}")

# GUI界面
root = tk.Tk()
root.title("1.促动器电压数据生成")
root.geometry('350x200')

tk.Label(root, text="数据组数(N)：").pack(pady=5)
entry_N = tk.Entry(root)
entry_N.pack(pady=5)

tk.Label(root, text="促动器数量(n)：").pack(pady=5)
entry_dim = tk.Entry(root)
entry_dim.pack(pady=5)

btn_generate = tk.Button(root, text="生成数据并保存Excel", command=generate_and_save)
btn_generate.pack(pady=10)

root.mainloop()
