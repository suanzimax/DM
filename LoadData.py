import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import numpy as np

selected_data_df = None
baseline_data = None

def select_excel_file():
    global selected_data_df
    filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
    if filepath:
        selected_data_df = pd.read_excel(filepath)
        messagebox.showinfo("成功", f"已加载Excel文件：{filepath}")
        update_preview()

def select_baseline_file():
    global baseline_data
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filepath:
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
            baseline_data = list(map(int, lines[0].strip().split("\t")))
            baseline_label.config(text=f"基准文件已加载：{filepath}")
            messagebox.showinfo("成功", f"已加载基准文件：{filepath}")
            update_preview()
        except Exception as e:
            messagebox.showerror("错误", f"加载基准文件失败：{str(e)}")

def update_preview():
    if selected_data_df is None or baseline_data is None:
        return
    
    try:
        i = int(entry_i.get())
        if not (1 <= i <= len(selected_data_df)):
            return
    except ValueError:
        return
    
    # 获取选中的实验数据
    selected_data = selected_data_df.iloc[i-1].tolist()
    
    # 确保数据长度匹配
    if len(selected_data) != len(baseline_data):
        messagebox.showerror("错误", f"数据长度不匹配！Excel数据有{len(selected_data)}个值，基准数据有{len(baseline_data)}个值")
        return
    
    # 计算电压变化
    voltage_changes = np.array(selected_data) - np.array(baseline_data)
    
    # 计算新的电压值
    new_voltages = np.array(baseline_data) + voltage_changes
    
    # 检查是否超出阈值
    # 检查是否超出阈值（直接检查selected_data）
    out_of_bounds = (np.array(selected_data) < -17000) | (np.array(selected_data) > 17000)
    
    # 更新表格显示
    update_table(baseline_data, selected_data, voltage_changes, out_of_bounds)
    
    # 更新按钮状态
    if np.any(out_of_bounds):
        btn_save.config(state='disabled', text="保存LHS数据 (有电压超限)")
        status_label.config(text="警告：检测到电压超限！", fg="red")
    else:
        btn_save.config(state='normal', text="保存LHS数据")
        status_label.config(text="所有电压值在安全范围内", fg="green")

def update_table(baseline, selected, changes,  out_of_bounds):
    # 清空表格
    for item in tree.get_children():
        tree.delete(item)
    
    # 填充数据
    for i in range(len(baseline)):
        values = (
            f"A{i}",
            baseline[i],
            selected[i],
            f"{changes[i]:+d}"
        )
        
        # 如果超出阈值，用红色标记
        if out_of_bounds[i]:
            tree.insert("", "end", values=values, tags=("warning",))
        else:
            tree.insert("", "end", values=values)
    
    # 配置标签样式
    tree.tag_configure("warning", background="lightcoral")

def on_experiment_change(*args):
    update_preview()

def save_selected_data():
    global selected_data_df, baseline_data
    if selected_data_df is None:
        messagebox.showerror("错误", "请先选择并加载Excel文件！")
        return
    
    if baseline_data is None:
        messagebox.showerror("错误", "请先选择并加载基准文件！")
        return

    try:
        i = int(entry_i.get())
        if not (1 <= i <= len(selected_data_df)):
            raise ValueError
    except ValueError:
        messagebox.showerror("错误", "请输入有效的实验序号！")
        return

    selected_data = selected_data_df.iloc[i-1].tolist()
    
    # 最终确认对话框
    result = messagebox.askyesno("确认保存", 
                                f"确认保存实验序号 {i} 的LHS数据吗？\n"
                                f"数据将保存为新的txt文件。")
    if not result:
        return

    current_time = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    current_time = f"{current_time}_{i}"
    display_time = datetime.now().strftime('%m/%d/%Y %I:%M:%S %p')

    lhs_dir = "LHS_txt"
    os.makedirs(lhs_dir, exist_ok=True)
    # 生成完整文件路径
    filename = f"{current_time}.txt"
    filepath = os.path.join(lhs_dir, filename)
    with open(filepath, 'w') as file:
        file.write('\t'.join(map(str, selected_data)) + '\n')
        file.write(f"Set-up\t{display_time}\n")

    messagebox.showinfo("成功", f"LHS数据文件 {filename} 已生成。")

# 创建主窗口
root = tk.Tk()
root.title("2.LHS txt数据生成v1.0")
root.geometry('800x600')

# 文件选择区域
file_frame = tk.Frame(root)
file_frame.pack(pady=10, fill='x', padx=10)

btn_browse_excel = tk.Button(file_frame, text="选择Excel文件 (LHS数据)", command=select_excel_file)
btn_browse_excel.pack(side='left', padx=5)

btn_browse_baseline = tk.Button(file_frame, text="选择基准文件 (.txt)", command=select_baseline_file)
btn_browse_baseline.pack(side='left', padx=5)

baseline_label = tk.Label(file_frame, text="未选择基准文件", fg="gray")
baseline_label.pack(side='left', padx=10)

# 实验序号选择
exp_frame = tk.Frame(root)
exp_frame.pack(pady=5)

tk.Label(exp_frame, text="实验序号：").pack(side='left')
entry_i = tk.Entry(exp_frame, width=10)
entry_i.pack(side='left', padx=5)
entry_i.bind('<KeyRelease>', on_experiment_change)

# 状态显示
status_label = tk.Label(root, text="请选择文件", fg="blue")
status_label.pack(pady=5)

# 数据对比表格
table_frame = tk.Frame(root)
table_frame.pack(fill='both', expand=True, padx=10, pady=5)

columns = ("执行器", "基准电压", "LHS电压", "电压变化")
tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)

# 设置列标题和宽度
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100, anchor='center')

# 添加滚动条
scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)

tree.pack(side='left', fill='both', expand=True)
scrollbar.pack(side='right', fill='y')

# 保存按钮
btn_save = tk.Button(root, text="保存LHS数据", command=save_selected_data, state='disabled')
btn_save.pack(pady=10)

# 说明文本
info_text = """
LHS生成txt数据v1.0
"""
info_label = tk.Label(root, text=info_text, justify='left', fg="gray")
info_label.pack(pady=5, padx=10)

root.mainloop()
