# BO 拆分版项目

## 运行方式

把本文件夹放到原始数据文件所在目录，确保同级目录下有：

- `lhs_data.csv`
- `00102.txt`

然后运行：

```bash
python main.py
```

## 文件结构

```text
bo_split_project/
├── main.py              # 主程序入口
├── gui.py               # GUI 主界面
├── config.py            # 参数配置，包括 Windows 共享路径
├── model.py             # 模型训练、约束、BO 推荐
├── file_io.py           # 文件保存、txt 生成、复制到 Windows 共享文件夹
├── utils.py             # 日志、训练评估、样本权重等
├── config_viewer.py     # 配置文件查看器
├── test_share.py        # 单独测试共享文件夹写入
└── README.md
```

## 安全阈值和优化维度

在 `config.py` 中修改：

```python
MAX_DELTA_FROM_BASELINE = 250
OPTIMIZED_DIM_INDICES = list(range(10))
```

含义：

- `MAX_DELTA_FROM_BASELINE`：每个优化维度相对 `00102.txt` 初始面形允许变化的最大幅度。
- `OPTIMIZED_DIM_INDICES`：允许 BO 修改的执行器维度，可以是任意多个维度。
- 非优化维度会被强制保持为初始面形对应维度的值。

例如优化前 20 个维度：

```python
OPTIMIZED_DIM_INDICES = list(range(20))
```

例如只优化指定维度：

```python
OPTIMIZED_DIM_INDICES = [0, 2, 4, 6, 8, 10, 12]
```

GUI 会在“当前推荐面形”表格和右侧柱状图中展示每个维度相对初始面形的变化量、安全占比和状态。

## Burst / 重复 shot

每个参数点的重复 shot 数不固定。GUI 左侧“每点重复 shots”可以手动输入任意正整数。

每一发输入后点击“记录当前 Shot”，程序会缓存当前参数点下的多发结果；达到设定数量后自动按均值提交，也可以提前点击“按均值提交当前点”。

保存到 `lhs_data.csv` 的统计列包括：

- `shot_mean`
- `shot_std`
- `shot_var`
- `repeat_count`
- `repeat_values`

## Windows 共享文件夹

当前配置为：

```python
WINDOWS_SHARE_DIR = "/Volumes/BO0612"
COPY_TXT_TO_WINDOWS_SHARE = True
```

对应 Windows 端：

```text
C:\BO0612
```

前提是 Mac 已经挂载：

```text
smb://192.168.10.2/BO0612
```

## 单独测试共享文件夹

```bash
python test_share.py
```

如果成功，Windows 的 `C:\BO0612` 里会出现一个 `mac_python_share_test_*.txt` 文件。

## txt 生成逻辑

点击 GUI 的“推荐新配置”后，会调用 `file_io.py` 里的：

```python
save_dm_txt(vector, shot_id)
```

它会做两件事：

1. 本地保存到 `config/xxxx.txt`
2. 自动复制到 `/Volumes/BO0612/xxxx.txt`

如果复制失败，会写入：

```text
suanfa/runtime_logs/operation_log.csv
```
