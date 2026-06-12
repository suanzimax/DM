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
