"""将初始面形 txt 文件导入为 Excel 表格，列格式与 lhs_data.csv 保持一致。"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_TXT = BASE_DIR / "00102.txt"
DEFAULT_TEMPLATE_CSV = BASE_DIR / "lhs_data.csv"
DEFAULT_OUTPUT_XLSX = BASE_DIR / "initial_surface.xlsx"
N_ACTUATORS = 52


def read_initial_surface(txt_path):
    """读取 txt 第一行的 52 维初始面形电压。"""
    with open(txt_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    values = [int(item) for item in first_line.split("\t") if item.strip()]
    if len(values) != N_ACTUATORS:
        raise ValueError(f"{txt_path} 维度错误：期望 {N_ACTUATORS} 维，实际 {len(values)} 维")
    return values


def load_template_columns(template_csv):
    """优先使用 lhs_data.csv 的列名，保证导出的 Excel 与训练数据格式一致。"""
    if template_csv.exists():
        return list(pd.read_csv(template_csv, nrows=0).columns)
    return [f"a{i}" for i in range(N_ACTUATORS)] + [
        "energy",
        "shot_mean",
        "shot_std",
        "repeat_count",
        "shot_var",
        "repeat_values",
    ]


def build_initial_surface_row(values, columns):
    """构建一行与 lhs_data.csv 同格式的初始面形数据。"""
    row = {column: "" for column in columns}
    for idx, value in enumerate(values):
        row[f"a{idx}"] = value

    # 初始面形不是一次实验测量，目标值相关列保持空值；重复次数记为 0。
    if "repeat_count" in row:
        row["repeat_count"] = 0
    if "shot_std" in row:
        row["shot_std"] = 0.0
    if "shot_var" in row:
        row["shot_var"] = 0.0
    return row


def export_initial_surface_to_excel(
    txt_path=DEFAULT_BASELINE_TXT,
    template_csv=DEFAULT_TEMPLATE_CSV,
    output_xlsx=DEFAULT_OUTPUT_XLSX,
):
    """导出初始面形 Excel 文件并返回输出路径。"""
    values = read_initial_surface(txt_path)
    columns = load_template_columns(template_csv)
    row = build_initial_surface_row(values, columns)
    df = pd.DataFrame([row], columns=columns)
    df.to_excel(output_xlsx, index=False)
    return output_xlsx


def main():
    output_path = export_initial_surface_to_excel()
    print(f"已导出初始面形 Excel: {output_path}")


if __name__ == "__main__":
    main()
