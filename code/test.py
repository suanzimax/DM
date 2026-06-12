from pathlib import Path
from datetime import datetime

# Windows 共享文件夹在 Mac 上的挂载路径
# 对应 Windows 里的 C:\BO0612
SHARE_DIR = Path("/Volumes/BO0612MATLAB")

def main():
    print("正在测试 Windows 共享文件夹写入...")
    print(f"目标共享路径: {SHARE_DIR}")

    # 1. 检查共享文件夹是否已经挂载
    if not SHARE_DIR.exists():
        print("错误：没有找到 /Volumes/BO0612")
        print("请先在 Mac Finder 中连接：smb://192.168.10.2/BO0612")
        return

    # 2. 检查是否是目录
    if not SHARE_DIR.is_dir():
        print("错误：/Volumes/BO0612 存在，但不是一个文件夹")
        return

    # 3. 生成测试 txt 文件名
    now = datetime.now()
    filename = f"test_share_{now.strftime('%Y%m%d_%H%M%S')}.txt"
    file_path = SHARE_DIR / filename

    # 4. 写入测试内容
    content = [
        "这是 Mac Python 写入 Windows 共享文件夹的测试文件。",
        f"写入时间：{now.strftime('%Y-%m-%d %H:%M:%S')}",
        "如果你能在 Windows 的 C:\\BO0612 里看到这个文件，说明共享成功。",
        "",
        "下面模拟一个 52 维 DM 电压向量：",
        "\t".join(str(i) for i in range(52)),
        "Set-up\t" + now.strftime("%m/%d/%Y %I:%M:%S %p"),
    ]

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")

        print("写入成功！")
        print(f"文件路径: {file_path}")
        print("请去 Windows 的 C:\\BO0612 查看这个 txt 文件。")

    except PermissionError:
        print("写入失败：权限不足。")
        print("请检查 Windows 共享权限和安全权限是否允许当前用户写入。")

    except Exception as e:
        print("写入失败，错误信息：")
        print(e)

if __name__ == "__main__":
    main()