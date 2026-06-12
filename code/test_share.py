"""单独测试能否写入 Windows 共享文件夹。"""

from file_io import test_windows_share_write

if __name__ == "__main__":
    path = test_windows_share_write()
    print("测试文件已写入 Windows 共享文件夹：")
    print(path)
