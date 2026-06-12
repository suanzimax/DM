"""主程序入口。运行：python main.py"""

import tkinter as tk

from gui import BO_GUI


if __name__ == "__main__":
    root = tk.Tk()
    app = BO_GUI(root)
    root.mainloop()
