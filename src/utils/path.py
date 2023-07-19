import tkinter as tk

from tkinter import filedialog


def select_path():
    """
    Select a path where e.g. some images are stored.
    :return: Path to folder
    """
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askdirectory()
    path = path + '/'
    print(path)
    return path