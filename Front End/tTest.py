import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)

        # Tạo một cửa sổ Toplevel mới
        new_window = tk.Toplevel(root)
        new_window.title("Image Viewer")

        # Hiển thị ảnh trong cửa sổ mới
        image_label = tk.Label(new_window, image=photo)
        image_label.pack()

root = tk.Tk()

browse_button = tk.Button(root, text="Browse", command=open_file)
browse_button.pack()

root.mainloop()
