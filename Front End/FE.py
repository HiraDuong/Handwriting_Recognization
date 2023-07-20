import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, Button, Tk
from img_lib import *
from camera_lib import *

def start_camera():
    global camera_label
        # Tạo frame camera
    camera_window = tk.Toplevel(window)
    camera_window.title("CAMERA")
    camera_frame = tk.Frame(camera_window)
    # Frame img


    # Tạo label để hiển thị hình ảnh từ camera
    camera_label = tk.Label(camera_frame)
    camera_label.pack()
    
    camera_frame.pack()
    #update_frame()
    cam_reg()


def start_img():
    
    def open_file():
        new_window.withdraw()
        global photo
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((600, 400))  # Thay đổi kích thước hình ảnh để hiển thị
            

        reg_img(image)
    
    new_window = tk.Toplevel(window)
    new_window.title("Image Wiewer")

    # Tạo nút "Browse" để mở hộp thoại chọn tệp
    browse_button = tk.Button(new_window, text="Browse", command=open_file)
    browse_button.pack(pady= 5)
   

    

# Tạo cửa sổ
window = tk.Tk()
window.title("HandWitting Recognization")

# Đặt kích thước cửa sổ
window.geometry("800x600")

# Tạo frame bắt đầu
start_frame = tk.Frame(window)
start_frame.pack()

# Tạo nút "Start" để chuyển đến frame camera
cam_button = tk.Button(start_frame, text="CAM", command=start_camera)
cam_button.pack(pady=20)

img_button = tk.Button(start_frame, text="IMG", command=start_img)
img_button.pack(pady= 10)


# Khởi tạo đối tượng VideoCapture để đọc camera
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có khả dụng không
if not cap.isOpened():
    raise RuntimeError("Không thể mở camera")

# Chạy vòng lặp chính
window.mainloop()

# Giải phóng tài nguyên camera
cap.release()
