import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
from img_lib import *
import numpy as np


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
    update_frame()

def update_frame():
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    if ret:
        # Lật ngược khung hình
        frame = cv2.flip(frame, 1)

        # Chuyển đổi khung hình thành định dạng hình ảnh
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((600, 400), Image.ANTIALIAS)  # Điều chỉnh kích thước hình ảnh

        # Cập nhật hình ảnh lên giao diện
        img_tk = ImageTk.PhotoImage(image=img)
        camera_label.configure(image=img_tk)
        camera_label.image = img_tk

    # Gọi lại hàm update_frame sau một khoảng thời gian
    camera_label.after(10, update_frame)



def start_img():
    
    def open_file():
        global photo
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((600, 400))  # Thay đổi kích thước hình ảnh để hiển thị
            
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
        reg_img(image)
            #cv2.imshow(image)
            #print (type(image))
    
    new_window = tk.Toplevel(window)
    new_window.title("Image Wiewer")

    # Tạo nút "Browse" để mở hộp thoại chọn tệp
    browse_button = tk.Button(new_window, text="Browse", command=open_file)
    browse_button.pack(pady= 5)
    #hien thi anh 
    image_label = tk.Label(new_window)
    image_label.pack()
    new_window.mainloop()

def reg_img(pil_image):
    image = np.array(pil_image)
    cropped_images = []
    #image = image = cv2.imread('test6.png')
    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Sobel để tăng cường biên cạnh
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.addWeighted(np.absolute(sobelx), 0.5, np.absolute(sobely), 0.5, 0)

    # Áp dụng phép làm mờ để loại bỏ nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện biên cạnh bằng phương pháp Canny
    edges = cv2.Canny(blurred, 100, 120,3)
    dilated = cv2.dilate(edges, None, iterations=2)
    # Tìm các đường viền trong ảnh
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lặp qua từng đường viền
    for contour in contours:
    # Tính diện tích của đường viền
        area = cv2.contourArea(contour)

    # Nếu diện tích đủ lớn (tùy chỉnh ngưỡng theo yêu cầu của bạn)
        if area > 100:
        # Vẽ đường viền xung quanh chữ cái
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            #luu vao list

            cropped_img = image[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, (28, 28))
            cropped_images.append(resized_img)

    # Hiển thị kết quả
    for i, cropped_img in enumerate(cropped_images):
        cv2.imshow(f"Cropped Image {i+1}", cropped_img)

    cv2.imshow("video",image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
