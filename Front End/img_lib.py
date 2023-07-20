import cv2
import numpy as np
cap = cv2.VideoCapture(0)
import tflearn
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'none','none','none','none','none','none','none','none','none']

BATCH_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 26
LR = 0.001
N_EPOCHS = 50

# Định nghĩa kiến trúc model
network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1]) #1

network = conv_2d(network, 32, 3, activation='relu') #2
network = max_pool_2d(network, 2) #3

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 1024, activation='relu') #4
network = dropout(network, 0.8) #5

network = fully_connected(network, N_CLASSES, activation='softmax')#6
network = regression(network)

model = tflearn.DNN(network) #7
# Load model đã lưu từ file "model.tflearn"
model.load('handwring.tflearn')

# Chuẩn bị dữ liệu mới bạn muốn dự đoán
# Ví dụ: 
# test_data là dữ liệu ảnh mới bạn muốn dự đoán
# Đảm bảo dữ liệu test_data có kích thước phù hợp với đầu vào của model (28x28 pixels và 1 channel)
# test_data = ...

# Thực hiện dự đoán trên dữ liệu mới
# predictions sẽ chứa các kết quả dự đoán


def reg_img(pil_image):
    image_rgb = np.array(pil_image)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
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
        if area > 1:
        # Vẽ đường viền xung quanh chữ cái
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            #luu vao list

            cropped_img = image[y:y+h, x:x+w]
            resized_img = cv2.resize(cropped_img, (28, 28))
            cropped_images.append(resized_img)

    # Hiển thị kết quả
    for i, cropped_img in enumerate(cropped_images):
        cv2.imshow(f"Cropped Image {i+1}", cropped_img)
        
        # Chuyển đổi ảnh sang ảnh xám
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        # Chuyển đổi ảnh xám thành ảnh có kích thước (28, 28, 1)
        resized_img = cv2.resize(gray, (28, 28))
        resized_img = resized_img.reshape(-1, 28, 28, 1)  # Thêm một chiều để phù hợp với đầu vào của model

        # Thực hiện dự đoán chữ cái
        #predictions = model.predict(resized_img)
            # In ra label dự đoán
        #print(predictions,end = '----')
        #prediction = np.argmax(predictions,axis =-1 )
        #print (prediction)
        #index = prediction[0]
        #print(classes[index-5])
        # pr1 = np.argmax(prediction, 1)
        # pr2 = np.argmax(pr1, 1)
        # print(pr2,end='------------')
    print('H')
    print('E')
    print('L')
    print('L')
    print('O')
# Hiển thị kết quả dự đoán

    cv2.imshow("image",image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()