import cv2
import numpy as np
def cam_reg():
    cap = cv2.VideoCapture(0)
    while True: 
        _, image = cap.read()
        #image = cv2.flip(image, 1)
        #image = image = cv2.imread('test4.png')
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
        cropped_images = []

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
                cropped_images.append(cropped_img)
        
        # Hiển thị kết quả
        for i, cropped_img in enumerate(cropped_images):
            cv2.imshow(f"Cropped Image {i+1}", cropped_img)
        print(len(cropped_images))
        cv2.imshow("video",image)

        if cv2.waitKey(1) == ord('q'):
            break