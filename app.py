import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

device = 'cpu'
# Load your models (assuming you have separate models for each task)
two_wheeler_model = YOLO("models/twowheeler_model_yolov8_final.pt").to(device)  # Replace with your actual two-wheeler model path
helmet_model = YOLO("models/helmet_model_yolov8_initial2.pt").to(device)  # Your provided helmet model
license_plate_model = YOLO("models/license_model_yolov8_initial1.pt").to(device)  # Replace with your actual license plate model

def detect_two_wheelers(image):
    results = two_wheeler_model.predict(source=image, conf=0.5)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            boxes.append((x1, y1, x2, y2))
    return boxes

def detect_helmet(two_wheeler_img):
    img_with_helmet = two_wheeler_img.copy()
    results = helmet_model.predict(source=img_with_helmet, conf=0.5, device=device) 
    has_helmet = False
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Assuming class 0 is 'helmet'
                has_helmet = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_with_helmet, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                st.write(f"Helmet detected at ({x1}, {y1}, {x2}, {y2})")
    return has_helmet, img_with_helmet

def get_license_plate(two_wheeler_img):
    results = license_plate_model.predict(source=two_wheeler_img, conf=0.5)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            license_plate_img = two_wheeler_img[y1:y2, x1:x2]
            return license_plate_img
    return None 

def process_image(image):
    img_array = np.array(image)
    two_wheeler_boxes = detect_two_wheelers(img_array)
    if two_wheeler_boxes:
        for x1, y1, x2, y2 in two_wheeler_boxes:
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(img_array, caption='Two wheelers detected', use_column_width=True)

    for box in two_wheeler_boxes:
        x1, y1, x2, y2 = box
        two_wheeler_img = img_array[y1:y2, x1:x2]
        has_helmet, helmet_highlighted_img = detect_helmet(two_wheeler_img)
        st.image(helmet_highlighted_img, caption='Helmet detection', use_column_width=True)
        if not has_helmet:
            license_plate_img = get_license_plate(two_wheeler_img)
            if license_plate_img is not None:
                license_plate_img_rgb = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2RGB)
                st.error("Violation detected!")
                st.image(license_plate_img_rgb, caption=f"License Plate for Two-Wheeler at ({x1}, {y1})", width=200)
            else:
                st.error("Violation detected but no license plate found!")
        else:
            st.success("No violation detected for this two-wheeler")
    
    return img_array, two_wheeler_boxes

def main():
    st.title("Two-Wheeler Helmet Violation Detector")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Process Image'):
            try:
                processed_img, boxes = process_image(image)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()
