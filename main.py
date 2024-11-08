import requests
import os
from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2

def getImage():
    url = 'http://172.17.40.97:8080/photo.jpg'
    try:
        response = requests.get(url=url)
        if response.status_code == 200:
            with open("downloaded_photo.jpg", "wb") as file:
                file.write(response.content)
            print("Photo downloaded successfully.")
        else:
            print("Failed to take photo. Status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

def readImage():
    directory = "/"  # 设置图片存储的目录
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 根据需要过滤文件类型
            image_path = os.path.join(directory, filename)
            print(f"正在显示: {image_path}")

def processImage():
    model = YOLO('yolov8n.pt')
    filepath = "downloaded_photo.jpg"  # 假设下载后的图片路径

    if os.path.exists(filepath):
        results = model(filepath)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": box.conf.item(),
                    "bbox": box.xyxy.tolist()[0]  # Bounding box coordinates
                })
        print(detections)
    else:
        print("图片文件不存在，无法进行处理。")
    
    

def main():
    getImage()     # 下载图片
    readImage()    # 读取目录中的图片
    processImage() # 处理图片

if __name__ == "__main__":
    main()
