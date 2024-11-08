import requests
import os
from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2
from PIL import Image, ImageDraw, ImageFont

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
    output_path = "output_with_detections.jpg"  # 输出图片的路径

    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
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
                # 绘制边框和标签
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                label = f"{class_name} {confidence:.2f}"
                text_size = draw.textsize(label, font)
                draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="red")
                draw.text((x1, y1 - text_size[1]), label, fill="white", font=font)
        img.save(output_path)
        print(f"处理完成，图片已保存至: {output_path}")
        print("检测结果:", detections)
        print(detections)
    else:
        print("图片文件不存在，无法进行处理。")
    
    

def main():
    getImage()     # 下载图片
    readImage()    # 读取目录中的图片
    processImage() # 处理图片

if __name__ == "__main__":
    main()
