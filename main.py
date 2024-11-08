import requests
import os
from ultralytics import YOLO
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

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
    directory = "/"  # Set the directory for image storage
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            print(f"Displaying: {image_path}")

def processImage():
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print("Failed to load YOLO model:", e)
        return

    filepath = "downloaded_photo.jpg"
    output_path = "output_with_detections.jpg"

    if not os.path.exists(filepath):
        print("Image file does not exist, cannot process.")
        return

    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    
    # Use a default truetype font if available for better readability
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    results = model(filepath)
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": box.conf.item(),
                "bbox": box.xyxy.tolist()[0]
            })
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label = f"{result.names[int(box.cls)]} {box.conf.item():.2f}"
            
            # Adjust handling of text size
            text_size = draw.textlength(label, font)
            if isinstance(text_size, tuple):
                text_width, text_height = text_size
            else:
                text_width, text_height = text_size, 10  # Default height if not tuple
            
            # Draw text background and label
            draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
            draw.text((x1, y1 - text_height), label, fill="white", font=font)

    img.save(output_path)
    print(f"Processing completed. Image saved at: {output_path}")
    print("Detections:", detections)
    
    return detections

@app.route('/process', methods=['POST'])
def process_route():
    getImage()
    detections = processImage()
    return jsonify(detections=detections), 200

def main():
    getImage()     # Download the image
    readImage()    # Read images in the directory
    processImage() # Process the image

if __name__ == "__main__":
    main()
    # To start the Flask server, uncomment the line below
    # app.run(debug=True, host='0.0.0.0', port=5000)