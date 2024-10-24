import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use yolov8n.pt, yolov8s.pt, etc.

# Initialize Flask app
app = Flask(__name__)

# Directory to temporarily save uploaded images
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a route for image upload and object detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Perform YOLOv8 inference
        results = model(filepath)

        # Extract detection results
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": box.conf.item(),
                    "bbox": box.xyxy.tolist()[0]  # Bounding box coordinates
                })

        # Clean up the uploaded file
        os.remove(filepath)

        # Return results as JSON
        return jsonify({"detections": detections})
    
    return jsonify({"error": "File upload failed"}), 500

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)