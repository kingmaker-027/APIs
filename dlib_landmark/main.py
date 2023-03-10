from flask import Flask, render_template, Response, request
import cv2
import base64
import datetime
import numpy as np
import dlib
import pandas as pd

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("nose_landmarks_4400_09_03_23.dat")

app = Flask(__name__)

@app.route('/')
def index():
    """Render the index HTML template."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Return the live video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate frames from the video stream."""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        # To flip vertically, replace 1 with 0
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/capture_image')
def capture_image():
    """Capture an image, save it to a file, and return it as a response."""
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    
    if success:
        frame = cv2.flip(frame, 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        image = buffer.tobytes()
        with open('image/captured_image.jpg', 'wb') as f:
            f.write(image)
    else:
        image = b''
    return Response(image, mimetype='image/jpeg')

@app.route('/detect_landmarks')
def detect_landmarks():
    """Detect landmarks in the captured image and return it as a response."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('nose_landmarks_4400_09_03_23.dat')
    img = cv2.imread('image/captured_image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    Xco =[]
    Yco =[]
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        for i in range(3):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            Xco.append(x)  
            Yco.append(y) 
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        df = pd.DataFrame({'x': Xco, 'y': Yco})
        df.to_csv('results/results.csv', index=False)    
        ret, buffer = cv2.imencode('.jpg', img)
        image = buffer.tobytes()
        with open('results/result_image.jpg', 'wb') as f:
            f.write(image)
    else:
        image = b''
    return Response(image, mimetype='image/jpeg')


@app.route('/display_dataframe')
def display_dataframe():
    """Create a dataframe and return it as an HTML table."""
    data = pd.read_csv('results/results.csv')
    df = pd.DataFrame(data)
    html = df.to_html()
    return html


if __name__ == '__main__':
    app.run(debug=True)
