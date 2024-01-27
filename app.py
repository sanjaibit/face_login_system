#app.py
from curses import flash
import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
import zlib
from base64 import b64decode
from flask import flash
import time
import face_recognition
from datetime import datetime
from werkzeug.utils import secure_filename
import atexit


# Initialize a global variable to store the detected name
detected_name = None
video_feed_is_running = True

app_face = Flask(__name__)

app_face.config['SECRET_KEY'] = 'your_secret_key'
app_face.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app_face)
migrate = Migrate(app_face, db)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    department = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    image = db.Column(db.String(500), nullable=True, default='default_image.jpg')


# A simple list to store registered users (in a real application, you would use a database)

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)

cap = None  # Initialize the camera capture as None


def initialize_camera_capture():
    global cap
    cap = cv2.VideoCapture(0)

# Register a function to be called on application shutdown
atexit.register(lambda: cap.release() if cap is not None else None)

def generate_frames():
    global detected_name

    while True:
            success, img = cap.read()
            #imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    detected_name = classNames[matchIndex]
                else:
                    detected_name = ""

            if detected_name:
                # Display the detected name at the top center of the frame
                cv2.putText(img, detected_name, (int(img.shape[1]/2) - 50, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app_face.route('/')
def new_home():
    return render_template('login.html', user_data=None)  # Pass user_data as None initially

@app_face.route('/login', methods=['POST'])
def login():
    global detected_name, video_feed_is_running  # Access the global variables

    if request.method == 'POST' and detected_name is not None:

        # Query the database to find the user by username
        user = User.query.filter_by(username=detected_name).first()

        if user:
            user_data = {
                'username': user.username,
                'name': user.name,
                'age': user.age,
                'phone': user.phone,
                'password': user.password,  # Note: Passwords should not be exposed like this in a real application
                'department': user.department
            }

            detected_name = None  # Reset the detected_name

            if video_feed_is_running:
                cap.release()  # Release the camera
                cv2.destroyAllWindows()  # Close OpenCV windows
                video_feed_is_running = False

            # User found, render the login.html template with user data
            return render_template('login.html', user_data=user_data, video_feed_is_running=video_feed_is_running)

    # Render the login.html template without user data if not logged in
    return render_template('login.html', video_feed_is_running=video_feed_is_running)

@app_face.route('/logout')
def logout():
    # Add any necessary logic to handle the logout action (e.g., clearing session data)
    return redirect('/')

@app_face.route('/lo', methods=['GET', 'POST'])
def home():
    global cap
    if cap is not None:
        cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close OpenCV windows
        cap = None

    initialize_camera_capture()
    return render_template('home.html')


offline_image_path = os.path.join("static", "C:\\Users\\sanja\\OneDrive\\Documents\\New folder\\offline_image.jpg")

@app_face.route('/video_feed')
def video_feed():
    if cap is None:
        img = cv2.imread(offline_image_path)
        if img is None:
            print("Error loading the static image")
            return Response(status=404)
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        return Response(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app_face.route('/reg', methods=['GET', 'POST'])
def register():
    global detected_name, video_feed_is_running  # Access the global variables
    if video_feed_is_running:
            cap.release()  # Release the camera
            cv2.destroyAllWindows()  # Close OpenCV windows
            video_feed_is_running = False


    if request.method == 'POST':

        # Capture user data and image
        name = request.form.get('name')
        age = request.form.get('age')
        department = request.form.get('department')
        phone = request.form.get('phone')
        username = request.form.get('username')
        password = request.form.get('password')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))

        # Store user data and image in the databas
        image_file = request.files['image']

        if image_file:
            # Create a unique filename based on the username with a .jpg extension
            filename = f'{username}.jpg'

            # Specify the directory where you want to save the image
            save_directory = "C:\\Users\\sanja\\OneDrive\\Documents\\New folder\\ImagesAttendance"

            # Create the full file path
            file_path = os.path.join(save_directory, filename)

            # Save the image file in JPEG format with the new filename
            image_file.save(file_path)
        user = User(
            username=username,
            password=generate_password_hash(password),
            name=name,
            age=age,
            department=department,
            phone=phone,
            # Store the base64-encoded image data
        )

        db.session.add(user)
        db.session.commit()

       

        flash('Registration successful. You can now log in.', 'success')

        return render_template('home.html')

    return render_template('register.html')

if _name_ == '__main__':
    
    app_face.run(debug=True)