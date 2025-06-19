from flask import Flask, request, render_template, redirect, url_for, session
import cv2
import os
from datetime import date, datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

#### Defining Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
    print("Attendance directory created.")
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
    print("static/faces directory created.")

#### Paths for CSV files
total_users_file = 'Attendance/total_users.csv'
if not os.path.exists(total_users_file):
    with open(total_users_file, 'w') as f:
        f.write('Name,Roll\n')
    print("total_users.csv created.")

#### Global variables
cap = None  # Initialize the video capture object globally

#### Get the number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### Extract the face from an image using Haar Cascade
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

#### Identify face using a pre-trained KNN model
def identify_face(facearray):
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return "Unknown Person"
    model = joblib.load('static/face_recognition_model.pkl')
    try:
        prediction = model.predict(facearray)
        return prediction[0]
    except:
        return "Unknown Person"

#### Train the model on all the faces available in the faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (160, 160))  # Resize to a standard size
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    print("Model trained and saved.")

#### Add Attendance of a specific user
def add_attendance(name):
    if name == "Unknown Person":
        print("Unknown person detected. Attendance not recorded.")
        return  # Do not add attendance for unknown persons
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Check if the user's attendance is already recorded for the day
    present_file = f'Attendance/Present-{datetoday}.csv'
    if not os.path.exists(present_file):
        with open(present_file, 'w') as f:
            f.write('Name,Roll,Time\n')
    
    df = pd.read_csv(present_file)
    if int(userid) not in df['Roll'].values:
        with open(present_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')
        print(f"Attendance recorded for {username} ({userid}) at {current_time}")

#### Generate Absentee List and Summary
def generate_daily_reports():
    try:
        # Get list of present users
        present_file = f'Attendance/Present-{datetoday}.csv'
        if not os.path.exists(present_file):
            present_users = set()
            print("No present users file found.")
        else:
            df_present = pd.read_csv(present_file)
            present_users = set(df_present['Roll'].astype(str))
            print(f"Present users: {present_users}")
        
        # Get list of all registered users
        df_total = pd.read_csv(total_users_file)
        all_users = set(df_total['Roll'].astype(str))
        print(f"All users: {all_users}")
        
        # Find absent users
        absent_users = all_users - present_users
        print(f"Absent users: {absent_users}")
        
        # Save absent users to a file
        absent_file = f'Attendance/Absent-{datetoday}.csv'
        with open(absent_file, 'w') as f:
            f.write('Name,Roll\n')
            for user in absent_users:
                username = df_total[df_total['Roll'] == int(user)]['Name'].values[0]
                f.write(f'{username},{user}\n')
        print(f"Absent users saved to {absent_file}")
        
        # Generate summary file
        summary_file = f'Attendance/Summary-{datetoday}.csv'
        total_users = len(all_users)
        present_count = len(present_users)
        absent_count = len(absent_users)
        with open(summary_file, 'w') as f:
            f.write('Total Students,Present Students,Absent Students\n')
            f.write(f'{total_users},{present_count},{absent_count}\n')
        print(f"Summary saved to {summary_file}")
    except Exception as e:
        print(f"Error generating daily reports: {e}")

#### Update Total Users CSV
def update_total_users(newusername, newuserid):
    df = pd.read_csv(total_users_file)
    if int(newuserid) not in df['Roll'].values:
        with open(total_users_file, 'a') as f:
            f.write(f'{newusername},{newuserid}\n')
        print(f"User {newusername} ({newuserid}) added to total_users.csv.")

################## ROUTING FUNCTIONS #########################

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html', totalreg=totalreg(), datetoday2=datetoday2, datetoday=datetoday) 
    
@app.route("/adds", methods=['GET', 'POST'])
def adds():
    return render_template("adds.html", datetoday2=datetoday2, totalreg=totalreg())

#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET', 'POST'])
def start():
    global cap
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('adds.html', totalreg=totalreg(), datetoday2=datetoday2)

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (160, 160))  # Resize for recognition
            identified_person = identify_face(face.reshape(1, -1))
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    # Generate daily reports after attendance is taken
    generate_daily_reports()
    
    return redirect(url_for('index'))

#### This function will stop the attendance process
@app.route('/stop', methods=['GET', 'POST'])
def stop():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    print("Generating daily reports...")
    generate_daily_reports()
    return redirect(url_for('index'))

#### This function will log out and redirect to the login page
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    session.clear()  # Clear the session data
    return redirect(url_for('login'))  # Redirect to the login page

@app.route("/attendance", methods=['GET', 'POST'])
def attendance():
    return render_template("attendance.html", datetoday2=datetoday2, totalreg=totalreg())

#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    
    # Check if user ID already exists
    df = pd.read_csv(total_users_file)
    if int(newuserid) in df['Roll'].values:
        return "Error: User ID already exists. Please choose a different ID."
    
    # Create user folder
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    # Capture user images
    global cap
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while i < 50:  # Capture 50 images
        _, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))  # Keep original frame size
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:  # Save an image every 5 frames (faster capture)
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit early
            break
    cap.release()
    cv2.destroyAllWindows()
    
    # Update total users CSV
    update_total_users(newusername, newuserid)
    
    # Train the model with the new user
    print('Training Model')
    train_model()
    
    return render_template('adds.html', totalreg=totalreg(), datetoday2=datetoday2)

#### This function will delete a user
@app.route('/delete_user', methods=['GET', 'POST'])
def delete_user():
    if request.method == 'POST':
        # Access the form data using the correct key
        userid_to_delete = request.form['deleteuserid']
        
        # Remove user folder
        user_folders = os.listdir('static/faces')
        for folder in user_folders:
            if folder.endswith(f'_{userid_to_delete}'):
                user_folder_path = os.path.join('static/faces', folder)
                if os.path.exists(user_folder_path):
                    for file in os.listdir(user_folder_path):
                        os.remove(os.path.join(user_folder_path, file))
                    os.rmdir(user_folder_path)
                    print(f"Deleted folder: {user_folder_path}")
        
        # Remove user from total_users.csv
        df = pd.read_csv(total_users_file)
        df = df[df['Roll'] != int(userid_to_delete)]
        df.to_csv(total_users_file, index=False)
        print(f"Removed user with ID {userid_to_delete} from total_users.csv")
        
        # Retrain the model
        print("Retraining model...")
        train_model()
        
        return redirect(url_for('adds'))
    
    return render_template("adds.html", totalreg=totalreg(), datetoday2=datetoday2)

@app.route("/results", methods=['GET', 'POST'])
def results():
    return render_template("results.html", datetoday2=datetoday2, totalreg=totalreg())

@app.route("/login", methods=['GET', 'POST'])
def login():
    return render_template("login.html", totalreg=totalreg())

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)