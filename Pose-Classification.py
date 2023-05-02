import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle

def export_to_cvs():
    directory = os.getcwd()
    path = directory + '/data'
    dir_list = os.listdir(path)
    print(dir_list)

    num_coords = 33
    landmarks = ['class']
    for val in range(1, num_coords + 1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    with open('coords.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

    for i in range(len(dir_list)):
        print(i)
    # for yoga_pose in ['Adho Mukha Svanasana', 'Navasana']:
        yoga_pose = dir_list[i]
        if (yoga_pose != "Poses.json" and yoga_pose != ".DS_Store") :
            class_name = yoga_pose
            cur_path = path + "/" + yoga_pose
            img_list = os.listdir(cur_path)

            for img in img_list:
                if img != ".DS_Store":
                    img_path = cur_path + "/" + img
                    # initialize Pose estimator
                    mp_drawing = mp.solutions.drawing_utils
                    mp_pose = mp.solutions.pose

                    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                    cap = cv2.imread(img_path)
                    frame = cap
                    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(RGB)

                    try:
                        # print(results.pose_landmarks)
                        landmarks = results.pose_landmarks.landmark
                        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())
                        # Append class name
                        pose_row.insert(0, class_name)

                        with open('coords.csv', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(pose_row)
                    except:
                        pass

def train_model():
    df = pd.read_csv('coords.csv')
    df.head()
    df.tail()
    X = df.drop('class', axis=1)  # features
    y = df['class']  # target value
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    pipelines = {
        'lr': make_pipeline(StandardScaler(), LogisticRegression()),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    # print(fit_models['rc'].predict(X_test))
    # fit_models['rc'].predict(X_test)

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))

    with open('body_language.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)

def pose_classification(video):
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video)
    while (cv2.waitKey(1) != ord('q')):
        _, frame = cap.read()
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(RGB)

        try:
            # print(results.pose_landmarks)
            landmarks = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

            # Make Detections
            X = pd.DataFrame([pose_row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, max(body_language_prob))

        except:
            pass

        if (max(body_language_prob) > 0.9):
            cv2.putText(frame, "Pose Name: " + body_language_class
                        , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if (body_language_class == "Adho Mukha Svanasana"):
                grade = str(grade_downdog(frame, mp_pose, landmarks))
                cv2.putText(frame, "Grade: " + grade
                            , (90, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif (body_language_class == "Pincha Mayurasana"):
                grade = str(grade_feathered(frame, mp_pose, landmarks))
                cv2.putText(frame, "Grade: " + grade
                            , (90, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Video', frame)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

def grade_downdog(frame, mp_pose, landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


    angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle2 = calculate_angle(left_hip, left_shoulder, left_elbow)
    angle3 = calculate_angle(left_hip, left_knee, left_ankle)
    angle4 = calculate_angle(right_shoulder, right_elbow, right_wrist)
    angle5 = calculate_angle(right_hip, right_shoulder, right_elbow)
    angle6 = calculate_angle(right_hip, right_knee, right_ankle)

    standard = [180, 180, 180, 180, 180, 180]  # angle1, angle2, angle3
    comparison = [angle1, angle2, angle3, angle4, angle5, angle6]

    cosine = np.dot(standard, comparison) / (norm(standard) * norm(comparison))
    grade = round(cosine * 100, 2)
    height, width = frame.shape[:2]

    cv2.putText(frame, str(round(angle1, 2)),
                tuple(np.multiply(left_elbow, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle2, 2)),
                tuple(np.multiply(left_shoulder, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle3, 2)),
                tuple(np.multiply(left_knee, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle4, 2)),
                tuple(np.multiply(right_elbow, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle5, 2)),
                tuple(np.multiply(right_shoulder, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle6, 2)),
                tuple(np.multiply(right_knee, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

    return grade

def grade_feathered(frame, mp_pose, landmarks):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    angle3 = calculate_angle(left_hip, left_knee, left_ankle)
    angle4 = calculate_angle(left_wrist, left_elbow, left_shoulder)
    angle5 = calculate_angle(left_shoulder, left_hip, left_knee)
    angle6 = calculate_angle(left_elbow, left_shoulder, left_hip)

    angle33 = calculate_angle(right_hip, right_knee, right_ankle)
    angle44 = calculate_angle(right_wrist, right_elbow, right_shoulder)
    angle55 = calculate_angle(right_shoulder, right_hip, right_knee)
    angle66 = calculate_angle(right_elbow, right_shoulder, right_hip)

    standard = [90, 180, 180, 180, 90, 180, 180, 180]  # angle4, angle6, angle5, angle3
    comparison = [angle4, angle6, angle5, angle3, angle44, angle66, angle55, angle33]
    cosine = np.dot(standard, comparison) / (norm(standard) * norm(comparison))
    grade = round(cosine * 100, 2)

    height, width = frame.shape[:2]

    cv2.putText(frame, str(round(angle3, 2)),
                tuple(np.multiply(left_elbow, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle4, 2)),
                tuple(np.multiply(left_shoulder, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle5, 2)),
                tuple(np.multiply(left_hip, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle6, 2)),
                tuple(np.multiply(left_knee, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

    cv2.putText(frame, str(round(angle33, 2)),
                tuple(np.multiply(right_elbow, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle44, 2)),
                tuple(np.multiply(right_shoulder, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle55, 2)),
                tuple(np.multiply(right_hip, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    cv2.putText(frame, str(round(angle66, 2)),
                tuple(np.multiply(right_knee, [width, height]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

    return grade

if __name__ == '__main__':
    # export_to_cvs()
    # train_model()
    video = 'downdog.mov'
    # video = 'video.mp4'
    # video = "feathered peacock.mov"
    pose_classification(video)

