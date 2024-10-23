import cv2
import dlib

# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    num_faces = len(faces)

    # Logic to determine if the scenario might be cheating
    if num_faces == 0:
        cheat_status = "No face detected - possible cheating"
    elif num_faces > 1:
        cheat_status = "Multiple faces detected - possible cheating"
    else:
        # Handle the case where exactly one face is detected
        cheat_status = "One face detected - analyzing direction"
        face = faces[0]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Facial landmarks
        landmarks = predictor(gray, face)
        nose_tip = landmarks.part(30).x
        left_eye = sum([landmarks.part(n).x for n in range(36, 42)]) // 6
        right_eye = sum([landmarks.part(n).x for n in range(42, 48)]) // 6

        # Determine the direction the face is looking
        if nose_tip < left_eye:
            direction = "Looking Left"
        elif nose_tip > right_eye:
            direction = "Looking Right"
        else:
            direction = "Looking Forward"
        
        cheat_status += f" - {direction}"

    # Draw this status on the frame
    cv2.putText(frame, cheat_status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
