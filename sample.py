import cv2

# Path to the pre-trained models
face_cascade_path = "/Users/_shahid_abdul/Downloads/Programs/Incomplete Projects/1-Day_7(march)/model/haarcascade_frontalface_default.xml"
age_net_path = "/Users/_shahid_abdul/Downloads/Programs/Incomplete Projects/1-Day_7(march)/model/age_net.caffemodel"
age_proto_path = "/Users/_shahid_abdul/Downloads/Programs/Incomplete Projects/1-Day_7(march)/model/deploy_age2.prototxt"
gender_net_path = "/Users/_shahid_abdul/Downloads/Programs/Incomplete Projects/1-Day_7(march)/model/gender_net.caffemodel"
gender_proto_path = "/Users/_shahid_abdul/Downloads/Programs/Incomplete Projects/1-Day_7(march)/model/deploy_gender2.prototxt"

# Load the pre-trained models
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
age_net = cv2.dnn.readNet(age_net_path, age_proto_path)
gender_net = cv2.dnn.readNet(gender_net_path, gender_proto_path)

# Function to detect face, estimate age, and classify gender
def detect_face_gender_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()

        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_preds[0] * 100

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display age and gender
        cv2.putText(frame, "Age: " + str(int(age)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Gender: " + gender, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# Main function to capture video from webcam
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_face_gender_age(frame)

        cv2.imshow('Face Gender Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
