import cv2
import face_recognition

# Load the reference image
reference_image = face_recognition.load_image_file("reference_image.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for face_encoding in face_encodings:
        # Compare the face encoding with the reference face encoding
        match = face_recognition.compare_faces([reference_encoding], face_encoding)
        
        # If a match is found, print "Match Found" and break the loop
        if match[0]:
            print("Match Found")
            break

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
