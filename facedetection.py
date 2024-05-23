import cv2

def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imwrite(output_path, img)
    print(f"Processed image saved as {output_path}")

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #'XVID' for .avi videos and 'mp4v' for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved as {output_path}")
    
def process_webcam(input_path):
    cap = cv2.VideoCapture(input_path)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(input_path, output_path):
    if isinstance(input_path, int):
        process_webcam(input_path)
    elif input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        process_image(input_path, output_path)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(input_path, output_path)
    else:
        print("Unsupported file format")

if __name__ == "__main__":
    input_path = 'input_image3.webp'  # Replace with the path to your input file or 0 for local webcam
    output_path = 'output_image2.jpg'  # Replace with the path to save the output file
    main(input_path, output_path)
