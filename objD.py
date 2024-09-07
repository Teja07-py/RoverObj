import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n-cls.pt')

cap = cv2.VideoCapture(0)  # '0' is usually the default webcam

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
out = cv2.VideoWriter('output79.avi', fourcc, 20.0, (640, 480))  
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model(frame)
    
    for result in results:
        annotated_frame = result.plot()  # Get the annotated frame

    # Display the resulting frame
    cv2.imshow('Frame', annotated_frame)
    
    out.write(annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()