import cv2
import numpy as np

def fullbody_detection_dnn(video_path, prototxt_path, model_path, confidence_threshold=0.6):
    # Load the Caffe model (MobileNet SSD)
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Define the class labels for the COCO dataset (MobileNet SSD is often trained on COCO)
    # The 'person' class is usually at index 15
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    print(f"Detecting people in video: {video_path} using MobileNet SSD.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            break

        (h, w) = frame.shape[:2]

        # MobileNet SSD requires a specific input size (e.g., 300x300)
        # and normalization.
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # Extract the confidence (probability) of the detection
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater than the threshold
            if confidence > confidence_threshold:
                # Extract the class label and ensure it's "person"
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    # Compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw the prediction on the frame
                    label = f"{CLASSES[idx]}: {confidence:.2f}"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Marathon People Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"{video_file} successfully terminated!")
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Define the paths to your video and the MobileNet SSD model files
    video_file = 'bodies.mp4'  # Your marathon footage video
    prototxt_file = 'MobileNetSSD_deploy.prototxt'
    model_file = 'MobileNetSSD_deploy.caffemodel'

    # Download these two files and place them in the same directory as your script
    # Or provide the full paths to them.

    # Example download links (you might need to search for updated/reliable sources):
    # prototxt: https://github.com/chuanqi305/MobileNet-SSD/blob/master/prototxt/MobileNetSSD_deploy.prototxt
    # caffemodel: http://download.deeplabv3.ai/models/MobileNetSSD_deploy.caffemodel

    print("Please ensure 'MobileNetSSD_deploy.prototxt' and 'MobileNetSSD_deploy.caffemodel'")
    print("are in the same directory as this script, or provide their full paths.")
    print("Adjust the 'confidence_threshold' if you see too many false positives or miss detections.")

    fullbody_detection_dnn(video_file, prototxt_file, model_file)