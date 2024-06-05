import cv2
from algorithm import ObjectRemovalDetector

def main():
    object_detector = ObjectRemovalDetector()
    rtspUrl = "rtsp://service:service@172.196.129.121:554/1/h264minor"
    cap = cv2.VideoCapture(rtspUrl)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        exit()

    object_detector.set_matching_threshold(12)  # Adjust this threshold based on your needs

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if object_detector.initial_frame is None:
            object_detector.set_initial_frame(frame)
        else:
            object_removed = object_detector.detect_object_removal(frame)
            if object_removed:
                print("Object removal detected!")

        object_detector.draw_rectangles(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
