import cv2
from algorithm import ObjectRemovalDetector

def main():
    # Create ObjectRemovalDetector object
    object_detector = ObjectRemovalDetector()
    
    # Open video capture
    cap = cv2.VideoCapture('bottle-detection.mp4')
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Set initial frame
        if object_detector.initial_frame is None:
            object_detector.set_initial_frame(frame)
        
        # Detect object removal
        if object_detector.detect_object_removal(frame):
            # Potential object removal detected
            print("Object removal detected!")
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
