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
        
        # object_detector.draw_rectangles(frame)
        # cv2.imshow("First", frame)
        # cv2.waitKey(0)

        # return

   
        
        # Set initial frame and fill the object images
        if object_detector.initial_frame is None:
            object_detector.set_initial_frame(frame)

        # Display all the images in objectImages
        # for idx, object in enumerate(object_detector.objectImages):
        #     cv2.imshow(f"Display{idx}", object)

        # cv2.waitKey(0)
        
        # Detect object removal
        object_detector.detect_object_removal(frame)
            # Potential object removal detected
            # print("Object removal detected!")
        
        # Display the frame
        object_detector.draw_rectangles(frame)
        cv2.imshow('Frame', frame)
    
        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
