import cv2
import numpy as np

class ObjectRemovalDetector:
    def __init__(self):
        self.initial_frame = None
        self.initial_keypoints = []
        self.initial_descriptors = []
        self.detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.rectangles = [[[648, 420], [728, 413], [773, 673], [705, 687]]]

    def set_initial_frame(self, frame):
        if self.initial_frame is None:
            self.initial_frame = frame.copy()
            self.initial_keypoints, self.initial_descriptors = self.compute_keypoints_descriptors(frame, self.rectangles)

    def compute_keypoints_descriptors(self, frame, rectangles):
        keypoints_list = []
        descriptors_list = []
        for points in rectangles:
            points = np.array(points)
            rect = cv2.boundingRect(points)
            x, y, width, height = rect
            object_image = frame[y:y+height, x:x+width]
            keypoints, descriptors = self.detector.detectAndCompute(object_image, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
        return keypoints_list, descriptors_list

    def detect_object_removal(self, frame):
        current_keypoints, current_descriptors = self.compute_keypoints_descriptors(frame, self.rectangles)
        for idx, (keypoints, descriptors) in enumerate(zip(current_keypoints, current_descriptors)):
            matches = self.matcher.match(self.initial_descriptors[idx], descriptors)
            print(f"Object {idx+1} matches: {len(matches)}")
            if len(matches) < self.matching_threshold:
                print(f"Object {idx+1} removed")
                return True
        return False

    def draw_rectangles(self, frame):
        for points in self.rectangles:
            points = np.array(points)
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    def set_matching_threshold(self, threshold):
        self.matching_threshold = threshold
