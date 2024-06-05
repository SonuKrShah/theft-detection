import cv2
import numpy as np
import matplotlib.pyplot as plt

class ObjectRemovalDetector:
    def __init__(self):
        # Initialize member variables
        self.initial_frame = None
        self.initial_histogram = []
        self.histogram_threshold = 0.60
        self.objectImages = []
        self.rectangles = [[[648, 420], [728, 413], [773, 673], [705, 687]]]

    def set_initial_frame(self, frame):
        """
        Set the initial frame.
        
        Args:
            frame: Initial frame.
        """
        if self.initial_frame is None:
            self.initial_frame = frame.copy()
            self.objectImages = self.crop_object_images(self.initial_frame, self.rectangles)

    # Function to draw rectangles on the frame based on given points
    def draw_rectangles(self, frame):
        for points in self.rectangles:
            # Convert points to numpy array
            points = np.array(points)
            
            # Draw the rectangle on the frame
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Function to construct rectangles from 4 points and crop object images
    def crop_object_images(self, frame, rectangles):
        object_images = []
        for points in rectangles:
            # Convert the points to a numpy array
            points = np.array(points)
            
            # Find the minimum bounding rectangle (MBR)
            rect = cv2.boundingRect(points)
            
            # Extract rectangle coordinates
            x, y, width, height = rect
            
            # Crop the object image from the frame
            object_image = frame[y:y+height, x:x+width]
            
            # Append the cropped object image to the list
            object_images.append(object_image)
        
        # After the object images have been cropped, compute the histograms for each of the object images and store them in initial_histogram
        for idx, object in enumerate(object_images):
            self.initial_histogram.append(self.compute_histogram(object))
            # self.plotHistogram(self.initial_histogram[idx])
        return object_images

    def plotHistogram(self, hist):
        # Plotting the histogram
        plt.bar(range(len(hist)), hist.flatten())

        # Customizing the plot
        plt.title('Histogram')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')

        # Show the plot
        plt.show()
    
    def compute_histogram(self, image):
        """
        Compute histogram of an input image.
        
        Args:
            image: Input image.
            
        Returns:
            Histogram of the input image.
        """
        # Convert image to YUV color space
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Extract chroma channels (U and V)
        chroma_channels = yuv_image[:,:,1:3]
        
        # Flatten and concatenate the chroma channels
        # chroma_data = chroma_channels.reshape(-1, 2)

        # Convert chroma_data to float32
        # chroma_data = np.float32(chroma_data)
        
        # Compute histogram
        hist = cv2.calcHist([chroma_channels], [0, 1], None, [256, 256], [0, 256, 0, 256])
        
        # Normalize histogram
        hist = cv2.normalize(hist, hist)
        
        return hist

    def compare_histograms(self, hist1, hist2):
        """
        Compare histograms using histogram intersection.
        
        Args:
            hist1: First histogram.
            hist2: Second histogram.
            
        Returns:
            Histogram intersection value.
        """
        # Compute histogram intersection
        intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        return intersection
    
    def detect_object_removal(self, frame):
        """
        Detect object removal by comparing the current frame with the initial frame.
        
        Args:
            frame: Current frame.
            
        Returns:
            Boolean indicating whether object removal is detected.
        """
        # Generate the rectangles from the current frame.
        currentRectangles = self.crop_object_images(frame, self.rectangles)

        # For each of the rectangles, in the current frame, compute the histogram and compare with the initial frame
        for idx, object in enumerate(currentRectangles):

            objHistogram = self.compute_histogram(object)
            histogram_difference = self.compare_histograms(self.initial_histogram[idx], objHistogram)
            # Check if histogram difference exceeds threshold
            print(f"Histogram difference: {idx}: {histogram_difference}")

            if histogram_difference > self.histogram_threshold:
                print(f"Object {idx+1} removed")
                # return True

        return False
