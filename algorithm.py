import cv2
import numpy as np
import matplotlib.pyplot as plt

class ObjectRemovalDetector:
    def __init__(self):
        # Initialize member variables
        self.initial_frame = None
        self.initial_histogram = None
        self.histogram_threshold = 0.2

    def set_initial_frame(self, frame):
        """
        Set the initial frame.
        
        Args:
            frame: Initial frame.
        """
        if self.initial_frame is None:
            self.initial_frame = frame.copy()

        # Compute histogram of initial frame
        self.initial_histogram = self.compute_histogram(frame)

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
        # Compute histogram of current frame
        current_histogram = self.compute_histogram(frame)
        
        
        # Compare histograms using histogram intersection
        histogram_difference = self.compare_histograms(current_histogram, self.initial_histogram)

        print("Histogram distance: ", histogram_difference)
        
        # Check if histogram difference exceeds threshold
        if histogram_difference > self.histogram_threshold:
            return True
        else:
            return False

