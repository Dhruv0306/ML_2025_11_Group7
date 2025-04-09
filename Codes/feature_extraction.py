import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog

class FeatureExtractor:
    def __init__(self):
        """
        Initialize feature extractors
        """
        # Initialize SIFT
        self.sift = cv2.SIFT_create()
        
        # Initialize ORB
        self.orb = cv2.ORB_create()
        
        # LBP parameters
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
        
        # HOG parameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
    
    def extract_sift(self, image):
        """
        Extract SIFT features from image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: SIFT features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # If no features found, return zero vector
        if descriptors is None:
            return np.zeros(128)
        
        # Return mean of descriptors
        return np.mean(descriptors, axis=0)
    
    def extract_hog(self, image):
        """
        Extract HOG features from image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: HOG features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract HOG features
        features = hog(gray, 
                      orientations=self.hog_orientations,
                      pixels_per_cell=self.hog_pixels_per_cell,
                      cells_per_block=self.hog_cells_per_block,
                      block_norm='L2-Hys')
        
        return features
    
    def extract_lbp(self, image):
        """
        Extract Local Binary Pattern features from image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: LBP features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract LBP features
        lbp = local_binary_pattern(gray, 
                                 self.lbp_n_points,
                                 self.lbp_radius,
                                 method='uniform')
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), 
                             bins=np.arange(0, self.lbp_n_points + 3),
                             density=True)
        
        return hist
    
    def extract_orb(self, image):
        """
        Extract ORB features from image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: ORB features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # If no features found, return zero vector
        if descriptors is None:
            return np.zeros(32)
        
        # Return mean of descriptors
        return np.mean(descriptors, axis=0)
    
    def extract_all_features(self, image):
        """
        Extract all features from image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Concatenated features
        """
        # Extract individual features
        sift_features = self.extract_sift(image)
        hog_features = self.extract_hog(image)
        lbp_features = self.extract_lbp(image)
        orb_features = self.extract_orb(image)
        
        # Concatenate all features
        all_features = np.concatenate([
            sift_features,
            hog_features,
            lbp_features,
            orb_features
        ])
        
        return all_features 