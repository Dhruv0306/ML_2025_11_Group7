import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
import os
from concurrent.futures import ThreadPoolExecutor
from joblib import Memory
from tqdm import tqdm
import time

class FeatureExtractor:
    def __init__(self):
        """
        Initialize feature extractors with caching
        """
        print("[DEBUG] Initializing FeatureExtractor...")
        # Initialize feature extractors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        
        # Parameters
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
        # Setup caching in current directory
        cache_dir = 'feature_cache'
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[DEBUG] Cache directory: {cache_dir}")
        self.memory = Memory(cache_dir, verbose=0)
        
        # Cache the feature extraction methods
        print("[DEBUG] Setting up feature caching...")
        self.cached_sift = self.memory.cache(self._extract_sift)
        self.cached_hog = self.memory.cache(self._extract_hog)
        self.cached_lbp = self.memory.cache(self._extract_lbp)
        self.cached_orb = self.memory.cache(self._extract_orb)
        print("[DEBUG] FeatureExtractor initialization complete")
    
    def _extract_sift(self, gray):
        """Internal SIFT feature extraction"""
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            print("[DEBUG] No SIFT features found, returning zero vector")
            return np.zeros(128)
        return np.mean(descriptors, axis=0)
    
    def _extract_hog(self, gray):
        """Internal HOG feature extraction"""
        return hog(gray, 
                  orientations=self.hog_orientations,
                  pixels_per_cell=self.hog_pixels_per_cell,
                  cells_per_block=self.hog_cells_per_block,
                  block_norm='L2-Hys')
    
    def _extract_lbp(self, gray):
        """Internal LBP feature extraction"""
        lbp = local_binary_pattern(gray, 
                                 self.lbp_n_points,
                                 self.lbp_radius,
                                 method='uniform')
        hist, _ = np.histogram(lbp.ravel(), 
                             bins=np.arange(0, self.lbp_n_points + 3),
                             density=True)
        return hist
    
    def _extract_orb(self, gray):
        """Internal ORB feature extraction"""
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None:
            print("[DEBUG] No ORB features found, returning zero vector")
            return np.zeros(32)
        return np.mean(descriptors, axis=0)
    
    def extract_all_features(self, image):
        """
        Extract all features from image using parallel processing
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Concatenated features
        """
        start_time = time.time()
        
        # Convert to grayscale once
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract features in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                'sift': executor.submit(self.cached_sift, gray),
                'hog': executor.submit(self.cached_hog, gray),
                'lbp': executor.submit(self.cached_lbp, gray),
                'orb': executor.submit(self.cached_orb, gray)
            }
            
            # Get results
            features = {name: future.result() for name, future in futures.items()}
        
        # Concatenate all features
        all_features = np.concatenate([
            features['sift'],
            features['hog'],
            features['lbp'],
            features['orb']
        ])
        
        return all_features
    
    def extract_features_batch(self, images, desc="Extracting features"):
        """
        Extract features from a batch of images with progress bar
        
        Args:
            images (list): List of images
            desc (str): Description for progress bar
            
        Returns:
            numpy.ndarray: Array of features
        """
        print(f"\n[DEBUG] Starting batch processing of {len(images)} images")
        start_time = time.time()
        
        features = []
        for img in tqdm(images, desc=desc, unit="img"):
            features.append(self.extract_all_features(img))
        
        total_time = time.time() - start_time
        print(f"\n[DEBUG] Batch processing complete:")
        print(f"[DEBUG] Total time: {total_time:.2f}s")
        print(f"[DEBUG] Average time per image: {total_time/len(images):.2f}s")
        print(f"[DEBUG] Final feature array shape: {np.array(features).shape}")
        
        return np.array(features) 