import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

def create_directory(path):
    """
    Create a directory if it doesn't exist
    
    Args:
        path (str): Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_voc_xml(xml_path):
    """
    Parse Pascal VOC XML file to get bounding box coordinates
    
    Args:
        xml_path (str): Path to XML file
        
    Returns:
        tuple: (xmin, ymin, xmax, ymax) coordinates
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get bounding box coordinates
    obj = root.find('object')
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    return (xmin, ymin, xmax, ymax)

def crop_image(image_path, bbox=None, output_size=(224, 224)):
    """
    Crop image using bounding box or center crop
    
    Args:
        image_path (str): Path to input image
        bbox (tuple, optional): (xmin, ymin, xmax, ymax) coordinates
        output_size (tuple): Desired output size (width, height)
        
    Returns:
        numpy.ndarray: Cropped image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        cropped = img[ymin:ymax, xmin:xmax]
    else:
        # Center crop
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        xmin = max(0, center_x - output_size[0] // 2)
        ymin = max(0, center_y - output_size[1] // 2)
        xmax = min(w, xmin + output_size[0])
        ymax = min(h, ymin + output_size[1])
        cropped = img[ymin:ymax, xmin:xmax]
    
    # Resize to standard size
    cropped = cv2.resize(cropped, output_size)
    return cropped

def extract_croc_id_from_filename(filename):
    """
    Extract crocodile ID from filename
    
    Args:
        filename (str): Input filename (e.g., 'Croc1_1.jpg')
        
    Returns:
        str: Crocodile ID (e.g., 'Croc1')
    """
    return filename.split('_')[0]

def validate_xml(xml_path):
    """Validate XML file for proper bounding box information"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = root.findall('.//object')
        if not objects:
            return False
        
        for obj in objects:
            bndbox = obj.find('bndbox')
            if bndbox is None:
                return False
            
            for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
                if bndbox.find(coord) is None:
                    return False
                try:
                    float(bndbox.find(coord).text)
                except (ValueError, TypeError):
                    return False
        
        return True
    except Exception as e:
        print(f"Error validating XML {xml_path}: {str(e)}")
        return False 