import os
import glob
import shutil
import xml.etree.ElementTree as ET
import random

def validate_xml(xml_path):
    """
    Validate XML file for proper bounding box information.
    
    Args:
        xml_path (str): Path to the XML file
        
    Returns:
        bool: True if XML is valid, False otherwise
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Check if there are any object tags
        objects = root.findall('.//object')
        if not objects:
            return False
        
        # Check if each object has valid bounding box
        for obj in objects:
            bndbox = obj.find('bndbox')
            if bndbox is None:
                return False
            
            # Check if all required coordinates are present and valid
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

def get_min_folder_size(directory):
    """
    Get the minimum number of valid image-XML pairs across all crocodile folders.
    
    Args:
        directory (str): Path to the training dataset directory
        
    Returns:
        int: Minimum number of valid pairs
    """
    print("\n[STEP 1] Analyzing folders to find minimum number of valid pairs...")
    min_size = float('inf')
    folder_counts = {}
    
    for root, dirs, files in os.walk(directory):
        if "removed" in root:
            continue
            
        jpg_files = set(f for f in files if f.endswith('.jpg'))
        xml_files = set(f.replace('.jpg', '.xml') for f in jpg_files)
        
        # Count only valid pairs
        valid_pairs = sum(1 for xml in xml_files if xml in files and validate_xml(os.path.join(root, xml)))
        
        if valid_pairs > 0:
            min_size = min(min_size, valid_pairs)
            folder_name = os.path.basename(root)
            folder_counts[folder_name] = valid_pairs
    
    # Print folder statistics
    print(f"\nFound {len(folder_counts)} folders with valid image-XML pairs:")
    for folder, count in sorted(folder_counts.items()):
        print(f"  - {folder}: {count} valid pairs")
    
    return min_size if min_size != float('inf') else 0

def clean_and_balance_dataset(directory):
    """
    Clean the dataset by moving invalid files to a removed directory and balance the number of images.
    Recursively checks all subfolders within the given directory.
    
    Args:
        directory (str): Path to the root training dataset directory
    """
    print("\n[STEP 2] Setting up removed directory structure...")
    
    # Use the existing removed directory
    removed_dir = os.path.join(os.path.dirname(os.path.dirname(directory)), "removed", "Training")
    os.makedirs(removed_dir, exist_ok=True)
    print(f"Using existing removed directory: {removed_dir}")
    
    # First, get the minimum folder size
    min_size = get_min_folder_size(directory)
    print(f"\n[STEP 3] Minimum number of valid pairs across all folders: {min_size}")
    
    total_moved_jpg = 0
    total_moved_xml = 0
    
    print("\n[STEP 4] Processing each folder to clean and balance the dataset...")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        # Skip the removed directory
        if "removed" in root:
            continue
            
        # Get the CrocID from the current directory path
        croc_id = os.path.basename(root)
        print(f"\n  Processing folder: {croc_id}")
        
        # Get all jpg and xml files in current directory
        jpg_files = set(glob.glob(os.path.join(root, "*.jpg")))
        xml_files = set(glob.glob(os.path.join(root, "*.xml")))
        
        print(f"    Found {len(jpg_files)} JPG files and {len(xml_files)} XML files")
        
        # Convert to sets of filenames without extensions
        jpg_bases = {os.path.splitext(os.path.basename(f))[0] for f in jpg_files}
        xml_bases = {os.path.splitext(os.path.basename(f))[0] for f in xml_files}
        
        # Create corresponding removed directory
        removed_croc_dir = os.path.join(removed_dir, croc_id)
        os.makedirs(removed_croc_dir, exist_ok=True)
        
        # Find orphaned JPG files (no corresponding XML)
        orphaned_jpg = jpg_bases - xml_bases
        
        # Move orphaned JPG files
        if orphaned_jpg:
            print(f"    Found {len(orphaned_jpg)} orphaned JPG files (no corresponding XML)")
            for base in orphaned_jpg:
                src_path = os.path.join(root, f"{base}.jpg")
                dst_path = os.path.join(removed_croc_dir, f"{base}.jpg")
                try:
                    shutil.move(src_path, dst_path)
                    print(f"      Moved orphaned image: {base}.jpg")
                    total_moved_jpg += 1
                except Exception as e:
                    print(f"      Error moving {base}.jpg: {str(e)}")
        
        # Get valid image-XML pairs
        valid_pairs = []
        invalid_pairs = []
        
        for base in xml_bases:
            xml_path = os.path.join(root, f"{base}.xml")
            jpg_path = os.path.join(root, f"{base}.jpg")
            
            # Check if both files exist and XML is valid
            if os.path.exists(jpg_path) and validate_xml(xml_path):
                valid_pairs.append((base, xml_path, jpg_path))
            else:
                invalid_pairs.append((base, xml_path, jpg_path))
        
        # Move invalid pairs
        if invalid_pairs:
            print(f"    Found {len(invalid_pairs)} invalid pairs (missing or invalid XML)")
            for base, xml_path, jpg_path in invalid_pairs:
                try:
                    if os.path.exists(xml_path):
                        xml_dst = os.path.join(removed_croc_dir, f"{base}.xml")
                        shutil.move(xml_path, xml_dst)
                        print(f"      Moved invalid XML: {base}.xml")
                        total_moved_xml += 1
                    
                    if os.path.exists(jpg_path):
                        jpg_dst = os.path.join(removed_croc_dir, f"{base}.jpg")
                        shutil.move(jpg_path, jpg_dst)
                        print(f"      Moved corresponding image: {base}.jpg")
                        total_moved_jpg += 1
                except Exception as e:
                    print(f"      Error moving files for {base}: {str(e)}")
        
        print(f"    Found {len(valid_pairs)} valid image-XML pairs")
        
        # If we have more valid pairs than the minimum size, randomly select pairs to move
        if len(valid_pairs) > min_size:
            excess_count = len(valid_pairs) - min_size
            print(f"    Need to move {excess_count} excess pairs to balance the dataset")
            pairs_to_move = random.sample(valid_pairs, excess_count)
            for base, xml_path, jpg_path in pairs_to_move:
                try:
                    # Move XML file
                    xml_dst = os.path.join(removed_croc_dir, f"{base}.xml")
                    shutil.move(xml_path, xml_dst)
                    print(f"      Moved excess XML: {base}.xml")
                    total_moved_xml += 1
                    
                    # Move corresponding JPG
                    jpg_dst = os.path.join(removed_croc_dir, f"{base}.jpg")
                    shutil.move(jpg_path, jpg_dst)
                    print(f"      Moved excess image: {base}.jpg")
                    total_moved_jpg += 1
                except Exception as e:
                    print(f"      Error moving excess files for {base}: {str(e)}")
        
        print(f"    Folder {croc_id} now has {min_size} valid pairs")
    
    # Print summary
    print("\n[STEP 5] Cleaning and balancing complete!")
    total_moved = total_moved_jpg + total_moved_xml
    if total_moved == 0:
        print("\nNo files needed to be moved. Dataset is clean and balanced!")
    else:
        print(f"\nMoved {total_moved} files to removed directory:")
        print(f"- {total_moved_jpg} JPG files")
        print(f"- {total_moved_xml} XML files")
        print(f"\nFiles have been moved to: {removed_dir}")
        print(f"Each folder now contains {min_size} valid image-XML pairs")

if __name__ == "__main__":
    # Hard-coded path to the Training directory
    training_dir = "dataset/Training"
    
    # Check if directory exists
    if not os.path.isdir(training_dir):
        print(f"Error: Directory '{training_dir}' does not exist!")
    else:
        # Ask for confirmation before starting
        print("\nThis will:")
        print("1. Check all subfolders for invalid files")
        print("2. Balance the number of images across all crocodile folders")
        print("3. Move excess files to the 'removed' directory")
        response = input("\nDo you want to proceed? (yes/no): ").lower()
        
        if response == 'yes':
            # Clean and balance the dataset
            clean_and_balance_dataset(training_dir)
        else:
            print("Operation cancelled.") 