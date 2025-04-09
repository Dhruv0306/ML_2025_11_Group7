import os

def count_files_in_subdirectories(root_dir):
    """
    Counts the number of files in each subdirectory of a given directory.

    Args:
        root_dir (str): The path to the root directory.

    Returns:
        dict: A dictionary where keys are subdirectory names and values are the 
              corresponding file counts.
    """
    subdir_counts = {}
    for subdir, _, files in os.walk(root_dir):
        if subdir != root_dir:  # Avoid counting files in the root directory itself
            subdir_name = os.path.basename(subdir)
            subdir_counts[subdir_name] = len(files)
    return subdir_counts

if __name__ == "__main__":
    directory_path = input("Enter the path to the directory: ")
    if os.path.isdir(directory_path):
        file_counts = count_files_in_subdirectories(directory_path)
        if file_counts:
            print("File counts per subdirectory:")
            for subdir, count in file_counts.items():
                print(f"- {subdir}: {count} files")
            
            # Find the folder with minimum number of files
            min_subdir = min(file_counts, key=file_counts.get)
            min_count = file_counts[min_subdir]
            print(f"\nFolder with minimum number of files: {min_subdir} with {min_count} files")
        else:
             print("No subdirectories found in the given directory.")
    else:
        print("Invalid directory path.")