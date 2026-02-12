import os

def delete_empty_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            try:
                # Check if folder is empty
                if not os.listdir(full_path):
                    os.rmdir(full_path)
                    print(f"Deleted empty folder: {full_path}")
            except Exception as e:
                print(f"Could not delete {full_path}: {e}")

# Example usage
ROOT_FOLDER = "t1_t2_patches"
delete_empty_folders(ROOT_FOLDER)
