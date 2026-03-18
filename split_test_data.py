import os
import shutil
import random

COLLECTED_DIR = "collected_faces"
TEST_DIR = "test_faces"

def split_data():
    if not os.path.exists(COLLECTED_DIR):
        print(f"Error: {COLLECTED_DIR} not found.")
        return

    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        print(f"Created {TEST_DIR} directory.")

    # Get all subdirectories (MSSV folders)
    subdirs = [d for d in os.listdir(COLLECTED_DIR) if os.path.isdir(os.path.join(COLLECTED_DIR, d))]
    
    # Filter out 'processed' if it exists
    subdirs = [d for d in subdirs if d != 'processed']

    print(f"Scanning {len(subdirs)} folders...")

    test_count = 0
    for mssv in subdirs:
        mssv_path = os.path.join(COLLECTED_DIR, mssv)
        # Get all images in this folder
        images = [f for f in os.listdir(mssv_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if len(images) > 1:
            # Pick one random image for testing
            test_img = random.choice(images)
            
            # Create subfolder in test_faces
            test_mssv_path = os.path.join(TEST_DIR, mssv)
            if not os.path.exists(test_mssv_path):
                os.makedirs(test_mssv_path)
            
            src_path = os.path.join(mssv_path, test_img)
            dst_path = os.path.join(test_mssv_path, test_img)
            
            # Move to test folder
            shutil.move(src_path, dst_path)
            test_count += 1
            # print(f"Moved {test_img} from {mssv} to test set.")

    print(f"Done! Extracted {test_count} images for testing.")
    print(f"Remaining images in {COLLECTED_DIR} will be used for indexing.")

if __name__ == "__main__":
    split_data()
