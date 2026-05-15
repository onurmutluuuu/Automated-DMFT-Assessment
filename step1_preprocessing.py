"""
Step 1: Data Preprocessing
This module is responsible for standardizing the dimensions and color channels 
of the input panoramic dental radiographs. It resizes the images to the optimal 
1280x1280 pixel resolution for the YOLOv11 architecture and ensures the RGB 
color space is maintained.

Note on Normalization: YOLOv11 automatically scales pixel intensities from 
[0, 255] to [0.0, 1.0] internally via the PyTorch DataLoader (img.float() / 255.0).
Therefore, explicit manual normalization is intentionally omitted here to prevent 
redundancy and data corruption during the training pipeline.
"""

import cv2
import os
from glob import glob

def run_preprocessing(subset="train", base_dir="data", target_size=(1280, 1280)):
    input_dir = os.path.join(base_dir, subset, "images")
    output_dir = os.path.join(base_dir, subset, "processed_images")
    
    print(f"--- Initiating Step 1: Data Preprocessing ---")
    print(f"Targeting dataset subset: {subset.upper()} (Path: {input_dir})")
    
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' not found. Skipping execution. (Please run the data split script first)")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = glob(os.path.join(input_dir, "*.*"))
    print(f"Found {len(image_paths)} images for preprocessing.")
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: 
            continue
            
        # Convert from BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to the target resolution
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)
        
        # Revert to BGR for saving with OpenCV
        cv2.imwrite(save_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        
    print(f"Preprocessing completed. Processed images are saved in: {output_dir}\n")

if __name__ == "__main__":
    run_preprocessing()
