"""
Step 2: Data Augmentation
To mitigate overfitting and enhance the model's robustness against variations 
in radiograph quality (e.g., radiation dose discrepancies, patient movement artifacts), 
this module defines an advanced augmentation pipeline using the Albumentations library.

The pipeline specifically introduces:
- Random Brightness and Contrast adjustments (-44% to +44%).
- Random Pixel Noise (4%) to simulate sensor noise and movement artifacts.

This configuration is automatically recognized and integrated by YOLOv11 
during the initialization of the training phase if implemented in the dataset pipeline.
"""

import albumentations as A

def get_augmentation_pipeline():
    print(f"--- Initiating Step 2: Data Augmentation Pipeline Configuration ---")
    
    # Define the custom Albumentations pipeline
    custom_transform = A.Compose([
        # Brightness adjustment limit: +/- 44%
        A.RandomBrightnessContrast(brightness_limit=0.44, contrast_limit=0.0, p=0.5),
        
        # Noise injection: Random 4% pixel corruption (Salt and Pepper noise)
        A.SaltAndPepper(amount=0.04, p=0.3)
    ])
    
    print("Augmentation pipeline successfully configured.")
    print("Techniques applied: RandomBrightnessContrast, SaltAndPepper Noise.\n")
    return custom_transform

def run_augmentation_setup():
    pipeline = get_augmentation_pipeline()
    # The pipeline is typically passed to PyTorch Dataset classes.

if __name__ == "__main__":
    run_augmentation_setup()
