"""
Step 4: Performance Evaluation Metrics
Following the training phase, this module assesses the model's predictive capabilities 
on an unseen test dataset. It extracts critical class-wise performance metrics, 
including Precision, Recall, F1-Score, and Mean Average Precision (mAP@0.5, mAP@0.5-0.95), 
for the specific target classes: "Healthy", "Decayed", and "Filled".

The module also verifies the generation of the visual confusion matrix.
"""

import os
import cv2
from ultralytics import YOLO

def run_evaluation(model_path="DMFT_Project/yolov11_dmft_run/weights/best.pt", data_yaml="DATA/data.yaml"):
    print(f"--- Initiating Step 4: Model Performance Evaluation ---")
    
    if not os.path.exists(model_path):
        print(f"Trained model weights '{model_path}' not found. Skipping evaluation.")
        print("(Please complete the training phase before executing evaluation.)\n")
        return
        
    print(f"Loading the optimized trained model from '{model_path}'...")
    model = YOLO(model_path)
    
    print("Executing validation pipeline on the test dataset split...")
    metrics = model.val(data=data_yaml, split="test", conf=0.25)
    
    print("\n=== Overall Model Performance Assessment ===")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
    
    print("\n=== Class-Specific Performance Analysis ===")
    class_names = metrics.names
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        class_name = class_names[class_idx]
        
        # Filtering strictly for the DMFT diagnostic classes
        if class_name in ["Healthy", "Decayed", "Filled"]:
            precision = metrics.box.p[i]
            recall = metrics.box.r[i]
            f1_score = metrics.box.f1[i]
            ap50 = metrics.box.ap50[i]
            
            print(f"Class: {class_name}")
            print(f"  - Precision: {precision:.4f}")
            print(f"  - Recall: {recall:.4f}")
            print(f"  - F1-Score: {f1_score:.4f}")
            print(f"  - mAP@0.5: {ap50:.4f}")
            
    cm_path = "DMFT_Project/yolov11_dmft_run/confusion_matrix.png"
    if os.path.exists(cm_path):
        print(f"\nVisual confusion matrix successfully located at: {cm_path}")
    else:
        print("\nConfusion matrix not found in the expected directory structure.")
        
    print("Performance evaluation fully completed.\n")

if __name__ == "__main__":
    run_evaluation()
