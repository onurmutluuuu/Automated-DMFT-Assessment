"""
Master Execution Script (run.py)

This root script orchestrates the sequential execution of all the individual modules 
in the DMFT Index Automation pipeline. It sequentially invokes Data Preprocessing, 
Augmentation Setup, Model Training, Performance Evaluation, and concludes with 
the DMFT Score Calculation.

Usage:
    python run.py
"""

import sys
import traceback


from step1_preprocessing import run_preprocessing
from step2_augmentation import run_augmentation_setup
from step3_training import run_training
from step4_evaluation import run_evaluation
from step5_dmft_calculation import run_dmft_calculation

def execute_pipeline():
    print("================================================================")
    print("   DMFT Index Automation via YOLOv11 - Pipeline Execution       ")
    print("================================================================\n")
    
    try:
        run_preprocessing()
        
        run_augmentation_setup()
        
        run_training()
        
        run_evaluation()
        
        run_dmft_calculation()
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An exception occurred during pipeline execution: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    print("================================================================")
    print("   Pipeline Execution Successfully Concluded                    ")
    print("================================================================")

if __name__ == "__main__":
    execute_pipeline()
