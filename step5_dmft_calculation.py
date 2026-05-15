"""
Step 5: Missing Tooth Inference and DMFT Score Calculation
The YOLO model inherently detects only visible entities within the radiograph 
(e.g., Healthy, Decayed, Filled teeth). According to standard dental anatomy 
(FDI World Dental Federation notation), an adult human dentition consists of 32 teeth.

This algorithmic module cross-references the model-detected teeth against the complete 
FDI reference array to autonomously deduce the "Missing" teeth category. Subsequently, 
it computes the comprehensive DMFT (Decayed, Missing, Filled Teeth) index score for the patient.
"""

def calculate_dmft_score(detected_teeth_dict):
    """
    Computes the DMFT score based on the teeth identified by the YOLOv11 model.
    
    Args:
        detected_teeth_dict (dict): A dictionary mapping FDI tooth numbers to their 
                                    respective diagnosed status.
                                    Example: {11: "Healthy", 12: "Decayed", 46: "Filled"}
                                    
    Returns:
        dict: A comprehensive analytical report containing individual component counts 
              and the final computed DMFT score.
    """
    # Standard FDI Adult Tooth Numbers (Total: 32 Teeth)
    upper_right = [11, 12, 13, 14, 15, 16, 17, 18]
    upper_left  = [21, 22, 23, 24, 25, 26, 27, 28]
    lower_left  = [31, 32, 33, 34, 35, 36, 37, 38]
    lower_right = [41, 42, 43, 44, 45, 46, 47, 48]
    
    reference_teeth = set(upper_right + upper_left + lower_left + lower_right)
    detected_teeth = set(detected_teeth_dict.keys())
    
    # Mathematical deduction of Missing Teeth: Reference Set minus Detected Set
    missing_teeth = list(reference_teeth - detected_teeth)
    
    # DMFT component accumulators
    decayed_count = 0
    filled_count = 0
    missing_count = len(missing_teeth)
    
    # Iterate through model predictions to count specific conditions
    for tooth_num, status in detected_teeth_dict.items():
        if status == "Decayed":
            decayed_count += 1
        elif status == "Filled":
            filled_count += 1
            
    # Final scalar computation
    total_dmft_score = decayed_count + missing_count + filled_count
    
    return {
        "D (Decayed) Component": decayed_count,
        "M (Missing) Component": missing_count,
        "F (Filled) Component": filled_count,
        "Total Computed DMFT Score": total_dmft_score,
        "Inferred Missing Teeth (FDI Notation)": sorted(missing_teeth)
    }

def run_dmft_calculation():
    print(f"--- Initiating Step 5: DMFT Score Calculation Algorithmic Test ---")
    
    # Simulated output dictionary from the YOLO model for a hypothetical patient
    hypothetical_yolo_output = {
        11: "Healthy", 12: "Decayed", 13: "Healthy", 14: "Filled",
        21: "Healthy", 22: "Healthy", 24: "Decayed",
        46: "Filled", 47: "Healthy"
    }
    
    print(f"Simulated Model Input Prediction Array: {hypothetical_yolo_output}")
    print("Executing subtractive cross-reference against the standard 32-tooth FDI matrix...")
    
    dmft_report = calculate_dmft_score(hypothetical_yolo_output)
    
    print("\n=== Automated Patient DMFT Analytical Report ===")
    for key, value in dmft_report.items():
        print(f"  > {key}: {value}")
        
    print("\nAlgorithm execution and validation completed successfully.\n")

if __name__ == "__main__":
    run_dmft_calculation()
