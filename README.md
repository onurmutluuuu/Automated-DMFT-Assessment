# Automated DMFT Assessment on Digital Panoramic Radiographs Using YOLOv11

## Project Overview
This repository contains the official implementation of a deep learning-based framework for automated assessment of the Decayed, Missing, and Filled Teeth (DMFT) index on digital panoramic radiographs (DPRs). Powered by the state-of-the-art YOLOv11 single-stage convolutional neural network (CNN) architecture, this system provides tooth-resolved diagnostic charting to support clinical decision-making and enable large-scale epidemiological surveillance.

## Key Features
* **YOLOv11 Object Detection Pipeline:** Utilizes an advanced single-stage detector to perform simultaneous tooth localization, FDI tooth numbering, and multi-class condition classification (**Healthy**, **Decayed**, **Filled**).
* **Algorithmic Missing-Tooth Inference:** Implements a rule-based post-processing strategy that identifies missing teeth by comparing detected tooth positions against a complete 32-tooth FDI reference framework, eliminating the need for direct bounding-box training on edentulous regions.
* **Interactive Clinical Interface:** Features a full-stack web application (GUI) that enables clinicians to upload panoramic radiographs, visualize model outputs, and automatically generate interactive linear DMFT tooth charts.

## Repository Structure & Pipeline Workflow
The project repository is structured to mirror the methodological steps outlined in the paper:

```text
├── DATA/                                             # Directory for panoramic images and YOLO TXT annotations
├── GUI/                                              # Full-stack clinical web application (Vue.js, FastAPI, MSSQL)
├── Data_Split_for_Training_Testing_and_Vali...py     # Partitions dataset (N=1,316) into train (n=1,053), val (n=132), and test (n=131)
├── step1_preprocessing.py                            # Rescaling images to 1280x1280 pixels and normalizing pixel intensities
├── step2_augmentation.py                             # Applies random brightness adjustment (±44%) and noise perturbation (4%)
├── step3_training.py                                 # Executes YOLOv11 model training using PyTorch and SGD optimization
├── step4_evaluation.py                               # Computes Precision, Recall, F1-Score, mAP@0.5, and generates confusion matrices
├── step5_dmft_calculation.py                         # Rule-based missing-tooth inference and final DMFT score derivation
└── run.py                                            # Main execution entry point for pipeline execution or GUI invocation
```

Model Performance
The YOLOv11 model demonstrated robust overall performance across all DMFT components, with notable excellence in detecting restored (Filled) teeth. The rule-based missing-tooth identification achieved an overall accuracy of 95.48%.

| Condition / Class | Precision | Recall | F1-Score | mAP@0.5 |
| :--- | :--- | :--- | :--- | :--- |
| **Healthy** | 0.805 | 0.899 | 0.849 | 0.863 |
| **Decayed** | 0.794 | 0.597 | 0.682 | 0.699 |
| **Filled** | 0.883 | 0.937 | 0.909 | 0.954 |
| **Overall** | 0.828 | 0.811 | 0.819 | 0.839 |


## Technology Stack
* **Deep Learning Framework:** Python (v3.11.10), PyTorch (v2.5.1+cu121), Ultralytics YOLOv11 (v8.3.158)
* **Frontend:** Vue.js
* **Backend API:** FastAPI
* **Database Management:** Microsoft SQL Server (MSSQL)

* Computational Hardware
Model development, hyperparameter tuning, and evaluation were conducted on a high-performance workstation with the following hardware specifications:
* **GPU:** NVIDIA GeForce RTX 3090
* **CPU:** AMD Ryzen 9 5900x
* **RAM:** 64 GB DDR4 3600MHz

* Data Availability
The fully annotated digital panoramic radiograph dataset (N=1,316) used in this study is openly accessible on Kaggle:

Kaggle Dataset: Automated DMFT Assessment Dataset **https://www.kaggle.com/datasets/ohyp96/automated-dmft-assessment**

Citation Request
If you find this repository, dataset, or clinical tool useful in your academic research or experiments, please cite our published paper:
@article{aslan2026dmft,
  title={DMF-T assessment on panoramic images using deep learning-based convolutional neural network algorithm},
  author={Aslan, Elif and Ulusoy, Ali Canberk and Mutlu, Onur and Onem, Erinc and Sener, Elif and Mert, Ali and Baksi, B. Guniz},
  journal={Scientific Reports},
  volume={16},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-026-63568-y}
}

## Authors
* **Elif Aslan** - Izmir Tinaztepe University, Department of Oral and Maxillofacial Radiology
* **Ali Canberk Ulusoy** - Sigma Dental Clinic
* **Onur Mutlu** - Karadeniz Technical University, Department of Computer Sciences
* **Erinç Önem** - Ege University, Department of Oral and Maxillofacial Radiology
* **Elif Şener** - Ege University, Department of Oral and Maxillofacial Radiology
* **Ali Mert** - Izmir Katip Celebi University, Department of Engineering Sciences
* **B. Güniz Baksi** - Ege University, Department of Oral and Maxillofacial Radiology
