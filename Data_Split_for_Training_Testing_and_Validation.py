import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from skmultilearn.model_selection import IterativeStratification

# --- CONFIGURATION: Please adjust this section according to your specific project requirements ---

# 1. Dataset Directory Paths
IMAGE_DIR = 'data/images/'  # Directory containing the raw image files
ANNOTATION_DIR = 'data/labels/'  # Directory containing the corresponding annotation files

# 2. Output Directories (The script will automatically generate these directories)
OUTPUT_DIR = 'data/'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
VAL_DIR = os.path.join(OUTPUT_DIR, 'validation')

TRAIN_SIZE = 0.7  # Allocation of 70% of the dataset for training
TEST_SIZE = 0.2   # Allocation of 20% of the dataset for testing
VAL_SIZE = 0.1    # Allocation of 10% of the dataset for validation

ANNOTATION_FORMAT = 'txt'


def find_corresponding_image(basename, image_dir):
    """Searches for the corresponding image file for a given basename across common image extensions."""
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for ext in supported_extensions:
        # Evaluate both lowercase and uppercase extension variants (e.g., .jpg vs .JPG)
        for case_ext in [ext.lower(), ext.upper()]:
            image_path = os.path.join(image_dir, f"{basename}{case_ext}")
            if os.path.exists(image_path):
                return image_path
    return None  # Return None if no corresponding image file is located


def copy_files(file_indices, all_ann_files, train_dir, test_dir, val_dir, image_dir, ann_dir):
    """Executes a robust file copy operation to distribute data into the train/test/validation directories."""
    train_indices, test_indices, val_indices = file_indices

    def _copy_subset(indices, target_dir):
        for i in indices:
            ann_filename = all_ann_files[i]
            basename = os.path.splitext(ann_filename)[0]
            
            # Retrieve the corresponding image file
            image_path = find_corresponding_image(basename, image_dir)

            if image_path:
                # Duplicate both the image and its associated annotation file
                shutil.copy(image_path, os.path.join(target_dir, 'images'))
                shutil.copy(os.path.join(ann_dir, ann_filename), os.path.join(target_dir, 'labels'))
            else:
                print(f"WARNING: No corresponding image found for annotation '{ann_filename}'. Skipping this entry.")

    # Copy the training subset
    _copy_subset(train_indices, train_dir)
    # Copy the testing subset
    _copy_subset(test_indices, test_dir)
    # Copy the validation subset
    _copy_subset(val_indices, val_dir)


# Note: Functions for parsing XML and TXT annotations remain structurally identical
def get_labels_from_xml(filepath, class_map):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        labels = set()
        for member in root.findall('object'):
            label_name = member.find('name').text
            labels.add(label_name)
        one_hot = np.zeros(len(class_map), dtype=int)
        for label in labels:
            if label in class_map:
                one_hot[class_map[label]] = 1
        return one_hot
    except ET.ParseError:
        print(f"WARNING: XML parsing error encountered at '{filepath}'. Skipping this file.")
        return None


def get_labels_from_txt(filepath, class_map):
    labels = set()
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                class_id = int(line.strip().split()[0])
                labels.add(class_id)
            except (ValueError, IndexError):
                continue  # Bypass corrupted or malformed lines
    one_hot = np.zeros(len(class_map), dtype=int)
    for label_id in labels:
        if label_id < len(class_map):
            one_hot[label_id] = 1
    return one_hot


def get_all_class_names(annotation_dir, ann_format):
    class_names = set()
    if ann_format.lower() == 'xml':
        for filename in os.listdir(annotation_dir):
            if filename.lower().endswith('.xml'):
                try:
                    tree = ET.parse(os.path.join(annotation_dir, filename))
                    root = tree.getroot()
                    for member in root.findall('object'):
                        class_names.add(member.find('name').text)
                except ET.ParseError:
                    continue  # Bypass corrupted XML files
        return sorted(list(class_names))
    elif ann_format.lower() == 'txt':
        max_id = -1
        for filename in os.listdir(annotation_dir):
            if filename.lower().endswith('.txt'):
                with open(os.path.join(annotation_dir, filename), 'r') as f:
                    for line in f.readlines():
                        try:
                            class_id = int(line.strip().split()[0])
                            if class_id > max_id:
                                max_id = class_id
                        except (ValueError, IndexError):
                            continue  # Bypass corrupted or malformed lines
        if max_id == -1: return []
        return list(range(max_id + 1))


def create_split_dirs(train_dir, test_dir, val_dir):
    # Purge existing target directories and recreate their internal structure (without deleting the parent data directory)
    for d in [train_dir, test_dir, val_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(os.path.join(d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'labels'), exist_ok=True)


def main():
    # Verify the existence of the source directories
    for dir_path in [IMAGE_DIR, ANNOTATION_DIR]:
        if not os.path.isdir(dir_path):
            print(f"ERROR: The specified path '{dir_path}' is either not a directory or could not be found.")
            return

    create_split_dirs(TRAIN_DIR, TEST_DIR, VAL_DIR)

    ann_ext = f'.{ANNOTATION_FORMAT.lower()}'
    all_annotation_files = [f for f in os.listdir(ANNOTATION_DIR) if f.lower().endswith(ann_ext)]

    if not all_annotation_files:
        print(f"ERROR: No files with the '{ann_ext}' extension were found in the '{ANNOTATION_DIR}' directory.")
        return

    all_classes = get_all_class_names(ANNOTATION_DIR, ANNOTATION_FORMAT)
    if not all_classes:
        print("ERROR: Annotation files were successfully parsed, but no class information was detected within them.")
        return

    if ANNOTATION_FORMAT.lower() == 'xml':
        class_map = {name: i for i, name in enumerate(all_classes)}
    else:
        class_map = {i: i for i in all_classes}
    print(f"Detected a total of {len(class_map)} unique taxonomic classes.")

    X, y, valid_files = [], [], []
    for ann_file in all_annotation_files:
        filepath = os.path.join(ANNOTATION_DIR, ann_file)
        if ANNOTATION_FORMAT.lower() == 'xml':
            labels = get_labels_from_xml(filepath, class_map)
        else:
            labels = get_labels_from_txt(filepath, class_map)

        if labels is not None:
            X.append(ann_file)
            y.append(labels)
            valid_files.append(ann_file)

    if not X:
        print("ERROR: Failed to process any valid annotation data.")
        return

    X_np = np.array(X).reshape(-1, 1)
    y_np = np.array(y)

    print(f"Successfully processed {len(X_np)} files with valid annotations. Ready for stratification.")
    print("Initiating the iterative stratification protocol...")

    # Phase 1: Partition the dataset into Test (20%) and the Remainder (80%) subsets
    stratifier1 = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[TEST_SIZE, 1.0 - TEST_SIZE])
    temp_indices, test_indices = next(stratifier1.split(X_np, y_np))

    X_temp = X_np[temp_indices]
    y_temp = y_np[temp_indices]

    # Phase 2: Partition the Remainder (80%) into Training (70%) and Validation (10%) subsets
    val_ratio_in_temp = VAL_SIZE / (VAL_SIZE + TRAIN_SIZE)
    stratifier2 = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[val_ratio_in_temp, 1.0 - val_ratio_in_temp])
    train_indices_of_temp, val_indices_of_temp = next(stratifier2.split(X_temp, y_temp))

    # Map the isolated indices back to the original dataset dimensions
    train_indices = temp_indices[train_indices_of_temp]
    val_indices = temp_indices[val_indices_of_temp]

    print(f"Training subset volume: {len(train_indices)} samples ({TRAIN_SIZE * 100:.0f}%)")
    print(f"Testing subset volume: {len(test_indices)} samples ({TEST_SIZE * 100:.0f}%)")
    print(f"Validation subset volume: {len(val_indices)} samples ({VAL_SIZE * 100:.0f}%)")

    print("Executing file transfer to the respective subset directories...")
    # Supply the original valid file list to the copy algorithm
    copy_files((train_indices, test_indices, val_indices), valid_files, TRAIN_DIR, TEST_DIR, VAL_DIR, IMAGE_DIR, ANNOTATION_DIR)

    print("\nDataset stratification procedure concluded successfully.")
    print(f"Structured datasets have been archived in: '{os.path.abspath(OUTPUT_DIR)}'.")


if __name__ == '__main__':
    main()


