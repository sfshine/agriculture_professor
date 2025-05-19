import os
import shutil
import sys

# Define the labels to keep (as specified in your query)
LABELS_TO_KEEP = {0, 3, 5, 9, 13, 15, 17, 21, 23}

# File paths
TRAIN_FILE = "AgriculturalDisease_trainingset/train_list.txt"
TRAIN_BACKUP = "AgriculturalDisease_trainingset/train_list_backup.txt"
TEST_FILE = "AgriculturalDisease_validationset/ttest_list.txt"
TEST_BACKUP = "AgriculturalDisease_validationset/ttest_list_backup.txt"
TRAIN_DIR = "AgriculturalDisease_trainingset/images"
TEST_DIR = "AgriculturalDisease_validationset/images"

# Function to restore or create backup
def manage_backup(file_path, backup_path, create_mode=False):
    # If we're restoring and backup doesn't exist, exit with error
    if not create_mode and not os.path.exists(backup_path):
        print(f"Error: Backup file {backup_path} does not exist. Cannot restore.")
        return False
    
    # If we're creating backup and it already exists, skip
    if create_mode and os.path.exists(backup_path):
        print(f"Backup file {backup_path} already exists. Skipping backup creation.")
        return True
    
    # Create backup or restore from backup
    try:
        if create_mode:
            shutil.copy2(file_path, backup_path)
            print(f"Created backup: {backup_path}")
        else:
            shutil.copy2(backup_path, file_path)
            print(f"Restored from backup: {file_path}")
        return True
    except Exception as e:
        print(f"Error with backup operation: {str(e)}")
        return False

# Helper function to filter file and collect images to keep or remove
def filter_data_file(file_path, images_dir, labels_to_keep):
    # Lists to store results
    kept_lines = []
    kept_images = []
    removed_lines = []
    removed_images = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return kept_lines, kept_images, removed_lines, removed_images
    
    # Read file content
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        
        img_path, label = parts[0], int(parts[1])
        
        if label in labels_to_keep:
            kept_lines.append(line)
            kept_images.append(img_path)
        else:
            removed_lines.append(line)
            removed_images.append(img_path)
    
    # Write filtered content back
    with open(file_path, 'w') as f:
        f.writelines(kept_lines)
    
    return kept_lines, kept_images, removed_lines, removed_images

# Function to remove image files
def remove_image_files(image_paths, base_dir):
    removed_count = 0
    failed_count = 0
    failed_files = []
    
    for img_path in image_paths:
        # Extract the relative path if it's a full path
        if '\\' in img_path:
            img_relative_path = img_path.split('\\')[1]
        else:
            img_relative_path = os.path.basename(img_path)
        
        # Full path to the image
        full_img_path = os.path.join(base_dir, img_relative_path)
        
        try:
            if os.path.exists(full_img_path):
                os.remove(full_img_path)
                removed_count += 1
            else:
                failed_count += 1
                failed_files.append(full_img_path)
        except Exception as e:
            failed_count += 1
            failed_files.append(f"{full_img_path} (Error: {str(e)})")
    
    return removed_count, failed_count, failed_files

# Function to find and remove orphaned image files
def remove_orphaned_images(list_file, images_dir):
    # Read all image paths from the list file
    valid_images = set()
    with open(list_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_path = parts[0]
                # Extract the relative path if it's a full path
                if '\\' in img_path:
                    img_relative_path = img_path.split('\\')[1]
                else:
                    img_relative_path = os.path.basename(img_path)
                valid_images.add(img_relative_path)
    
    # Find all image files in the directory
    removed_count = 0
    failed_count = 0
    failed_files = []
    
    for filename in os.listdir(images_dir):
        if filename not in valid_images:
            full_path = os.path.join(images_dir, filename)
            try:
                os.remove(full_path)
                removed_count += 1
            except Exception as e:
                failed_count += 1
                failed_files.append(f"{full_path} (Error: {str(e)})")
    
    return removed_count, failed_count, failed_files

# Main function
def main():
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--restore":
            print("Restoring original files from backups...")
            train_restored = manage_backup(TRAIN_FILE, TRAIN_BACKUP, create_mode=False)
            test_restored = manage_backup(TEST_FILE, TEST_BACKUP, create_mode=False)
            
            if train_restored and test_restored:
                print("Restore completed successfully.")
            else:
                print("Restore failed.")
            return
        elif sys.argv[1] == "--backup":
            print("Creating backups of current files...")
            train_backed_up = manage_backup(TRAIN_FILE, TRAIN_BACKUP, create_mode=True)
            test_backed_up = manage_backup(TEST_FILE, TEST_BACKUP, create_mode=True)
            
            if train_backed_up and test_backed_up:
                print("Backup completed successfully.")
            else:
                print("Backup failed.")
            return
    
    # Check if backups exist, if not create them
    print("Checking for backups...")
    train_backed_up = manage_backup(TRAIN_FILE, TRAIN_BACKUP, create_mode=True)
    test_backed_up = manage_backup(TEST_FILE, TEST_BACKUP, create_mode=True)
    
    # Restore from backups to start with original data
    print("Restoring from backups to ensure we have all original data...")
    train_restored = manage_backup(TRAIN_FILE, TRAIN_BACKUP, create_mode=False)
    test_restored = manage_backup(TEST_FILE, TEST_BACKUP, create_mode=False)
    
    if not (train_restored and test_restored):
        print("Error: Could not restore from backups. Exiting.")
        return
    
    print("\nStarting to filter data files...")
    
    # Filter training data
    train_kept_lines, train_kept_images, train_removed_lines, train_removed_images = filter_data_file(TRAIN_FILE, TRAIN_DIR, LABELS_TO_KEEP)
    
    # Filter test data
    test_kept_lines, test_kept_images, test_removed_lines, test_removed_images = filter_data_file(TEST_FILE, TEST_DIR, LABELS_TO_KEEP)
    
    # Remove unwanted image files
    print("\nRemoving unwanted image files...")
    train_removed_count, train_failed_count, train_failed_files = remove_image_files(train_removed_images, TRAIN_DIR)
    test_removed_count, test_failed_count, test_failed_files = remove_image_files(test_removed_images, TEST_DIR)
    
    # Remove orphaned image files
    print("\nRemoving orphaned image files...")
    train_orphaned_count, train_orphaned_failed, train_orphaned_failed_files = remove_orphaned_images(TRAIN_FILE, TRAIN_DIR)
    test_orphaned_count, test_orphaned_failed, test_orphaned_failed_files = remove_orphaned_images(TEST_FILE, TEST_DIR)
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Training file: {len(train_kept_lines)} lines kept, {len(train_removed_lines)} lines removed")
    print(f"Training images: {train_removed_count} images removed, {train_failed_count} failed")
    print(f"Training orphaned images: {train_orphaned_count} removed, {train_orphaned_failed} failed")
    print(f"Test file: {len(test_kept_lines)} lines kept, {len(test_removed_lines)} lines removed")
    print(f"Test images: {test_removed_count} images removed, {test_failed_count} failed")
    print(f"Test orphaned images: {test_orphaned_count} removed, {test_orphaned_failed} failed")
    
    # Print sample of removed records (first 20)
    print("\n--- Sample Removed Records (Training) ---")
    for i, line in enumerate(train_removed_lines[:20]):
        print(line.strip())
        if i >= 19:
            break
    
    print("\n--- Sample Removed Records (Test) ---")
    for i, line in enumerate(test_removed_lines[:20]):
        print(line.strip())
        if i >= 19:
            break
            
    print("\n--- Label Distribution (Training) ---")
    train_labels = {}
    for line in train_kept_lines:
        label = int(line.strip().split()[1])
        train_labels[label] = train_labels.get(label, 0) + 1
    
    for label, count in sorted(train_labels.items()):
        print(f"Label {label}: {count} images")
    
    print("\n--- Label Distribution (Test) ---")
    test_labels = {}
    for line in test_kept_lines:
        label = int(line.strip().split()[1])
        test_labels[label] = test_labels.get(label, 0) + 1
    
    for label, count in sorted(test_labels.items()):
        print(f"Label {label}: {count} images")
    
    # Print failed files (first 20)
    if train_failed_count > 0 or train_orphaned_failed > 0:
        print("\n--- Failed to Remove (Training) ---")
        for i, file_path in enumerate(train_failed_files[:10] + train_orphaned_failed_files[:10]):
            print(file_path)
            if i >= 19:
                break
    
    if test_failed_count > 0 or test_orphaned_failed > 0:
        print("\n--- Failed to Remove (Test) ---")
        for i, file_path in enumerate(test_failed_files[:10] + test_orphaned_failed_files[:10]):
            print(file_path)
            if i >= 19:
                break

if __name__ == "__main__":
    main() 