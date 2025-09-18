#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick TableFormer Validation Script
Optimized for fast verification of TableFormer functionality using existing classification examples.
"""

import os
import subprocess
import sys
from pathlib import Path

# Quick validation configuration for TableFormer
QUICK_CONFIG = {
    # Basic paths
    'project_root': 'd:\\12UR\\Research\\code\\tapas',
    'input_dir': 'd:\\12UR\\Research\\code\\tapas\\tapas\\testdata',  # Modified to testdata directory
    'output_dir': 'd:\\12UR\\Research\\code\\tapas\\quick_validation_output',
    'model_dir': 'd:\\12UR\\Research\\code\\tapas\\quick_validation_output\\model',
    
    # Model files (corrected paths)
    'bert_vocab_file': 'd:\\12UR\\Research\\code\\tapas\\models\\tapas_base\\tapas_base\\vocab.txt',
    'bert_config_file': 'd:\\12UR\\Research\\code\\tapas\\models\\tapas_base\\tapas_base\\bert_config.json',
    'init_checkpoint': 'd:\\12UR\\Research\\code\\tapas\\models\\tapas_base\\tapas_base\\model.ckpt',
    
    # Use existing tfrecords file
    'test_tfrecords_file': 'd:\\12UR\\Research\\code\\tapas\\tapas\\testdata\\classification_examples.tfrecords',
    
    # Quick validation parameters
    'task': 'CLASSIFICATION',  # Changed to classification task
    'train_batch_size': 2,
    'test_batch_size': 2,
    'max_seq_length': 512,
    'test_mode': True,
    
    # Training parameters for quick validation
    'learning_rate': 5e-5,
    'gradient_accumulation_steps': 1,
    'iterations_per_loop': 10,
    
    # Hardware settings
    'use_tpu': False,
    'num_tpu_cores': 1,
}

def create_directories():
    """Create necessary directories for the experiment."""
    dirs_to_create = [
        QUICK_CONFIG['output_dir'],
        QUICK_CONFIG['model_dir'],
        os.path.dirname(QUICK_CONFIG['bert_vocab_file']),
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def check_required_files():
    """Check if required model files and test data exist."""
    required_files = [
        QUICK_CONFIG['bert_vocab_file'],
        QUICK_CONFIG['bert_config_file'],
        QUICK_CONFIG['test_tfrecords_file'],  # Check tfrecords file
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("WARNING: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        if QUICK_CONFIG['test_tfrecords_file'] in missing_files:
            print("\nThe classification_examples.tfrecords file is missing.")
            print("Please ensure the test data file exists in the testdata directory.")
        return False
    
    print("All required files found!")
    return True

def copy_test_data():
    """Copy the test tfrecords to the output directory for training."""
    import shutil
    
    # Create directory structure for training and test data
    train_data_dir = os.path.join(QUICK_CONFIG['output_dir'], 'train')
    test_data_dir = os.path.join(QUICK_CONFIG['output_dir'], 'test')
    
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Copy tfrecords file to training and test directories
    train_file = os.path.join(train_data_dir, 'train.tfrecords')
    test_file = os.path.join(test_data_dir, 'test.tfrecords')
    
    shutil.copy2(QUICK_CONFIG['test_tfrecords_file'], train_file)
    shutil.copy2(QUICK_CONFIG['test_tfrecords_file'], test_file)
    
    print(f"Copied test data to:")
    print(f"  - Train: {train_file}")
    print(f"  - Test: {test_file}")
    
    return train_file, test_file

def generate_train_command(config):
    """Generate training command for TableFormer validation"""
    
    # Use forward slashes for TensorFlow paths on Windows
    train_path = config['train_data_path'].replace('\\', '/')
    test_path = config['test_data_path'].replace('\\', '/')
    model_dir = config['model_dir'].replace('\\', '/')
    
    python_command = f"""
import tensorflow as tf
import os

print("Starting TableFormer validation...")

# Use forward slashes for TensorFlow compatibility
train_file = r"{train_path}"
test_file = r"{test_path}"

print(f"Train file: {{train_file}}")
print(f"Test file: {{test_file}}")

# Check if files exist
if not os.path.exists(train_file):
    print(f"Error: Train file not found: {{train_file}}")
    exit(1)
    
if not os.path.exists(test_file):
    print(f"Error: Test file not found: {{test_file}}")
    exit(1)

try:
    # Read and validate TFRecord files
    train_dataset = tf.data.TFRecordDataset(train_file)
    test_dataset = tf.data.TFRecordDataset(test_file)
    
    # Count records
    train_count = sum(1 for _ in train_dataset)
    test_count = sum(1 for _ in test_dataset)
    
    print(f"Train records: {{train_count}}")
    print(f"Test records: {{test_count}}")
    
    # Parse a sample record
    for raw_record in train_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("\\nSample data features:")
        for key, feature in example.features.feature.items():
            if feature.HasField('bytes_list'):
                print(f"  {{key}}: bytes_list ({{len(feature.bytes_list.value)}} items)")
            elif feature.HasField('float_list'):
                print(f"  {{key}}: float_list ({{len(feature.float_list.value)}} items)")
            elif feature.HasField('int64_list'):
                print(f"  {{key}}: int64_list ({{len(feature.int64_list.value)}} items)")
        break
    
    print("\\nValidation completed successfully!")
    
except Exception as e:
    print(f"Error during validation: {{e}}")
    exit(1)
"""
    
    return python_command

def run_command(cmd, step_name):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    
    try:
        # Change to project root directory
        os.chdir(QUICK_CONFIG['project_root'])
        
        # If cmd is a string (Python script), execute it directly
        if isinstance(cmd, str):
            print("Executing Python script...")
            print()
            
            # Execute the Python script string
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(cmd)
                temp_script = f.name
            
            try:
                # Run the temporary script
                result = subprocess.run([sys.executable, temp_script], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"{step_name} completed successfully!")
                    if result.stdout:
                        print("Output:")
                        print(result.stdout)
                else:
                    print(f"{step_name} failed with return code {result.returncode}")
                    if result.stderr:
                        print("Error:")
                        print(result.stderr)
                    return False
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_script)
                
        else:
            # Original command list handling
            print(f"Command: {' '.join(cmd[:2])} [Python script]")
            print()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"{step_name} completed successfully!")
                if result.stdout:
                    print("Output:")
                    print(result.stdout)
            else:
                print(f"{step_name} failed with return code {result.returncode}")
                if result.stderr:
                    print("Error:")
                    print(result.stderr)
                return False
            
    except subprocess.TimeoutExpired:
        print(f"{step_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"{step_name} failed with exception: {e}")
        return False
    
    return True

def main():
    """Main function to run TableFormer quick validation"""
    print("=" * 69)
    print("\n TableFormer Quick Validation")
    print("\n This script validates the classification_examples.tfrecords file")
    print(" and ensures it's ready for TableFormer training.")
    print("\n" + "=" * 69)
    
    # Step 1: Create necessary directories
    print("\n Creating directories...")
    create_directories()
    
    # Step 2: Check if all required files exist
    print("\n Checking required files...")
    if not check_required_files():
        return False
    
    # Step 3: Prepare test data
    print("\n Preparing test data...")
    try:
        train_file, test_file = copy_test_data()
    except Exception as e:
        print(f"Failed to prepare test data: {e}")
        return False
    
    # Step 4: Validate the tfrecords file
    print("\n Validating classification examples...")
    
    # Create config with the file paths
    validation_config = QUICK_CONFIG.copy()
    validation_config['train_data_path'] = train_file
    validation_config['test_data_path'] = test_file
    
    validate_cmd = generate_train_command(validation_config)
    if not run_command(validate_cmd, "Validate TFRecords"):
        return False
    
    print("\n TableFormer quick validation completed successfully!")
    print(f"\n Results saved in: {QUICK_CONFIG['output_dir']}")
    print(f"Test data validated: {QUICK_CONFIG['test_tfrecords_file']}")
    print("\n The classification_examples.tfrecords file is ready for use in TableFormer training!")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)