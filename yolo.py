"""
YOLO Fine-tuning Script for Number Plate Detection
This script converts XML annotations to YOLO format and fine-tunes YOLO11n model
"""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = Path(r"C:\Users\Admin\Desktop\comparison_study")
DATA_DIR = Path(r"C:\Users\Admin\Downloads\archive (7)")
IMAGES_DIR = DATA_DIR / "Indian_Number_Plates" / "Sample_Images"
ANNOTATIONS_DIR = DATA_DIR / "Annotations" / "Annotations"
MODEL_PATH = BASE_DIR / "yolo11n.pt"

# Output paths for YOLO format dataset
YOLO_DATASET_DIR = BASE_DIR / "yolo_dataset"
TRAIN_IMAGES_DIR = YOLO_DATASET_DIR / "images" / "train"
VAL_IMAGES_DIR = YOLO_DATASET_DIR / "images" / "val"
TRAIN_LABELS_DIR = YOLO_DATASET_DIR / "labels" / "train"
VAL_LABELS_DIR = YOLO_DATASET_DIR / "labels" / "val"


def create_directories():
    """Create necessary directories for YOLO dataset"""
    for dir_path in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")


def convert_xml_to_yolo(xml_file, img_width, img_height):
    """
    Convert XML annotation to YOLO format
    YOLO format: class_id center_x center_y width height (normalized 0-1)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall('object'):
        # Class name (we'll use 0 for number_plate)
        class_id = 0
        
        # Get bounding box
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO format (center_x, center_y, width, height) - normalized
        center_x = ((xmin + xmax) / 2) / img_width
        center_y = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations


def get_image_dimensions(xml_file):
    """Extract image dimensions from XML file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    return width, height


def prepare_dataset(test_size=0.2):
    """
    Convert XML annotations to YOLO format and split into train/val sets
    """
    print("📊 Preparing dataset...")
    
    # Get all XML files
    xml_files = list(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")
    
    if len(xml_files) == 0:
        print("❌ No XML files found!")
        return
    
    # Split into train and validation sets
    train_files, val_files = train_test_split(xml_files, test_size=test_size, random_state=42)
    
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    
    # Process training set
    for xml_file in train_files:
        process_file(xml_file, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    
    # Process validation set
    for xml_file in val_files:
        process_file(xml_file, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    
    print("✓ Dataset preparation complete!")


def process_file(xml_file, images_dir, labels_dir):
    """Process a single XML file and copy image to appropriate directory"""
    # Get corresponding image file
    image_name = xml_file.stem + ".jpg"
    image_path = IMAGES_DIR / image_name
    
    if not image_path.exists():
        print(f"⚠️ Warning: Image not found: {image_name}")
        return
    
    # Get image dimensions
    img_width, img_height = get_image_dimensions(xml_file)
    
    # Convert XML to YOLO format
    yolo_annotations = convert_xml_to_yolo(xml_file, img_width, img_height)
    
    # Copy image to destination
    dest_image = images_dir / image_name
    shutil.copy2(image_path, dest_image)
    
    # Save YOLO format labels
    label_file = labels_dir / (xml_file.stem + ".txt")
    with open(label_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))


def create_data_yaml():
    """Create data.yaml file for YOLO training"""
    data_yaml = {
        'path': str(YOLO_DATASET_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # number of classes
        'names': ['number_plate']  # class names
    }
    
    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✓ Created data.yaml at {yaml_path}")
    return yaml_path


def train_model(data_yaml_path, epochs=50, batch_size=16, img_size=640):
    """
    Fine-tune YOLO model on the prepared dataset
    """
    print("\n🚀 Starting model training...")
    
    # Load pretrained model
    model = YOLO(MODEL_PATH)
    print(f"✓ Loaded model: {MODEL_PATH}")
    
    # Train the model
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='number_plate_finetuning',
        patience=10,  # Early stopping patience
        save=True,
        device='cpu',  # Use CPU for training (changed from '0')
        optimizer='AdamW',
        lr0=0.001,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # Box loss gain
        cls=0.5,  # Classification loss gain
        dfl=1.5,  # Distribution focal loss gain
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation augmentation
        hsv_v=0.4,  # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,  # Scale augmentation
        shear=0.0,  # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,  # Vertical flip probability
        fliplr=0.5,  # Horizontal flip probability
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.0,  # Mixup augmentation probability
        copy_paste=0.0,  # Copy-paste augmentation probability
        verbose=True,
        plots=True  # Save training plots
    )
    
    print("\n✅ Training completed!")
    print(f"Best model saved at: {model.trainer.best}")
    
    return results


def validate_model(model_path, data_yaml_path):
    """Validate the trained model"""
    print("\n📈 Validating model...")
    model = YOLO(model_path)
    results = model.val(data=str(data_yaml_path))
    print("✓ Validation complete!")
    return results


def main():
    """Main execution function"""
    print("=" * 60)
    print("YOLO Fine-tuning for Number Plate Detection")
    print("=" * 60)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Prepare dataset (convert XML to YOLO format)
    prepare_dataset(test_size=0.2)
    
    # Step 3: Create data.yaml
    data_yaml_path = create_data_yaml()
    
    # Step 4: Train model
    train_results = train_model(
        data_yaml_path=data_yaml_path,
        epochs=100,  # Adjust as needed
        batch_size=8,  # Adjust based on your GPU memory
        img_size=640
    )
    
    # Step 5: Validate best model
    best_model_path = BASE_DIR / "runs" / "detect" / "number_plate_finetuning" / "weights" / "best.pt"
    if best_model_path.exists():
        validate_model(best_model_path, data_yaml_path)
    
    print("\n" + "=" * 60)
    print("🎉 Fine-tuning pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

