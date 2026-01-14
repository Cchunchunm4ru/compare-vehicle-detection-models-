"""
Test Script for Fine-tuned YOLO Model
This script tests the fine-tuned model on validation images
"""
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# Paths
BASE_DIR = Path(r"C:\Users\Admin\Desktop\comparison_study")
YOLO_DATASET_DIR = BASE_DIR / "yolo_dataset"
VAL_IMAGES_DIR = YOLO_DATASET_DIR / "images" / "val"

# Model path (adjust based on your training run name)
MODEL_PATH = BASE_DIR / "runs" / "detect" / "number_plate_finetuning" / "weights" / "best.pt"


def test_on_validation_set(model_path, confidence_threshold=0.25):
    """
    Test the fine-tuned model on validation images
    """
    print("🔍 Testing fine-tuned model...")
    
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please run the training script first!")
        return
    
    # Load model
    model = YOLO(model_path)
    print(f"✓ Loaded model: {model_path}")
    
    # Get validation images
    val_images = list(VAL_IMAGES_DIR.glob("*.jpg"))
    
    if len(val_images) == 0:
        print("❌ No validation images found!")
        return
    
    print(f"Found {len(val_images)} validation images")
    
    # Create output directory
    output_dir = BASE_DIR / "finetuning" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test on each image
    detection_count = 0
    for img_path in val_images:
        # Run inference
        results = model(img_path, conf=confidence_threshold)
        
        # Count detections
        for result in results:
            if len(result.boxes) > 0:
                detection_count += 1
        
        # Save annotated image
        for i, result in enumerate(results):
            output_path = output_dir / f"result_{img_path.name}"
            result.save(filename=str(output_path))
    
    print(f"\n📊 Results:")
    print(f"  Total images tested: {len(val_images)}")
    print(f"  Images with detections: {detection_count}")
    print(f"  Detection rate: {detection_count/len(val_images)*100:.1f}%")
    print(f"\n✓ Results saved to: {output_dir}")


def test_single_image(model_path, image_path, confidence_threshold=0.25):
    """
    Test the model on a single image and display results
    """
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path, conf=confidence_threshold)
    
    # Display results
    for result in results:
        print(f"\nDetections: {len(result.boxes)}")
        for box in result.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            xyxy = box.xyxy[0].tolist()
            print(f"  Class: number_plate, Confidence: {conf:.2f}, Box: {xyxy}")
        
        # Show annotated image
        result.show()


def batch_predict(model_path, images_dir, output_dir, confidence_threshold=0.25):
    """
    Run batch predictions on a directory of images
    """
    model = YOLO(model_path)
    
    # Get all images
    image_paths = list(Path(images_dir).glob("*.jpg"))
    image_paths.extend(list(Path(images_dir).glob("*.png")))
    
    print(f"Processing {len(image_paths)} images...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in image_paths:
        results = model(img_path, conf=confidence_threshold)
        
        for result in results:
            output_path = Path(output_dir) / f"pred_{img_path.name}"
            result.save(filename=str(output_path))
    
    print(f"✓ Predictions saved to: {output_dir}")


def export_model(model_path, export_format='onnx'):
    """
    Export the fine-tuned model to different formats
    Supported formats: onnx, torchscript, coreml, tflite, etc.
    """
    print(f"📦 Exporting model to {export_format}...")
    
    model = YOLO(model_path)
    export_path = model.export(format=export_format)
    
    print(f"✓ Model exported to: {export_path}")
    return export_path


# Model architecture (must match training script)
class SimpleFasterRCNN(nn.Module):
    """Simplified detection model based on ResNet18"""
    
    def __init__(self, num_classes):
        super(SimpleFasterRCNN, self).__init__()
        
        # Use pretrained ResNet18 as backbone
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Detection head
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Classification and regression heads
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.bbox_head = nn.Conv2d(256, 4, kernel_size=1)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Detection head
        x = self.conv1(features)
        x = self.relu(x)
        
        # Predictions
        class_logits = self.cls_head(x)
        bbox_pred = self.bbox_head(x)
        
        return class_logits, bbox_pred


def convert_to_onnx():
    """Convert PyTorch model to ONNX"""
    
    # Paths
    pytorch_model_path = Path(r"C:\Users\Admin\trafficcamnet\number_plate_dataset\best_model.pth")
    onnx_model_path = Path(r"C:\Users\Admin\trafficcamnet\number_plate_dataset\best_model.onnx")
    
    print(f"🔄 Converting PyTorch model to ONNX...")
    print(f"Input: {pytorch_model_path}")
    print(f"Output: {onnx_model_path}")
    
    # Model configuration
    NUM_CLASSES = 5
    INPUT_WIDTH = 960
    INPUT_HEIGHT = 544
    
    # Load PyTorch model
    print("\n📥 Loading PyTorch model...")
    model = SimpleFasterRCNN(num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    
    # Export to ONNX
    print("\n📤 Exporting to ONNX...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_model_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['class_logits', 'bbox_pred'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'class_logits': {0: 'batch_size'},
                'bbox_pred': {0: 'batch_size'}
            },
            verbose=False
        )
    
    print(f"✓ ONNX model saved to: {onnx_model_path}")
    
    # Verify the exported model
    try:
        print("\n🔍 Verifying ONNX model...")
        import onnx
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except ImportError:
        print("⚠ onnx package not available for verification, but export completed")
    
    return onnx_model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test fine-tuned YOLO model')
    parser.add_argument('--mode', type=str, default='validation', 
                        choices=['validation', 'single', 'batch', 'export'],
                        help='Testing mode')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--images-dir', type=str, help='Directory of images (for batch mode)')
    parser.add_argument('--output-dir', type=str, help='Output directory (for batch mode)')
    parser.add_argument('--confidence', type=float, default=0.25, 
                        help='Confidence threshold')
    parser.add_argument('--export-format', type=str, default='onnx',
                        help='Export format (for export mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'validation':
        test_on_validation_set(MODEL_PATH, args.confidence)
    elif args.mode == 'single':
        if args.image:
            test_single_image(MODEL_PATH, args.image, args.confidence)
        else:
            print("❌ Please provide --image path for single mode")
    elif args.mode == 'batch':
        if args.images_dir and args.output_dir:
            batch_predict(MODEL_PATH, args.images_dir, args.output_dir, args.confidence)
        else:
            print("❌ Please provide --images-dir and --output-dir for batch mode")
    elif args.mode == 'export':
        export_model(MODEL_PATH, args.export_format)
    try:
        onnx_path = convert_to_onnx()
        print(f"\n✅ Conversion complete!")
        print(f"You can now use the ONNX model at: {onnx_path}")
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
