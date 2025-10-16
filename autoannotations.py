from ultralytics import YOLO
import os
import glob
import cv2

def yolo_normalize_bbox(x1, y1, x2, y2, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO normalized format
    """
    # Calculate center coordinates
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    
    # Calculate width and height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height

def infer_and_save_annotations():
    """
    Perform inference on all images using the trained YOLO model
    and save YOLO-formatted annotations
    """
    # Configuration 
    # add path of your last.pt file here
    weights_path = "D:/annotation_tool_videos_to_training_data/last.pt"
    images_root = "DW_Frame"
    output_root = "DW_Frame"
    
    # Check if model exists
    if not os.path.exists(weights_path):
        print(f"Error: Model file {weights_path} not found!")
        return
    
    # Check if images directory exists
    if not os.path.exists(images_root):
        print(f"Error: Images directory {images_root} not found!")
        return
    
    # Load YOLO model
    try:
        model = YOLO(weights_path)
        print(f"Loaded YOLO model from {weights_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output root directory if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        print(f"Created output directory: {output_root}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_root, '**', extension), recursive=True))
    
    print(f"Found {len(image_files)} images for inference")
    
    processed_count = 0
    
    for image_path in image_files:
        try:
            # Load image to get dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: {image_path}")
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Perform inference
            results = model(image_path)
            
            annotations = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Convert to YOLO format
                        x_center, y_center, width, height = yolo_normalize_bbox(
                            x1, y1, x2, y2, img_width, img_height
                        )
                        
                        # Class ID is always 1 as requested
                        class_id = 1
                        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Create output annotation path
            rel_path = os.path.relpath(image_path, images_root)
            annotation_rel_path = os.path.splitext(rel_path)[0] + '.txt'
            annotation_path = os.path.join(output_root, annotation_rel_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
            
            # Save annotations to file
            with open(annotation_path, 'w') as f:
                for annotation in annotations:
                    f.write(annotation + '\n')
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\nInference completed! Processed {processed_count} images")
    print(f"Annotations saved to: {output_root}")

def main():
    """
    Main function to run inference and save annotations
    """
    print("Starting YOLO inference...")
    infer_and_save_annotations()
    print("Done!")

if __name__ == "__main__":
    main()