#!/usr/bin/env python3
"""
Detect Custom Objects with Prototype Matching
==============================================

Zero-shot object detection using OWL-ViT + CLIP prototype matching.
No training required - just provide reference images of your objects.

Use Cases:
- Inventory tracking with custom products
- Research with specific experimental materials
- Quality control with custom parts
- Wildlife identification
- Art/artifact recognition

Example Usage:
    # Create prototypes from reference images
    python detect_custom_objects.py create-prototypes \\
        --images reference_images/ \\
        --output prototypes.h5
    
    # Detect objects in video frames
    python detect_custom_objects.py detect \\
        --input video.mp4 \\
        --prototypes prototypes.h5 \\
        --output detections/
    
    # Detect in image folder
    python detect_custom_objects.py detect \\
        --input frames/ \\
        --prototypes prototypes.h5 \\
        --categories '{"toy": ["toy", "object"], "tool": ["tool", "instrument"]}'
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_prototypes(args):
    """Create prototype embeddings from reference images."""
    import torch
    import h5py
    from PIL import Image
    from tqdm import tqdm
    
    try:
        from transformers import AutoModel, AutoImageProcessor
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        return 1
    
    images_path = Path(args.images)
    output_path = Path(args.output)
    
    print("=" * 60)
    print("Creating Object Prototypes")
    print("=" * 60)
    print(f"Images directory: {images_path}")
    print(f"Output file: {output_path}")
    print(f"Model: {args.model}")
    print()
    
    # Load CLIP model
    print("Loading CLIP model...")
    model = AutoModel.from_pretrained(args.model).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)
    
    if torch.cuda.is_available() and args.device == "cuda":
        model = model.cuda()
    
    # Discover images organized by category
    # Expected structure: images_dir/category_name/image1.jpg
    categories = {}
    
    if images_path.is_dir():
        for category_dir in images_path.iterdir():
            if category_dir.is_dir():
                image_files = list(category_dir.glob("*.jpg")) + \
                             list(category_dir.glob("*.png")) + \
                             list(category_dir.glob("*.jpeg"))
                if image_files:
                    categories[category_dir.name] = image_files
    
    if not categories:
        print("Error: No image categories found.")
        print("Expected structure: images_dir/category_name/image1.jpg")
        return 1
    
    print(f"Found {len(categories)} categories:")
    for cat, files in categories.items():
        print(f"  - {cat}: {len(files)} images")
    print()
    
    # Create embeddings
    print("Extracting embeddings...")
    embeddings = {}
    
    with torch.no_grad():
        for category, image_files in tqdm(categories.items()):
            cat_embeddings = []
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = processor(img, return_tensors="pt")
                    
                    if torch.cuda.is_available() and args.device == "cuda":
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # Get image embedding
                    if hasattr(model, "get_image_features"):
                        # CLIP model
                        embedding = model.get_image_features(**inputs)
                    else:
                        # Generic vision model
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0]
                    
                    embedding = embedding.cpu().numpy()
                    cat_embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"  Warning: Could not process {img_path}: {e}")
            
            if cat_embeddings:
                import numpy as np
                # Average embeddings for this category
                embeddings[category] = np.mean(np.vstack(cat_embeddings), axis=0)
    
    # Save to HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        for category, embedding in embeddings.items():
            f.create_dataset(category, data=embedding)
        f.attrs["model"] = args.model
        f.attrs["num_categories"] = len(embeddings)
    
    print(f"\nPrototypes saved to: {output_path}")
    print(f"Categories: {list(embeddings.keys())}")
    
    return 0


def detect_objects(args):
    """Detect objects using prototypes."""
    import torch
    import cv2
    import h5py
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    prototypes_path = Path(args.prototypes)
    
    print("=" * 60)
    print("Object Detection with Prototype Matching")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Prototypes: {prototypes_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load prototypes
    print("Loading prototypes...")
    with h5py.File(prototypes_path, "r") as f:
        prototype_names = list(f.keys())
        prototypes = {name: torch.tensor(f[name][:]) for name in prototype_names}
        model_name = f.attrs.get("model", "openai/clip-vit-base-patch32")
    
    print(f"Loaded {len(prototypes)} prototype categories: {prototype_names}")
    
    # Parse category queries for first stage
    first_stage_queries = ["object", "thing"]
    if args.categories:
        categories = json.loads(args.categories)
        first_stage_queries = []
        for queries in categories.values():
            first_stage_queries.extend(queries)
    
    # Load models
    print("Loading detection models...")
    try:
        from naturallab.gaze_analysis.object_detection.owlv2 import OWLv2Detector
        from transformers import AutoModel, AutoImageProcessor
    except ImportError as e:
        print(f"Error importing: {e}")
        return 1
    
    # Initialize detector
    owl_detector = OWLv2Detector(device=args.device)
    
    # Load CLIP for second stage
    clip_model = AutoModel.from_pretrained(model_name).eval()
    clip_processor = AutoImageProcessor.from_pretrained(model_name)
    if torch.cuda.is_available() and args.device == "cuda":
        clip_model = clip_model.cuda()
    
    # Stack prototypes for matching
    proto_tensor = torch.stack(list(prototypes.values()))
    proto_tensor = proto_tensor / proto_tensor.norm(dim=-1, keepdim=True)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process input
    if input_path.suffix in [".mp4", ".avi", ".mov"]:
        # Video input
        process_video(input_path, output_path, owl_detector, clip_model, 
                     clip_processor, proto_tensor, prototype_names,
                     first_stage_queries, args)
    else:
        # Image folder input
        image_files = list(input_path.glob("*.jpg")) + \
                     list(input_path.glob("*.png")) + \
                     list(input_path.glob("*.jpeg"))
        
        all_detections = []
        
        for img_path in tqdm(image_files, desc="Processing images"):
            detections = process_image(
                img_path, owl_detector, clip_model, clip_processor,
                proto_tensor, prototype_names, first_stage_queries, args
            )
            for det in detections:
                det["image"] = img_path.name
            all_detections.extend(detections)
        
        # Save results
        if all_detections:
            df = pd.DataFrame(all_detections)
            df.to_csv(output_path / "detections.csv", index=False)
            print(f"\nSaved {len(df)} detections to detections.csv")
    
    return 0


def process_image(img_path, owl_detector, clip_model, clip_processor, 
                 proto_tensor, prototype_names, queries, args):
    """Process a single image and return detections."""
    import torch
    from PIL import Image
    from torchvision.transforms.functional import crop
    
    img = Image.open(img_path).convert("RGB")
    
    # First stage: OWL-ViT detection
    first_stage = owl_detector.detect(img, queries, threshold=args.threshold)
    
    detections = []
    
    # Second stage: CLIP prototype matching
    with torch.no_grad():
        for box, score, label in zip(first_stage["boxes"], 
                                     first_stage["scores"], 
                                     first_stage["labels"]):
            x, y, w, h = box
            
            # Crop detection
            cropped = crop(img, top=int(y), left=int(x), 
                          height=int(h), width=int(w))
            
            # Get CLIP embedding
            inputs = clip_processor(cropped, return_tensors="pt")
            if torch.cuda.is_available() and args.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            if hasattr(clip_model, "get_image_features"):
                embedding = clip_model.get_image_features(**inputs)
            else:
                outputs = clip_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0]
            
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            # Match to prototypes
            similarities = (embedding @ proto_tensor.T).squeeze()
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            
            if best_score >= args.match_threshold:
                detections.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "detection_score": float(score),
                    "category": prototype_names[best_idx],
                    "match_score": float(best_score)
                })
    
    return detections


def process_video(video_path, output_path, owl_detector, clip_model,
                 clip_processor, proto_tensor, prototype_names, queries, args):
    """Process video and save detections."""
    import cv2
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames @ {fps} FPS")
    
    all_detections = []
    frame_idx = 0
    
    # Process every Nth frame
    frame_skip = args.frame_skip or 1
    
    for _ in tqdm(range(0, total_frames, frame_skip), desc="Processing video"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to PIL
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect
        detections = process_image(
            img, owl_detector, clip_model, clip_processor,
            proto_tensor, prototype_names, queries, args
        )
        
        for det in detections:
            det["frame"] = frame_idx
            det["timestamp"] = frame_idx / fps if fps > 0 else 0
        
        all_detections.extend(detections)
        frame_idx += frame_skip
    
    cap.release()
    
    # Save results
    if all_detections:
        df = pd.DataFrame(all_detections)
        df.to_csv(output_path / "detections.csv", index=False)
        
        # Summary statistics
        summary = df.groupby("category").agg({
            "frame": "count",
            "match_score": "mean"
        }).rename(columns={"frame": "count", "match_score": "avg_confidence"})
        summary.to_csv(output_path / "detection_summary.csv")
        
        print(f"\nSaved {len(df)} detections")
        print("\nDetection summary:")
        print(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot object detection with prototype matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create prototypes command
    proto_parser = subparsers.add_parser("create-prototypes",
                                         help="Create prototype embeddings from images")
    proto_parser.add_argument("--images", "-i", required=True,
                             help="Directory of reference images (organized by category)")
    proto_parser.add_argument("--output", "-o", required=True,
                             help="Output HDF5 file for prototypes")
    proto_parser.add_argument("--model", default="openai/clip-vit-large-patch14-336",
                             help="CLIP model to use")
    proto_parser.add_argument("--device", default="cuda",
                             help="Device (cuda/cpu)")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect",
                                          help="Detect objects using prototypes")
    detect_parser.add_argument("--input", "-i", required=True,
                              help="Input video or image directory")
    detect_parser.add_argument("--prototypes", "-p", required=True,
                              help="Prototype embeddings file (HDF5)")
    detect_parser.add_argument("--output", "-o", required=True,
                              help="Output directory")
    detect_parser.add_argument("--categories", type=str,
                              help="JSON dict of first-stage category queries")
    detect_parser.add_argument("--threshold", type=float, default=0.1,
                              help="Detection threshold (default: 0.1)")
    detect_parser.add_argument("--match-threshold", type=float, default=0.3,
                              help="Prototype match threshold (default: 0.3)")
    detect_parser.add_argument("--frame-skip", type=int, default=1,
                              help="Process every Nth frame (default: 1)")
    detect_parser.add_argument("--device", default="cuda",
                              help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.command == "create-prototypes":
        return create_prototypes(args)
    elif args.command == "detect":
        return detect_objects(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
