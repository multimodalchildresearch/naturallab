#!/usr/bin/env python3
"""
Detect Custom Objects from Reference Images
============================================

Reference-image object detection using OWLv2 + CLIP prototype matching.
No gradient-based training is required: provide photographs of the objects.

Use Cases:
- Inventory tracking with custom products
- Research with specific experimental materials
- Quality control with custom parts
- Wildlife identification
- Art/artifact recognition

Example Usage:
    # Create prototypes from reference images
    python detect_custom_objects.py create-prototypes \\
        --images private-study-data/reference_images/ \\
        --output private-study-data/prototypes.h5
    
    # Detect objects in video frames
    python detect_custom_objects.py detect \\
        --input private-study-data/video.mp4 \\
        --prototypes private-study-data/prototypes.h5 \\
        --output private-study-data/detection-run-01/
    
    # Detect in image folder
    python detect_custom_objects.py detect \\
        --input private-study-data/frames/ \\
        --prototypes private-study-data/prototypes.h5 \\
        --output private-study-data/image-run-01/ \\
        --categories '{"toy": ["toy", "object"], "tool": ["tool", "instrument"]}'
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
)
VIDEO_SUFFIXES = frozenset(
    {
        ".avi",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".webm",
    }
)
BASE_DETECTION_COLUMNS = (
    "x",
    "y",
    "width",
    "height",
    "detection_score",
    "category",
    "match_score",
)
VIDEO_DETECTION_COLUMNS = (
    *BASE_DETECTION_COLUMNS,
    "frame",
    "timestamp",
    "timestamp_seconds",
    "timestamp_ns",
    "timestamp_source",
)


def discover_media_files(directory, suffixes):
    """Discover supported files case-insensitively in natural numeric order."""
    from naturallab.media import natural_path_sort_key

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        ),
        key=natural_path_sort_key,
    )


def classify_detection_input(input_path):
    """Return the supported input kind and any discovered image files."""
    input_path = Path(input_path)
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            return "images", [input_path]
        if suffix in VIDEO_SUFFIXES:
            return "video", []
        raise ValueError(
            f"unsupported input file type: {input_path.suffix or '(none)'}"
        )
    if input_path.is_dir():
        image_files = discover_media_files(input_path, IMAGE_SUFFIXES)
        if not image_files:
            raise ValueError(f"no supported images found in {input_path}")
        return "images", image_files
    raise ValueError(f"input does not exist: {input_path}")


def require_empty_output_directory(output_path):
    """Refuse to mix a new run with files from an earlier configuration."""
    output_path = Path(output_path)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"output path is not a directory: {output_path}")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(
            f"output directory is not empty: {output_path}; choose a new run name"
        )


def annotated_image_filename(image_path, image_index):
    """Return a stable, collision-free preview name for one source image."""
    safe_stem = Path(image_path).stem or "image"
    return f"image_{image_index:06d}_{safe_stem}_detections.jpg"


def write_detection_tables(detections, output_path, media_kind):
    """Write stable CSV schemas, including for valid zero-detection runs."""
    import pandas as pd

    output_path = Path(output_path)
    if media_kind == "video":
        frame = pd.DataFrame(detections, columns=VIDEO_DETECTION_COLUMNS)
    elif media_kind == "images":
        frame = pd.DataFrame(
            detections,
            columns=(*BASE_DETECTION_COLUMNS, "image"),
        )
    else:
        raise ValueError(f"unsupported media kind: {media_kind}")
    frame.to_csv(output_path / "detections.csv", index=False)

    if media_kind == "video":
        summary = frame.groupby("category").agg(
            count=("frame", "count"),
            avg_confidence=("match_score", "mean"),
        )
        summary.to_csv(output_path / "detection_summary.csv")
        return frame, summary
    return frame, None


def parse_category_queries(value):
    """Validate optional JSON search phrases and return one flat query list."""
    if not value:
        return ["object", "thing"]
    try:
        categories = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"--categories is not valid JSON: {error}") from error
    if not isinstance(categories, dict) or not categories or not all(
        isinstance(queries, list)
        and queries
        and all(
            isinstance(query, str) and query.strip()
            for query in queries
        )
        for queries in categories.values()
    ):
        raise ValueError(
            "--categories must map each label to a non-empty list of search "
            "phrases"
        )
    return [query for queries in categories.values() for query in queries]


def resolve_torch_device(requested, torch_module):
    """Resolve ``auto`` and reject unavailable explicitly requested CUDA."""
    if requested == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        ):
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is unavailable")
    return requested


def draw_detection_preview(image, detections):
    """Return a copy of a PIL image with readable detection overlays."""
    from PIL import ImageDraw

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for detection in detections:
        x = int(detection["x"])
        y = int(detection["y"])
        width = int(detection["width"])
        height = int(detection["height"])
        right = x + width
        bottom = y + height
        draw.rectangle((x, y, right, bottom), outline="lime", width=3)
        label = (
            f'{detection["category"]} '
            f'{float(detection["match_score"]):.2f}'
        )
        text_y = max(0, y - 14)
        draw.text((x + 2, text_y), label, fill="lime", stroke_fill="black",
                  stroke_width=2)
    return preview


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
    
    try:
        args.device = resolve_torch_device(args.device, torch)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    # Discover images organized by category
    # Expected structure: images_dir/category_name/image1.jpg
    categories = {}

    category_directories = (
        sorted(
            (path for path in images_path.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        if images_path.is_dir()
        else []
    )
    if not category_directories:
        print("Error: No image categories found.")
        print("Expected structure: images_dir/category_name/image1.jpg")
        return 1

    empty_categories = []
    for category_dir in category_directories:
        image_files = discover_media_files(category_dir, IMAGE_SUFFIXES)
        if image_files:
            categories[category_dir.name] = image_files
        else:
            empty_categories.append(category_dir.name)
    if empty_categories:
        print(
            "Error: Every category folder must contain at least one supported "
            "image. Empty categories: " + ", ".join(empty_categories)
        )
        return 1

    # Load CLIP only after the folder layout has passed validation.
    print("Loading CLIP model...")
    model = AutoModel.from_pretrained(args.model).eval().to(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model)
    
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
                    
                    inputs = {k: v.to(args.device) for k, v in inputs.items()}
                    
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

    failed_categories = sorted(set(categories) - set(embeddings))
    if failed_categories:
        print(
            "Error: No usable reference image could be decoded for: "
            + ", ".join(failed_categories)
        )
        print("No prototype file was written.")
        return 1
    
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
    import h5py
    from PIL import Image
    from tqdm import tqdm
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    prototypes_path = Path(args.prototypes)

    if args.frame_skip < 1:
        print("Error: --frame-skip must be positive")
        return 1
    if args.frame_interval < 1:
        print("Error: --frame-interval must be positive")
        return 1
    if not prototypes_path.is_file():
        print(f"Error: Prototype file does not exist: {prototypes_path}")
        return 1
    try:
        input_kind, image_files = classify_detection_input(input_path)
        require_empty_output_directory(output_path)
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    
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

    if not prototypes:
        print("Error: Prototype file contains no categories")
        return 1
    
    print(f"Loaded {len(prototypes)} prototype categories: {prototype_names}")
    
    # Parse category queries for first stage
    try:
        first_stage_queries = parse_category_queries(args.categories)
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    
    # Load models
    print("Loading detection models...")
    try:
        from naturallab.gaze_analysis.object_detection.owlv2 import OWLv2Detector
        from transformers import AutoModel, AutoImageProcessor
    except ImportError as e:
        print(f"Error importing: {e}")
        return 1
    
    try:
        requested_device = resolve_torch_device(args.device, torch)
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    args.device = requested_device

    # Initialize detector
    owl_detector = OWLv2Detector(device=args.device)
    
    # Load CLIP for second stage
    clip_model = AutoModel.from_pretrained(model_name).eval()
    clip_processor = AutoImageProcessor.from_pretrained(model_name)
    clip_model = clip_model.to(args.device)
    
    # Stack prototypes for matching
    proto_tensor = torch.stack(list(prototypes.values()))
    proto_tensor = proto_tensor / proto_tensor.norm(dim=-1, keepdim=True)
    proto_tensor = proto_tensor.to(args.device)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process input
    if input_kind == "video":
        # Video input
        try:
            process_video(input_path, output_path, owl_detector, clip_model,
                          clip_processor, proto_tensor, prototype_names,
                          first_stage_queries, args)
        except (RuntimeError, ValueError) as error:
            print(f"Error: {error}")
            return 1
    else:
        # Single-image or image-folder input
        all_detections = []
        
        preview_dir = output_path / "annotated_frames"
        if args.save_frames:
            preview_dir.mkdir(parents=True, exist_ok=True)

        for image_index, img_path in enumerate(
            tqdm(image_files, desc="Processing images"),
            start=1,
        ):
            try:
                with Image.open(img_path) as opened_image:
                    source_image = opened_image.convert("RGB").copy()
            except Exception as error:
                print(f"Error: could not decode image {img_path}: {error}")
                return 1
            detections = process_image(
                source_image, owl_detector, clip_model, clip_processor,
                proto_tensor, prototype_names, first_stage_queries, args
            )
            for det in detections:
                det["image"] = img_path.name
            all_detections.extend(detections)
            if args.save_frames:
                preview = draw_detection_preview(source_image, detections)
                preview.save(
                    preview_dir
                    / annotated_image_filename(img_path, image_index),
                    quality=90,
                )

        df, _summary = write_detection_tables(
            all_detections,
            output_path,
            "images",
        )
        print(f"\nSaved {len(df)} detections to detections.csv")
    
    return 0


def process_image(img_path, owl_detector, clip_model, clip_processor, 
                 proto_tensor, prototype_names, queries, args):
    """Process a single image and return detections."""
    import torch
    from PIL import Image
    from torchvision.transforms.functional import crop
    
    if isinstance(img_path, Image.Image):
        img = img_path.convert("RGB")
    else:
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
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            if hasattr(clip_model, "get_image_features"):
                embedding = clip_model.get_image_features(**inputs)
            else:
                outputs = clip_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0]
            
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            # Match to prototypes
            similarities = (embedding @ proto_tensor.T).reshape(-1)
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
    import math
    from PIL import Image
    from tqdm import tqdm

    from naturallab.media import VideoFileSource
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video file: {video_path}")
    reported_total_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = (
        int(reported_total_frames)
        if math.isfinite(reported_total_frames)
        and reported_total_frames > 0
        else 0
    )
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = (
        reported_fps
        if math.isfinite(reported_fps) and reported_fps > 0
        else None
    )
    cap.release()
    
    print(
        "Video: "
        f"{total_frames if total_frames else 'unknown'} frames @ "
        f"{fps if fps is not None else 'unknown'} nominal FPS"
    )
    
    all_detections = []
    processed_frames = 0
    preview_dir = output_path / "annotated_frames"
    if args.save_frames:
        preview_dir.mkdir(parents=True, exist_ok=True)
    
    # Process every Nth frame
    frame_skip = args.frame_skip or 1
    expected_frames = (
        (total_frames + frame_skip - 1) // frame_skip
        if total_frames
        else None
    )
    frame_source = VideoFileSource(video_path, step=frame_skip)

    for packet in tqdm(
        frame_source,
        total=expected_frames,
        desc="Processing video",
    ):
        frame_idx = packet.frame_index
        frame = packet.image
        processed_frames += 1

        # Convert to PIL
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Detect
        detections = process_image(
            img, owl_detector, clip_model, clip_processor,
            proto_tensor, prototype_names, queries, args
        )
        
        for det in detections:
            det["frame"] = frame_idx
            # Keep the legacy column while making its unit and source explicit.
            det["timestamp"] = packet.source_timestamp
            det["timestamp_seconds"] = packet.source_timestamp
            det["timestamp_ns"] = packet.timestamp_ns
            det["timestamp_source"] = packet.metadata.get(
                "timestamp_source"
            )
        
        all_detections.extend(detections)

        if args.save_frames and frame_idx % args.frame_interval == 0:
            preview = draw_detection_preview(img, detections)
            preview.save(
                preview_dir / f"frame_{frame_idx:08d}.jpg",
                quality=90,
            )
    
    if processed_frames == 0:
        raise RuntimeError(f"video contains no decodable frames: {video_path}")

    # Always write result tables, including their headers when no detection
    # passes the configured thresholds.
    df, summary = write_detection_tables(
        all_detections,
        output_path,
        "video",
    )

    print(f"\nSaved {len(df)} detections")
    print("\nDetection summary:")
    print(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Reference-image object detection with prototype matching",
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
    proto_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device (default: auto)",
    )
    
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
    detect_parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated JPEG frames for visual review",
    )
    detect_parser.add_argument(
        "--frame-interval",
        type=int,
        default=100,
        help="For video, save one annotated frame every N source frames",
    )
    detect_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device (default: auto)",
    )
    
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
