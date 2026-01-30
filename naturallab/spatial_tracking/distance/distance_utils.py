import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import logging



def analyze_track_attachment(floor_positions_file, output_file, attachment_threshold=1500):
    """
    Analyze attachment between tracks by measuring distance distributions
    
    Args:
        floor_positions_file: Path to the enhanced_floor_positions.csv file
        output_file: Path to output the attachment analysis results
        attachment_threshold: Distance threshold in mm to consider tracks as attached (default: 1500mm)
    """
    close_threshold = attachment_threshold
    
    print(f"Loading floor position data from {floor_positions_file}")
    # Load the floor positions data
    df = pd.read_csv(floor_positions_file)
    
    # Get unique track IDs
    track_ids = df['track_id'].unique()
    
    # Dictionary to store distances between each pair of tracks
    track_distances = defaultdict(list)
    
    # Dictionary to store when tracks are "attached" (within threshold)
    attachment_frames = defaultdict(int)
    attachment_durations = defaultdict(list)
    
    # Configure attachment thresholds (in mm)
    close_threshold = 1500  # 1.5 meters
    
    # Current attachment state for each pair
    current_attachment = {}
    attachment_start = {}
    
    # Process data frame by frame
    frames = df['frame'].unique()
    frames.sort()
    
    for frame in frames:
        # Get data for this frame
        frame_data = df[df['frame'] == frame]
        
        # For each pair of tracks in this frame
        tracks_in_frame = frame_data['track_id'].unique()
        
        # Skip frames with less than 2 tracks
        if len(tracks_in_frame) < 2:
            continue
        
        # Process each pair of tracks
        for i, track1 in enumerate(tracks_in_frame):
            for track2 in tracks_in_frame[i+1:]:
                pair_key = tuple(sorted([track1, track2]))
                
                # Get positions for both tracks
                track1_data = frame_data[frame_data['track_id'] == track1].iloc[0]
                track2_data = frame_data[frame_data['track_id'] == track2].iloc[0]
                
                # Calculate distance between smoothed positions
                distance = np.sqrt(
                    (track1_data['smoothed_x'] - track2_data['smoothed_x'])**2 +
                    (track1_data['smoothed_z'] - track2_data['smoothed_z'])**2
                )
                
                # Record distance for this pair
                track_distances[pair_key].append(distance)
                
                # Check if tracks are "attached" (within threshold)
                is_attached = distance <= close_threshold
                
                # Track attachment states and durations
                if pair_key not in current_attachment:
                    current_attachment[pair_key] = is_attached
                    if is_attached:
                        attachment_start[pair_key] = frame
                elif current_attachment[pair_key] != is_attached:
                    # State changed
                    if current_attachment[pair_key]:  # Was attached, now detached
                        duration = frame - attachment_start[pair_key]
                        attachment_durations[pair_key].append(duration)
                    elif is_attached:  # Was detached, now attached
                        attachment_start[pair_key] = frame
                    
                    current_attachment[pair_key] = is_attached
                
                # Count frames where tracks are close
                if is_attached:
                    attachment_frames[pair_key] += 1
    
    # Finalize any ongoing attachments at the end
    for pair_key, is_attached in current_attachment.items():
        if is_attached and pair_key in attachment_start:
            duration = frames[-1] - attachment_start[pair_key]
            attachment_durations[pair_key].append(duration)
    
    # Compute statistics for each track pair
    results = []
    
    for pair_key, distances in track_distances.items():
        track1, track2 = pair_key
        
        distances_array = np.array(distances)
        total_frames = len(distances)
        
        # Calculate attachment metrics
        attached_frames = attachment_frames.get(pair_key, 0)
        attachment_percentage = (attached_frames / total_frames) * 100 if total_frames > 0 else 0
        
        # Calculate distance statistics
        avg_distance = np.mean(distances_array)
        min_distance = np.min(distances_array)
        max_distance = np.max(distances_array)
        std_distance = np.std(distances_array)
        
        # Calculate percentiles
        p25 = np.percentile(distances_array, 25)
        p50 = np.percentile(distances_array, 50)  # median
        p75 = np.percentile(distances_array, 75)
        
        # Calculate attachment duration statistics
        durations = attachment_durations.get(pair_key, [0])
        avg_attachment_duration = np.mean(durations) if durations else 0
        max_attachment_duration = np.max(durations) if durations else 0
        attachment_count = len(durations) if durations[0] != 0 else 0
        
        # Add to results
        results.append({
            'track1': track1,
            'track2': track2,
            'frames_analyzed': total_frames,
            'avg_distance_mm': avg_distance,
            'min_distance_mm': min_distance,
            'max_distance_mm': max_distance,
            'std_distance_mm': std_distance,
            'p25_distance_mm': p25,
            'median_distance_mm': p50,
            'p75_distance_mm': p75,
            'attached_frames': attached_frames,
            'attachment_percentage': attachment_percentage,
            'attachment_count': attachment_count,
            'avg_attachment_duration_frames': avg_attachment_duration,
            'max_attachment_duration_frames': max_attachment_duration
        })
    
    # Create DataFrame and save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"Attachment analysis saved to {output_file}")
    
    # Print summary
    if results:
        print("\nATTACHMENT SUMMARY:")
        for result in results:
            print(f"Tracks {result['track1'][:6]} and {result['track2'][:6]}:")
            print(f"  Average distance: {result['avg_distance_mm']:.2f}mm")
            print(f"  Attachment percentage: {result['attachment_percentage']:.2f}%")
            print(f"  Attachment count: {result['attachment_count']}")
            print(f"  Average attachment duration: {result['avg_attachment_duration_frames']:.1f} frames")
    
    return results


def visualize_attachment(floor_positions_file, output_dir, attachment_threshold=1500):
    """
    Create visualizations for track attachment analysis
    
    Args:
        floor_positions_file: Path to the enhanced_floor_positions.csv file
        output_dir: Directory to save visualization plots
        attachment_threshold: Distance threshold in mm to consider tracks as attached (default: 1500mm)
    """
    threshold_meters = attachment_threshold / 1000
    
    print(f"Generating attachment visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the floor positions data
    df = pd.read_csv(floor_positions_file)
    
    # Get unique track IDs and frames
    track_ids = df['track_id'].unique()
    frames = df['frame'].unique()
    frames.sort()
    
    # Create track pairs
    track_pairs = []
    for i, track1 in enumerate(track_ids):
        for track2 in track_ids[i+1:]:
            track_pairs.append((track1, track2))
    
    # Dictionary to store distances by frame
    distances_by_frame = {pair: [] for pair in track_pairs}
    frames_by_pair = {pair: [] for pair in track_pairs}
    
    # Process each frame
    for frame in frames:
        frame_data = df[df['frame'] == frame]
        
        # For each pair of tracks
        for track1, track2 in track_pairs:
            # Check if both tracks exist in this frame
            track1_data = frame_data[frame_data['track_id'] == track1]
            track2_data = frame_data[frame_data['track_id'] == track2]
            
            if len(track1_data) > 0 and len(track2_data) > 0:
                # Calculate distance between smoothed positions
                distance = np.sqrt(
                    (track1_data.iloc[0]['smoothed_x'] - track2_data.iloc[0]['smoothed_x'])**2 +
                    (track1_data.iloc[0]['smoothed_z'] - track2_data.iloc[0]['smoothed_z'])**2
                )
                
                # Store distance and frame
                distances_by_frame[(track1, track2)].append(distance)
                frames_by_pair[(track1, track2)].append(frame)
    
    # Generate plots
    for pair, distances in distances_by_frame.items():
        if not distances:
            continue
            
        track1, track2 = pair
        pair_frames = frames_by_pair[pair]
        
        # Convert to numpy arrays
        distances_array = np.array(distances)
        frames_array = np.array(pair_frames)
        
        # Create plot for this pair
        plt.figure(figsize=(12, 6))
        
        # Plot distance over time
        plt.plot(frames_array, distances_array / 1000, 'b-', linewidth=2)
        
        # Add threshold line
        plt.axhline(y=threshold_meters, color='r', linestyle='--', 
                label=f'{threshold_meters}m threshold')
        
        # Add annotations
        plt.fill_between(frames_array, 0, threshold_meters, 
                    where=distances_array/1000 <= threshold_meters, 
                    color='green', alpha=0.3, label='Attached')
        
        # Set labels and title
        plt.xlabel('Frame')
        plt.ylabel('Distance (meters)')
        plt.title(f'Distance Between Tracks {track1[:6]} and {track2[:6]} Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Calculate statistics for this pair
        avg_distance = np.mean(distances_array) / 1000  # Convert to meters
        attached_percentage = np.sum(distances_array/1000 <= threshold_meters) / len(distances_array) * 100
        
        # Add text box with statistics
        stats_text = (
            f'Statistics:\n'
            f'Average distance: {avg_distance:.2f}m\n'
            f'Time attached (<1.5m): {attached_percentage:.1f}%\n'
            f'Frames analyzed: {len(distances_array)}'
        )
        plt.figtext(0.15, 0.15, stats_text, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        plot_filename = os.path.join(output_dir, f'attachment_{track1[:6]}_{track2[:6]}.png')
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        
        # Create distance distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(distances_array/1000, bins=30, alpha=0.7, color='blue')
        plt.axvline(x=threshold_meters, color='r', linestyle='--', 
                label=f'{threshold_meters}m threshold')
        plt.axvline(x=avg_distance, color='g', linestyle='-', label=f'Mean: {avg_distance:.2f}m')
        
        # Set labels and title
        plt.xlabel('Distance (meters)')
        plt.ylabel('Frequency')
        plt.title(f'Distance Distribution Between Tracks {track1[:6]} and {track2[:6]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save histogram
        hist_filename = os.path.join(output_dir, f'distance_hist_{track1[:6]}_{track2[:6]}.png')
        plt.savefig(hist_filename, dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    return True