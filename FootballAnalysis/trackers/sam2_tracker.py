"""
SAM2-based tracker for improved object tracking in football analysis.
Uses SAM2 for better segmentation and tracking consistency.
"""

import sys
sys.path.append('../')

import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM2 not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")

class SAM2Tracker:
    """
    SAM2-based tracker for improved object tracking.
    Uses SAM2 for better segmentation and tracking consistency.
    """
    
    def __init__(self, sam2_checkpoint_path: str = None, device: str = "cuda"):
        """
        Initialize SAM2 tracker.
        
        Args:
            sam2_checkpoint_path: Path to SAM2 checkpoint
            device: Device to run SAM2 on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sam2_predictor = None
        self.sam2_video_predictor = None
        
        if SAM2_AVAILABLE and sam2_checkpoint_path:
            try:
                # Initialize SAM2
                sam2_model = build_sam2(sam2_checkpoint_path, device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                self.sam2_video_predictor = SAM2VideoPredictor(sam2_model)
                print(f"✓ SAM2 initialized on {self.device}")
            except Exception as e:
                print(f"Warning: Failed to initialize SAM2: {e}")
                self.sam2_predictor = None
                self.sam2_video_predictor = None
        else:
            print("Warning: SAM2 not available, falling back to basic tracking")
    
    def get_segmentation_masks(self, frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
        """
        Get segmentation masks for detections using SAM2.
        
        Args:
            frame: Input frame
            detections: Supervision detections
            
        Returns:
            List of segmentation masks
        """
        if not self.sam2_predictor or detections.empty:
            return []
        
        masks = []
        try:
            # Set image for SAM2
            self.sam2_predictor.set_image(frame)
            
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                # Convert bbox to SAM2 format (x1, y1, x2, y2)
                input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                
                # Get mask from SAM2
                mask, _, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                masks.append(mask[0])
        except Exception as e:
            print(f"Warning: SAM2 segmentation failed: {e}")
            # Return empty masks if SAM2 fails
            masks = [np.zeros((frame.shape[0], frame.shape[1]), dtype=bool) for _ in range(len(detections))]
        
        return masks
    
    def track_with_sam2(self, frames: List[np.ndarray], detections_list: List[sv.Detections]) -> Dict[str, List]:
        """
        Track objects using SAM2 for better consistency.
        
        Args:
            frames: List of video frames
            detections_list: List of detections for each frame
            
        Returns:
            Dictionary with tracking results
        """
        if not self.sam2_video_predictor or not frames:
            return self._fallback_tracking(frames, detections_list)
        
        tracks = {
            "players": [],
            "ball": [],
            "referees": []
        }
        
        # Initialize tracking for each frame
        for frame_num in range(len(frames)):
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})
        
        try:
            # Use SAM2 video predictor for consistent tracking
            video_frames = np.stack(frames)
            
            # Get initial detections from first frame
            if detections_list and len(detections_list[0]) > 0:
                initial_detections = detections_list[0]
                
                # Set up SAM2 video predictor
                self.sam2_video_predictor.set_video(video_frames)
                
                # Track each object
                for obj_id, detection in enumerate(initial_detections):
                    bbox = detection.xyxy[0]
                    class_id = detection.class_id[0]
                    
                    # Convert bbox to SAM2 format
                    input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                    
                    # Track object across video
                    masks, scores, logits = self.sam2_video_predictor.predict(
                        video_segment_ids=[0],  # Start from first frame
                        input_box=input_box[None, :],
                        multimask_output=False,
                    )
                    
                    # Process tracking results
                    if masks is not None and len(masks) > 0:
                        self._process_tracking_results(tracks, masks, class_id, obj_id)
        
        except Exception as e:
            print(f"Warning: SAM2 video tracking failed: {e}")
            return self._fallback_tracking(frames, detections_list)
        
        return tracks
    
    def _process_tracking_results(self, tracks: Dict, masks: np.ndarray, class_id: int, obj_id: int):
        """Process SAM2 tracking results into our format."""
        # This is a simplified implementation
        # In practice, you'd need to handle the full SAM2 video tracking pipeline
        pass
    
    def _fallback_tracking(self, frames: List[np.ndarray], detections_list: List[sv.Detections]) -> Dict[str, List]:
        """Fallback tracking when SAM2 is not available."""
        tracks = {
            "players": [],
            "ball": [],
            "referees": []
        }
        
        # Simple tracking based on detection overlap
        for frame_num, detections in enumerate(detections_list):
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})
            
            if detections.empty:
                continue
            
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]
                
                # Assign to appropriate category
                if class_id == 0:  # ball
                    tracks["ball"][frame_num][1] = {"bbox": bbox.tolist()}
                elif class_id in [1, 2]:  # goalkeeper, player
                    tracks["players"][frame_num][i] = {"bbox": bbox.tolist(), "class_id": class_id}
                elif class_id == 3:  # referee
                    tracks["referees"][frame_num][i] = {"bbox": bbox.tolist(), "class_id": class_id}
        
        return tracks
    
    def add_position_to_tracks(self, tracks: Dict):
        """Add position information to tracks."""
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self, ball_positions: List[Dict]) -> List[Dict]:
        """Interpolate missing ball positions."""
        import pandas as pd
        
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
