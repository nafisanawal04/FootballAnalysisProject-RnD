"""
Enhanced tracker that integrates SAM2 with RFDETR-SEG for improved tracking.
Maintains compatibility with existing RFDETR-SEG format while adding SAM2 capabilities.
"""

import sys
sys.path.append('../')

import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
from typing import List, Dict, Any, Optional, Tuple
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from detectors.rfdetr_seg_detector import RFDETRSegDetector
from trackers.sam2_tracker import SAM2Tracker

class EnhancedTracker:
    """
    Enhanced tracker that combines RFDETR-SEG detection with SAM2 tracking.
    """
    
    def __init__(self, api_key: str, workspace: str, project: str, version: int, 
                 sam2_checkpoint_path: str = None):
        """
        Initialize enhanced tracker.
        
        Args:
            api_key: Roboflow API key for RFDETR-SEG
            workspace: Roboflow workspace name
            project: Roboflow project name
            version: Model version number
            sam2_checkpoint_path: Path to SAM2 checkpoint (optional)
        """
        # Initialize RFDETR-SEG detector
        self.detector = RFDETRSegDetector(api_key, workspace, project, version)
        
        # Initialize SAM2 tracker
        self.sam2_tracker = SAM2Tracker(sam2_checkpoint_path)
        
        # Fallback to ByteTrack if SAM2 not available
        if not self.sam2_tracker.sam2_video_predictor:
            print("Using ByteTrack as fallback (SAM2 not available)")
            self.tracker = sv.ByteTrack()
            self.use_sam2 = False
        else:
            self.use_sam2 = True
    
    def detect_frames(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect objects in frames using RFDETR-SEG.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection dictionaries
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            predictions = self.detector.predict_frames_batch(batch_frames, confidence=0.1)
            detections += predictions
        return detections
    
    def get_object_tracks(self, frames: List[np.ndarray], read_from_stub: bool = False, 
                         stub_path: str = None) -> Dict[str, List]:
        """
        Get object tracks using enhanced tracking.
        
        Args:
            frames: List of video frames
            read_from_stub: Whether to read from stub file
            stub_path: Path to stub file
            
        Returns:
            Dictionary containing tracking results
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        # Get detections from RFDETR-SEG
        detections = self.detect_frames(frames)
        
        # Convert to supervision format
        detections_list = []
        for detection in detections:
            supervision_detection = self.detector.convert_to_supervision_format(detection)
            detections_list.append(supervision_detection)
        
        # Use SAM2 tracking if available, otherwise fallback to ByteTrack
        if self.use_sam2:
            tracks = self.sam2_tracker.track_with_sam2(frames, detections_list)
        else:
            tracks = self._track_with_bytetrack(frames, detections_list)
        
        # Save to stub if requested
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        return tracks
    
    def _track_with_bytetrack(self, frames: List[np.ndarray], 
                             detections_list: List[sv.Detections]) -> Dict[str, List]:
        """Fallback tracking using ByteTrack."""
        tracks = {
            "players": [],
            "ball": [],
            "referees": []
        }
        
        for frame_num, detection in enumerate(detections_list):
            # Handle RFDETR-SEG format
            cls_names_inv = self.detector.class_names_inv
            
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection)
            
            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})
            
            # Process tracked detections (players and goalkeepers)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                # Check if it's a player or goalkeeper
                if cls_id in [cls_names_inv.get('player', 2), cls_names_inv.get('goalkeeper', 1)]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox, "class_id": cls_id}
                # Check if it's a referee
                elif cls_id == cls_names_inv.get('referee', 3):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox, "class_id": cls_id}
            
            # Process untracked detections (ball)
            for frame_detection in detection:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv.get('ball', 0):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        
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
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def draw_ellipse(self, frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int], 
                    track_id: Optional[int] = None) -> np.ndarray:
        """Draw ellipse annotation for player."""
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y2 - rectangle_height//2) + 15
            y2_rect = (y2 + rectangle_height//2) + 15
            
            cv2.rectangle(frame,
                         (int(x1_rect), int(y1_rect)),
                         (int(x2_rect), int(y2_rect)),
                         color,
                         cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame
    
    def draw_triangle(self, frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int]) -> np.ndarray:
        """Draw triangle annotation for ball."""
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        
        return frame
    
    def draw_annotations(self, video_frames: List[np.ndarray], tracks: Dict, 
                        team_ball_control: np.ndarray, goalkeeper_saves: List[Dict] = None) -> List[np.ndarray]:
        """Draw annotations on video frames."""
        output_video_frames = []
        total_frames = len(video_frames)
        
        print(f"Processing {total_frames} frames for annotation...")
        
        for frame_num, frame in enumerate(video_frames):
            try:
                # Progress indicator
                if frame_num % 100 == 0:
                    print(f"  Processing frame {frame_num}/{total_frames}...")
                
                frame = frame.copy()
                
                player_dict = tracks["players"][frame_num]
                ball_dict = tracks["ball"][frame_num]
                referee_dict = tracks["referees"][frame_num]
                
                # Draw Players and Goalkeepers
                for track_id, player in player_dict.items():
                    class_id = player.get("class_id", 2)  # Default to player
                    
                    # Determine color based on class
                    if class_id == 1:  # Goalkeeper
                        color = (0, 0, 0)  # Black for goalkeeper
                    else:  # Regular player
                        # Get team color for the player
                        team_color = player.get("team_color", (255, 0, 0))
                        if isinstance(team_color, np.ndarray):
                            color = tuple(map(int, team_color))
                        else:
                            color = team_color
                    
                    frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                    
                    # Draw pass accuracy if available
                    if 'pass_accuracy' in player:
                        accuracy = player.get('pass_accuracy', 0)
                        accuracy_text = f"{accuracy:.0f}%"
                        x_center, _ = get_center_of_bbox(player["bbox"])
                        y_pos = int(player["bbox"][3]) + 35
                        
                        # Draw background rectangle for text
                        text_size = cv2.getTextSize(accuracy_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        cv2.rectangle(frame, 
                                    (x_center - text_size[0]//2 - 2, y_pos - 12),
                                    (x_center + text_size[0]//2 + 2, y_pos + 2),
                                    (255, 255, 255), -1)
                        
                        cv2.putText(frame, accuracy_text,
                                    (x_center - text_size[0]//2, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    
                    # Draw ball possession indicator
                    if player.get('has_ball', False):
                        frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))
                        
                        # Draw SAVE! annotation if goalkeeper has the ball
                        if class_id == 1 and goalkeeper_saves is not None:
                            # Check if this frame has a save event for this goalkeeper
                            for save in goalkeeper_saves:
                                if save['frame_num'] == frame_num and save['goalkeeper_id'] == track_id:
                                    # Draw "SAVE!" text above the goalkeeper
                                    x_center, _ = get_center_of_bbox(player["bbox"])
                                    y_pos = int(player["bbox"][1]) - 20  # Above the player
                                    
                                    save_text = "SAVE!"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 1.0
                                    thickness = 2
                                    
                                    # Get text size for background
                                    text_size = cv2.getTextSize(save_text, font, font_scale, thickness)[0]
                                    
                                    # Draw bright yellow background rectangle
                                    padding = 8
                                    cv2.rectangle(frame,
                                                (x_center - text_size[0]//2 - padding, y_pos - text_size[1] - padding),
                                                (x_center + text_size[0]//2 + padding, y_pos + padding),
                                                (0, 255, 255), -1)  # Bright yellow
                                    
                                    # Draw black border
                                    cv2.rectangle(frame,
                                                (x_center - text_size[0]//2 - padding, y_pos - text_size[1] - padding),
                                                (x_center + text_size[0]//2 + padding, y_pos + padding),
                                                (0, 0, 0), 2)
                                    
                                    # Draw text in red
                                    cv2.putText(frame, save_text,
                                              (x_center - text_size[0]//2, y_pos),
                                              font, font_scale, (0, 0, 255), thickness)
                                    break

                
                # Draw Referees
                for track_id, referee in referee_dict.items():
                    color = (0, 255, 255)  # Yellow for referee
                    frame = self.draw_ellipse(frame, referee["bbox"], color, track_id)
                
                # Draw ball
                for track_id, ball in ball_dict.items():
                    frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))
                
                output_video_frames.append(frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                import traceback
                traceback.print_exc()
                # Append the original frame without annotations to continue processing
                output_video_frames.append(frame.copy())
        
        print(f"Completed processing all {total_frames} frames.")
        return output_video_frames
