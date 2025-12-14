import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path, "../"))
from utils import get_foot_position, measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 400
        self.height = 200

        # Football field dimensions in meters (standard FIFA field)
        self.actual_width_in_meters = 105  # 105m length
        self.actual_height_in_meters = 68   # 68m width

        # Football field keypoints based on the trained model configuration
        # 35 points total: 15 orange (left), 5 pink (center), 15 blue (right)
        # Serialized from top to bottom, left to right
        
        # Calculate field dimensions and positions
        field_width = self.width
        field_height = self.height
        center_x = field_width // 2
        center_y = field_height // 2
        
        # Goal area dimensions (approximate proportions)
        goal_area_width = int(field_width * 0.15)  # 15% of field width
        penalty_area_width = int(field_width * 0.25)  # 25% of field width
        goal_area_height = int(field_height * 0.3)  # 30% of field height
        penalty_area_height = int(field_height * 0.4)  # 40% of field height
        
        self.key_points = [
            # LEFT GOAL AREA (Orange dots - 15 points)
            # Top row (left to right)
            (0, 0),  # 1: Top-left corner of field
            (goal_area_width, 0),  # 2: Top-right of goal area
            (penalty_area_width, 0),  # 3: Top-right of penalty area
            
            # Middle rows (left to right)
            (0, center_y - goal_area_height//2),  # 4: Left goal area top
            (goal_area_width, center_y - goal_area_height//2),  # 5: Goal area top-right
            (penalty_area_width, center_y - penalty_area_height//2),  # 6: Penalty area top-right
            
            (0, center_y),  # 7: Left goal area center
            (goal_area_width, center_y),  # 8: Goal area center
            (penalty_area_width, center_y),  # 9: Penalty area center
            
            (0, center_y + goal_area_height//2),  # 10: Left goal area bottom
            (goal_area_width, center_y + goal_area_height//2),  # 11: Goal area bottom-right
            (penalty_area_width, center_y + penalty_area_height//2),  # 12: Penalty area bottom-right
            
            # Bottom row (left to right)
            (0, field_height),  # 13: Bottom-left corner of field
            (goal_area_width, field_height),  # 14: Bottom-right of goal area
            (penalty_area_width, field_height),  # 15: Bottom-right of penalty area
            
            # CENTER AREA (Pink dots - 5 points)
            (center_x, 0),  # 16: Top center
            (center_x - 20, center_y),  # 17: Center circle left
            (center_x, center_y),  # 18: Center point
            (center_x + 20, center_y),  # 19: Center circle right
            (center_x, field_height),  # 20: Bottom center
            
            # RIGHT GOAL AREA (Blue dots - 15 points)
            # Top row (left to right)
            (field_width - penalty_area_width, 0),  # 21: Top-left of penalty area
            (field_width - goal_area_width, 0),  # 22: Top-left of goal area
            (field_width, 0),  # 23: Top-right corner of field
            
            # Middle rows (left to right)
            (field_width - penalty_area_width, center_y - penalty_area_height//2),  # 24: Penalty area top-left
            (field_width - goal_area_width, center_y - goal_area_height//2),  # 25: Goal area top-left
            (field_width, center_y - goal_area_height//2),  # 26: Right goal area top
            
            (field_width - penalty_area_width, center_y),  # 27: Penalty area center
            (field_width - goal_area_width, center_y),  # 28: Goal area center
            (field_width, center_y),  # 29: Right goal area center
            
            (field_width - penalty_area_width, center_y + penalty_area_height//2),  # 30: Penalty area bottom-left
            (field_width - goal_area_width, center_y + goal_area_height//2),  # 31: Goal area bottom-left
            (field_width, center_y + goal_area_height//2),  # 32: Right goal area bottom
            
            # Bottom row (left to right)
            (field_width - penalty_area_width, field_height),  # 33: Bottom-left of penalty area
            (field_width - goal_area_width, field_height),  # 34: Bottom-left of goal area
            (field_width, field_height),  # 35: Bottom-right corner of field
        ]

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.
        
        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.
        
        Returns:
            List[bool]: A list indicating whether each frame's keypoints are valid.
        """

        keypoints_list = deepcopy(keypoints_list)
        num_reference_points = len(self.key_points)

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            # Check if keypoints exist and have data
            if frame_keypoints is None or not hasattr(frame_keypoints, 'xy') or frame_keypoints.xy is None:
                continue
                
            try:
                frame_keypoints_list = frame_keypoints.xy.tolist()
                if len(frame_keypoints_list) == 0:
                    continue
                frame_keypoints = frame_keypoints_list[0]
            except (IndexError, AttributeError):
                continue
            
            # Get indices of detected keypoints (not (0, 0))
            # Add bounds checking to ensure index is within reference points range
            detected_indices = [i for i, kp in enumerate(frame_keypoints) 
                              if kp[0] > 0 and kp[1] > 0 and i < num_reference_points]
            
            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue
            
            invalid_keypoints = []
            # Validate each detected keypoint
            for i in detected_indices:
                # Skip if this is (0, 0)
                if frame_keypoints[i][0] == 0 and frame_keypoints[i][1] == 0:
                    continue

                # Choose two other random detected keypoints
                other_indices = [idx for idx in detected_indices 
                               if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                # Take first two other indices for simplicity
                j, k = other_indices[0], other_indices[1]

                # Calculate distances between detected keypoints
                d_ij = measure_distance(frame_keypoints[i], frame_keypoints[j])
                d_ik = measure_distance(frame_keypoints[i], frame_keypoints[k])
                
                # Calculate distances between corresponding tactical keypoints
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                # Calculate and compare proportions with 50% error margin
                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    if error > 0.8:  # 80% error margin                        
                        keypoints_list[frame_idx].xy[0][i] *= 0
                        keypoints_list[frame_idx].xyn[0][i] *= 0
                        invalid_keypoints.append(i)
            
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        Uses a simplified approach that works with any number of detected keypoints.
        
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        
        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            # Check if keypoints exist and have data
            if frame_keypoints is None or not hasattr(frame_keypoints, 'xy') or frame_keypoints.xy is None:
                tactical_player_positions.append(tactical_positions)
                continue
                
            try:
                frame_keypoints_list = frame_keypoints.xy.tolist()
                if len(frame_keypoints_list) == 0:
                    tactical_player_positions.append(tactical_positions)
                    continue
                frame_keypoints = frame_keypoints_list[0]
            except (IndexError, AttributeError):
                tactical_player_positions.append(tactical_positions)
                continue

            # Skip frames with insufficient keypoints
            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Get detected keypoints for this frame
            detected_keypoints = frame_keypoints
            
            # Filter valid keypoints (non-zero coordinates)
            valid_keypoints = [(kp[0], kp[1]) for kp in detected_keypoints if kp[0] > 0 and kp[1] > 0]
            
            if len(valid_keypoints) < 4:
                tactical_player_positions.append(tactical_positions)
                continue
            
            try:
                # Use a simple approach: map detected keypoints to tactical view corners
                # Find the bounding box of detected keypoints
                min_x = min(kp[0] for kp in valid_keypoints)
                max_x = max(kp[0] for kp in valid_keypoints)
                min_y = min(kp[1] for kp in valid_keypoints)
                max_y = max(kp[1] for kp in valid_keypoints)
                
                # Create source points (corners of detected field)
                source_points = np.array([
                    [min_x, min_y],  # Top-left
                    [max_x, min_y],  # Top-right
                    [min_x, max_y],  # Bottom-left
                    [max_x, max_y]   # Bottom-right
                ], dtype=np.float32)
                
                # Create target points (corners of tactical view)
                target_points = np.array([
                    [0, 0],                    # Top-left
                    [self.width, 0],           # Top-right
                    [0, self.height],          # Bottom-left
                    [self.width, self.height]  # Bottom-right
                ], dtype=np.float32)
                
                # Create homography transformer
                homography = Homography(source_points, target_points)
                
                # Transform each player's position
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    # Use bottom center of bounding box as player position
                    player_position = np.array([get_foot_position(bbox)])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(player_position)

                    # If tactical position is within reasonable bounds, include it
                    if (tactical_position[0][0] >= -50 and tactical_position[0][0] <= self.width + 50 and 
                        tactical_position[0][1] >= -50 and tactical_position[0][1] <= self.height + 50):
                        tactical_positions[player_id] = tactical_position[0].tolist()
                    
            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                print(f"Homography failed for frame {frame_idx}: {e}")
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions
