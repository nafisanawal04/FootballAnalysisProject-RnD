import numpy as np
from collections import deque
from utils import measure_distance, get_center_of_bbox

class ImprovedBallAcquisitionDetector:
    """
    Improved ball acquisition detector with temporal consistency and better logic.
    """

    def __init__(self):
        self.possession_threshold = 25  # Increased threshold for better detection
        self.min_possession_frames = 3  # Minimum frames for possession confirmation
        self.max_possession_gap = 5    # Maximum frames between possession events
        self.ball_velocity_threshold = 2.0  # Minimum ball velocity for possession change
        
        # Track possession history for temporal consistency
        self.possession_history = deque(maxlen=10)
        self.ball_trajectory = deque(maxlen=5)
        self.player_positions = {}

    def get_bottom_edge_points(self, player_bbox):
        """Get the bottom edge points of a player's bounding box (foot area)."""
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        
        # Return left, middle, and right points of the bottom edge
        return [
            (x1, y2),                      # bottom left
            (x1 + width//2, y2),          # bottom center  
            (x2, y2)                       # bottom right
        ]

    def calculate_ball_velocity(self, ball_positions):
        """Calculate ball velocity from recent positions."""
        if len(ball_positions) < 2:
            return 0.0
        
        recent_positions = list(ball_positions)[-3:]  # Last 3 positions
        if len(recent_positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(recent_positions)):
            distance = measure_distance(recent_positions[i-1], recent_positions[i])
            total_distance += distance
        
        return total_distance / (len(recent_positions) - 1)

    def is_ball_moving_towards_player(self, ball_center, player_bbox, ball_velocity):
        """Check if ball is moving towards the player."""
        if ball_velocity < self.ball_velocity_threshold:
            return True  # If ball is slow, consider it as potentially possessed
        
        # Get player center
        player_center = get_center_of_bbox(player_bbox)
        
        # Calculate direction from ball to player
        dx = player_center[0] - ball_center[0]
        dy = player_center[1] - ball_center[1]
        
        # If ball is very close, consider it possessed
        distance = measure_distance(ball_center, player_center)
        if distance < self.possession_threshold:
            return True
        
        return False

    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox, frame_num):
        """Find the best candidate for ball possession with improved logic."""
        candidates = []
        
        # Update ball trajectory
        self.ball_trajectory.append(ball_center)
        
        # Calculate ball velocity
        ball_velocity = self.calculate_ball_velocity(self.ball_trajectory)
        
        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get('bbox', [])
            if not player_bbox:
                continue
            
            # Calculate distance to ball
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)
            
            # Check if ball is moving towards player
            is_moving_towards = self.is_ball_moving_towards_player(ball_center, player_bbox, ball_velocity)
            
            # Calculate possession score (lower is better)
            possession_score = min_distance
            
            # Bonus for players the ball is moving towards
            if is_moving_towards:
                possession_score *= 0.8
            
            # Only consider players within reasonable distance
            if min_distance < self.possession_threshold * 1.5:  # More lenient threshold
                candidates.append((player_id, possession_score, min_distance))
        
        if not candidates:
            return -1
        
        # Sort by possession score (lower is better)
        candidates.sort(key=lambda x: x[1])
        best_candidate = candidates[0]
        
        # Only assign possession if distance is reasonable
        if best_candidate[2] < self.possession_threshold:
            return best_candidate[0]
        
        return -1

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """Find minimum distance from ball to player's foot area."""
        bottom_points = self.get_bottom_edge_points(player_bbox)
        return min(measure_distance(ball_center, point) for point in bottom_points)

    def apply_temporal_consistency(self, possession_list):
        """Apply temporal consistency to possession detection."""
        if len(possession_list) < 3:
            return possession_list
        
        # Smooth possession changes
        smoothed_possession = possession_list.copy()
        
        for i in range(1, len(possession_list) - 1):
            current = possession_list[i]
            prev = possession_list[i-1]
            next_poss = possession_list[i+1]
            
            # If current frame has no possession but adjacent frames have same player
            if current == -1 and prev == next_poss and prev != -1:
                smoothed_possession[i] = prev
            
            # If possession changes too frequently, keep previous possession
            elif current != prev and current != -1:
                # Check if this is a brief change (1-2 frames)
                if i < len(possession_list) - 2:
                    if possession_list[i+1] == prev and possession_list[i+2] == prev:
                        smoothed_possession[i] = prev
        
        return smoothed_possession

    def detect_ball_possession(self, player_tracks, ball_tracks):
        """Detect ball possession with improved logic and temporal consistency."""
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        
        print(f"\nDetecting ball possession for {num_frames} frames...")
        
        for frame_num in range(num_frames):
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue
                
            ball_bbox = ball_info.get('bbox', [])
            if not ball_bbox:
                continue
                
            ball_center = get_center_of_bbox(ball_bbox)
            
            best_player_id = self.find_best_candidate_for_possession(
                ball_center, 
                player_tracks[frame_num], 
                ball_bbox,
                frame_num
            )
            
            if best_player_id != -1:
                possession_list[frame_num] = best_player_id
        
        # Apply temporal consistency
        possession_list = self.apply_temporal_consistency(possession_list)
        
        # Print possession statistics
        possession_changes = sum(1 for i in range(1, len(possession_list)) 
                               if possession_list[i] != possession_list[i-1])
        total_possessions = sum(1 for p in possession_list if p != -1)
        
        print(f"Ball possession detection completed:")
        print(f"  - Total frames with possession: {total_possessions}")
        print(f"  - Possession changes: {possession_changes}")
        
        return possession_list

    def get_possession_statistics(self, possession_list, player_assignment):
        """Get detailed possession statistics."""
        stats = {
            'total_frames': len(possession_list),
            'possession_frames': sum(1 for p in possession_list if p != -1),
            'possession_percentage': 0,
            'team_possession': {1: 0, 2: 0},
            'player_possession': {}
        }
        
        if stats['total_frames'] > 0:
            stats['possession_percentage'] = (stats['possession_frames'] / stats['total_frames']) * 100
        
        # Calculate team possession
        for frame_num, player_id in enumerate(possession_list):
            if player_id != -1 and frame_num < len(player_assignment):
                team = player_assignment[frame_num].get(player_id, -1)
                if team in [1, 2]:
                    stats['team_possession'][team] += 1
                    stats['player_possession'][player_id] = stats['player_possession'].get(player_id, 0) + 1
        
        return stats
