import sys 
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class BallAcquisitionDetector:
    """
    Detects ball acquisition by players in a football game.

    This class determines which player is most likely in possession of the ball
    by measuring the distance from the ball to the bottom edge (foot area) of 
    each player's bounding box.
    """

    def __init__(self):
        """
        Initialize the BallAcquisitionDetector with default thresholds.

        Attributes:
            possession_threshold (int): Maximum distance (in pixels) at which
                a player can be considered to have the ball.
        """
        self.possession_threshold = 15
        
    def get_bottom_edge_points(self, player_bbox):
        """
        Get the bottom edge points of a player's bounding box (foot area).

        Args:
            player_bbox (tuple or list): A bounding box in the format (x1, y1, x2, y2).

        Returns:
            list of tuple: A list of (x, y) coordinates representing points
            along the bottom edge of the bounding box.
        """
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        
        # Return left, middle, and right points of the bottom edge
        return [
            (x1, y2),                      # bottom left
            (x1 + width//2, y2),          # bottom center  
            (x2, y2)                       # bottom right
        ]

    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        """
        Compute the minimum distance from the ball to the bottom edge (foot area)
        of a player's bounding box.

        Args:
            ball_center (tuple): (x, y) coordinates of the center of the ball.
            player_bbox (tuple): A bounding box (x1, y1, x2, y2) for the player.

        Returns:
            float: The smallest distance from the ball center to
            any point on the bottom edge of the player's bounding box.
        """
        bottom_points = self.get_bottom_edge_points(player_bbox)
        return min(measure_distance(ball_center, point) for point in bottom_points)

    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):
        """
        Determine which player in a single frame is most likely to have the ball
        based on distance to the bottom edge of their bounding box.

        Args:
            ball_center (tuple): (x, y) coordinates of the ball center.
            player_tracks_frame (dict): Mapping from player_id to info about that player,
                including a 'bbox' key with (x1, y1, x2, y2).
            ball_bbox (tuple): Bounding box for the ball (x1, y1, x2, y2).

        Returns:
            int: (best_player_id), or (-1) if none found.
        """
        candidates = []
        
        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get('bbox', [])
            if not player_bbox:
                continue
                
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)
            
            # Only consider players within the possession threshold
            if min_distance < self.possession_threshold:
                candidates.append((player_id, min_distance))

        # Return the player with the minimum distance to the ball
        if candidates:
            best_candidate = min(candidates, key=lambda x: x[1])
            return best_candidate[0]
                
        return -1
    
    def detect_ball_possession(self, player_tracks, ball_tracks):
        """
        Detect which player has the ball in each frame based on distance to bottom edge.

        Loops through all frames, looks up ball bounding boxes and player bounding boxes,
        and uses find_best_candidate_for_possession to determine who has the ball.
        Simple frame-by-frame detection without consecutive frame requirements.

        Args:
            player_tracks (list): A list of dictionaries for each frame, where each dictionary
                maps player_id to player information including 'bbox'.
            ball_tracks (list): A list of dictionaries for each frame, where each dictionary
                maps ball_id to ball information including 'bbox'.

        Returns:
            list: A list of length num_frames with the player_id who has possession,
            or -1 if no one is determined to have possession in that frame.
        """
        num_frames = len(ball_tracks)
        possession_list = [-1] * num_frames
        
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
                ball_bbox
            )

            if best_player_id != -1:
                possession_list[frame_num] = best_player_id
    
        return possession_list
