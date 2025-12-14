"""
Goalkeeper Save Detector

Detects when goalkeepers make saves by identifying when they have ball possession.
"""

import sys
sys.path.append('../')

class GoalkeeperSaveDetector:
    """
    Detects goalkeeper saves by identifying when goalkeepers have ball possession.
    """
    
    def __init__(self, goalkeeper_class_id=1):
        """
        Initialize the GoalkeeperSaveDetector.
        
        Args:
            goalkeeper_class_id (int): Class ID for goalkeepers (default: 1)
        """
        self.goalkeeper_class_id = goalkeeper_class_id
        self.save_events = []
    
    def detect_saves(self, player_tracks, ball_acquisition):
        """
        Detect goalkeeper saves from player tracks and ball acquisition data.
        
        Args:
            player_tracks (list): List of dictionaries for each frame containing player tracking data
            ball_acquisition (list): List of player IDs who have ball possession per frame
            
        Returns:
            list: List of save events, each containing:
                - frame_num: Frame number where save occurred
                - goalkeeper_id: ID of the goalkeeper making the save
                - class_id: Class ID confirming it's a goalkeeper
        """
        save_events = []
        
        for frame_num in range(len(ball_acquisition)):
            player_with_ball = ball_acquisition[frame_num]
            
            # Skip if no one has the ball
            if player_with_ball == -1:
                continue
            
            # Check if the player with the ball is in the current frame's tracks
            if frame_num >= len(player_tracks):
                continue
                
            frame_players = player_tracks[frame_num]
            
            if player_with_ball in frame_players:
                player_info = frame_players[player_with_ball]
                class_id = player_info.get('class_id', 2)  # Default to player (2)
                
                # Check if it's a goalkeeper
                if class_id == self.goalkeeper_class_id:
                    save_events.append({
                        'frame_num': frame_num,
                        'goalkeeper_id': player_with_ball,
                        'class_id': class_id,
                        'bbox': player_info.get('bbox', [])
                    })
        
        self.save_events = save_events
        return save_events
    
    def get_save_statistics(self, save_events, player_assignment):
        """
        Calculate statistics about goalkeeper saves.
        
        Args:
            save_events (list): List of save events from detect_saves
            player_assignment (list): List of dictionaries mapping player_id to team per frame
            
        Returns:
            dict: Statistics including:
                - total_saves: Total number of save frames
                - saves_by_goalkeeper: Dictionary mapping goalkeeper_id to save count
                - saves_by_team: Dictionary mapping team to save count
        """
        if not save_events:
            return {
                'total_saves': 0,
                'saves_by_goalkeeper': {},
                'saves_by_team': {1: 0, 2: 0}
            }
        
        saves_by_goalkeeper = {}
        saves_by_team = {1: 0, 2: 0}
        
        for save in save_events:
            goalkeeper_id = save['goalkeeper_id']
            frame_num = save['frame_num']
            
            # Count saves per goalkeeper
            if goalkeeper_id not in saves_by_goalkeeper:
                saves_by_goalkeeper[goalkeeper_id] = 0
            saves_by_goalkeeper[goalkeeper_id] += 1
            
            # Count saves per team
            if frame_num < len(player_assignment):
                team = player_assignment[frame_num].get(goalkeeper_id, -1)
                if team in saves_by_team:
                    saves_by_team[team] += 1
        
        return {
            'total_saves': len(save_events),
            'saves_by_goalkeeper': saves_by_goalkeeper,
            'saves_by_team': saves_by_team
        }
    
    def is_save_frame(self, frame_num):
        """
        Check if a specific frame contains a save event.
        
        Args:
            frame_num (int): Frame number to check
            
        Returns:
            dict or None: Save event data if frame contains a save, None otherwise
        """
        for save in self.save_events:
            if save['frame_num'] == frame_num:
                return save
        return None
