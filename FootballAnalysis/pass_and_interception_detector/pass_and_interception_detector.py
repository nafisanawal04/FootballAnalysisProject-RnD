class PassAndInterceptionDetector():
    """
    A class that detects passes between teammates and interceptions by opposing teams.
    """
    def __init__(self):
        pass 

    def detect_passes(self, ball_acquisition, player_assignment):
        """
        Detects successful passes between players of the same team.
        Improved to handle frames where ball is in the air (no holder = -1).
        """
        passes = [-1] * len(ball_acquisition)
        
        prev_holder = -1
        prev_holder_frame = -1
        
        for frame in range(len(ball_acquisition)):
            current_holder = ball_acquisition[frame]
            
            # Update previous holder when someone has the ball
            if current_holder != -1:
                # Check if ball changed hands
                if prev_holder != -1 and prev_holder != current_holder:
                    # Get teams (check within reasonable frame window)
                    if prev_holder_frame < len(player_assignment) and frame < len(player_assignment):
                        prev_team = player_assignment[prev_holder_frame].get(prev_holder, -1)
                        current_team = player_assignment[frame].get(current_holder, -1)
                        
                        # Same team = successful pass
                        if prev_team == current_team and prev_team != -1:
                            passes[frame] = prev_team
                
                # Update tracking
                prev_holder = current_holder
                prev_holder_frame = frame
        
        return passes

    def detect_interceptions(self, ball_acquisition, player_assignment):
        """
        Detects interceptions where the ball possession changes between opposing teams.

        Args:
            ball_acquisition (list): A list indicating which player has possession of the ball in each frame.
            player_assignment (list): A list of dictionaries indicating team assignments for each player
                in the corresponding frame.

        Returns:
            list: A list where each element indicates if an interception occurred in that frame
                (-1: no interception, 1: Team 1 interception, 2: Team 2 interception).
        """
        interceptions = [-1] * len(ball_acquisition)
        prev_holder = -1
        previous_frame = -1
        
        for frame in range(1, len(ball_acquisition)):
            if ball_acquisition[frame - 1] != -1:
                prev_holder = ball_acquisition[frame - 1]
                previous_frame = frame - 1

            current_holder = ball_acquisition[frame]
            
            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_assignment[previous_frame].get(prev_holder, -1)
                current_team = player_assignment[frame].get(current_holder, -1)
                
                if prev_team != current_team and prev_team != -1 and current_team != -1:
                    interceptions[frame] = current_team
        
        return interceptions
    def calculate_pass_accuracy_per_player(self, ball_acquisition, player_assignment, passes, interceptions):
        """
        Calculate pass accuracy for each player based on detected passes and interceptions.
        Only includes players that are currently visible in the frame.
        """
        player_stats = {}
        
        # Get all valid player IDs (players that appear in current frames)
        valid_players = set()
        for frame_assignment in player_assignment:
            valid_players.update(frame_assignment.keys())
        
        # Count successful passes
        for frame in range(len(passes)):
            if passes[frame] != -1:
                if frame > 0:
                    passer = ball_acquisition[frame - 1]
                   
                    if passer != -1 and passer in valid_players:
                        if passer not in player_stats:
                            player_stats[passer] = {'successful': 0, 'failed': 0, 'accuracy': 0.0}
                        player_stats[passer]['successful'] += 1
        
        # Count failed passes (interceptions)
        for frame in range(len(interceptions)):
            if interceptions[frame] != -1:
                if frame > 0:
                    passer = ball_acquisition[frame - 1]
                   
                    if passer != -1 and passer in valid_players:
                        if passer not in player_stats:
                            player_stats[passer] = {'successful': 0, 'failed': 0, 'accuracy': 0.0}
                        player_stats[passer]['failed'] += 1
        
        # Calculate accuracy
        for player_id in player_stats:
            total = player_stats[player_id]['successful'] + player_stats[player_id]['failed']
            if total > 0:
                player_stats[player_id]['accuracy'] = (player_stats[player_id]['successful'] / total) * 100
        
        return player_stats