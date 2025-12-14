import cv2
import numpy as np
from utils import get_center_of_bbox

class GoalDetector:
    def __init__(self):
        # Core validation parameters
        self.goal_line_threshold = 15      # Pixels near goal line
        self.goal_height_min = 0.35        # % of frame height (goal post bottom)
        self.goal_height_max = 0.65        # % of frame height (goal post top)
        self.min_ball_speed = 7            # Pixels per frame
        self.min_frames_in_goal = 4        # Consecutive frames for confirmation
        self.goal_cooldown = 120           # Minimum frames between goals
        self.ball_history_size = 10        # Frames to track ball movement
        
        # State tracking
        self.ball_history = []             # Recent ball positions
        self.last_goal_frame = -1

    def is_ball_in_left_goal(self, ball_center, frame_width, frame_height):
        """
        Check if ball is in left goal area.
        Left goal is at x=0 side of the frame.
        """
        x, y = ball_center
        
        # Goal area dimensions (as percentage of frame)
        goal_width = frame_width * 0.05  # 5% of frame width
        goal_height_start = frame_height * 0.3  # 30% from top
        goal_height_end = frame_height * 0.7    # 70% from top
        
        # Check if ball is in goal area
        in_x_range = x < goal_width
        in_y_range = goal_height_start < y < goal_height_end
        
        return in_x_range and in_y_range
    
    def is_ball_in_right_goal(self, ball_center, frame_width, frame_height):
        """
        Check if ball is in right goal area.
        Right goal is at x=frame_width side.
        """
        x, y = ball_center
        
        # Goal area dimensions (as percentage of frame)
        goal_width = frame_width * 0.05  # 5% of frame width
        goal_height_start = frame_height * 0.3  # 30% from top
        goal_height_end = frame_height * 0.7    # 70% from top
        
        # Check if ball is in goal area
        in_x_range = x > (frame_width - goal_width)
        in_y_range = goal_height_start < y < goal_height_end
        
        return in_x_range and in_y_range
        
    def calculate_ball_trajectory(self, ball_positions):
        """Calculate ball direction and speed"""
        if len(ball_positions) < 2:
            return 0, 0, 0
        
        last_pos = np.array(ball_positions[-1])
        prev_pos = np.array(ball_positions[-2])
        
        velocity = last_pos - prev_pos
        speed = np.linalg.norm(velocity)
        direction = np.arctan2(velocity[1], velocity[0])
        
        return velocity, speed, direction
    
    def is_valid_shot_trajectory(self, ball_pos, velocity, goal_side):
        """Validate if ball movement is consistent with a shot"""
        if goal_side == 'left':
            # Ball should be moving left (negative x velocity)
            return velocity[0] < -self.min_ball_speed
        else:
            # Ball should be moving right (positive x velocity)
            return velocity[0] > self.min_ball_speed
    
    def is_goalkeeper_save(self, ball_pos, player_positions, goal_side):
        """Check if a goalkeeper is between ball and goal"""
        for player_pos in player_positions:
            if goal_side == 'left':
                if (player_pos[0] < ball_pos[0] and 
                    abs(player_pos[1] - ball_pos[1]) < 50):
                    return True
            else:
                if (player_pos[0] > ball_pos[0] and 
                    abs(player_pos[1] - ball_pos[1]) < 50):
                    return True
        return False
    
    def detect_goals(self, ball_tracks, frame_width, frame_height, player_assignment, ball_acquisition):
        """
        Enhanced goal detection with strict validation rules
        """
        goals = []
        in_goal_counter = 0
        potential_goal_frame = -1
        potential_scorer = -1
        potential_team = -1
        
        print("\nStarting strict goal detection...")
        
        for frame_num, ball_frame in enumerate(ball_tracks):
            # Skip if no ball detected
            if 1 not in ball_frame:
                self.ball_history = []
                continue
            
            ball_bbox = ball_frame[1].get('bbox', [])
            if not ball_bbox:
                continue
            
            # Get ball position
            ball_center = get_center_of_bbox(ball_bbox)
            self.ball_history.append(ball_center)
            if len(self.ball_history) > self.ball_history_size:
                self.ball_history.pop(0)
            
            # Calculate ball movement
            if len(self.ball_history) >= 2:
                velocity, speed, direction = self.calculate_ball_trajectory(self.ball_history)
                
                # Check if ball is in either goal
                in_left_goal = self.is_ball_in_left_goal(ball_center, frame_width, frame_height)
                in_right_goal = self.is_ball_in_right_goal(ball_center, frame_width, frame_height)
                
                if in_left_goal or in_right_goal:
                    goal_side = 'left' if in_left_goal else 'right'
                    
                    # New potential goal
                    if in_goal_counter == 0:
                        # Validate ball trajectory
                        if not self.is_valid_shot_trajectory(ball_center, velocity, goal_side):
                            print(f"Frame {frame_num}: Invalid shot trajectory")
                            continue
                        
                        # Check if enough time has passed since last goal
                        if (frame_num - self.last_goal_frame) < self.goal_cooldown:
                            continue
                        
                        potential_goal_frame = frame_num
                        
                        # Identify scorer (last player with ball possession)
                        for i in range(frame_num - 1, max(0, frame_num - 30), -1):
                            if i < len(ball_acquisition) and ball_acquisition[i] != -1:
                                potential_scorer = ball_acquisition[i]
                                if i < len(player_assignment):
                                    potential_team = player_assignment[i].get(potential_scorer, -1)
                                break
                    
                    in_goal_counter += 1
                    
                    # Confirm goal after minimum frames
                    if in_goal_counter >= self.min_frames_in_goal:
                        # Validate scoring team matches goal side
                        valid_team = (
                            (goal_side == 'right' and potential_team == 1) or
                            (goal_side == 'left' and potential_team == 2)
                        )
                        
                        if valid_team:
                            goal_event = {
                                'frame': potential_goal_frame,
                                'team': potential_team,
                                'scorer': potential_scorer,
                                'side': goal_side,
                                'speed': speed
                            }
                            
                            goals.append(goal_event)
                            self.last_goal_frame = frame_num
                            
                            print(f"\n⚽ Goal confirmed at frame {potential_goal_frame}")
                            print(f"   Scored by: Player {potential_scorer} (Team {potential_team})")
                            print(f"   Goal side: {goal_side}")
                            print(f"   Shot speed: {speed:.2f} pixels/frame")
                        
                        in_goal_counter = 0
                else:
                    in_goal_counter = 0
        
        return goals