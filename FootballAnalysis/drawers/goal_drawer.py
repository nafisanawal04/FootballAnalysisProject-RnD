import cv2
import numpy as np

class GoalDrawer:
    """
    Draws goal detection overlays and statistics on video frames.
    """
    
    def __init__(self):
        self.goal_flash_duration = 48  # Show "GOAL!" for 3 seconds at 24fps
        self.team_colors = {
            1: (255, 255, 255),  # White team
            2: (0, 0, 255)       # Red team
        }
    
    def draw_goal_flash(self, frame, goal_team, frames_since_goal):
        """
        Draw a large "GOAL!" overlay when a goal is scored.
        """
        if frames_since_goal > self.goal_flash_duration:
            return frame
            
        frame_height, frame_width = frame.shape[:2]
        
        # Fade out effect
        alpha = max(0, 1 - (frames_since_goal / self.goal_flash_duration))
        
        # Semi-transparent overlay
        overlay = frame.copy()
        color = self.team_colors.get(goal_team, (0, 255, 0))
        
        # Draw large rectangle in center
        rect_height = int(frame_height * 0.3)
        rect_width = int(frame_width * 0.6)
        rect_x = (frame_width - rect_width) // 2
        rect_y = (frame_height - rect_height) // 2
        
        cv2.rectangle(overlay, (rect_x, rect_y), 
                     (rect_x + rect_width, rect_y + rect_height),
                     color, -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - (alpha * 0.7), 0, frame)
        
        # Draw "GOAL!" text
        goal_text = "GOAL!"
        font = cv2.FONT_HERSHEY_DUPLEX  
        font_scale = 5.0
        font_thickness = 15
        text_size = cv2.getTextSize(goal_text, font, font_scale, font_thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        
        # Draw text outline
        cv2.putText(frame, goal_text, (text_x, text_y),
                   font, font_scale, (0, 0, 0), font_thickness + 4)
        # Draw text
        cv2.putText(frame, goal_text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    def draw_goal_scoreboard(self, frame, goals, frame_num, team_assignments):
        """
        Draw goal scoreboard in top-right corner.
        """
        # Count goals per team up to current frame
        team1_goals = sum(1 for g in goals if g['frame'] <= frame_num and g['team'] == 1)
        team2_goals = sum(1 for g in goals if g['frame'] <= frame_num and g['team'] == 2)
        
        frame_height, frame_width = frame.shape[:2]
        
        # Scoreboard position (top-right)
        board_width = 200
        board_height = 80
        board_x = frame_width - board_width - 20
        board_y = 20
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (board_x, board_y),
                     (board_x + board_width, board_y + board_height),
                     (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (board_x, board_y),
                     (board_x + board_width, board_y + board_height),
                     (0, 0, 0), 3)
        
        # Draw scores
        font = cv2.FONT_HERSHEY_DUPLEX  
        font_scale = 1.5
        font_thickness = 3
        
        score_text = f"{team1_goals}  -  {team2_goals}"
        text_size = cv2.getTextSize(score_text, font, font_scale, font_thickness)[0]
        text_x = board_x + (board_width - text_size[0]) // 2
        text_y = board_y + board_height // 2 + text_size[1] // 2
        
        cv2.putText(frame, score_text, (text_x, text_y),
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # Draw team labels
        label_font_scale = 0.5
        cv2.putText(frame, "Team 1", (board_x + 10, board_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (100, 100, 100), 1)
        cv2.putText(frame, "Team 2", (board_x + board_width - 60, board_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (100, 100, 100), 1)
        
        return frame
    
    def draw(self, video_frames, goals, team_assignments):
        """
        Draw goal overlays on all video frames.
        """
        output_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            # Check if a goal was just scored
            for goal in goals:
                frames_since_goal = frame_num - goal['frame']
                if 0 <= frames_since_goal <= self.goal_flash_duration:
                    frame = self.draw_goal_flash(frame, goal['team'], frames_since_goal)
            
            # Draw scoreboard
            frame = self.draw_goal_scoreboard(frame, goals, frame_num, team_assignments)
            
            output_frames.append(frame)
        
        return output_frames