import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from detectors.rfdetr_seg_detector import RFDETRSegDetector

class Tracker:
    def __init__(self, api_key, workspace, project, version):
        """
        Initialize tracker with RFDETR-SEG model.
        
        Args:
            api_key: Roboflow API key for RFDETR-SEG
            workspace: Roboflow workspace name
            project: Roboflow project name  
            version: Model version number
        """
        self.detector = RFDETRSegDetector(api_key, workspace, project, version)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        # Use RFDETR-SEG detector
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            predictions = self.detector.predict_frames_batch(batch_frames, confidence=0.1)
            detections += predictions
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "ball":[],
            "referees":[]
        }

        for frame_num, detection in enumerate(detections):
            # Handle RFDETR-SEG format
            # Convert to supervision format
            detection_supervision = self.detector.convert_to_supervision_format(detection)
            
            # Get class names from detector
            cls_names_inv = self.detector.class_names_inv

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            # Process tracked detections (players and goalkeepers)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Check if it's a player or goalkeeper (both are considered players for tracking)
                if cls_id in [cls_names_inv.get('player', 2), cls_names_inv.get('goalkeeper', 1)]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox, "class_id": cls_id}
                # Check if it's a referee
                elif cls_id == cls_names_inv.get('referee', 3):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox, "class_id": cls_id}

            # Process untracked detections (ball)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv.get('ball', 0):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw team ball control statistics on a single frame with consistent styling.
        """
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        # Overlay Position (positioned on the right side to avoid overlap with pass/interception stats)
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.69) 
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.95)  
        rect_y2 = int(frame_height * 0.90)
        # Text positions
        text_x = int(frame_width * 0.71)  
        text_y1 = int(frame_height * 0.80)  
        text_y2 = int(frame_height * 0.88)

        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255,255,255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        
        if team_1_num_frames + team_2_num_frames > 0:
            team_1_percentage = team_1_num_frames/(team_1_num_frames+team_2_num_frames) * 100
            team_2_percentage = team_2_num_frames/(team_1_num_frames+team_2_num_frames) * 100
        else:
            team_1_percentage = 0
            team_2_percentage = 0

        cv2.putText(
            frame, 
            f"Team 1 (White) Ball Control: {team_1_percentage:.1f}%",
            (text_x, text_y1), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0,0,0), 
            font_thickness
        )
        
        cv2.putText(
            frame, 
            f"Team 2 (Mint) Ball Control: {team_2_percentage:.1f}%",
            (text_x, text_y2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (0,0,0), 
            font_thickness
        )

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
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
                # Inside draw_annotations, after drawing ellipse:
                if 'pass_accuracy' in player:
                    accuracy = player.get('pass_accuracy', 0)
                    # Draw accuracy below player ID
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
                                

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))
            
            # Draw Referees
            for track_id, referee in referee_dict.items():
                color = (0, 255, 255)  # Yellow for referee
                frame = self.draw_ellipse(frame, referee["bbox"], color, track_id)
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames