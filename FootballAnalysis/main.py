from utils import read_video, save_video, get_center_of_bbox
from trackers import Tracker
from trackers.enhanced_tracker import EnhancedTracker
import cv2
import numpy as np
import argparse
from team_assigner import TeamAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from pass_and_interception_detector import PassAndInterceptionDetector
from drawers.pass_and_interceptions_drawer import PassInterceptionDrawer
from ball_acquisition import BallAcquisitionDetector
from ball_acquisition.improved_ball_acquisition_detector import ImprovedBallAcquisitionDetector
from team_assigner.improved_team_assigner import ImprovedTeamAssigner
from team_assigner.siglip_team_assigner import SigLIPTeamAssigner
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter
from drawers.tactical_view_drawer import TacticalViewDrawer
from heatmap_generator import HeatmapGenerator
from goal_detector import GoalDetector
from goal_detector.goalkeeper_save_detector import GoalkeeperSaveDetector
from drawers.goal_drawer import GoalDrawer
from pass_network_generator import PassNetworkGenerator
from config.rfdetr_config import RFDETR_CONFIG
from analysis.match_analysis import MatchAnalyzer

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Football Analysis System')
    parser.add_argument('--video', type=str, default='input_videos/0bfacc_1.mp4', help='Path to input video')
    parser.add_argument('--speed', action='store_true', help='Enable Speed and Distance estimation')
    parser.add_argument('--heatmap', action='store_true', help='Enable Heatmap generation')
    parser.add_argument('--goals', action='store_true', help='Enable Goal detection')
    parser.add_argument('--possession', action='store_true', help='Enable Ball Acquisition and Possession stats')
    parser.add_argument('--passing', action='store_true', help='Enable Pass and Interception detection')
    parser.add_argument('--network', action='store_true', help='Enable Pass Network generation')
    parser.add_argument('--tactical', action='store_true', help='Enable Tactical View')
    parser.add_argument('--camera', action='store_true', help='Enable Camera Movement overlay')
    parser.add_argument('--analysis', action='store_true', help='Enable Match Analysis Report')
    parser.add_argument('--all', action='store_true', help='Enable ALL features')
    
    args = parser.parse_args()
    
    # Handle dependencies
    if args.all:
        args.speed = args.heatmap = args.goals = args.possession = args.passing = args.network = args.tactical = args.camera = args.analysis = True

    # Passing and Network imply Possession, which implies Team Assignment
    if args.network: args.passing = True
    if args.passing: args.possession = True
    
    # Features requiring Team Assignment
    needs_team_assignment = args.possession or args.goals or args.tactical or args.analysis or args.heatmap
    
    # Read Video
    video_frames = read_video(args.video)

    # Initialize Enhanced Tracker with SAM2 (if available)
    sam2_checkpoint = None 
    tracker = EnhancedTracker(
        api_key=RFDETR_CONFIG["api_key"],
        workspace=RFDETR_CONFIG["workspace"],
        project=RFDETR_CONFIG["project"],
        version=RFDETR_CONFIG["version"],
        sam2_checkpoint_path=sam2_checkpoint
    )

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)
    
    # Detect Court Keypoints
    court_keypoint_detector = CourtKeypointDetector('models/football_keypoint_detector.pt')
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames,
                                                                 read_from_stub=True,
                                                                 stub_path='stubs/court_keypoints_stub.pkl')
    
    # Debug: Print keypoint detection results
    valid_keypoints = 0
    for i, kp in enumerate(court_keypoints):
        if kp is not None and hasattr(kp, 'xy') and kp.xy is not None:
            kp_list = kp.xy.tolist()
            if len(kp_list) > 0 and len(kp_list[0]) > 0:
                valid_keypoints += 1

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # --- Conditional Features ---

    # Speed and distance estimator
    speed_and_distance_estimator = None
    if args.speed or args.analysis: # Analysis might need speed data
        print("Running Speed and Distance Estimator...")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assignments = {1: [], 2: []}
    player_assignment = []
    
    if needs_team_assignment:
        print("Running Team Assigner...")
        team_assigner = SigLIPTeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            frame_player_assignment = {}
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(video_frames[frame_num],   
                                                     track['bbox'],
                                                     player_id)
                tracks['players'][frame_num][player_id]['team'] = team 
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
                
                if player_id not in team_assignments[team]:
                    team_assignments[team].append(player_id)
                
                frame_player_assignment[player_id] = team
            player_assignment.append(frame_player_assignment)

    # Ball Acquisition
    ball_acquisition = []
    team_ball_control = []
    ball_acquisition_detector = None
    goalkeeper_saves = None
    
    if args.possession:
        print("Running Ball Acquisition Detector...")
        ball_acquisition_detector = ImprovedBallAcquisitionDetector()
        ball_acquisition = ball_acquisition_detector.detect_ball_possession(tracks['players'], tracks['ball'])
        
        for frame_num, player_track in enumerate(tracks['players']):
            assigned_player = ball_acquisition[frame_num] if frame_num < len(ball_acquisition) else -1
            
            if assigned_player != -1:
                if assigned_player in player_track:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                else:
                    if len(team_ball_control) > 0: team_ball_control.append(team_ball_control[-1])
                    else: team_ball_control.append(1)
            else:
                if len(team_ball_control) > 0: team_ball_control.append(team_ball_control[-1])
                else: team_ball_control.append(1)
        
        team_ball_control = np.array(team_ball_control)
        
        # Possession Stats
        possession_stats = ball_acquisition_detector.get_possession_statistics(ball_acquisition, player_assignment)
        print(f"\nBall Possession Statistics:")
        print(f"  - Frames with possession: {possession_stats['possession_frames']}/{possession_stats['total_frames']} ({possession_stats['possession_percentage']:.1f}%)")
        
        # Goalkeeper Saves Detection
        print("\nDetecting Goalkeeper Saves...")
        goalkeeper_save_detector = GoalkeeperSaveDetector()
        goalkeeper_saves = goalkeeper_save_detector.detect_saves(tracks['players'], ball_acquisition)
        save_stats = goalkeeper_save_detector.get_save_statistics(goalkeeper_saves, player_assignment)
        
        print(f"\nGoalkeeper Save Statistics:")
        print(f"  - Total save frames: {save_stats['total_saves']}")
        print(f"  - Saves by Team 1: {save_stats['saves_by_team'][1]}")
        print(f"  - Saves by Team 2: {save_stats['saves_by_team'][2]}")
        if save_stats['saves_by_goalkeeper']:
            print(f"  - Saves by Goalkeeper:")
            for gk_id, count in save_stats['saves_by_goalkeeper'].items():
                print(f"    - Goalkeeper {gk_id}: {count} frames")

    # Goals
    goals = []
    goal_detector = None
    if args.goals:
        print("Running Goal Detector...")
        goal_detector = GoalDetector()
        frame_height, frame_width = video_frames[0].shape[:2]
        # Note: Goal detector needs ball acquisition and player assignment. 
        # If possession was not enabled, we might have empty ball_acquisition.
        # Let's ensure ball_acquisition is computed if goals are enabled, 
        # but we already set needs_team_assignment. 
        # We also need ball_acquisition for goals.
        if not args.possession:
             # If possession wasn't explicitly requested but goals were, we need to run acquisition logic minimally or just pass what we have.
             # The current GoalDetector signature uses ball_acquisition.
             # We should probably run ball acquisition if goals are enabled too.
             # Let's update the dependency logic at the top? 
             # Or just run it here if empty.
             if not ball_acquisition:
                 ball_acquisition_detector = ImprovedBallAcquisitionDetector()
                 ball_acquisition = ball_acquisition_detector.detect_ball_possession(tracks['players'], tracks['ball'])

        goals = goal_detector.detect_goals(
            tracks['ball'], 
            frame_width, 
            frame_height,
            player_assignment,
            ball_acquisition
        )
        print(f"Total goals detected: {len(goals)}")

    # Pass and Interception
    passes = []
    interceptions = []
    pass_accuracy_stats = {}
    
    if args.passing:
        print("Running Pass and Interception Detector...")
        pass_interception_detector = PassAndInterceptionDetector()
        passes = pass_interception_detector.detect_passes(ball_acquisition, player_assignment)
        interceptions = pass_interception_detector.detect_interceptions(ball_acquisition, player_assignment)
        
        pass_accuracy_stats = pass_interception_detector.calculate_pass_accuracy_per_player(
            ball_acquisition, player_assignment, passes, interceptions)
            
        # Add to tracks for visualization
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id in player_track.keys():
                if player_id in pass_accuracy_stats:
                    tracks['players'][frame_num][player_id]['pass_accuracy'] = pass_accuracy_stats[player_id]['accuracy']

    # Pass Network
    if args.network:
        print("Generating Pass Networks...")
        pass_network_gen = PassNetworkGenerator()
        try:
            network_image, team_networks = pass_network_gen.generate_networks(
                passes, ball_acquisition, player_assignment, tracks
            )
            cv2.imwrite('output_videos/pass_networks.png', network_image)
        except Exception as e:
            print(f"Error generating pass networks: {e}")

    # Tactical View
    tactical_player_positions = []
    tactical_view_converter = None
    
    if args.tactical or args.analysis: # Analysis might need tactical positions
        print("Preparing Tactical View...")
        tactical_view_converter = TacticalViewConverter('football_field.png')
        if valid_keypoints > 0:
            tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints, tracks['players'])
        else:
            # Fallback
            for frame_tracks in tracks['players']:
                tactical_positions = {}
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    center_x, center_y = get_center_of_bbox(bbox)
                    tactical_x = (center_x / video_frames[0].shape[1]) * tactical_view_converter.width
                    tactical_y = (center_y / video_frames[0].shape[0]) * tactical_view_converter.height
                    tactical_positions[player_id] = [tactical_x, tactical_y]
                tactical_player_positions.append(tactical_positions)

    # Match Analysis Report
    if args.analysis:
        print("Generating Match Analysis Report...")
        # Ensure we have necessary data
        if not ball_acquisition:
             ball_acquisition_detector = ImprovedBallAcquisitionDetector()
             ball_acquisition = ball_acquisition_detector.detect_ball_possession(tracks['players'], tracks['ball'])
             
        analyzer = MatchAnalyzer(tracks, team_ball_control, player_assignment, ball_acquisition)
        analyzer.save_report('output_videos/match_analysis_report.txt')

    # Heatmaps
    if args.heatmap:
        print("Generating Heatmaps...")
        heatmap_gen = HeatmapGenerator(pitch_width=105, pitch_height=68)
        heatmap_gen.save_heatmap(heatmap_gen.generate_team_heatmap(tracks, view_transformer, team_id=1), 'output_videos/team1_heatmap.png')
        heatmap_gen.save_heatmap(heatmap_gen.generate_team_heatmap(tracks, view_transformer, team_id=2), 'output_videos/team2_heatmap.png')

    # --- Drawing Output ---
    print("Drawing output video...")
    
    # Basic Tracks
    try:
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control if args.possession else None, goalkeeper_saves)
        print("✓ Completed drawing annotations")
    except Exception as e:
        print(f"✗ Error drawing annotations: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Camera Movement
    if args.camera:
        try:
            print("Drawing camera movement...")
            output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
            print("✓ Completed drawing camera movement")
        except Exception as e:
            print(f"✗ Error drawing camera movement: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Speed
    if args.speed and speed_and_distance_estimator:
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
        
    # Passing
    if args.passing:
        pass_interception_drawer = PassInterceptionDrawer()
        pass_interception_drawer.pass_accuracy_stats = pass_accuracy_stats  
        output_video_frames = pass_interception_drawer.draw(output_video_frames, passes, interceptions,
                                                             ball_acquisition, player_assignment, tracks)
                                                             
    # Goals
    if args.goals and goals:
        goal_drawer = GoalDrawer()
        output_video_frames = goal_drawer.draw(output_video_frames, goals, player_assignment)
        
    # Tactical View
    if args.tactical and tactical_view_converter:
        tactical_view_drawer = TacticalViewDrawer()
        output_video_frames = tactical_view_drawer.draw(output_video_frames,
                                                       'football_field.png',
                                                       tactical_view_converter.width,
                                                       tactical_view_converter.height,
                                                       tactical_view_converter.key_points,
                                                       tactical_player_positions,
                                                       player_assignment,
                                                       ball_acquisition)
                                                       
    # Save video
    try:
        print("Saving output video...")
        save_video(output_video_frames, 'output_videos/output_video.avi')
        print("✓ Video saved successfully")
    except Exception as e:
        print(f"✗ Error saving video: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Done!")

if __name__ == '__main__':
    main()