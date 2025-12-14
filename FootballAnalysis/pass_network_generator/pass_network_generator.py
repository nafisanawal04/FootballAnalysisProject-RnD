import cv2
import numpy as np
from collections import defaultdict

class PassNetworkGenerator:
    """
    Generates pass network visualizations showing connections between players.
    """
    
    def __init__(self):
        self.min_passes = 1  # Minimum passes to show connection
        self.tactical_width = 400
        self.tactical_height = 200
        
    def build_pass_network(self, passes, ball_acquisition, player_assignment, tracks):
        """
        Build pass network data structure with improved pass detection.
        """
        team_networks = {1: defaultdict(int), 2: defaultdict(int)}
        possession_window = []  # Track possession over multiple frames
        
        print("\n=== Building Pass Network ===")
        print(f"Total frames: {len(passes)}")
        
        # Detect passes based on ball possession changes
        for frame_num in range(1, len(ball_acquisition)):
            current_holder = ball_acquisition[frame_num]
            previous_holder = ball_acquisition[frame_num - 1]
            
            # Detect change in possession
            if current_holder != previous_holder and current_holder != -1 and previous_holder != -1:
                # Get team information
                if frame_num < len(player_assignment):
                    current_team = player_assignment[frame_num].get(current_holder, -1)
                    previous_team = player_assignment[frame_num-1].get(previous_holder, -1)
                    
                    # Same team = successful pass
                    if current_team == previous_team and current_team in [1, 2]:
                        pair = tuple(sorted([previous_holder, current_holder]))
                        team_networks[current_team][pair] += 1
                        print(f"Pass detected: Frame {frame_num}, Player {previous_holder} → {current_holder} (Team {current_team})")
        
        # Print detailed network summary
        print("\nPass Network Summary:")
        for team in [1, 2]:
            print(f"\nTeam {team} Connections:")
            if team_networks[team]:
                total_passes = sum(team_networks[team].values())
                print(f"Total passes: {total_passes}")
                for (p1, p2), count in sorted(team_networks[team].items(), key=lambda x: x[1], reverse=True):
                    print(f"  Players {p1} ↔ {p2}: {count} passes")
            else:
                print("  No passes detected")
        
        return team_networks
            
    def get_player_average_position(self, player_id, tracks):
        """
        Calculate average position of a player across all frames.
        """
        positions = []
        
        for frame_tracks in tracks['players']:
            if player_id in frame_tracks:
                bbox = frame_tracks[player_id].get('bbox')
                if bbox is not None:
                    # Get center of bounding box
                    x = (bbox[0] + bbox[2]) / 2
                    y = (bbox[1] + bbox[3]) / 2
                    
                    # Scale to tactical view size
                    x_scaled = x * (self.tactical_width / 1920)
                    y_scaled = y * (self.tactical_height / 1080)
                    
                    positions.append((x_scaled, y_scaled))
        
        if not positions:
            return None
            
        avg_x = np.mean([p[0] for p in positions])
        avg_y = np.mean([p[1] for p in positions])
        return (avg_x, avg_y)
    
    def draw_tactical_pitch(self):
        """Create a tactical view of the pitch."""
        img = np.ones((self.tactical_height, self.tactical_width, 3), dtype=np.uint8) * 50  # Dark green
        
        # Field lines (white)
        cv2.rectangle(img, (5, 5), (self.tactical_width-5, self.tactical_height-5), (255, 255, 255), 1)
        cv2.line(img, (self.tactical_width//2, 5), (self.tactical_width//2, self.tactical_height-5), (255, 255, 255), 1)
        
        # Center circle
        cv2.circle(img, (self.tactical_width//2, self.tactical_height//2), 30, (255, 255, 255), 1)
        
        # Penalty areas
        penalty_width = 60
        penalty_height = 80
        # Left penalty area
        cv2.rectangle(img, (5, (self.tactical_height-penalty_height)//2),
                     (penalty_width, (self.tactical_height+penalty_height)//2), (255, 255, 255), 1)
        # Right penalty area
        cv2.rectangle(img, (self.tactical_width-penalty_width, (self.tactical_height-penalty_height)//2),
                     (self.tactical_width-5, (self.tactical_height+penalty_height)//2), (255, 255, 255), 1)
        
        return img
    
    def draw_network(self, network_data, tracks, team_id):
        """Draw pass network for a team."""
        # Create pitch background
        img = self.draw_tactical_pitch()
        
        # Get player positions
        player_positions = {}
        all_player_ids = set([p for pair in network_data.keys() for p in pair])
        
        print(f"\nProcessing Team {team_id} Network")
        print(f"Players to plot: {all_player_ids}")
        
        for player_id in all_player_ids:
            pos = self.get_player_average_position(player_id, tracks)
            if pos is not None:
                player_positions[player_id] = pos
                print(f"Player {player_id} position: ({pos[0]:.1f}, {pos[1]:.1f})")
        
        if not player_positions:
            print(f"No valid positions found for Team {team_id}")
            return img
        
        # Draw pass lines
        max_passes = max(network_data.values()) if network_data else 1
        print(f"Maximum passes between any pair: {max_passes}")
        
        for (player1, player2), pass_count in network_data.items():
            if pass_count < self.min_passes:
                continue
            
            if player1 not in player_positions or player2 not in player_positions:
                continue
                
            pos1 = player_positions[player1]
            pos2 = player_positions[player2]
            
            # Draw connection line
            thickness = max(1, int((pass_count / max_passes) * 5))
            color = (255, 255, 255) if team_id == 1 else (0, 0, 255)
            cv2.line(img, 
                    (int(pos1[0]), int(pos1[1])),
                    (int(pos2[0]), int(pos2[1])),
                    color, thickness)
            
            # Draw pass count
            mid_x = int((pos1[0] + pos2[0]) / 2)
            mid_y = int((pos1[1] + pos2[1]) / 2)
            cv2.putText(img, str(pass_count), (mid_x-8, mid_y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw player nodes
        for player_id, pos in player_positions.items():
            # Draw player circle
            cv2.circle(img, (int(pos[0]), int(pos[1])), 8,
                      (255, 255, 255) if team_id == 1 else (0, 0, 255), -1)
            cv2.circle(img, (int(pos[0]), int(pos[1])), 8, (0, 0, 0), 1)
            
            # Draw player number
            cv2.putText(img, str(player_id), (int(pos[0])-6, int(pos[1])+4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        return img
    
    def generate_networks(self, passes, ball_acquisition, player_assignment, tracks):
        """Generate pass networks for both teams."""
        # Build network data
        team_networks = self.build_pass_network(passes, ball_acquisition, player_assignment, tracks)
        
        # Generate visualizations
        team1_img = self.draw_network(team_networks[1], tracks, 1)
        team2_img = self.draw_network(team_networks[2], tracks, 2)
        
        # Create side-by-side comparison
        comparison = np.zeros((self.tactical_height, self.tactical_width*2 + 10, 3), dtype=np.uint8)
        comparison[:, :self.tactical_width] = team1_img
        comparison[:, self.tactical_width+10:] = team2_img
        
        # Add labels
        cv2.putText(comparison, "Team 1 (White)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, "Team 2 (Red)", (self.tactical_width+20, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return comparison, team_networks