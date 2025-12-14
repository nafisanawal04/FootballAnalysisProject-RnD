import numpy as np
import os

class MatchAnalyzer:
    def __init__(self, tracks, team_ball_control, player_assignment, ball_acquisition, fps=24, pitch_width=105, pitch_height=68):
        self.tracks = tracks
        self.team_ball_control = team_ball_control
        self.player_assignment = player_assignment
        self.ball_acquisition = ball_acquisition
        self.fps = fps
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        
    def analyze(self):
        report = []
        report.append("MATCH ANALYSIS REPORT")
        report.append("=====================")
        
        # 1. Distance Covered
        report.append("\n1. DISTANCE COVERED")
        report.append("-------------------")
        team_distances = {1: 0, 2: 0}
        player_distances = {}
        
        for frame_tracks in self.tracks['players']:
            for player_id, track in frame_tracks.items():
                if 'distance' in track:
                    # distance is cumulative in the track? 
                    # Let's check how speed_and_distance_estimator works. 
                    # Assuming 'distance' is total distance so far, we take the max value for each player.
                    pass

        # Actually, let's iterate through players to get their max distance
        # We need to reconstruct player tracks across frames
        player_final_stats = {}
        
        for frame_num, frame_tracks in enumerate(self.tracks['players']):
            for player_id, track in frame_tracks.items():
                if player_id not in player_final_stats:
                    player_final_stats[player_id] = {'team': track.get('team', -1), 'distance': 0, 'speed_sum': 0, 'count': 0}
                
                if 'distance' in track:
                    player_final_stats[player_id]['distance'] = max(player_final_stats[player_id]['distance'], track['distance'])
                
                if 'speed' in track:
                    player_final_stats[player_id]['speed_sum'] += track['speed']
                    player_final_stats[player_id]['count'] += 1
                    
        # Aggregate by team
        for player_id, stats in player_final_stats.items():
            team = stats['team']
            if team in team_distances:
                team_distances[team] += stats['distance']
            player_distances[player_id] = stats['distance']
            
        report.append(f"Team 1 Total Distance: {team_distances[1]:.2f} m")
        report.append(f"Team 2 Total Distance: {team_distances[2]:.2f} m")
        
        report.append("\nTop 5 Players by Distance:")
        sorted_players = sorted(player_distances.items(), key=lambda x: x[1], reverse=True)[:5]
        for pid, dist in sorted_players:
            team = player_final_stats[pid]['team']
            report.append(f"  Player {pid} (Team {team}): {dist:.2f} m")

        # 2. Zone Analysis
        report.append("\n2. ZONE ANALYSIS (Time spent in %)")
        report.append("----------------------------------")
        # Zones: Defensive (< 1/3), Midfield (1/3 - 2/3), Attacking (> 2/3)
        # We need tactical positions for this. 
        # If tactical positions are not available in tracks, we might need to rely on transformed positions if available.
        # main.py adds transformed positions: view_transformer.add_transformed_position_to_tracks(tracks)
        # Let's check if 'position_transformed' is in tracks.
        
        team_zones = {1: {'Defensive': 0, 'Midfield': 0, 'Attacking': 0}, 
                      2: {'Defensive': 0, 'Midfield': 0, 'Attacking': 0}}
        total_player_frames = {1: 0, 2: 0}
        
        for frame_tracks in self.tracks['players']:
            for player_id, track in frame_tracks.items():
                team = track.get('team', -1)
                if team not in [1, 2]: continue
                
                # Check for transformed position (meters)
                pos = track.get('position_transformed')
                if pos is not None:
                    y = pos[1] # y is usually length in football analysis (0-105 or similar)? 
                    # Wait, ViewTransformer usually transforms to (x, y). 
                    # Let's assume x is width (0-pitch_width) and y is length (0-pitch_height) or vice versa.
                    # Standard pitch: 105m long, 68m wide.
                    # Usually x is along the length.
                    x = pos[0]
                    
                    # Normalize x to 0-1 range if needed, or use pitch dimensions
                    # Let's assume x is in meters.
                    
                    # Determine zone relative to TEAM direction.
                    # This is tricky without knowing which side is which.
                    # Usually Team 1 is left, Team 2 is right?
                    # For now, let's just do absolute zones: Thirds of the pitch.
                    
                    third = self.pitch_width / 3
                    if x < third:
                        zone = 'Zone 1 (Left)'
                    elif x < 2 * third:
                        zone = 'Zone 2 (Center)'
                    else:
                        zone = 'Zone 3 (Right)'
                        
                    # Map to Team zones (assuming Team 1 defends Left, Team 2 defends Right)
                    # This is a simplification.
                    if team == 1:
                        if x < third: zone_type = 'Defensive'
                        elif x < 2 * third: zone_type = 'Midfield'
                        else: zone_type = 'Attacking'
                    else: # Team 2
                        if x > 2 * third: zone_type = 'Defensive'
                        elif x > third: zone_type = 'Midfield'
                        else: zone_type = 'Attacking'
                        
                    team_zones[team][zone_type] += 1
                    total_player_frames[team] += 1

        for team in [1, 2]:
            report.append(f"\nTeam {team} Spatial Distribution:")
            if total_player_frames[team] > 0:
                for zone in ['Defensive', 'Midfield', 'Attacking']:
                    pct = (team_zones[team][zone] / total_player_frames[team]) * 100
                    report.append(f"  {zone}: {pct:.1f}%")
            else:
                report.append("  No position data available.")

        # 3. Off-Ball Runs (Heuristic)
        report.append("\n3. OFF-BALL RUNS (High Speed without Ball)")
        report.append("------------------------------------------")
        # Heuristic: Speed > 4 m/s (approx 14.4 km/h) AND not having ball
        
        high_speed_threshold = 4.0 # m/s
        off_ball_runs = {1: 0, 2: 0}
        
        for frame_num, frame_tracks in enumerate(self.tracks['players']):
            ball_holder = self.ball_acquisition[frame_num] if frame_num < len(self.ball_acquisition) else -1
            
            for player_id, track in frame_tracks.items():
                team = track.get('team', -1)
                if team not in [1, 2]: continue
                
                speed = track.get('speed', 0)
                if speed is None: speed = 0
                
                if speed > high_speed_threshold and player_id != ball_holder:
                    off_ball_runs[team] += 1
                    
        # Convert frames to seconds (approx)
        for team in [1, 2]:
            seconds = off_ball_runs[team] / self.fps
            report.append(f"Team {team}: {seconds:.1f} seconds of high-intensity off-ball movement")

        # 4. PER-PLAYER DETAILED ANALYSIS
        report.append("\n4. PER-PLAYER DETAILED ANALYSIS")
        report.append("================================")
        
        # Calculate per-player zone distribution
        player_zones = {}
        player_off_ball_runs = {}
        player_possession_frames = {}
        
        for frame_num, frame_tracks in enumerate(self.tracks['players']):
            ball_holder = self.ball_acquisition[frame_num] if frame_num < len(self.ball_acquisition) else -1
            
            for player_id, track in frame_tracks.items():
                if player_id not in player_zones:
                    player_zones[player_id] = {'Defensive': 0, 'Midfield': 0, 'Attacking': 0, 'total': 0}
                    player_off_ball_runs[player_id] = 0
                    player_possession_frames[player_id] = 0
                
                team = track.get('team', -1)
                if team not in [1, 2]: continue
                
                # Zone tracking
                pos = track.get('position_transformed')
                if pos is not None:
                    x = pos[0]
                    third = self.pitch_width / 3
                    
                    if team == 1:
                        if x < third: zone_type = 'Defensive'
                        elif x < 2 * third: zone_type = 'Midfield'
                        else: zone_type = 'Attacking'
                    else:
                        if x > 2 * third: zone_type = 'Defensive'
                        elif x > third: zone_type = 'Midfield'
                        else: zone_type = 'Attacking'
                    
                    player_zones[player_id][zone_type] += 1
                    player_zones[player_id]['total'] += 1
                
                # Off-ball runs
                speed = track.get('speed', 0)
                if speed is None: speed = 0
                if speed > high_speed_threshold and player_id != ball_holder:
                    player_off_ball_runs[player_id] += 1
                
                # Possession tracking
                if player_id == ball_holder:
                    player_possession_frames[player_id] += 1
        
        # Sort players by team and then by distance
        team_1_players = [(pid, stats) for pid, stats in player_final_stats.items() if stats['team'] == 1]
        team_2_players = [(pid, stats) for pid, stats in player_final_stats.items() if stats['team'] == 2]
        
        team_1_players.sort(key=lambda x: x[1]['distance'], reverse=True)
        team_2_players.sort(key=lambda x: x[1]['distance'], reverse=True)
        
        for team_id, team_players in [(1, team_1_players), (2, team_2_players)]:
            report.append(f"\n--- TEAM {team_id} PLAYERS ---")
            
            for player_id, stats in team_players:
                report.append(f"\nPlayer {player_id}:")
                
                # Distance
                distance = stats['distance']
                report.append(f"  Distance Covered: {distance:.2f} m")
                
                # Average Speed
                if stats['count'] > 0:
                    avg_speed = stats['speed_sum'] / stats['count']
                    report.append(f"  Average Speed: {avg_speed:.2f} m/s ({avg_speed * 3.6:.2f} km/h)")
                
                # Zone Distribution
                if player_id in player_zones and player_zones[player_id]['total'] > 0:
                    zones = player_zones[player_id]
                    report.append(f"  Zone Distribution:")
                    for zone in ['Defensive', 'Midfield', 'Attacking']:
                        pct = (zones[zone] / zones['total']) * 100
                        report.append(f"    {zone}: {pct:.1f}%")
                
                # Possession Time
                if player_id in player_possession_frames:
                    possession_seconds = player_possession_frames[player_id] / self.fps
                    report.append(f"  Ball Possession: {possession_seconds:.1f} seconds ({player_possession_frames[player_id]} frames)")
                
                # Off-ball runs
                if player_id in player_off_ball_runs:
                    off_ball_seconds = player_off_ball_runs[player_id] / self.fps
                    report.append(f"  High-Intensity Off-Ball Movement: {off_ball_seconds:.1f} seconds")

        return "\n".join(report)

    def save_report(self, filename="match_analysis_report.txt"):
        content = self.analyze()
        with open(filename, "w") as f:
            f.write(content)
        print(f"Match analysis report saved to {filename}")
