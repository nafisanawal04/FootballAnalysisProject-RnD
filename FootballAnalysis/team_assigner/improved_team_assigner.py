import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class ImprovedTeamAssigner:
    def __init__(self):
        # Define reference team colors in BGR format (OpenCV uses BGR)
        self.TEAM_WHITE = (255, 255, 255)  # White team
        self.TEAM_RED = (0, 0, 255)        # Red team
        self.TEAM_DARK_GREEN = (0, 100, 0)  # Dark green team
        
        # Color tolerance for matching
        self.COLOR_TOLERANCE = 80
        
        self.player_team_dict = {}
        self.team_colors = {
            1: self.TEAM_WHITE,
            2: self.TEAM_LIGHT_GREEN
        }
        
        # Store color samples for better team assignment
        self.team_color_samples = {1: [], 2: []}

    def extract_jersey_color(self, frame, bbox):
        """Extract the dominant jersey color from player bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return None
        
        # Focus on upper body (jersey area) - top 60% of the bbox
        jersey_height = int((y2 - y1) * 0.6)
        jersey_region = player_region[:jersey_height, :]
        
        # Convert to HSV for better color analysis
        hsv_region = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # Create mask to exclude background (black/dark areas)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        dark_mask = cv2.inRange(hsv_region, lower_dark, upper_dark)
        
        # Invert mask to get non-dark areas
        jersey_mask = cv2.bitwise_not(dark_mask)
        
        if np.sum(jersey_mask) == 0:
            return None
        
        # Get dominant color from non-dark areas
        pixels = jersey_region[jersey_mask > 0]
        if len(pixels) == 0:
            return None
        
        # Use K-means to find dominant color
        if len(pixels) > 10:  # Need enough pixels for clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the most common cluster (dominant color)
            labels = kmeans.labels_
            dominant_cluster = Counter(labels).most_common(1)[0][0]
            dominant_color = kmeans.cluster_centers_[dominant_cluster]
            
            return tuple(map(int, dominant_color))
        
        return None

    def calculate_color_similarity(self, color1, color2):
        """Calculate color similarity using Euclidean distance in RGB space."""
        if color1 is None or color2 is None:
            return float('inf')
        
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

    def assign_team_by_color(self, player_color):
        """Assign team based on color analysis."""
        if player_color is None:
            return 2  # Default to team 2 if color detection fails
        
        # Calculate similarity to reference colors
        white_similarity = self.calculate_color_similarity(player_color, self.TEAM_WHITE)
        red_similarity = self.calculate_color_similarity(player_color, self.TEAM_RED)
        
        # Check if color is close to white (more lenient threshold)
        if white_similarity < self.COLOR_TOLERANCE * 1.5:  # More lenient
            return 1  # White team
        
        # Check if color is close to red
        elif red_similarity < self.COLOR_TOLERANCE:
            return 2  # Red team
        
        # If we have team samples, use them for comparison
        if len(self.team_color_samples[1]) > 0 and len(self.team_color_samples[2]) > 0:
            # Calculate average similarity to each team's samples
            team1_avg_similarity = np.mean([
                self.calculate_color_similarity(player_color, sample) 
                for sample in self.team_color_samples[1]
            ])
            
            team2_avg_similarity = np.mean([
                self.calculate_color_similarity(player_color, sample) 
                for sample in self.team_color_samples[2]
            ])
            
            if team1_avg_similarity < team2_avg_similarity:
                return 1
            else:
                return 2
        
        # Enhanced brightness and color analysis
        brightness = sum(player_color) / 3
        r, g, b = player_color
        
        # Check for white-like colors (high brightness, balanced RGB)
        if brightness > 180 and abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:
            return 1  # White team
        
        # Check for green-like colors (higher green component)
        elif g > r and g > b and g > 100:
            return 2  # Light green team
        
        # Default assignment based on brightness
        elif brightness > 150:  # Bright colors likely white
            return 1
        else:
            return 2

    def assign_team_color(self, frame, player_detections):
        """Assign team colors to all players in the first frame."""
        print("\nDetecting player colors and assigning teams:")
        
        # First pass: collect all colors
        player_colors = {}
        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.extract_jersey_color(frame, bbox)
            player_colors[player_id] = player_color
            print(f"Player {player_id}: RGB color {player_color}")
        
        # Second pass: assign teams with improved logic
        for player_id, player_color in player_colors.items():
            team = self.assign_team_by_color(player_color)
            self.player_team_dict[player_id] = team
            
            # Store color sample for future reference
            if player_color is not None:
                self.team_color_samples[team].append(player_color)
            
            print(f"Player {player_id}: Team {team}")
        
        # Check if all players are assigned to the same team
        teams = list(self.player_team_dict.values())
        if len(set(teams)) == 1:
            print(f"\nWarning: All players assigned to Team {teams[0]}. Reassigning some players...")
            self._rebalance_teams(player_colors)

    def get_player_team(self, frame, player_bbox, player_id):
        """Get team for a specific player."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # If not in dictionary, detect color and assign team
        player_color = self.extract_jersey_color(frame, player_bbox)
        team_id = self.assign_team_by_color(player_color)
        self.player_team_dict[player_id] = team_id
        
        # Store color sample
        if player_color is not None:
            self.team_color_samples[team_id].append(player_color)
        
        return team_id

    def _rebalance_teams(self, player_colors):
        """Rebalance teams if all players are assigned to the same team."""
        # Sort players by brightness (brightest players likely to be white team)
        player_brightness = []
        for player_id, color in player_colors.items():
            if color is not None:
                brightness = sum(color) / 3
                player_brightness.append((player_id, brightness, color))
        
        # Sort by brightness (brightest first)
        player_brightness.sort(key=lambda x: x[1], reverse=True)
        
        # Assign brightest half to team 1, darker half to team 2
        total_players = len(player_brightness)
        half_point = total_players // 2
        
        for i, (player_id, brightness, color) in enumerate(player_brightness):
            if i < half_point:
                self.player_team_dict[player_id] = 1
                if color is not None:
                    self.team_color_samples[1].append(color)
                print(f"Player {player_id}: Reassigned to Team 1 (brightness: {brightness:.1f})")
            else:
                self.player_team_dict[player_id] = 2
                if color is not None:
                    self.team_color_samples[2].append(color)
                print(f"Player {player_id}: Reassigned to Team 2 (brightness: {brightness:.1f})")

    def update_team_assignments(self, frame, player_detections):
        """Update team assignments based on new detections."""
        for player_id, player_detection in player_detections.items():
            if player_id not in self.player_team_dict:
                bbox = player_detection["bbox"]
                player_color = self.extract_jersey_color(frame, bbox)
                team_id = self.assign_team_by_color(player_color)
                self.player_team_dict[player_id] = team_id
                
                if player_color is not None:
                    self.team_color_samples[team_id].append(player_color)
