from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        # Define reference team colors as tuples (RGB format for color comparison)
        self.TEAM_WHITE = (255, 255, 255)  # White team
        self.TEAM_RED = (0, 0, 255)        # Red team
        self.ORANGE = (255, 165, 0)        # Orange color (RGB)
        self.BLACK = (0, 0, 0)             # Black color (RGB)
        self.player_team_dict = {}
        self.team_colors = {
            1: self.TEAM_WHITE,  # White team
            2: self.TEAM_RED     # Red team
        }

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        # Take middle half of the image (25% to 75% of height)
        height = image.shape[0]
        start_row = int(height * 0.25)
        end_row = int(height * 0.75)
        middle_half_image = image[start_row:end_row, :]

        # Get Clustering model
        kmeans = self.get_clustering_model(middle_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(middle_half_image.shape[0], middle_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def color_distance(self, color1, color2):
        """Calculate Euclidean distance between two colors"""
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

    def assign_team_by_color(self, player_color):
        """Assign team based on color priority logic"""
        # Convert player_color to tuple if it's numpy array
        if isinstance(player_color, np.ndarray):
            player_color = tuple(map(int, player_color))
        
        # Calculate distances to reference colors
        white_distance = self.color_distance(player_color, self.TEAM_WHITE)
        black_distance = self.color_distance(player_color, self.BLACK)
        orange_distance = self.color_distance(player_color, self.ORANGE)
        
        # Priority-based team assignment:
        # 1. If white color detected → White team
        if white_distance < 100:
            return 1  # White team
        
        # 2. If no white but black color detected → White team
        elif black_distance < 100:
            return 1  # White team
        
        # 3. If no white but orange color detected → Red team
        elif orange_distance < 100:
            return 2  # Red team
        
        # 4. All other players → Red team
        else:
            return 2  # Red team

    def assign_team_color(self, frame, player_detections):
        print("\nDetecting player colors and assigning teams:")
        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            # Convert to tuple for consistent color format
            player_color = tuple(map(int, player_color))
            team = self.assign_team_by_color(player_color)
            print(f"Player {player_id}: RGB color {player_color} -> Team {team}")
            self.player_team_dict[player_id] = team


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.assign_team_by_color(player_color)
        self.player_team_dict[player_id] = team_id
        
        return team_id
