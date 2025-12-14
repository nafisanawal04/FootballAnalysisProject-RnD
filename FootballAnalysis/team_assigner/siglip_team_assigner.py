"""
SigLIP+UMAP+Kmeans-based team assigner for improved player team detection.
Uses SigLIP for feature extraction, UMAP for dimensionality reduction, and Kmeans for clustering.
"""

import sys
sys.path.append('../')

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    import umap
    from sklearn.cluster import KMeans
    from transformers import CLIPProcessor, CLIPModel
    UMAP_KMEANS_AVAILABLE = True
except ImportError:
    UMAP_KMEANS_AVAILABLE = False
    print("Warning: UMAP/Kmeans not available. Install with: pip install umap-learn scikit-learn")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

class SigLIPTeamAssigner:
    """
    SigLIP+UMAP+Kmeans-based team assigner for improved player team detection.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the SigLIP team assigner.
        
        Args:
            device: Device to run models on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_model = None
        self.clip_processor = None
        self.player_features = {}
        self.team_clusters = {1: [], 2: []}
        self.team_centroids = {1: None, 2: None}
        self.umap_reducer = None
        
        # Team colors for visualization
        self.team_colors = {
            1: (255, 255, 255),  # White team
            2: (0, 0, 255)       # Red team
        }
        
        if CLIP_AVAILABLE:
            try:
                # Load SigLIP model (using CLIP as fallback)
                self.clip_model, self.clip_processor = clip.load("ViT-B/32", device=self.device)
                print(f"✓ SigLIP/CLIP model loaded on {self.device}")
            except Exception as e:
                print(f"Warning: Failed to load CLIP model: {e}")
                self.clip_model = None
                self.clip_processor = None
        else:
            print("Warning: CLIP not available, using color-based fallback")
    
    def extract_jersey_features(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extract SigLIP features from player jersey region.
        
        Args:
            frame: Input frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            Feature vector or None if extraction fails
        """
        if not self.clip_model or not self.clip_processor:
            return None
        
        try:
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
            
            # Convert to PIL Image
            jersey_pil = Image.fromarray(cv2.cvtColor(jersey_region, cv2.COLOR_BGR2RGB))
            
            # Extract features using CLIP
            with torch.no_grad():
                # Preprocess image for CLIP
                image_input = self.clip_processor(jersey_pil).unsqueeze(0).to(self.device)
                image_features = self.clip_model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return image_features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"Warning: Feature extraction failed: {e}")
            return None
    
    def extract_color_features(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract color-based features as fallback.
        
        Args:
            frame: Input frame
            bbox: Player bounding box
            
        Returns:
            Color feature vector
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(10)  # Return zero vector if bbox is invalid
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return np.zeros(10)
        
        # Focus on upper body (jersey area)
        jersey_height = int((y2 - y1) * 0.6)
        jersey_region = player_region[:jersey_height, :]
        
        # Convert to HSV for better color analysis
        hsv_region = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # Extract color statistics
        features = []
        
        # Mean and std of each channel
        for channel in range(3):
            channel_data = hsv_region[:, :, channel].flatten()
            features.extend([np.mean(channel_data), np.std(channel_data)])
        
        # Dominant color (most frequent hue)
        hue_values = hsv_region[:, :, 0].flatten()
        if len(hue_values) > 0:
            dominant_hue = Counter(hue_values).most_common(1)[0][0]
            features.append(dominant_hue)
        else:
            features.append(0)
        
        # Brightness
        brightness = np.mean(cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY))
        features.append(brightness)
        
        return np.array(features)
    
    def cluster_players_with_umap_kmeans(self, player_features: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Cluster players into teams using UMAP + Kmeans.
        
        Args:
            player_features: Dictionary mapping player_id to feature vector
            
        Returns:
            Dictionary mapping player_id to team (1 or 2)
        """
        if not UMAP_KMEANS_AVAILABLE or len(player_features) < 2:
            # Fallback to simple clustering
            return self._simple_clustering(player_features)
        
        try:
            # Prepare feature matrix
            player_ids = list(player_features.keys())
            feature_matrix = np.array([player_features[pid] for pid in player_ids])
            
            # Apply UMAP for dimensionality reduction
            if feature_matrix.shape[1] > 2:
                reducer = umap.UMAP(n_components=2, random_state=42)
                reduced_features = reducer.fit_transform(feature_matrix)
            else:
                reduced_features = feature_matrix
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_features)
            
            # Assign teams based on cluster labels
            team_assignments = {}
            for i, player_id in enumerate(player_ids):
                team_assignments[player_id] = cluster_labels[i] + 1  # Convert 0,1 to 1,2
            
            # Store cluster centroids for future reference
            self.team_centroids[1] = kmeans.cluster_centers_[0]
            self.team_centroids[2] = kmeans.cluster_centers_[1]
            
            # Store the UMAP reducer for future use
            self.umap_reducer = reducer
            
            return team_assignments
        
        except Exception as e:
            print(f"Warning: UMAP+Kmeans clustering failed: {e}")
            return self._simple_clustering(player_features)
    
    def _simple_clustering(self, player_features: Dict[int, np.ndarray]) -> Dict[int, int]:
        """Simple clustering fallback."""
        player_ids = list(player_features.keys())
        if len(player_ids) < 2:
            return {pid: 1 for pid in player_ids}
        
        # Simple brightness-based clustering
        brightness_values = []
        for pid in player_ids:
            if len(player_features[pid]) > 0:
                brightness_values.append(np.mean(player_features[pid]))
            else:
                brightness_values.append(0)
        
        # Sort by brightness and assign teams
        sorted_indices = np.argsort(brightness_values)
        team_assignments = {}
        
        for i, idx in enumerate(sorted_indices):
            player_id = player_ids[idx]
            team_assignments[player_id] = 1 if i < len(player_ids) // 2 else 2
        
        return team_assignments
    
    def assign_team_color(self, frame: np.ndarray, player_detections: Dict[int, Dict]) -> None:
        """
        Assign team colors to all players using SigLIP+UMAP+Kmeans.
        
        Args:
            frame: Input frame
            player_detections: Dictionary of player detections
        """
        print("\nDetecting player teams using SigLIP+UMAP+Kmeans:")
        
        # Extract features for all players
        player_features = {}
        for player_id, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            
            # Try SigLIP features first
            features = self.extract_jersey_features(frame, bbox)
            
            # Fallback to color features if SigLIP fails
            if features is None:
                features = self.extract_color_features(frame, bbox)
                print(f"Player {player_id}: Using color features (SigLIP unavailable)")
            else:
                print(f"Player {player_id}: Using SigLIP features")
            
            player_features[player_id] = features
        
        # Cluster players into teams
        team_assignments = self.cluster_players_with_umap_kmeans(player_features)
        
        # Store assignments
        self.player_team_dict = team_assignments
        
        # Print results
        team_counts = {1: 0, 2: 0}
        for player_id, team in team_assignments.items():
            team_counts[team] += 1
            print(f"Player {player_id}: Team {team}")
        
        print(f"\nTeam Distribution:")
        print(f"  Team 1: {team_counts[1]} players")
        print(f"  Team 2: {team_counts[2]} players")
    
    def get_player_team(self, frame: np.ndarray, player_bbox: List[float], player_id: int) -> int:
        """
        Get team for a specific player.
        
        Args:
            frame: Input frame
            player_bbox: Player bounding box
            player_id: Player ID
            
        Returns:
            Team ID (1 or 2)
        """
        if hasattr(self, 'player_team_dict') and player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Extract features for new player
        features = self.extract_jersey_features(frame, player_bbox)
        if features is None:
            features = self.extract_color_features(frame, player_bbox)
        
        # Assign to closest team centroid if available
        if self.team_centroids[1] is not None and self.team_centroids[2] is not None and self.umap_reducer is not None:
            try:
                # Reduce features to same dimensionality as centroids
                features_reshaped = np.array(features).reshape(1, -1)
                reduced_features = self.umap_reducer.transform(features_reshaped).flatten()
                
                # Calculate distances to centroids
                centroid1 = np.array(self.team_centroids[1]).flatten()
                centroid2 = np.array(self.team_centroids[2]).flatten()
                
                dist1 = np.linalg.norm(reduced_features - centroid1)
                dist2 = np.linalg.norm(reduced_features - centroid2)
                team = 1 if dist1 < dist2 else 2
            except Exception as e:
                # Fallback assignment if UMAP transformation fails
                team = 1 if np.mean(features) > 100 else 2
        else:
            # Fallback assignment
            team = 1 if np.mean(features) > 100 else 2
        
        if not hasattr(self, 'player_team_dict'):
            self.player_team_dict = {}
        self.player_team_dict[player_id] = team
        
        return team
    
    def update_team_assignments(self, frame: np.ndarray, player_detections: Dict[int, Dict]) -> None:
        """Update team assignments for new players."""
        for player_id, player_detection in player_detections.items():
            if player_id not in self.player_team_dict:
                bbox = player_detection["bbox"]
                team = self.get_player_team(frame, bbox, player_id)
                self.player_team_dict[player_id] = team
