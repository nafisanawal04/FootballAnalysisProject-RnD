import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class HeatmapGenerator:
    def __init__(self, pitch_width=105, pitch_height=68):
        """
        Initialize heatmap generator.
        
        Args:
            pitch_width: Width of football pitch in meters (default 105m)
            pitch_height: Height of football pitch in meters (default 68m)
        """
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.heatmap_resolution = (680, 1050)  # 10 pixels per meter
        
    def generate_team_heatmap(self, tracks, view_transformer, team_id):
        """
        Generate heatmap for an entire team.
        
        Args:
            tracks: Dictionary containing player tracking data
            view_transformer: ViewTransformer object for coordinate transformation
            team_id: Team ID (1 or 2)
            
        Returns:
            numpy.ndarray: Heatmap image
        """
        # Collect all positions for team
        all_positions = []
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, player in player_track.items():
                # Check if player belongs to the team
                if player.get('team', -1) == team_id:
                    bbox = player['bbox']
                    # Get foot position (bottom center of bbox)
                    foot_position = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                    
                    # Transform to pitch coordinates
                    foot_position_array = np.array([foot_position])
                    transformed_positions = view_transformer.transform_points(foot_position_array)
                    
                    if transformed_positions is not None and len(transformed_positions) > 0:
                        all_positions.append(transformed_positions[0])
        
        # Generate heatmap
        if len(all_positions) == 0:
            return self._create_empty_heatmap()
        
        heatmap = self._create_heatmap_from_positions(all_positions)
        return heatmap
    
    def generate_player_heatmap(self, tracks, view_transformer, player_id):
        """
        Generate heatmap for a specific player.
        
        Args:
            tracks: Dictionary containing player tracking data
            view_transformer: ViewTransformer object for coordinate transformation
            player_id: Specific player ID to generate heatmap for
            
        Returns:
            numpy.ndarray: Heatmap image
        """
        # Collect all positions for specific player
        player_positions = []
        
        for frame_num, player_track in enumerate(tracks['players']):
            if player_id in player_track:
                player = player_track[player_id]
                bbox = player['bbox']
                
                # Get foot position (bottom center of bbox)
                foot_position = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                
                # Transform to pitch coordinates
                foot_position_array = np.array([foot_position])
                transformed_positions = view_transformer.transform_points(foot_position_array)
                
                if transformed_positions is not None and len(transformed_positions) > 0:
                    player_positions.append(transformed_positions[0])
        
        # Generate heatmap
        if len(player_positions) == 0:
            return self._create_empty_heatmap()
        
        heatmap = self._create_heatmap_from_positions(player_positions)
        return heatmap
    
    def _create_heatmap_from_positions(self, positions):
        """
        Create heatmap from list of positions.
        
        Args:
            positions: List of (x, y) coordinates in meters
            
        Returns:
            numpy.ndarray: Heatmap image
        """
        # Create empty heatmap grid
        heatmap_grid = np.zeros(self.heatmap_resolution, dtype=np.float32)
        
        # Convert positions to grid coordinates
        for pos in positions:
            x_meters, y_meters = pos
            
            # Convert meters to pixels (10 pixels per meter)
            x_pixel = int(x_meters * 10)
            y_pixel = int(y_meters * 10)
            
            # Ensure within bounds
            if 0 <= x_pixel < self.heatmap_resolution[1] and 0 <= y_pixel < self.heatmap_resolution[0]:
                heatmap_grid[y_pixel, x_pixel] += 1
        
        # Apply Gaussian blur for smooth heatmap
        heatmap_smooth = gaussian_filter(heatmap_grid, sigma=20)
        
        # Normalize to 0-255
        if heatmap_smooth.max() > 0:
            heatmap_normalized = (heatmap_smooth / heatmap_smooth.max() * 255).astype(np.uint8)
        else:
            heatmap_normalized = heatmap_smooth.astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = self._apply_heatmap_colormap(heatmap_normalized)
        
        return heatmap_colored
    
    def _apply_heatmap_colormap(self, heatmap_gray):
        """
        Apply color map to grayscale heatmap.
        
        Args:
            heatmap_gray: Grayscale heatmap (0-255)
            
        Returns:
            numpy.ndarray: Colored heatmap (BGR format for OpenCV)
        """
        # Create custom colormap (blue -> green -> yellow -> red)
        colors = ['#000033', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('heatmap', colors, N=n_bins)
        
        # Apply colormap using matplotlib
        heatmap_colored = cmap(heatmap_gray / 255.0)
        
        # Convert to BGR (OpenCV format) and scale to 0-255
        heatmap_bgr = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2BGR)
        
        return heatmap_bgr
    
    def _create_empty_heatmap(self):
        """Create empty heatmap when no data available."""
        return np.zeros((self.heatmap_resolution[0], self.heatmap_resolution[1], 3), dtype=np.uint8)
    
    def overlay_heatmap_on_pitch(self, heatmap, pitch_image_path='pitch.png', alpha=0.6):
        """
        Overlay heatmap on pitch image.
        
        Args:
            heatmap: Generated heatmap
            pitch_image_path: Path to pitch background image
            alpha: Transparency of heatmap (0-1)
            
        Returns:
            numpy.ndarray: Combined image
        """
        # Load pitch image or create green background
        try:
            pitch_img = cv2.imread(pitch_image_path)
            pitch_img = cv2.resize(pitch_img, (self.heatmap_resolution[1], self.heatmap_resolution[0]))
        except:
            # Create green pitch background
            pitch_img = np.ones((self.heatmap_resolution[0], self.heatmap_resolution[1], 3), dtype=np.uint8)
            pitch_img[:] = (34, 139, 34)  # Green color
            
            # Draw pitch lines
            self._draw_pitch_lines(pitch_img)
        
        # Create mask for non-zero heatmap values
        heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        mask = heatmap_gray > 5
        
        # Overlay heatmap on pitch
        result = pitch_img.copy()
        result[mask] = cv2.addWeighted(pitch_img[mask], 1-alpha, heatmap[mask], alpha, 0)
        
        return result
    
    def _draw_pitch_lines(self, pitch_img):
        """Draw basic pitch lines on green background."""
        height, width = pitch_img.shape[:2]
        white = (255, 255, 255)
        thickness = 2
        
        # Outer boundary
        cv2.rectangle(pitch_img, (50, 50), (width-50, height-50), white, thickness)
        
        # Center line
        cv2.line(pitch_img, (width//2, 50), (width//2, height-50), white, thickness)
        
        # Center circle
        cv2.circle(pitch_img, (width//2, height//2), 91, white, thickness)  # 9.1m radius
        
        # Penalty areas (approximate)
        # Left penalty area
        cv2.rectangle(pitch_img, (50, height//2-165), (50+165, height//2+165), white, thickness)
        # Right penalty area
        cv2.rectangle(pitch_img, (width-50-165, height//2-165), (width-50, height//2+165), white, thickness)
        
        # Goal areas
        # Left goal area
        cv2.rectangle(pitch_img, (50, height//2-55), (50+55, height//2+55), white, thickness)
        # Right goal area
        cv2.rectangle(pitch_img, (width-50-55, height//2-55), (width-50, height//2+55), white, thickness)
    
    def save_heatmap(self, heatmap, output_path, add_pitch=True):
        """
        Save heatmap to file.
        
        Args:
            heatmap: Generated heatmap
            output_path: Path to save image
            add_pitch: Whether to overlay on pitch background
        """
        if heatmap is None:
            print(f"Warning: No heatmap data for {output_path}, creating empty pitch")
            # Create empty pitch image
            pitch_img = np.zeros((self.pitch_height, self.pitch_width, 3), dtype=np.uint8)
            pitch_img[:] = (34, 139, 34)  # Green color
            self._draw_pitch_lines(pitch_img)
            cv2.imwrite(output_path, pitch_img)
            return
        
        if add_pitch:
            final_image = self.overlay_heatmap_on_pitch(heatmap)
        else:
            final_image = heatmap
        
        cv2.imwrite(output_path, final_image)
        print(f"Heatmap saved to: {output_path}")