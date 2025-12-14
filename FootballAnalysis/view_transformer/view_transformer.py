import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([[110, 1035], 
                                        [265, 275], 
                                        [910, 260], 
                                        [1640, 915]])

        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_points(self, points):
        # Ensure points are not empty or invalid
        if points is None or len(points) == 0:
            print("Invalid points detected, skipping transformation.")
            return None

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        
        # Perform the perspective transform
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.persepctive_trasnformer)
        
        # Ensure the transformation is valid
        if transformed_points is None:
            print("Perspective transform failed, returning None.")
            return None
        
        return transformed_points.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                positions = []
                track_ids = []

                # Collect all positions and track_ids for this frame
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    positions.append(np.array(position))
                    track_ids.append(track_id)
                
                # Convert positions to numpy array
                positions = np.array(positions)

                # Transform all points at once
                transformed_positions = self.transform_points(positions)

                # Only proceed if the transformation was successful
                if transformed_positions is not None:
                    # Store transformed positions back into tracks
                    for track_id, transformed_position in zip(track_ids, transformed_positions):
                        tracks[object][frame_num][track_id]['position_transformed'] = transformed_position.tolist()
                else:
                    print(f"Skipping frame {frame_num} for object {object} due to transformation failure.")
