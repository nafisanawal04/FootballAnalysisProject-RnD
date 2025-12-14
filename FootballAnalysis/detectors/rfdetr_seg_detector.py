import sys
sys.path.append('../')
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import json
from typing import List, Dict, Any

class RFDETRSegDetector:
    """
    RFDETR-SEG detector for football analysis.
    Handles detection of ball, goalkeeper, player, and referee classes.
    """
    
    def __init__(self, api_key: str, workspace: str, project: str, version: int):
        """
        Initialize the RFDETR-SEG detector.
        
        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Roboflow project name
            version: Model version number
        """
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace(workspace).project(project)
        self.model = self.project.version(version).model
        
        # Class mapping: 0=ball, 1=goalkeeper, 2=player, 3=referee
        self.class_names = {
            0: 'ball',
            1: 'goalkeeper', 
            2: 'player',
            3: 'referee'
        }
        
        # Reverse mapping for class names to IDs
        self.class_names_inv = {v: k for k, v in self.class_names.items()}
        
    def predict_frame(self, frame: np.ndarray, confidence: float = 0.4) -> Dict[str, Any]:
        """
        Predict objects in a single frame.
        
        Args:
            frame: Input frame as numpy array
            confidence: Confidence threshold for detections
            
        Returns:
            Dictionary containing predictions in the specified format
        """
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        # Run prediction
        result = self.model.predict(frame_rgb, confidence=confidence).json()
        
        return result
    
    def predict_frames_batch(self, frames: List[np.ndarray], confidence: float = 0.4) -> List[Dict[str, Any]]:
        """
        Predict objects in a batch of frames.
        
        Args:
            frames: List of input frames
            confidence: Confidence threshold for detections
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for frame in frames:
            prediction = self.predict_frame(frame, confidence)
            predictions.append(prediction)
            
        return predictions
    
    def convert_to_supervision_format(self, prediction: Dict[str, Any]) -> sv.Detections:
        """
        Convert RFDETR-SEG prediction format to supervision format.
        
        Args:
            prediction: Prediction dictionary from RFDETR-SEG
            
        Returns:
            supervision.Detections object
        """
        if not prediction or 'predictions' not in prediction:
            return sv.Detections.empty()
            
        predictions = prediction['predictions']
        if not predictions:
            return sv.Detections.empty()
        
        # Extract data for supervision format
        xyxy = []
        confidence = []
        class_id = []
        
        for pred in predictions:
            # Convert center-based bbox to xyxy format
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            x1 = x - width / 2
            y1 = y - height / 2
            x2 = x + width / 2
            y2 = y + height / 2
            
            xyxy.append([x1, y1, x2, y2])
            confidence.append(pred['confidence'])
            class_id.append(pred['class_id'])
        
        if not xyxy:
            return sv.Detections.empty()
            
        # Create supervision detections
        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )
        
        return detections
    
    def filter_detections_by_class(self, detections: sv.Detections, target_classes: List[str]) -> sv.Detections:
        """
        Filter detections to only include specific classes.
        
        Args:
            detections: supervision.Detections object
            target_classes: List of class names to keep
            
        Returns:
            Filtered supervision.Detections object
        """
        if detections.empty:
            return detections
            
        # Get class IDs for target classes
        target_class_ids = [self.class_names_inv[cls] for cls in target_classes if cls in self.class_names_inv]
        
        if not target_class_ids:
            return sv.Detections.empty()
        
        # Create mask for target classes
        mask = np.isin(detections.class_id, target_class_ids)
        
        if not np.any(mask):
            return sv.Detections.empty()
        
        # Filter detections
        filtered_detections = detections[mask]
        
        return filtered_detections
    
    def get_ball_detections(self, detections: sv.Detections) -> sv.Detections:
        """Get only ball detections."""
        return self.filter_detections_by_class(detections, ['ball'])
    
    def get_player_detections(self, detections: sv.Detections) -> sv.Detections:
        """Get only player and goalkeeper detections."""
        return self.filter_detections_by_class(detections, ['player', 'goalkeeper'])
    
    def get_referee_detections(self, detections: sv.Detections) -> sv.Detections:
        """Get only referee detections."""
        return self.filter_detections_by_class(detections, ['referee'])
