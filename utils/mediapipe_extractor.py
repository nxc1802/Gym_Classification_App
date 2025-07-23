import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import suppress_warnings

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class MediaPipeExtractor:
    def __init__(self):
        """Initialize MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        
        # List of joint names in order (from original Mediapipe.py)
        self.joint_names = [name for name in self.mp_pose.PoseLandmark.__members__.keys()]
        
        # Initialize pose detector
        self.pose = None
        self._init_pose_detector()
    
    def _init_pose_detector(self):
        """Initialize or reinitialize pose detector"""
        if self.pose is not None:
            try:
                self.pose.close()
            except:
                pass
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0
        )
    
    def extract_from_video(self, video_path):
        """
        Extract pose landmarks from video file
        Returns: DataFrame with pose landmarks (same format as original pipeline)
        """
        logger.info(f"Extracting pose landmarks from: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Reinitialize pose detector for each video to avoid reuse issues
        self._init_pose_detector()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties - FPS: {fps}, Frames: {total_frames}, Size: {width}x{height}")
        
        frame_idx = 0
        detected = 0
        records = []
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                frame_idx += 1
                h, w = frame.shape[:2]
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    detected += 1
                    values = []
                    
                    # Extract landmarks in pixel coordinates (x, y) and original z, visibility
                    for landmark in results.pose_landmarks.landmark:
                        # Convert normalized coordinates to pixel coordinates
                        x_px = int(landmark.x * w)  # multiply by width 
                        y_px = int(landmark.y * h)  # multiply by height
                        # Keep z and visibility as original values
                        values.extend([x_px, y_px, landmark.z, landmark.visibility])
                else:
                    # Fill with zeros if no pose detected
                    values = [0] * (len(self.joint_names) * 4)
                
                # Add frame index and landmark values
                records.append([frame_idx] + values)
        
        finally:
            cap.release()
        
        # Create DataFrame with proper column names
        columns = ["frame"]
        for joint_name in self.joint_names:
            for suffix in ["x", "y", "z", "visibility"]:
                columns.append(f"{joint_name}_{suffix}")
        
        df = pd.DataFrame(records, columns=columns)
        
        logger.info(f"Extraction complete - Total frames: {frame_idx}, Detected: {detected}")
        logger.info(f"DataFrame shape: {df.shape}")
        
        return df
    
    def process_landmarks_for_stgcn(self, df):
        """
        Process landmarks for ST-GCN model
        Returns data in format [T, V, C] where T=time, V=joints, C=channels
        """
        # ST-GCN joint order from original code
        JOINT_NAMES_STGCN = [
            'NOSE','LEFT_EYE_INNER','LEFT_EYE','LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER','RIGHT_EYE','RIGHT_EYE_OUTER',
            'LEFT_EAR','RIGHT_EAR','MOUTH_LEFT','MOUTH_RIGHT',
            'LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW',
            'LEFT_WRIST','RIGHT_WRIST','LEFT_PINKY','RIGHT_PINKY',
            'LEFT_INDEX','RIGHT_INDEX','LEFT_THUMB','RIGHT_THUMB',
            'LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE',
            'LEFT_ANKLE','RIGHT_ANKLE','LEFT_HEEL','RIGHT_HEEL'
        ]
        
        DIMS = ["x", "y", "z", "visibility"]
        V, C = len(JOINT_NAMES_STGCN), len(DIMS)
        
        # Build column names
        cols = [f"{j}_{d}" for j in JOINT_NAMES_STGCN for d in DIMS]
        
        # Extract data and fill missing columns with 0
        data = np.zeros((len(df), len(cols)), dtype=np.float32)
        for i, col in enumerate(cols):
            if col in df.columns:
                data[:, i] = df[col].fillna(0.0).values
        
        T = data.shape[0]
        data = data.reshape(T, V, C)
        
        # Normalize to nose position (relative coordinates)
        nose_idx = JOINT_NAMES_STGCN.index("NOSE")
        coords = data[..., :3]  # x, y, z
        nose_pos = coords[:, nose_idx:nose_idx+1, :]  # [T, 1, 3]
        coords = coords - nose_pos  # Make relative to nose
        
        # Combine normalized coords with visibility
        data = np.concatenate([coords, data[..., 3:4]], axis=-1)
        
        return data
    
    def process_landmarks_for_transformer_12rel(self, df):
        """
        Process landmarks for 12-relative transformer model
        """
        RAW_POINTS_13 = [
            "NOSE","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
            "LEFT_WRIST","RIGHT_WRIST","LEFT_HIP","RIGHT_HIP","LEFT_KNEE",
            "RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"
        ]
        DIMS_TF = ["x","y","z","visibility"]
        
        # Get base point (NOSE) coordinates
        base_cols = [f"NOSE_{d}" for d in DIMS_TF]
        base_data = df[base_cols].fillna(0.0).values
        
        # Process other points relative to nose
        rel_features = []
        for joint in RAW_POINTS_13:
            if joint == "NOSE":
                continue
            
            for i, dim in enumerate(DIMS_TF):
                col_name = f"{joint}_{dim}"
                if col_name in df.columns:
                    values = df[col_name].fillna(0.0).values
                    
                    # Make relative to nose for x, y, z but keep visibility as is
                    if dim != "visibility":
                        values = values - base_data[:, i]
                    
                    rel_features.append(values)
                else:
                    # Fill with zeros if column missing
                    rel_features.append(np.zeros(len(df)))
        
        # Stack into array [T, 48] (12 joints * 4 dims)
        rel_data = np.stack(rel_features, axis=1).astype(np.float32)
        
        return rel_data
    
    def process_landmarks_for_transformer_angle(self, df):
        """
        Process landmarks for angle-branch transformer model
        Returns: (rel_branch_data, angle_branch_data)
        """
        from itertools import combinations
        
        RAW_POINTS_13 = [
            "NOSE","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
            "LEFT_WRIST","RIGHT_WRIST","LEFT_HIP","RIGHT_HIP","LEFT_KNEE",
            "RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE"
        ]
        
        # 1. Get 12-relative features (same as transformer_12rel)
        rel_data = self.process_landmarks_for_transformer_12rel(df)
        
        # 2. Get angle features
        dims_rel = ["x","y","z","visibility"]
        cols_rel = [f"{pt}_{d}" for pt in RAW_POINTS_13 for d in dims_rel]
        cols_ang = [f"{pt}_{d}" for pt in RAW_POINTS_13 for d in ["x","y"]]
        
        # Get coordinates for angle calculation (only x, y)
        coord_data = np.zeros((len(df), len(RAW_POINTS_13), 2))
        for i, joint in enumerate(RAW_POINTS_13):
            for j, dim in enumerate(["x", "y"]):
                col_name = f"{joint}_{dim}"
                if col_name in df.columns:
                    coord_data[:, i, j] = df[col_name].fillna(0.0).values
        
        # Calculate angles between all combinations of 3 points
        COMB3 = np.array(list(combinations(range(13), 3)), dtype=np.int32)
        
        T, _, _ = coord_data.shape
        A_pts = coord_data[:, COMB3[:,0], :]  # [T, n_combinations, 2]
        B_pts = coord_data[:, COMB3[:,1], :]  # [T, n_combinations, 2]
        C_pts = coord_data[:, COMB3[:,2], :]  # [T, n_combinations, 2]
        
        # Calculate vectors
        v1 = A_pts - B_pts  # Vector from B to A
        v2 = C_pts - B_pts  # Vector from B to C
        
        # Calculate angles
        dot_product = np.sum(v1 * v2, axis=2)
        norms = (np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2)) + 1e-6
        angles = np.arccos(np.clip(dot_product / norms, -1, 1))  # [T, 286]
        
        return rel_data, angles
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
                self.pose = None
            except Exception as e:
                logger.warning(f"Error during MediaPipe cleanup: {e}")
                self.pose = None 