#!/usr/bin/env python3
"""
Data Loader for Synchronized Markerless Prosthetic Control Data
Loads and processes synchronized camera, IMU, and glove data for model training.
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SynchronizedDataLoader:
    """Load and process synchronized multi-modal data"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.metadata = None
        self.timestamps = None
        self.imu_data = None
        self.glove_data = None
        self.frames_path = None
        
    def load_task_data(self, task_name: str) -> bool:
        """Load synchronized data for a specific task"""
        task_dir = self.data_path / task_name
        
        if not task_dir.exists():
            logger.error(f"Task directory not found: {task_dir}")
            return False
        
        try:
            # Load synchronized data
            sync_file = task_dir / "synchronized_data.npz"
            if sync_file.exists():
                data = np.load(sync_file)
                self.timestamps = data['timestamps']
                self.imu_data = data['imu_quaternions']
                self.glove_data = data['glove_sensors']
            
            # Load metadata
            meta_file = task_dir / "sync_metadata.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    self.metadata = json.load(f)
            
            self.frames_path = task_dir
            
            logger.info(f"Loaded {len(self.timestamps)} synchronized samples for {task_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading task data: {e}")
            return False
    
    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get camera frame at specific index"""
        if self.frames_path is None:
            return None
        
        frame_file = self.frames_path / f"frame_{index:06d}.jpg"
        if frame_file.exists():
            return cv2.imread(str(frame_file))
        return None
    
    def get_sample(self, index: int) -> Dict:
        """Get complete synchronized sample at index"""
        if index >= len(self.timestamps):
            return {}
        
        sample = {
            'timestamp': self.timestamps[index],
            'frame': self.get_frame(index)
        }
        
        if len(self.imu_data) > 0:
            sample['imu_quaternion'] = self.imu_data[index]
        
        if len(self.glove_data) > 0:
            sample['glove_sensors'] = self.glove_data[index]
        
        return sample
    
    def get_batch(self, start_idx: int, batch_size: int) -> List[Dict]:
        """Get batch of synchronized samples"""
        batch = []
        for i in range(start_idx, min(start_idx + batch_size, len(self.timestamps))):
            batch.append(self.get_sample(i))
        return batch
    
    def get_time_window(self, start_time: float, duration: float) -> List[Dict]:
        """Get samples within time window"""
        start_idx = np.searchsorted(self.timestamps, start_time)
        end_time = start_time + duration
        end_idx = np.searchsorted(self.timestamps, end_time)
        
        return [self.get_sample(i) for i in range(start_idx, end_idx)]
    
    def get_statistics(self) -> Dict:
        """Get data statistics"""
        if self.metadata is None:
            return {}
        
        stats = self.metadata.copy()
        
        if len(self.imu_data) > 0:
            stats['imu_stats'] = {
                'mean_quaternion': np.mean(self.imu_data, axis=0).tolist(),
                'std_quaternion': np.std(self.imu_data, axis=0).tolist()
            }
        
        if len(self.glove_data) > 0:
            stats['glove_stats'] = {
                'mean_sensors': np.mean(self.glove_data, axis=0).tolist(),
                'std_sensors': np.std(self.glove_data, axis=0).tolist(),
                'min_sensors': np.min(self.glove_data, axis=0).tolist(),
                'max_sensors': np.max(self.glove_data, axis=0).tolist()
            }
        
        return stats


def load_session_data(session_path: str) -> Dict[str, SynchronizedDataLoader]:
    """Load all tasks from a session"""
    session_dir = Path(session_path)
    loaders = {}
    
    for task_dir in session_dir.iterdir():
        if task_dir.is_dir():
            loader = SynchronizedDataLoader(session_dir)
            if loader.load_task_data(task_dir.name):
                loaders[task_dir.name] = loader
    
    return loaders


# Example usage
if __name__ == "__main__":
    # Load session data
    session_path = "collected_data/subject1_20241201_143000"
    loaders = load_session_data(session_path)
    
    for task_name, loader in loaders.items():
        print(f"\nTask: {task_name}")
        print(f"Samples: {len(loader.timestamps)}")
        print(f"Duration: {loader.metadata['duration']:.2f}s")
        print(f"Sample rate: {loader.metadata['sample_rate']:.1f} Hz")
        
        # Get first sample
        sample = loader.get_sample(0)
        print(f"Frame shape: {sample['frame'].shape if sample['frame'] is not None else 'None'}")
        print(f"IMU quaternion: {sample.get('imu_quaternion', 'None')}")
        print(f"Glove sensors: {len(sample.get('glove_sensors', []))} sensors")