#!/usr/bin/env python3
"""
Data Collection System for Markerless Prosthetic Control
Collects synchronized data from webcam, XIMU3 IMU, and 5DT dataglove
with task-specific prompts and animations.

Tasks:
1. Wrist Extension-Flexion
2. Wrist Pronation-Supination  
3. Handopen-Powergrasp

Author: Vision Bionics Team
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy import interpolate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCollector:
    """Main data collection coordinator"""
    
    def __init__(self, output_dir: str = "collected_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data queues for thread-safe collection
        self.camera_queue = queue.Queue()
        self.imu_queue = queue.Queue()
        self.glove_queue = queue.Queue()
        
        # Collection control
        self.collecting = False
        self.session_id = None
        self.collection_start_time = None
        self.target_fps = 30  # Target synchronization rate
        
        # Initialize devices
        self.camera = CameraCapture()
        self.imu = XIMU3Interface()
        self.glove = DataGlove5DT()
        self.display = TaskDisplay()
        
        # Set parent reference for collection control
        self.camera._parent = self
        self.imu._parent = self
        self.glove._parent = self
        
    def start_session(self, subject_id: str) -> str:
        """Start new data collection session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{subject_id}_{timestamp}"
        
        session_dir = self.output_dir / self.session_id
        session_dir.mkdir(exist_ok=True)
        
        logger.info(f"Started session: {self.session_id}")
        return self.session_id
    
    def collect_task_data(self, task_name: str, rest_duration: float = 5.0, 
                         task_duration: float = 10.0) -> bool:
        """Collect data for a specific task"""
        if not self.session_id:
            raise ValueError("No active session. Call start_session() first.")
        
        logger.info(f"Starting task: {task_name}")
        
        # Show rest prompt
        self.display.show_rest_prompt(task_name, rest_duration)
        time.sleep(rest_duration)
        
        # Start data collection with synchronized timing
        self.collection_start_time = time.time()
        self.collecting = True
        threads = self._start_collection_threads()
        
        # Show task animation and collect data
        self.display.show_task_animation(task_name, task_duration)
        
        # Stop collection
        self.collecting = False
        self._wait_for_threads(threads)
        
        # Save collected data
        return self._save_task_data(task_name)
    
    def _start_collection_threads(self) -> List[threading.Thread]:
        """Start data collection threads"""
        threads = [
            threading.Thread(target=self.camera.collect_frames, args=(self.camera_queue,)),
            threading.Thread(target=self.imu.collect_quaternions, args=(self.imu_queue,)),
            threading.Thread(target=self.glove.collect_data, args=(self.glove_queue,))
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        return threads
    
    def _wait_for_threads(self, threads: List[threading.Thread]):
        """Wait for collection threads to finish"""
        for thread in threads:
            thread.join(timeout=1.0)
    
    def _save_task_data(self, task_name: str) -> bool:
        """Save synchronized data to files"""
        try:
            task_dir = self.output_dir / self.session_id / task_name
            task_dir.mkdir(exist_ok=True)
            
            # Collect all data
            camera_data = []
            while not self.camera_queue.empty():
                camera_data.append(self.camera_queue.get())
            
            imu_data = []
            while not self.imu_queue.empty():
                imu_data.append(self.imu_queue.get())
            
            glove_data = []
            while not self.glove_queue.empty():
                glove_data.append(self.glove_queue.get())
            
            # Save raw data first
            self._save_raw_data(camera_data, imu_data, glove_data, task_dir)
            
            # Synchronize data to common timeline
            synchronized_data = self._synchronize_data(camera_data, imu_data, glove_data)
            
            # Save synchronized data
            self._save_synchronized_data(synchronized_data, task_dir)
            
            logger.info(f"Saved raw data: {len(camera_data)} frames, {len(imu_data)} IMU, {len(glove_data)} glove")
            logger.info(f"Synchronized and saved {len(synchronized_data['timestamps'])} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def _synchronize_data(self, camera_data: List[Dict], imu_data: List[Dict], glove_data: List[Dict]) -> Dict:
        """Synchronize all sensor data to common timeline"""
        if not camera_data:
            return {'timestamps': [], 'frames': [], 'imu': [], 'glove': []}
        
        # Convert to relative timestamps
        cam_times = np.array([d['timestamp'] - self.collection_start_time for d in camera_data])
        imu_times = np.array([d['timestamp'] - self.collection_start_time for d in imu_data]) if imu_data else np.array([])
        glove_times = np.array([d['timestamp'] - self.collection_start_time for d in glove_data]) if glove_data else np.array([])
        
        # Create common timeline based on camera frames (lowest rate)
        common_times = cam_times
        
        # Interpolate IMU data to camera timeline
        imu_sync = []
        if len(imu_data) > 1:
            imu_quats = np.array([[d['quaternion']['w'], d['quaternion']['x'], 
                                 d['quaternion']['y'], d['quaternion']['z']] for d in imu_data])
            
            for i in range(4):  # w, x, y, z components
                f = interpolate.interp1d(imu_times, imu_quats[:, i], 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                if i == 0:
                    imu_sync = f(common_times).reshape(-1, 1)
                else:
                    imu_sync = np.hstack([imu_sync, f(common_times).reshape(-1, 1)])
        
        # Interpolate glove data to camera timeline
        glove_sync = []
        if len(glove_data) > 1:
            glove_sensors = np.array([d['sensors'] for d in glove_data])
            
            for i in range(glove_sensors.shape[1]):  # Each sensor
                f = interpolate.interp1d(glove_times, glove_sensors[:, i], 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
                if i == 0:
                    glove_sync = f(common_times).reshape(-1, 1)
                else:
                    glove_sync = np.hstack([glove_sync, f(common_times).reshape(-1, 1)])
        
        return {
            'timestamps': common_times.tolist(),
            'frames': [d['frame'] for d in camera_data],
            'imu': imu_sync.tolist() if len(imu_sync) > 0 else [],
            'glove': glove_sync.tolist() if len(glove_sync) > 0 else []
        }
    
    def _save_synchronized_data(self, sync_data: Dict, output_dir: Path):
        """Save synchronized data"""
        # Save frames with synchronized indices
        for i, frame in enumerate(sync_data['frames']):
            filename = output_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(filename), frame)
        
        # Save synchronized data as NPZ for efficient loading
        np.savez_compressed(
            output_dir / "synchronized_data.npz",
            timestamps=np.array(sync_data['timestamps']),
            imu_quaternions=np.array(sync_data['imu']) if sync_data['imu'] else np.array([]),
            glove_sensors=np.array(sync_data['glove']) if sync_data['glove'] else np.array([])
        )
        
        # Also save as JSON for readability
        metadata = {
            'num_samples': len(sync_data['timestamps']),
            'duration': sync_data['timestamps'][-1] if sync_data['timestamps'] else 0,
            'sample_rate': len(sync_data['timestamps']) / sync_data['timestamps'][-1] if sync_data['timestamps'] else 0
        }
        
        with open(output_dir / "sync_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_raw_data(self, camera_data: List[Dict], imu_data: List[Dict], glove_data: List[Dict], output_dir: Path):
        """Save raw unsynchronized data for verification"""
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        # Save raw camera data
        if camera_data:
            cam_timestamps = [d['timestamp'] - self.collection_start_time for d in camera_data]
            with open(raw_dir / "camera_raw.json", 'w') as f:
                json.dump({
                    'timestamps': cam_timestamps,
                    'frame_count': len(camera_data),
                    'fps': len(camera_data) / (cam_timestamps[-1] - cam_timestamps[0]) if len(cam_timestamps) > 1 else 0
                }, f, indent=2)
        
        # Save raw IMU data
        if imu_data:
            imu_raw = {
                'timestamps': [d['timestamp'] - self.collection_start_time for d in imu_data],
                'quaternions': [[d['quaternion']['w'], d['quaternion']['x'], 
                               d['quaternion']['y'], d['quaternion']['z']] for d in imu_data],
                'sample_count': len(imu_data)
            }
            with open(raw_dir / "imu_raw.json", 'w') as f:
                json.dump(imu_raw, f, indent=2)
        
        # Save raw glove data
        if glove_data:
            glove_raw = {
                'timestamps': [d['timestamp'] - self.collection_start_time for d in glove_data],
                'sensors': [d['sensors'] for d in glove_data],
                'sample_count': len(glove_data)
            }
            with open(raw_dir / "glove_raw.json", 'w') as f:
                json.dump(glove_raw, f, indent=2)


class CameraCapture:
    """Webcam capture interface"""
    
    def __init__(self, camera_id: int = 0, fps: int = 30):
        self.camera_id = camera_id
        self.fps = fps
        self.cap = None
        
    def initialize(self) -> bool:
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info(f"Camera initialized at {self.fps} FPS")
        return True
    
    def collect_frames(self, data_queue: queue.Queue):
        """Collect camera frames in separate thread"""
        if not self.initialize():
            return
        
        frame_interval = 1.0 / self.fps
        
        while self._parent.collecting:
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.time()
                data_queue.put({
                    'timestamp': timestamp,
                    'frame': frame.copy(),
                    'frame_id': len(data_queue.queue)
                })
            
            time.sleep(frame_interval)
        
        self.cap.release()


class XIMU3Interface:
    """XIMU3 IMU sensor interface using dataCollectionAmpConcise approach"""
    
    def __init__(self, ip: str = "192.168.42.2", port: int = 7000):
        self.ip = ip
        self.port = port
        self.reader = None
    
    def initialize(self) -> bool:
        """Initialize XIMU3 connection"""
        try:
            from ximu3_reader import XIMU3Reader
            self.reader = XIMU3Reader(self.ip, self.port)
            self.reader.open()
            logger.info(f"XIMU3 initialized on {self.ip}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize XIMU3: {e}")
            return False
    
    def collect_quaternions(self, data_queue: queue.Queue):
        """Collect quaternion data from XIMU3"""
        if not self.initialize():
            return
        
        while self._parent.collecting:
            try:
                sample = self.reader.get_latest()
                timestamp = time.time()
                data_queue.put({
                    'timestamp': timestamp,
                    'quaternion': {
                        'w': sample.qw,
                        'x': sample.qx,
                        'y': sample.qy,
                        'z': sample.qz
                    },
                    'sample_id': len(data_queue.queue)
                })
            except Exception as e:
                logger.warning(f"XIMU3 read error: {e}")
            
            time.sleep(0.01)
        
        if self.reader:
            self.reader.close()


class DataGlove5DT:
    """5DT Data Glove interface using dataCollectionAmpConcise approach"""
    
    def __init__(self, dll: str = "fglove64.dll", dev: str = "USB0", hz: int = 1000):
        self.dll = dll
        self.dev = dev
        self.hz = hz
        self.glove = None
    
    def initialize(self) -> bool:
        """Initialize 5DT glove connection"""
        try:
            from fivedt_direct import FiveDTglove
            self.glove = FiveDTglove(dll=self.dll, dev=self.dev, hz=self.hz, stale_ms=10)
            self.glove.start()
            logger.info(f"5DT Glove initialized on {self.dev}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize 5DT glove: {e}")
            return False
    
    def collect_data(self, data_queue: queue.Queue):
        """Collect sensor data from 5DT glove"""
        if not self.initialize():
            return
        
        while self._parent.collecting:
            try:
                values = self.glove.raw()
                if values is not None:
                    timestamp = time.time()
                    data_queue.put({
                        'timestamp': timestamp,
                        'sensors': values
                    })
            except Exception as e:
                logger.warning(f"Glove read error: {e}")
            
            time.sleep(0.01)
        
        if self.glove:
            self.glove.close()


class TaskDisplay:
    """Display task prompts and animations"""
    
    def __init__(self, window_name: str = "Task Display"):
        self.window_name = window_name
        self.window_size = (800, 600)
        
    def show_rest_prompt(self, task_name: str, duration: float):
        """Show rest prompt before task"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            remaining = int(end_time - time.time())
            
            # Create display image
            img = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            
            # Add text
            text1 = f"Prepare for: {task_name}"
            text2 = f"Stay at rest: {remaining}s"
            
            cv2.putText(img, text1, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, text2, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, img)
            cv2.waitKey(100)
    
    def show_task_animation(self, task_name: str, duration: float):
        """Show task animation during data collection"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            remaining = duration - (time.time() - (end_time - duration))
            progress = remaining / duration
            
            # Create animation based on task
            img = self._create_task_animation(task_name, progress)
            
            # Add timer
            timer_text = f"Time: {remaining:.1f}s"
            cv2.putText(img, timer_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow(self.window_name, img)
            cv2.waitKey(50)
        
        cv2.destroyWindow(self.window_name)
    
    def _create_task_animation(self, task_name: str, progress: float) -> np.ndarray:
        """Create task-specific animation"""
        img = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        
        center_x, center_y = self.window_size[0] // 2, self.window_size[1] // 2
        
        if "Extension-Flexion" in task_name:
            # Vertical movement animation
            offset = int(100 * np.sin(progress * 4 * np.pi))
            cv2.circle(img, (center_x, center_y + offset), 30, (0, 255, 0), -1)
            cv2.putText(img, "Wrist Extension-Flexion", (center_x - 150, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif "Pronation-Supination" in task_name:
            # Rotation animation
            angle = progress * 4 * np.pi
            x_offset = int(100 * np.cos(angle))
            cv2.ellipse(img, (center_x, center_y), (100, 30), int(np.degrees(angle)), 
                       0, 360, (0, 0, 255), -1)
            cv2.putText(img, "Wrist Pronation-Supination", (center_x - 170, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        elif "Hand" in task_name:
            # Opening/closing animation
            radius = int(20 + 40 * (0.5 + 0.5 * np.sin(progress * 4 * np.pi)))
            cv2.circle(img, (center_x, center_y), radius, (255, 0, 0), -1)
            cv2.putText(img, "Hand Open-Power Grasp", (center_x - 150, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img


def main():
    """Main data collection workflow"""
    # Initialize system
    collector = DataCollector()
    
    # Define tasks
    tasks = [
        "Wrist Extension-Flexion",
        "Wrist Pronation-Supination", 
        "Handopen-PG"
    ]
    
    # Get subject information
    subject_id = input("Enter subject ID: ").strip()
    if not subject_id:
        subject_id = "test_subject"
    
    # Start session
    session_id = collector.start_session(subject_id)
    print(f"Session started: {session_id}")
    
    # Collect data for each task
    for task in tasks:
        print(f"\nPreparing for task: {task}")
        input("Press Enter when ready...")
        
        success = collector.collect_task_data(
            task_name=task,
            rest_duration=10.0,
            task_duration=60.0
        )
        
        if success:
            print(f"✓ Task '{task}' completed successfully")
        else:
            print(f"✗ Task '{task}' failed")
    
    print(f"\nData collection complete. Files saved to: {collector.output_dir / session_id}")


if __name__ == "__main__":
    main()