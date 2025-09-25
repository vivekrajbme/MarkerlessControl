#!/usr/bin/env python3
"""
Synchronization Verification Tool
Analyzes and visualizes synchronization accuracy between raw and synchronized data.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SyncVerifier:
    """Verify synchronization accuracy"""
    
    def __init__(self, task_dir: str):
        self.task_dir = Path(task_dir)
        self.raw_data = {}
        self.sync_data = {}
        
    def load_data(self) -> bool:
        """Load raw and synchronized data"""
        try:
            # Load raw data
            raw_dir = self.task_dir / "raw"
            if raw_dir.exists():
                for file in ["camera_raw.json", "imu_raw.json", "glove_raw.json"]:
                    filepath = raw_dir / file
                    if filepath.exists():
                        with open(filepath, 'r') as f:
                            self.raw_data[file.split('_')[0]] = json.load(f)
            
            # Load synchronized data
            sync_file = self.task_dir / "synchronized_data.npz"
            if sync_file.exists():
                data = np.load(sync_file)
                self.sync_data = {
                    'timestamps': data['timestamps'],
                    'imu': data['imu_quaternions'] if 'imu_quaternions' in data else np.array([]),
                    'glove': data['glove_sensors'] if 'glove_sensors' in data else np.array([])
                }
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def analyze_timing(self) -> Dict:
        """Analyze timing accuracy"""
        analysis = {}
        
        # Camera timing analysis
        if 'camera' in self.raw_data:
            cam_times = np.array(self.raw_data['camera']['timestamps'])
            cam_intervals = np.diff(cam_times)
            analysis['camera'] = {
                'raw_samples': len(cam_times),
                'sync_samples': len(self.sync_data['timestamps']),
                'mean_interval': np.mean(cam_intervals),
                'std_interval': np.std(cam_intervals),
                'expected_fps': self.raw_data['camera'].get('fps', 0),
                'actual_fps': 1.0 / np.mean(cam_intervals) if len(cam_intervals) > 0 else 0
            }
        
        # IMU timing analysis
        if 'imu' in self.raw_data:
            imu_times = np.array(self.raw_data['imu']['timestamps'])
            imu_intervals = np.diff(imu_times)
            analysis['imu'] = {
                'raw_samples': len(imu_times),
                'sync_samples': len(self.sync_data['imu']) if len(self.sync_data['imu']) > 0 else 0,
                'mean_interval': np.mean(imu_intervals),
                'std_interval': np.std(imu_intervals),
                'sample_rate': 1.0 / np.mean(imu_intervals) if len(imu_intervals) > 0 else 0
            }
        
        # Glove timing analysis
        if 'glove' in self.raw_data:
            glove_times = np.array(self.raw_data['glove']['timestamps'])
            glove_intervals = np.diff(glove_times)
            analysis['glove'] = {
                'raw_samples': len(glove_times),
                'sync_samples': len(self.sync_data['glove']) if len(self.sync_data['glove']) > 0 else 0,
                'mean_interval': np.mean(glove_intervals),
                'std_interval': np.std(glove_intervals),
                'sample_rate': 1.0 / np.mean(glove_intervals) if len(glove_intervals) > 0 else 0
            }
        
        return analysis
    
    def check_interpolation_accuracy(self) -> Dict:
        """Check interpolation accuracy by comparing raw vs interpolated values"""
        accuracy = {}
        
        # Check IMU interpolation accuracy
        if 'imu' in self.raw_data and len(self.sync_data['imu']) > 0:
            raw_times = np.array(self.raw_data['imu']['timestamps'])
            raw_quats = np.array(self.raw_data['imu']['quaternions'])
            sync_times = self.sync_data['timestamps']
            sync_quats = self.sync_data['imu']
            
            # Find closest matches
            errors = []
            for i, sync_time in enumerate(sync_times):
                closest_idx = np.argmin(np.abs(raw_times - sync_time))
                if np.abs(raw_times[closest_idx] - sync_time) < 0.1:  # Within 100ms
                    error = np.linalg.norm(raw_quats[closest_idx] - sync_quats[i])
                    errors.append(error)
            
            accuracy['imu'] = {
                'mean_error': np.mean(errors) if errors else float('inf'),
                'max_error': np.max(errors) if errors else float('inf'),
                'matched_samples': len(errors)
            }
        
        # Check glove interpolation accuracy
        if 'glove' in self.raw_data and len(self.sync_data['glove']) > 0:
            raw_times = np.array(self.raw_data['glove']['timestamps'])
            raw_sensors = np.array(self.raw_data['glove']['sensors'])
            sync_times = self.sync_data['timestamps']
            sync_sensors = self.sync_data['glove']
            
            # Find closest matches
            errors = []
            for i, sync_time in enumerate(sync_times):
                closest_idx = np.argmin(np.abs(raw_times - sync_time))
                if np.abs(raw_times[closest_idx] - sync_time) < 0.1:  # Within 100ms
                    error = np.mean(np.abs(raw_sensors[closest_idx] - sync_sensors[i]))
                    errors.append(error)
            
            accuracy['glove'] = {
                'mean_error': np.mean(errors) if errors else float('inf'),
                'max_error': np.max(errors) if errors else float('inf'),
                'matched_samples': len(errors)
            }
        
        return accuracy
    
    def plot_timing_analysis(self, save_path: str = None):
        """Plot timing analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Synchronization Analysis', fontsize=16)
        
        # Plot 1: Sample rates comparison
        ax1 = axes[0, 0]
        devices = []
        raw_rates = []
        sync_rates = []
        
        analysis = self.analyze_timing()
        for device, data in analysis.items():
            devices.append(device.upper())
            raw_rates.append(data.get('sample_rate', data.get('actual_fps', 0)))
            sync_rate = data['sync_samples'] / (self.sync_data['timestamps'][-1] - self.sync_data['timestamps'][0]) if len(self.sync_data['timestamps']) > 1 else 0
            sync_rates.append(sync_rate)
        
        x = np.arange(len(devices))
        width = 0.35
        ax1.bar(x - width/2, raw_rates, width, label='Raw', alpha=0.8)
        ax1.bar(x + width/2, sync_rates, width, label='Synchronized', alpha=0.8)
        ax1.set_xlabel('Device')
        ax1.set_ylabel('Sample Rate (Hz)')
        ax1.set_title('Sample Rates Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(devices)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Timeline visualization
        ax2 = axes[0, 1]
        colors = ['blue', 'red', 'green']
        labels = ['Camera', 'IMU', 'Glove']
        
        for i, (device, color, label) in enumerate(zip(['camera', 'imu', 'glove'], colors, labels)):
            if device in self.raw_data:
                times = np.array(self.raw_data[device]['timestamps'])
                y_pos = np.full_like(times, i)
                ax2.scatter(times, y_pos, c=color, alpha=0.6, s=10, label=f'{label} Raw')
        
        # Plot synchronized timeline
        sync_times = self.sync_data['timestamps']
        y_sync = np.full_like(sync_times, 3)
        ax2.scatter(sync_times, y_sync, c='black', alpha=0.8, s=15, marker='|', label='Synchronized')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Device')
        ax2.set_title('Timeline Visualization')
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['Camera', 'IMU', 'Glove', 'Sync'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Interpolation errors
        ax3 = axes[1, 0]
        accuracy = self.check_interpolation_accuracy()
        
        devices = []
        mean_errors = []
        max_errors = []
        
        for device, data in accuracy.items():
            devices.append(device.upper())
            mean_errors.append(data['mean_error'])
            max_errors.append(data['max_error'])
        
        if devices:
            x = np.arange(len(devices))
            width = 0.35
            ax3.bar(x - width/2, mean_errors, width, label='Mean Error', alpha=0.8)
            ax3.bar(x + width/2, max_errors, width, label='Max Error', alpha=0.8)
            ax3.set_xlabel('Device')
            ax3.set_ylabel('Interpolation Error')
            ax3.set_title('Interpolation Accuracy')
            ax3.set_xticks(x)
            ax3.set_xticklabels(devices)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # Plot 4: Sample count comparison
        ax4 = axes[1, 1]
        devices = []
        raw_counts = []
        sync_counts = []
        
        for device, data in analysis.items():
            devices.append(device.upper())
            raw_counts.append(data['raw_samples'])
            sync_counts.append(data['sync_samples'])
        
        x = np.arange(len(devices))
        width = 0.35
        ax4.bar(x - width/2, raw_counts, width, label='Raw Samples', alpha=0.8)
        ax4.bar(x + width/2, sync_counts, width, label='Sync Samples', alpha=0.8)
        ax4.set_xlabel('Device')
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Sample Count Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(devices)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> str:
        """Generate synchronization verification report"""
        analysis = self.analyze_timing()
        accuracy = self.check_interpolation_accuracy()
        
        report = "SYNCHRONIZATION VERIFICATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Timing analysis
        report += "TIMING ANALYSIS:\n"
        report += "-" * 20 + "\n"
        for device, data in analysis.items():
            report += f"{device.upper()}:\n"
            report += f"  Raw samples: {data['raw_samples']}\n"
            report += f"  Sync samples: {data['sync_samples']}\n"
            if 'actual_fps' in data:
                report += f"  Actual FPS: {data['actual_fps']:.2f}\n"
            elif 'sample_rate' in data:
                report += f"  Sample rate: {data['sample_rate']:.2f} Hz\n"
            report += f"  Timing std: {data['std_interval']:.4f}s\n\n"
        
        # Interpolation accuracy
        report += "INTERPOLATION ACCURACY:\n"
        report += "-" * 25 + "\n"
        for device, data in accuracy.items():
            report += f"{device.upper()}:\n"
            report += f"  Mean error: {data['mean_error']:.6f}\n"
            report += f"  Max error: {data['max_error']:.6f}\n"
            report += f"  Matched samples: {data['matched_samples']}\n\n"
        
        # Overall assessment
        report += "OVERALL ASSESSMENT:\n"
        report += "-" * 20 + "\n"
        
        # Check if all devices have similar sample counts after sync
        sync_counts = [data['sync_samples'] for data in analysis.values()]
        if len(set(sync_counts)) == 1:
            report += "✓ All devices have matching sample counts after synchronization\n"
        else:
            report += "✗ Sample count mismatch detected\n"
        
        # Check interpolation errors
        max_errors = [data['mean_error'] for data in accuracy.values() if data['mean_error'] != float('inf')]
        if max_errors and max(max_errors) < 0.01:
            report += "✓ Interpolation errors are within acceptable range\n"
        elif max_errors:
            report += "⚠ High interpolation errors detected\n"
        
        return report


def verify_task_synchronization(task_dir: str, save_plots: bool = True) -> str:
    """Verify synchronization for a task"""
    verifier = SyncVerifier(task_dir)
    
    if not verifier.load_data():
        return "Failed to load data"
    
    # Generate plots
    if save_plots:
        plot_path = Path(task_dir) / "sync_verification.png"
        verifier.plot_timing_analysis(str(plot_path))
    
    # Generate report
    report = verifier.generate_report()
    
    # Save report
    report_path = Path(task_dir) / "sync_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report


# Example usage
if __name__ == "__main__":
    task_dir = "collected_data/subject1_20241201_143000/Wrist Extension-Flexion"
    report = verify_task_synchronization(task_dir)
    print(report)