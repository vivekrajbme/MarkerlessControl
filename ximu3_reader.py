# ximu3_reader.py
from typing import Optional, List, Tuple
import threading
import time
from dataclasses import dataclass

# Import these conditionally based on USE_XIMU3 elsewhere
try:
    from ximu3 import Connection, TcpConnectionInfo
except ImportError:
    Connection = None
    TcpConnectionInfo = None

@dataclass
class ImuSample:
    timestamp: float
    roll: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    qw: Optional[float] = None
    qx: Optional[float] = None
    qy: Optional[float] = None
    qz: Optional[float] = None
    r: Optional[List[float]] = None

class XIMU3Reader:
    def __init__(self, ip: str, port: int) -> None:
        if Connection is None or TcpConnectionInfo is None:
            raise RuntimeError("xIMU3 SDK not available.")
        self.connection = Connection(TcpConnectionInfo(ip, port))
        self.lock = threading.Lock()
        self.latest: Optional[ImuSample] = None
        self._opened = False

    def open(self) -> None:
        self.connection.open()
        self._opened = True
        self.connection.add_euler_angles_callback(self._on_euler)
        self.connection.add_quaternion_callback(self._on_quaternion)
        self.connection.add_rotation_matrix_callback(self._on_rotation_matrix)

    def close(self) -> None:
        if self._opened:
            self.connection.close()
            self._opened = False

    def _ensure_latest(self) -> None:
        if self.latest is None:
            with self.lock:
                if self.latest is None:
                    self.latest = ImuSample(timestamp=time.perf_counter())

    def _on_euler(self, msg) -> None:
        self._ensure_latest()
        with self.lock:
            t = time.perf_counter()
            self.latest.timestamp = t
            self.latest.roll = getattr(msg, "roll", None)
            self.latest.pitch = getattr(msg, "pitch", None)
            self.latest.yaw = getattr(msg, "yaw", None)

    def _on_quaternion(self, msg) -> None:
        self._ensure_latest()
        with self.lock:
            t = time.perf_counter()
            self.latest.timestamp = t
            self.latest.qw = getattr(msg, "w", None)
            self.latest.qx = getattr(msg, "x", None)
            self.latest.qy = getattr(msg, "y", None)
            self.latest.qz = getattr(msg, "z", None)

    def _on_rotation_matrix(self, msg) -> None:
        self._ensure_latest()
        with self.lock:
            t = time.perf_counter()
            self.latest.timestamp = t
            self.latest.r = [
                getattr(msg, "xx", None), getattr(msg, "xy", None), getattr(msg, "xz", None),
                getattr(msg, "yx", None), getattr(msg, "yy", None), getattr(msg, "yz", None),
                getattr(msg, "zx", None), getattr(msg, "zy", None), getattr(msg, "zz", None),
            ]

    def get_latest(self) -> ImuSample:
        self._ensure_latest()
        with self.lock:
            return ImuSample(
                timestamp=self.latest.timestamp,
                # roll=self.latest.roll,
                # pitch=self.latest.pitch,
                # yaw=self.latest.yaw,
                qw=self.latest.qw,
                qx=self.latest.qx,
                qy=self.latest.qy,
                qz=self.latest.qz,
                # r=list(self.latest.r) if self.latest.r is not None else None,
            )