# fivedt_direct.py
import ctypes as C, os, threading, time

class FiveDTglove:
    def __init__(self, dll="fglove64.dll", dev="USB0", hz=1000, stale_ms=10):
        self.d = C.WinDLL(os.path.abspath(dll) if os.path.sep in dll else dll)
        self.d.fdOpen.argtypes = [C.c_char_p]; self.d.fdOpen.restype = C.c_void_p
        self.d.fdClose.argtypes = [C.c_void_p]
        self.d.fdGetNumSensors.argtypes = [C.c_void_p]; self.d.fdGetNumSensors.restype = C.c_int
        self.d.fdGetSensorScaledAll.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
        self.d.fdGetSensorScaledAll.restype = None

        # Add raw function signature
        self.d.fdGetSensorRawAll.argtypes = [C.c_void_p, C.POINTER(C.c_short)]
        self.d.fdGetSensorRawAll.restype = None

        self.h = self.d.fdOpen(dev.encode()); assert self.h
        self.n = self.d.fdGetNumSensors(self.h)
        self._buf = (C.c_float * self.n)()
        self._raw_buf = (C.c_short * self.n)()
        self.dt = 1.0 / max(1, hz); self.stale = stale_ms / 1000.0
        self.last = None
        self.last_raw = None
        self._run = False

    def start(self):
        self._run = True
        threading.Thread(target=self._poll, daemon=True).start()
        return self

    def _poll(self):
        t = time.perf_counter()
        while self._run:
            self.d.fdGetSensorRawAll(self.h, self._raw_buf)
            raw_vals = [int(x) for x in self._raw_buf]
            self.d.fdGetSensorScaledAll(self.h, self._buf)
            scaled_vals = [float(x) for x in self._buf]
            now = time.perf_counter()
            self.last = (now, scaled_vals)
            self.last_raw = (now, raw_vals)
            t += self.dt
            sl = t - time.perf_counter()
            if sl > 0:
                time.sleep(sl)

    def scaled(self):
        x = self.last
        return None if (not x) or (time.perf_counter() - x[0] > self.stale) else x[1]

    def raw(self):
        x = self.last_raw
        if (not x) or (time.perf_counter() - x[0] > self.stale):
            return None
        uniq_idxs = [0, 3, 6, 9, 12, 17]  # Adjust if you verify more or fewer active sensors
        raw_full = x[1]
        # Only take unique sensor values
        return [raw_full[i] for i in uniq_idxs]
        # return None if (not x) or (time.perf_counter() - x[0] > self.stale) else x[1]

    def close(self):
        self._run = False
        time.sleep(0.01)
        try:
            self.d.fdClose(self.h)
        except Exception:
            pass

