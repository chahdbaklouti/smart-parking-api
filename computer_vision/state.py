# state.py
import threading

class ParkingState:
    """
    Simple memory storage for latest detection results.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def update(self, parking_id, frame, stats):
        with self._lock:
            self._data[parking_id] = {
                "frame": frame,
                "stats": stats
            }

    def get(self, parking_id):
        with self._lock:
            return self._data.get(parking_id)

    def get_all_stats(self):
        with self._lock:
            return [v["stats"] for v in self._data.values()]


# SINGLE GLOBAL OBJECT (used everywhere)
state = ParkingState()