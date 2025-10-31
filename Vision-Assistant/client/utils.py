import cv2
import time
import threading
from queue import Queue, Empty, Full
from collections import deque
import concurrent.futures

class FrameBuffer:
    def __init__(self, buffer_size=5):
        self.frame_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None

    def start_capture(self, cap):
        def capture_frames():
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    with self.lock:
                        self.frame_buffer.append(frame)
                        self.latest_frame = frame
                time.sleep(0.001)  # Tiny sleep to prevent CPU overload

        self.capture_thread = threading.Thread(target=capture_frames, daemon=True)
        self.capture_thread.start()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()

class ProcessingThread:
    def __init__(self, max_workers=2):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = True

    def start_processing(self):
        def process_frames():
            while self.running:
                try:
                    frame, process_func = self.processing_queue.get(timeout=1)
                    if frame is not None and process_func is not None:
                        result = process_func(frame)
                        self.result_queue.put(result)
                except Empty:
                    continue

        self.process_thread = threading.Thread(target=process_frames, daemon=True)
        self.process_thread.start()

    def submit_frame(self, frame, process_func):
        try:
            self.processing_queue.put_nowait((frame, process_func))
        except Full:
            pass  # Skip frame if queue is full

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        self.executor.shutdown() 