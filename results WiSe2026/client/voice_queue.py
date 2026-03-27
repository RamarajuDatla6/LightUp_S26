"""
Shared queue module for voice commands.
This exists outside of Streamlit's session state so it can be accessed from background threads.
"""
from queue import Queue

# Global queue that persists across Streamlit reruns
_command_queue = Queue()

def get_command_queue():
    """Get the global command queue"""
    return _command_queue

def put_command(command):
    """Put a command in the queue (thread-safe)"""
    _command_queue.put(command)
    
def get_command():
    """Get a command from the queue (thread-safe)"""
    return _command_queue.get_nowait()
    
def queue_size():
    """Get the current queue size"""
    return _command_queue.qsize()
    
def is_empty():
    """Check if queue is empty"""
    return _command_queue.empty()
