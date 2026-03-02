import threading
import time
import os
import signal
from kiosk_db import DatabaseHandler
from anti_spoof import AntiSpoof

# --- GLOBAL STATE ---
class KioskState:
    def __init__(self):
        self.frame = None
        self.clean_snapshot = None # Bản ảnh cực sạch để lưu DB
        self.lock = threading.Lock()
        self.status = "SCANNING"  # SCANNING, PROCESSING, CONFIRM, SUCCESS
        self.progress = 0
        self.student_data = None 
        self.last_scan_time = 0
        self.process_start_time = 0
        self.db = DatabaseHandler()
        self.anti_spoof = AntiSpoof()
        self.running = True # Cờ kiểm soát vòng lặp
        self.pending_crop = None
        
        # Liveness State
        self.is_live = False
        self.fas_score = 0.0
        self.blink_count = 0
        self.last_blink_time = 0
        self.blink_threshold = 0.20
        
        # Verification State
        self.consecutive_match_count = 0
        self.last_recognized_sid = None
        self.is_near = False

def setup_signals(state):
    """Cấu hình phím nóng để tắt hệ thống dứt khoát"""
    def signal_handler(sig, frame):
        print('\n👋 Đang tắt hệ thống (FastAPI) NGAY LẬP TỨC...')
        state.running = False
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
