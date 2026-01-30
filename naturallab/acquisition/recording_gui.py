#!/usr/bin/env python3
"""
Recording Setup GUI for Ubuntu - Multi-Device Support
======================================================
A comprehensive GUI for managing LSL streaming with multiple Neon devices, LabRecorder, and XDF extraction.
Designed for easy use by interns with Caregiver/Child role assignment.

Usage:
    python recording_gui.py

Features:
- Start/stop LSL streaming with configurable parameters for multiple Neon devices
- Role-based device assignment (Caregiver/Child)
- Launch LabRecorder
- Extract XDF files to different drives
- Automatic cleanup after successful extraction
- Real-time status monitoring
- Configuration management
"""

import os
import sys
import json
import subprocess
import threading
import time
import shutil
import signal
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

class RecordingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking Recording Setup - Multi-Device")
        self.root.geometry("1100x750")
        
        # Configuration file
        self.config_file = os.path.expanduser("~/.recording_setup_config.json")
        self.config = self.load_config()
        
        # Process tracking
        self.lsl_process = None
        self.labrecorder_process = None
        self.streaming_active = False
        
        # Setup GUI
        self.setup_gui()
        self.load_saved_settings()
        
        # Start status update thread
        self.update_thread_running = True
        self.status_thread = threading.Thread(target=self.update_status_loop, daemon=True)
        self.status_thread.start()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "lsl_script_path": "",
            "conda_env_name": "",
            "extraction_drive": "/media",
            "labrecorder_path": "/usr/local/bin/LabRecorder",
            "labrecorder_config": "",
            "caregiver_ip": "YOUR_IP_ADDRESS",
            "child_ip": "YOUR_IP_ADDRESS",
            "last_recording_dir": os.path.expanduser("~/recordings"),
            "auto_extract": True,
            "delete_after_extract": True,
            "depth_interval": 30,
            "keep_raw_depth": True,
            "max_neon_devices": 2,
            # Individual camera settings
            "rtsp_user": "admin",
            "rtsp_pass": "password",
            "rtsp_stream": "stream1",
            "cam1_ip": "YOUR_IP_ADDRESS",
            "cam1_name": "Camera1",
            "cam1_enabled": True,
            "cam2_ip": "YOUR_IP_ADDRESS",
            "cam2_name": "Camera2",
            "cam2_enabled": True,
            "cam3_ip": "YOUR_IP_ADDRESS",
            "cam3_name": "Camera3",
            "cam3_enabled": True,
            "cam4_ip": "YOUR_IP_ADDRESS",
            "cam4_name": "Camera4",
            "cam4_enabled": True
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Update with any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.log(f"Error saving config: {e}")
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main Control Tab
        self.setup_main_tab(notebook)
        
        # Configuration Tab
        self.setup_config_tab(notebook)
        
        # File Management Tab
        self.setup_file_tab(notebook)
        
        # Logs Tab
        self.setup_logs_tab(notebook)
    
    def setup_main_tab(self, notebook):
        """Setup the main control tab"""
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Recording Control")
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.lsl_status_label = ttk.Label(status_frame, text="LSL Streaming: Not Running", 
                                         foreground="red")
        self.lsl_status_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        self.labrecorder_status_label = ttk.Label(status_frame, text="LabRecorder: Not Running", 
                                                 foreground="red")
        self.labrecorder_status_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Control buttons section
        control_frame = ttk.LabelFrame(main_frame, text="Recording Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # LSL Controls
        lsl_frame = ttk.Frame(control_frame)
        lsl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lsl_frame, text="1. LSL Streaming:").pack(side=tk.LEFT)
        
        self.start_lsl_btn = ttk.Button(lsl_frame, text="Start LSL Streaming", 
                                       command=self.start_lsl_streaming)
        self.start_lsl_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_lsl_btn = ttk.Button(lsl_frame, text="Stop LSL Streaming", 
                                      command=self.stop_lsl_streaming, state=tk.DISABLED)
        self.stop_lsl_btn.pack(side=tk.LEFT, padx=5)
        
        # LabRecorder Controls
        labrecorder_frame = ttk.Frame(control_frame)
        labrecorder_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(labrecorder_frame, text="2. LabRecorder:").pack(side=tk.LEFT)
        
        self.start_labrecorder_btn = ttk.Button(labrecorder_frame, text="Open LabRecorder", 
                                               command=self.start_labrecorder)
        self.start_labrecorder_btn.pack(side=tk.LEFT, padx=5)
        
        self.close_labrecorder_btn = ttk.Button(labrecorder_frame, text="Close LabRecorder", 
                                               command=self.close_labrecorder, state=tk.DISABLED)
        self.close_labrecorder_btn.pack(side=tk.LEFT, padx=5)
        
        # Quick Setup
        quick_frame = ttk.Frame(control_frame)
        quick_frame.pack(fill=tk.X, pady=10)
        
        self.quick_setup_btn = ttk.Button(quick_frame, text="🚀 Quick Setup (Start Both)", 
                                         command=self.quick_setup, style="Accent.TButton")
        self.quick_setup_btn.pack(side=tk.LEFT, padx=5)
        
        self.emergency_stop_btn = ttk.Button(quick_frame, text="🛑 Emergency Stop All", 
                                           command=self.emergency_stop, style="Accent.TButton")
        self.emergency_stop_btn.pack(side=tk.LEFT, padx=5)
        
        # LSL Kill Switch
        kill_frame = ttk.Frame(control_frame)
        kill_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(kill_frame, text="LSL Kill Switch:").pack(side=tk.LEFT)
        
        ttk.Button(kill_frame, text="🔍 Check LSL Processes", 
                  command=self.check_lsl_processes).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(kill_frame, text="💀 Force Kill All LSL", 
                  command=self.force_kill_all_lsl).pack(side=tk.LEFT, padx=5)
        
        # Recording parameters - Updated for multi-device
        params_frame = ttk.LabelFrame(main_frame, text="Multi-Device Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Device assignment section
        device_frame = ttk.LabelFrame(params_frame, text="Device Assignment", padding=5)
        device_frame.pack(fill=tk.X, pady=5)
        
        # Caregiver IP
        caregiver_frame = ttk.Frame(device_frame)
        caregiver_frame.pack(fill=tk.X, pady=2)
        ttk.Label(caregiver_frame, text="Caregiver IP:", width=15).pack(side=tk.LEFT)
        self.caregiver_ip_var = tk.StringVar()
        ttk.Entry(caregiver_frame, textvariable=self.caregiver_ip_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(caregiver_frame, text="(e.g., YOUR_IP_ADDRESS)", foreground="gray").pack(side=tk.LEFT, padx=5)
        
        # Child IP
        child_frame = ttk.Frame(device_frame)
        child_frame.pack(fill=tk.X, pady=2)
        ttk.Label(child_frame, text="Child IP:", width=15).pack(side=tk.LEFT)
        self.child_ip_var = tk.StringVar()
        ttk.Entry(child_frame, textvariable=self.child_ip_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(child_frame, text="(e.g., YOUR_IP_ADDRESS)", foreground="gray").pack(side=tk.LEFT, padx=5)
        
        # Max devices
        max_devices_frame = ttk.Frame(device_frame)
        max_devices_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_devices_frame, text="Max Devices:", width=15).pack(side=tk.LEFT)
        self.max_devices_var = tk.IntVar()
        ttk.Spinbox(max_devices_frame, from_=1, to=4, textvariable=self.max_devices_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(max_devices_frame, text="(auto-assigns if IPs not specified)", foreground="gray").pack(side=tk.LEFT, padx=5)
        
        # Camera configuration - Individual fields for each camera
        camera_frame = ttk.LabelFrame(params_frame, text="Additional Cameras (RTSP)", padding=5)
        camera_frame.pack(fill=tk.X, pady=5)
        
        # Header
        header_frame = ttk.Frame(camera_frame)
        header_frame.pack(fill=tk.X, pady=2)
        ttk.Label(header_frame, text="Camera", width=10).grid(row=0, column=0, padx=5)
        ttk.Label(header_frame, text="IP Address", width=15).grid(row=0, column=1, padx=5)
        ttk.Label(header_frame, text="Display Name", width=20).grid(row=0, column=2, padx=5)
        ttk.Label(header_frame, text="Enable", width=8).grid(row=0, column=3, padx=5)
        
        # Camera variables
        self.camera_vars = {}
        
        # Default camera settings from LSL script
        camera_defaults = [
            {"ip": "YOUR_IP_ADDRESS", "name": "Camera1"},
            {"ip": "YOUR_IP_ADDRESS", "name": "Camera2"},
            {"ip": "YOUR_IP_ADDRESS", "name": "Camera3"},
            {"ip": "YOUR_IP_ADDRESS", "name": "Camera4"}
        ]
        
        # Create fields for each camera
        for i, defaults in enumerate(camera_defaults, 1):
            cam_frame = ttk.Frame(camera_frame)
            cam_frame.pack(fill=tk.X, pady=1)
            
            # Camera label
            ttk.Label(cam_frame, text=f"Cam {i}:", width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
            
            # IP address field
            ip_var = tk.StringVar(value=defaults["ip"])
            ttk.Entry(cam_frame, textvariable=ip_var, width=15).grid(row=0, column=1, padx=5)
            
            # Name field
            name_var = tk.StringVar(value=defaults["name"])
            ttk.Entry(cam_frame, textvariable=name_var, width=25).grid(row=0, column=2, padx=5)
            
            # Enable checkbox
            enable_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(cam_frame, variable=enable_var).grid(row=0, column=3, padx=5)
            
            # Store variables
            self.camera_vars[f"cam{i}"] = {
                "ip": ip_var,
                "name": name_var,
                "enabled": enable_var
            }
        
        # Camera credentials and test button
        cred_frame = ttk.Frame(camera_frame)
        cred_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cred_frame, text="RTSP Credentials:", width=15).pack(side=tk.LEFT)
        ttk.Label(cred_frame, text="User:").pack(side=tk.LEFT, padx=(10, 2))
        self.rtsp_user_var = tk.StringVar(value="admin")
        ttk.Entry(cred_frame, textvariable=self.rtsp_user_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(cred_frame, text="Pass:").pack(side=tk.LEFT, padx=(10, 2))
        self.rtsp_pass_var = tk.StringVar(value="password")
        ttk.Entry(cred_frame, textvariable=self.rtsp_pass_var, width=10, show="*").pack(side=tk.LEFT, padx=2)
        ttk.Label(cred_frame, text="Stream:").pack(side=tk.LEFT, padx=(10, 2))
        self.rtsp_stream_var = tk.StringVar(value="stream1")
        ttk.Entry(cred_frame, textvariable=self.rtsp_stream_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # Camera utility buttons
        btn_frame = ttk.Frame(camera_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="🔗 Show Generated URLs", 
                  command=self.show_camera_urls).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✅ Enable All", 
                  command=self.enable_all_cameras).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="❌ Disable All", 
                  command=self.disable_all_cameras).pack(side=tk.LEFT, padx=5)
        
        # Checkboxes for optional components
        options_frame = ttk.LabelFrame(params_frame, text="Stream Options", padding=5)
        options_frame.pack(fill=tk.X, pady=5)
        
        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill=tk.X, pady=2)
        
        self.no_realsense_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Disable RealSense", 
                       variable=self.no_realsense_var).pack(side=tk.LEFT, padx=5)
        
        self.no_audio_var = tk.BooleanVar()
        ttk.Checkbutton(options_row1, text="Disable Audio", 
                       variable=self.no_audio_var).pack(side=tk.LEFT, padx=5)
        
        options_row2 = ttk.Frame(options_frame)
        options_row2.pack(fill=tk.X, pady=2)
        
        self.no_imu_var = tk.BooleanVar()
        ttk.Checkbutton(options_row2, text="Disable IMU", 
                       variable=self.no_imu_var).pack(side=tk.LEFT, padx=5)
        
        self.no_eye_events_var = tk.BooleanVar()
        ttk.Checkbutton(options_row2, text="Disable Eye Events", 
                       variable=self.no_eye_events_var).pack(side=tk.LEFT, padx=5)
        
        # Stream preview - UPDATED to match fixed script
        preview_frame = ttk.LabelFrame(main_frame, text="Expected LSL Streams", padding=10)
        preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        preview_text = """With 2 Neon devices, you'll get these streams:
• NeonGaze_Caregiver, NeonGaze_Child (gaze data)
• NeonVideo_Caregiver, NeonVideo_Child (scene camera video)
• NeonIMU_Caregiver, NeonIMU_Child (motion data)
• NeonFixations_Child (fixation events - child only)
• NeonSaccades_Child (saccade events - child only)
• NeonAudio_Caregiver, NeonAudio_Child (audio from both)
• RealSense_Color, RealSense_Depth (if enabled)
• Camera streams (if configured)

Note: Eye events (fixations/saccades) only work for Child device due to API limitations."""
        
        ttk.Label(preview_frame, text=preview_text, foreground="blue").pack(anchor=tk.W)
    
    def setup_config_tab(self, notebook):
        """Setup the configuration tab"""
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configuration")
        
        # Scrollable frame
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Paths configuration
        paths_frame = ttk.LabelFrame(scrollable_frame, text="Paths Configuration", padding=10)
        paths_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # LSL Script Path
        row = 0
        ttk.Label(paths_frame, text="LSL Script Path:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.lsl_script_var = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.lsl_script_var, width=50).grid(row=row, column=1, padx=5)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_file(self.lsl_script_var, "Python files", "*.py")).grid(row=row, column=2, padx=5)
        
        # Conda Environment
        row += 1
        ttk.Label(paths_frame, text="Conda Environment:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.conda_env_var = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.conda_env_var, width=50).grid(row=row, column=1, padx=5)
        ttk.Button(paths_frame, text="Detect", command=self.detect_conda_envs).grid(row=row, column=2, padx=5)
        
        # LabRecorder Path
        row += 1
        ttk.Label(paths_frame, text="LabRecorder Path:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.labrecorder_path_var = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.labrecorder_path_var, width=50).grid(row=row, column=1, padx=5)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_file(self.labrecorder_path_var, "All files", "*")).grid(row=row, column=2, padx=5)
        
        # LabRecorder Config
        row += 1
        ttk.Label(paths_frame, text="LabRecorder Config:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.labrecorder_config_var = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.labrecorder_config_var, width=50).grid(row=row, column=1, padx=5)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_file(self.labrecorder_config_var, "Config files", "*.cfg")).grid(row=row, column=2, padx=5)
        
        # Create config button
        config_helper_frame = ttk.Frame(paths_frame)
        config_helper_frame.grid(row=row+1, column=0, columnspan=3, sticky=tk.W, pady=5)
        ttk.Button(config_helper_frame, text="Create Multi-Device LabRecorder Config", 
                  command=self.create_labrecorder_config).pack(side=tk.LEFT, padx=5)
        
        # Extraction Drive
        row += 2
        ttk.Label(paths_frame, text="Extraction Drive:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.extraction_drive_var = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.extraction_drive_var, width=50).grid(row=row, column=1, padx=5)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.extraction_drive_var)).grid(row=row, column=2, padx=5)
        
        # Recording Directory
        row += 1
        ttk.Label(paths_frame, text="Recording Directory:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.recording_dir_var = tk.StringVar()
        ttk.Entry(paths_frame, textvariable=self.recording_dir_var, width=50).grid(row=row, column=1, padx=5)
        ttk.Button(paths_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.recording_dir_var)).grid(row=row, column=2, padx=5)
        
        # Extraction settings
        extract_frame = ttk.LabelFrame(scrollable_frame, text="Extraction Settings", padding=10)
        extract_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.auto_extract_var = tk.BooleanVar()
        ttk.Checkbutton(extract_frame, text="Auto-extract XDF files after recording", 
                       variable=self.auto_extract_var).pack(anchor=tk.W)
        
        self.delete_after_extract_var = tk.BooleanVar()
        ttk.Checkbutton(extract_frame, text="Delete XDF after successful extraction", 
                       variable=self.delete_after_extract_var).pack(anchor=tk.W)
        
        self.keep_raw_depth_var = tk.BooleanVar()
        ttk.Checkbutton(extract_frame, text="Keep raw depth data (recommended for measurements)", 
                       variable=self.keep_raw_depth_var).pack(anchor=tk.W)
        
        # Depth interval
        depth_frame = ttk.Frame(extract_frame)
        depth_frame.pack(fill=tk.X, pady=5)
        ttk.Label(depth_frame, text="Save depth PNG every N frames:").pack(side=tk.LEFT)
        self.depth_interval_var = tk.IntVar()
        ttk.Spinbox(depth_frame, from_=1, to=100, textvariable=self.depth_interval_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Save/Load configuration
        config_buttons_frame = ttk.Frame(scrollable_frame)
        config_buttons_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(config_buttons_frame, text="Save Configuration", 
                  command=self.save_current_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_buttons_frame, text="Load Configuration", 
                  command=self.load_saved_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_buttons_frame, text="Reset to Defaults", 
                  command=self.reset_config).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_file_tab(self, notebook):
        """Setup the file management tab"""
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="File Management")
        
        # XDF file selection
        select_frame = ttk.LabelFrame(file_frame, text="XDF File Management", padding=10)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Directory selection
        dir_frame = ttk.Frame(select_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dir_frame, text="Recording Directory:").pack(side=tk.LEFT)
        self.file_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.file_dir_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="Browse", 
                  command=lambda: self.browse_directory(self.file_dir_var)).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="Refresh", command=self.refresh_file_list).pack(side=tk.LEFT, padx=5)
        
        # File list
        list_frame = ttk.Frame(select_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Treeview for files
        self.file_tree = ttk.Treeview(list_frame, columns=("size", "modified"), show="tree headings")
        self.file_tree.heading("#0", text="Filename")
        self.file_tree.heading("size", text="Size")
        self.file_tree.heading("modified", text="Modified")
        self.file_tree.column("#0", width=300)
        self.file_tree.column("size", width=100)
        self.file_tree.column("modified", width=150)
        
        file_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=file_scrollbar.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # File operations
        operations_frame = ttk.Frame(select_frame)
        operations_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(operations_frame, text="Extract Selected", 
                  command=self.extract_selected_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(operations_frame, text="Extract All XDF Files", 
                  command=self.extract_all_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(operations_frame, text="Delete Selected", 
                  command=self.delete_selected_file).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.extraction_progress = ttk.Progressbar(select_frame, mode='indeterminate')
        self.extraction_progress.pack(fill=tk.X, pady=5)
        
    def setup_logs_tab(self, notebook):
        """Setup the logs tab"""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="Logs")
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=30, width=120)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log controls
        log_controls = ttk.Frame(logs_frame)
        log_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(log_controls, text="Clear Logs", command=self.clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(log_controls, text="Save Logs", command=self.save_logs).pack(side=tk.LEFT, padx=5)
        
        # Initial log message
        self.log("Multi-Device Recording Setup GUI initialized")
        self.log(f"Configuration loaded from: {self.config_file}")
    
    def create_labrecorder_config(self):
        """Create a default LabRecorder config file for multi-device setup - UPDATED"""
        # Ask user where to save the config
        default_path = os.path.join(os.path.dirname(self.lsl_script_var.get()) if self.lsl_script_var.get() else os.getcwd(), "LabRecorder_MultiDevice.cfg")
        
        config_path = filedialog.asksaveasfilename(
            title="Save Multi-Device LabRecorder Config",
            defaultextension=".cfg",
            filetypes=[("Config files", "*.cfg"), ("All files", "*.*")],
            initialname="LabRecorder_MultiDevice.cfg",
            initialdir=os.path.dirname(default_path)
        )
        
        if not config_path:
            return
        
        # Create a comprehensive config template for multi-device
        recordings_dir = self.recording_dir_var.get() or "./recordings"
        
        config_content = f'''<?xml version="1.0" encoding="utf-8"?>
<settings>
    <!-- LabRecorder Configuration for Multi-Device Eye Tracking Setup -->
    
    <!-- Required streams - recording won't start without these -->
    <requiredstreams>
        <stream>NeonVideo_Caregiver</stream>
        <stream>NeonGaze_Caregiver</stream>
        <stream>NeonVideo_Child</stream>
        <stream>NeonGaze_Child</stream>
    </requiredstreams>
    
    <!-- Streams to enable by default -->
    <onlinestreams>
        <!-- Caregiver Neon streams -->
        <stream>NeonVideo_Caregiver</stream>
        <stream>NeonGaze_Caregiver</stream>
        <stream>NeonIMU_Caregiver</stream>
        <stream>NeonAudio_Caregiver</stream>
        
        <!-- Child Neon streams -->
        <stream>NeonVideo_Child</stream>
        <stream>NeonGaze_Child</stream>
        <stream>NeonIMU_Child</stream>
        <stream>NeonAudio_Child</stream>
        
        <!-- Eye events - ONLY from Child device -->
        <stream>NeonFixations_Child</stream>
        <stream>NeonSaccades_Child</stream>
        
        <!-- RealSense camera streams -->
        <stream>RealSense_Color</stream>
        <stream>RealSense_Depth</stream>
        <stream>RealSense_Metadata</stream>
        
        <!-- Additional RTSP cameras -->
        <stream>Camera1</stream>
        <stream>Camera2</stream>
        <stream>Camera3</stream>
        <stream>Camera4</stream>
    </onlinestreams>
    
    <!-- Storage settings -->
    <storageLocation>{recordings_dir}</storageLocation>
    <studyRoot>{recordings_dir}</studyRoot>
    
    <!-- Recording settings -->
    <sessionBlocks>1</sessionBlocks>
    <legacyMode>false</legacyMode>
    <autoStart>false</autoStart>
    <enableScriptableInterface>true</enableScriptableInterface>
    
    <!-- File format settings -->
    <fileFormat>xdf</fileFormat>
    <compressionLevel>1</compressionLevel>
    
    <!-- UI settings -->
    <showAdvancedUI>false</showAdvancedUI>
    <showConsole>true</showConsole>
    
    <!-- Stream synchronization -->
    <enableClockSync>true</enableClockSync>
    <syncAccuracy>0.001</syncAccuracy>
    
    <!-- Buffer settings for high-throughput streams (important for multi-device) -->
    <bufferLength>120</bufferLength>
    <maxBufferSize>20000000</maxBufferSize>
    
    <!-- Multi-device specific settings -->
    <chunkSize>32</chunkSize>
    <maxCachedChunks>128</maxCachedChunks>
</settings>'''
        
        try:
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Set this as the config file
            self.labrecorder_config_var.set(config_path)
            
            self.log(f"Created multi-device LabRecorder config: {config_path}")
            messagebox.showinfo("Config Created", 
                               f"Multi-device LabRecorder config created successfully!\n\nPath: {config_path}\n\nThis config includes:\n- Role-based stream names (Caregiver/Child)\n- Audio from both devices\n- Eye events only from Child device\n- Optimized settings for multiple high-throughput streams\n- Proper recording directory\n\nThe config is now set as your LabRecorder config file.")
            
        except Exception as e:
            self.log(f"Error creating config: {e}")
            messagebox.showerror("Error", f"Failed to create config file: {e}")
    
    def show_camera_urls(self):
        """Show the generated RTSP URLs for debugging"""
        user = self.rtsp_user_var.get() or "admin"
        password = self.rtsp_pass_var.get() or "password"
        stream = self.rtsp_stream_var.get() or "stream1"
        
        urls_info = "Generated RTSP URLs:\n\n"
        
        for i in range(1, 5):
            cam_key = f"cam{i}"
            if cam_key in self.camera_vars:
                ip = self.camera_vars[cam_key]["ip"].get().strip()
                name = self.camera_vars[cam_key]["name"].get().strip()
                enabled = self.camera_vars[cam_key]["enabled"].get()
                
                if ip and name:
                    rtsp_url = f"rtsp://USERNAME:PASSWORD@{ip}/{stream}"
                    status = "✅ ENABLED" if enabled else "❌ DISABLED"
                    urls_info += f"Camera {i}: {name}\n"
                    urls_info += f"  URL: {rtsp_url}\n"
                    urls_info += f"  Status: {status}\n\n"
                else:
                    urls_info += f"Camera {i}: ⚠️ INCOMPLETE (missing IP or name)\n\n"
        
        # Create popup window
        url_window = tk.Toplevel(self.root)
        url_window.title("Generated Camera URLs")
        url_window.geometry("600x400")
        
        text_widget = scrolledtext.ScrolledText(url_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(1.0, urls_info)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(url_window, text="Close", command=url_window.destroy).pack(pady=5)
    
    def enable_all_cameras(self):
        """Enable all cameras"""
        for i in range(1, 5):
            cam_key = f"cam{i}"
            if cam_key in self.camera_vars:
                self.camera_vars[cam_key]["enabled"].set(True)
        self.log("All cameras enabled")
    
    def disable_all_cameras(self):
        """Disable all cameras"""
        for i in range(1, 5):
            cam_key = f"cam{i}"
            if cam_key in self.camera_vars:
                self.camera_vars[cam_key]["enabled"].set(False)
        self.log("All cameras disabled")
    
    def browse_file(self, var, file_desc, pattern):
        """Browse for a file"""
        filename = filedialog.askopenfilename(
            title=f"Select {file_desc}",
            filetypes=[(file_desc, pattern), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)
    
    def browse_directory(self, var):
        """Browse for a directory"""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def detect_conda_envs(self):
        """Detect available conda environments"""
        try:
            result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                envs = []
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) > 0:
                            env_name = parts[0]
                            if env_name not in ['base', '*']:
                                envs.append(env_name)
                
                if envs:
                    # Show selection dialog
                    env_window = tk.Toplevel(self.root)
                    env_window.title("Select Conda Environment")
                    env_window.geometry("400x300")
                    
                    ttk.Label(env_window, text="Available Conda Environments:").pack(pady=10)
                    
                    listbox = tk.Listbox(env_window)
                    for env in envs:
                        listbox.insert(tk.END, env)
                    listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                    
                    def select_env():
                        selection = listbox.curselection()
                        if selection:
                            self.conda_env_var.set(envs[selection[0]])
                            env_window.destroy()
                    
                    ttk.Button(env_window, text="Select", command=select_env).pack(pady=10)
                else:
                    messagebox.showinfo("No Environments", "No conda environments found")
            else:
                messagebox.showerror("Error", "Could not detect conda environments")
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting conda environments: {e}")
    
    def load_saved_settings(self):
        """Load saved settings into GUI"""
        self.caregiver_ip_var.set(self.config.get("caregiver_ip", ""))
        self.child_ip_var.set(self.config.get("child_ip", ""))
        self.max_devices_var.set(self.config.get("max_neon_devices", 2))
        self.lsl_script_var.set(self.config.get("lsl_script_path", ""))
        self.conda_env_var.set(self.config.get("conda_env_name", ""))
        self.labrecorder_path_var.set(self.config.get("labrecorder_path", ""))
        self.labrecorder_config_var.set(self.config.get("labrecorder_config", ""))
        self.extraction_drive_var.set(self.config.get("extraction_drive", ""))
        self.recording_dir_var.set(self.config.get("last_recording_dir", ""))
        self.file_dir_var.set(self.config.get("last_recording_dir", ""))
        self.auto_extract_var.set(self.config.get("auto_extract", True))
        self.delete_after_extract_var.set(self.config.get("delete_after_extract", True))
        self.depth_interval_var.set(self.config.get("depth_interval", 30))
        self.keep_raw_depth_var.set(self.config.get("keep_raw_depth", True))
        
        # Load camera settings
        self.rtsp_user_var.set(self.config.get("rtsp_user", "admin"))
        self.rtsp_pass_var.set(self.config.get("rtsp_pass", "password"))
        self.rtsp_stream_var.set(self.config.get("rtsp_stream", "stream1"))
        
        # Load individual camera settings
        for i in range(1, 5):
            cam_key = f"cam{i}"
            if cam_key in self.camera_vars:
                self.camera_vars[cam_key]["ip"].set(self.config.get(f"cam{i}_ip", ""))
                self.camera_vars[cam_key]["name"].set(self.config.get(f"cam{i}_name", ""))
                self.camera_vars[cam_key]["enabled"].set(self.config.get(f"cam{i}_enabled", True))
    
    def save_current_config(self):
        """Save current GUI settings to config"""
        self.config["caregiver_ip"] = self.caregiver_ip_var.get()
        self.config["child_ip"] = self.child_ip_var.get()
        self.config["max_neon_devices"] = self.max_devices_var.get()
        self.config["lsl_script_path"] = self.lsl_script_var.get()
        self.config["conda_env_name"] = self.conda_env_var.get()
        self.config["labrecorder_path"] = self.labrecorder_path_var.get()
        self.config["labrecorder_config"] = self.labrecorder_config_var.get()
        self.config["extraction_drive"] = self.extraction_drive_var.get()
        self.config["last_recording_dir"] = self.recording_dir_var.get()
        self.config["auto_extract"] = self.auto_extract_var.get()
        self.config["delete_after_extract"] = self.delete_after_extract_var.get()
        self.config["depth_interval"] = self.depth_interval_var.get()
        self.config["keep_raw_depth"] = self.keep_raw_depth_var.get()
        
        # Save camera credentials
        self.config["rtsp_user"] = self.rtsp_user_var.get()
        self.config["rtsp_pass"] = self.rtsp_pass_var.get()
        self.config["rtsp_stream"] = self.rtsp_stream_var.get()
        
        # Save individual camera settings
        for i in range(1, 5):
            cam_key = f"cam{i}"
            if cam_key in self.camera_vars:
                self.config[f"cam{i}_ip"] = self.camera_vars[cam_key]["ip"].get()
                self.config[f"cam{i}_name"] = self.camera_vars[cam_key]["name"].get()
                self.config[f"cam{i}_enabled"] = self.camera_vars[cam_key]["enabled"].get()
        
        self.save_config()
        self.log("Configuration saved")
        messagebox.showinfo("Saved", "Configuration saved successfully")
    
    def reset_config(self):
        """Reset configuration to defaults"""
        if messagebox.askyesno("Reset Configuration", "Are you sure you want to reset to default settings?"):
            self.config = self.load_config()  # This loads defaults if file doesn't exist
            self.load_saved_settings()
            self.log("Configuration reset to defaults")
    
    def start_lsl_streaming(self):
        """Start LSL streaming with multi-device support"""
        if self.streaming_active:
            messagebox.showwarning("Already Running", "LSL streaming is already active")
            return
        
        script_path = self.lsl_script_var.get()
        if not script_path or not os.path.exists(script_path):
            messagebox.showerror("Error", "Please configure a valid LSL script path")
            return
        
        conda_env = self.conda_env_var.get()
        caregiver_ip = self.caregiver_ip_var.get()
        child_ip = self.child_ip_var.get()
        
        if not caregiver_ip and not child_ip:
            if not messagebox.askyesno("No Device IPs", "No device IPs specified. Continue with auto-discovery?"):
                return
        
        try:
            # Build command
            if conda_env:
                cmd = ["conda", "run", "-n", conda_env, "python", script_path]
            else:
                cmd = ["python3", script_path]
            
            # Add multi-device arguments
            if caregiver_ip:
                cmd.extend(["--caregiver-ip", caregiver_ip])
            if child_ip:
                cmd.extend(["--child-ip", child_ip])
            
            cmd.extend(["--max-neon-devices", str(self.max_devices_var.get())])
            
            # Build RTSP URLs and camera names from individual camera settings
            rtsp_urls = []
            camera_names = []
            
            user = self.rtsp_user_var.get() or "admin"
            password = self.rtsp_pass_var.get() or "password"
            stream = self.rtsp_stream_var.get() or "stream1"
            
            for i in range(1, 5):
                cam_key = f"cam{i}"
                if cam_key in self.camera_vars and self.camera_vars[cam_key]["enabled"].get():
                    ip = self.camera_vars[cam_key]["ip"].get().strip()
                    name = self.camera_vars[cam_key]["name"].get().strip()
                    
                    if ip and name:
                        rtsp_url = f"rtsp://USERNAME:PASSWORD@{ip}/{stream}"
                        rtsp_urls.append(rtsp_url)
                        camera_names.append(name)
            
            # Add camera arguments if any cameras are enabled
            if rtsp_urls:
                cmd.extend(["--rtsp-urls", ",".join(rtsp_urls)])
                cmd.extend(["--camera-names", ",".join(camera_names)])
                self.log(f"Added {len(rtsp_urls)} cameras: {', '.join(camera_names)}")
            else:
                self.log("No cameras enabled")
            
            # Add optional flags
            if self.no_realsense_var.get():
                cmd.append("--no-realsense")
            if self.no_audio_var.get():
                cmd.append("--no-audio")
            if self.no_imu_var.get():
                cmd.append("--no-imu")
            if self.no_eye_events_var.get():
                cmd.append("--no-eye-events")
            
            self.log(f"Starting multi-device LSL streaming: {' '.join(cmd)}")
            
            # Start process
            self.lsl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=os.path.dirname(script_path) if script_path else None
            )
            
            self.streaming_active = True
            self.start_lsl_btn.config(state=tk.DISABLED)
            self.stop_lsl_btn.config(state=tk.NORMAL)
            
            # Start thread to read output
            threading.Thread(target=self.read_lsl_output, daemon=True).start()
            
            self.log("Multi-device LSL streaming started successfully")
            
        except Exception as e:
            self.log(f"Error starting LSL streaming: {e}")
            messagebox.showerror("Error", f"Failed to start LSL streaming: {e}")
    
    def stop_lsl_streaming(self):
        """Stop LSL streaming"""
        self.log("=== STOP LSL STREAMING CALLED ===")
        self.log(f"streaming_active: {self.streaming_active}")
        self.log(f"lsl_process: {self.lsl_process}")
        
        if self.lsl_process:
            try:
                pid = self.lsl_process.pid
                self.log(f"Found LSL process with PID: {pid}")
                
                # Check if process is still running
                if self.lsl_process.poll() is None:
                    self.log("Process is still running, attempting to stop...")
                    
                    # Send SIGINT - same as Ctrl+C
                    self.log("Sending SIGINT signal...")
                    self.lsl_process.send_signal(signal.SIGINT)
                    self.log("SIGINT sent successfully")
                    
                    # Wait for graceful shutdown
                    self.log("Waiting for graceful shutdown (5 seconds)...")
                    try:
                        self.lsl_process.wait(timeout=5)
                        self.log("LSL process stopped gracefully")
                    except subprocess.TimeoutExpired:
                        self.log("Process didn't respond to SIGINT, trying SIGTERM...")
                        self.lsl_process.terminate()
                        try:
                            self.lsl_process.wait(timeout=3)
                            self.log("LSL process terminated with SIGTERM")
                        except subprocess.TimeoutExpired:
                            self.log("Force killing LSL process with SIGKILL...")
                            self.lsl_process.kill()
                            self.lsl_process.wait()
                            self.log("LSL process force killed")
                else:
                    self.log("Process was already dead")
                
            except Exception as e:
                self.log(f"ERROR stopping LSL process: {e}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")
                
            finally:
                self.lsl_process = None
                self.log("lsl_process set to None")
        else:
            self.log("No LSL process to stop")
        
        self.streaming_active = False
        self.start_lsl_btn.config(state=tk.NORMAL)
        self.stop_lsl_btn.config(state=tk.DISABLED)
        self.log("GUI state updated: streaming_active=False, buttons updated")
        self.log("=== STOP LSL STREAMING COMPLETED ===")
    
    def read_lsl_output(self):
        """Read output from LSL process"""
        try:
            while self.streaming_active and self.lsl_process:
                line = self.lsl_process.stdout.readline()
                if line:
                    self.log(f"LSL: {line.strip()}")
                elif self.lsl_process.poll() is not None:
                    break
        except Exception as e:
            self.log(f"Error reading LSL output: {e}")
    
    def start_labrecorder(self):
        """Start LabRecorder with config file support"""
        labrecorder_path = self.labrecorder_path_var.get()
        config_path = self.labrecorder_config_var.get()
        
        if not labrecorder_path or not os.path.exists(labrecorder_path):
            # Try to find LabRecorder in common locations
            common_paths = [
                "/usr/local/bin/LabRecorder",
                "/usr/bin/LabRecorder",
                "/opt/LabRecorder/LabRecorder",
                os.path.expanduser("~/LabRecorder/LabRecorder")
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    labrecorder_path = path
                    self.labrecorder_path_var.set(path)
                    break
            
            if not labrecorder_path:
                messagebox.showerror("Error", "LabRecorder not found. Please configure the path in Configuration tab")
                return
        
        try:
            # Build command
            cmd = [labrecorder_path]
            
            # Add config file if specified
            if config_path and os.path.exists(config_path):
                cmd.extend(["-c", config_path])
                self.log(f"Starting LabRecorder with multi-device config: {labrecorder_path} -c {config_path}")
            else:
                self.log(f"Starting LabRecorder: {labrecorder_path}")
                if config_path:
                    self.log(f"Warning: Config file not found: {config_path}")
            
            self.labrecorder_process = subprocess.Popen(cmd)
            
            self.start_labrecorder_btn.config(state=tk.DISABLED)
            self.close_labrecorder_btn.config(state=tk.NORMAL)
            
            self.log("LabRecorder started successfully")
            
        except Exception as e:
            self.log(f"Error starting LabRecorder: {e}")
            messagebox.showerror("Error", f"Failed to start LabRecorder: {e}")
    
    def close_labrecorder(self):
        """Close LabRecorder"""
        if self.labrecorder_process:
            try:
                self.labrecorder_process.terminate()
                self.labrecorder_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.labrecorder_process.kill()
            except Exception as e:
                self.log(f"Error closing LabRecorder: {e}")
            
            self.labrecorder_process = None
        
        self.start_labrecorder_btn.config(state=tk.NORMAL)
        self.close_labrecorder_btn.config(state=tk.DISABLED)
        self.log("LabRecorder closed")
    
    def quick_setup(self):
        """Quick setup - start both LSL and LabRecorder"""
        self.log("Starting multi-device quick setup...")
        
        # Start LSL streaming
        if not self.streaming_active:
            self.start_lsl_streaming()
            time.sleep(3)  # Give LSL time to start multiple devices
        
        # Start LabRecorder
        if not self.labrecorder_process:
            self.start_labrecorder()
        
        self.log("Multi-device quick setup completed")
        messagebox.showinfo("Quick Setup", "Both multi-device LSL streaming and LabRecorder have been started.\n\nNext steps:\n1. In LabRecorder, click 'Update' to see all streams\n2. Select streams for both Caregiver and Child\n3. Click 'Start' to begin recording\n\nLook for streams ending with '_Caregiver' and '_Child'\n\nNote: Eye events (fixations/saccades) only available from Child device.")
    
    def emergency_stop(self):
        """Emergency stop - stop everything"""
        self.log("Emergency stop initiated...")
        
        # Try graceful stop first for any LSL processes
        try:
            result = subprocess.run(["pgrep", "-f", "lsl_streams.py"], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                self.log(f"Sending SIGINT to LSL processes: {pids}")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGINT)
                    except:
                        pass
                
                time.sleep(2)
                
                # Check if any are still running and force kill
                result = subprocess.run(["pgrep", "-f", "lsl_streams.py"], capture_output=True, text=True)
                if result.stdout.strip():
                    remaining_pids = result.stdout.strip().split('\n')
                    self.log(f"Force killing remaining LSL processes: {remaining_pids}")
                    for pid in remaining_pids:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                        except:
                            pass
        except:
            pass
        
        # Stop LSL streaming
        if self.streaming_active:
            self.stop_lsl_streaming()
        
        # Close LabRecorder
        if self.labrecorder_process:
            self.close_labrecorder()
        
        self.log("Emergency stop completed")
        messagebox.showinfo("Emergency Stop", "All processes have been stopped")
    
    def check_lsl_processes(self):
        """Check what LSL processes are actually running"""
        self.log("=== CHECKING LSL PROCESSES ===")
        
        try:
            result = subprocess.run(
                ["pgrep", "-f", "lsl_streams.py"], 
                capture_output=True, text=True
            )
            
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                self.log(f"Found LSL processes with PIDs: {pids}")
                
                for pid in pids:
                    try:
                        ps_result = subprocess.run(
                            ["ps", "-p", pid, "-o", "pid,ppid,cmd"], 
                            capture_output=True, text=True
                        )
                        lines = ps_result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            self.log(f"Process {pid}: {lines[1]}")
                    except Exception as e:
                        self.log(f"Error getting details for PID {pid}: {e}")
            else:
                self.log("No LSL processes found")
            
            self.log(f"GUI thinks streaming_active: {self.streaming_active}")
            if self.lsl_process:
                if self.lsl_process.poll() is None:
                    self.log(f"GUI tracked process {self.lsl_process.pid}: RUNNING")
                else:
                    self.log(f"GUI tracked process {self.lsl_process.pid}: DEAD (return code: {self.lsl_process.poll()})")
            else:
                self.log("GUI: No process tracked")
                
        except Exception as e:
            self.log(f"Error checking processes: {e}")
    
    def force_kill_all_lsl(self):
        """Force kill all LSL processes"""
        self.log("=== FORCE KILL ALL LSL ===")
        
        try:
            result = subprocess.run(["pgrep", "-f", "lsl_streams.py"], capture_output=True, text=True)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                self.log(f"Found LSL processes to kill: {pids}")
                
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        self.log(f"Force killed process {pid}")
                    except Exception as e:
                        self.log(f"Could not kill process {pid}: {e}")
            else:
                self.log("No LSL processes found to kill")
                
            # Reset GUI state
            self.lsl_process = None
            self.streaming_active = False
            self.start_lsl_btn.config(state=tk.NORMAL)
            self.stop_lsl_btn.config(state=tk.DISABLED)
            self.log("GUI state reset")
            
            messagebox.showinfo("Force Kill Complete", "All LSL processes have been force killed!")
            
        except Exception as e:
            self.log(f"Error in force kill: {e}")
    
    def refresh_file_list(self):
        """Refresh the file list"""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        directory = self.file_dir_var.get()
        if not directory or not os.path.exists(directory):
            return
        
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.xdf'):
                    filepath = os.path.join(directory, filename)
                    stat = os.stat(filepath)
                    size = f"{stat.st_size / (1024*1024):.1f} MB"
                    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    
                    self.file_tree.insert("", tk.END, text=filename, values=(size, modified))
        except Exception as e:
            self.log(f"Error refreshing file list: {e}")
    
    def extract_selected_file(self):
        """Extract the selected XDF file"""
        selection = self.file_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an XDF file to extract")
            return
        
        filename = self.file_tree.item(selection[0])['text']
        self.extract_file(filename)
    
    def extract_all_files(self):
        """Extract all XDF files in the directory"""
        directory = self.file_dir_var.get()
        if not directory or not os.path.exists(directory):
            messagebox.showerror("Error", "Please select a valid directory")
            return
        
        xdf_files = [f for f in os.listdir(directory) if f.endswith('.xdf')]
        if not xdf_files:
            messagebox.showinfo("No Files", "No XDF files found in the selected directory")
            return
        
        if messagebox.askyesno("Extract All", f"Extract {len(xdf_files)} XDF files?"):
            for filename in xdf_files:
                self.extract_file(filename)
    
    def extract_file(self, filename):
        """Extract a specific XDF file"""
        directory = self.file_dir_var.get()
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File not found: {filepath}")
            return
        
        # Start extraction in a separate thread
        threading.Thread(target=self._extract_file_thread, args=(filepath,), daemon=True).start()
    
    def _extract_file_thread(self, filepath):
        """Extract file in a separate thread"""
        try:
            self.extraction_progress.start()
            
            # Determine output directory
            extraction_drive = self.extraction_drive_var.get()
            if extraction_drive and os.path.exists(extraction_drive):
                # Create timestamped directory on extraction drive
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                basename = os.path.splitext(os.path.basename(filepath))[0]
                output_dir = os.path.join(extraction_drive, f"{basename}_{timestamp}")
            else:
                # Use same directory as XDF file
                output_dir = os.path.join(os.path.dirname(filepath), 
                                         os.path.splitext(os.path.basename(filepath))[0] + "_extracted")
            
            self.log(f"Extracting {filepath} to {output_dir}")
            
            # Build extraction command
            script_dir = os.path.dirname(self.lsl_script_var.get()) if self.lsl_script_var.get() else "."
            extract_script = os.path.join(script_dir, "xdf_extract.py")
            
            if not os.path.exists(extract_script):
                # Look for extract script in same directory as this GUI
                extract_script = os.path.join(os.path.dirname(__file__), "xdf_extract.py")
            
            if not os.path.exists(extract_script):
                raise FileNotFoundError("xdf_extract.py not found")
            
            conda_env = self.conda_env_var.get()
            if conda_env:
                cmd = ["conda", "run", "-n", conda_env, "python", extract_script]
            else:
                cmd = ["python3", extract_script]
            
            cmd.extend(["--file", filepath, "--outdir", output_dir])
            
            if not self.keep_raw_depth_var.get():
                cmd.append("--no-raw-depth")
            
            cmd.extend(["--depth-interval", str(self.depth_interval_var.get())])
            
            # Run extraction
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log(f"Extraction completed successfully: {output_dir}")
                
                # Delete XDF file if requested
                if self.delete_after_extract_var.get():
                    try:
                        os.remove(filepath)
                        self.log(f"Deleted XDF file: {filepath}")
                    except Exception as e:
                        self.log(f"Error deleting XDF file: {e}")
                
                # Refresh file list
                self.root.after(0, self.refresh_file_list)
                self.root.after(0, lambda: messagebox.showinfo("Extraction Complete", 
                                                               f"Extraction completed successfully!\nOutput: {output_dir}"))
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                self.log(f"Extraction failed: {error_msg}")
                self.root.after(0, lambda: messagebox.showerror("Extraction Failed", 
                                                               f"Extraction failed:\n{error_msg}"))
        
        except Exception as e:
            self.log(f"Error during extraction: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error during extraction: {e}"))
        
        finally:
            self.extraction_progress.stop()
    
    def delete_selected_file(self):
        """Delete the selected XDF file"""
        selection = self.file_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an XDF file to delete")
            return
        
        filename = self.file_tree.item(selection[0])['text']
        filepath = os.path.join(self.file_dir_var.get(), filename)
        
        if messagebox.askyesno("Delete File", f"Are you sure you want to delete {filename}?"):
            try:
                os.remove(filepath)
                self.log(f"Deleted file: {filepath}")
                self.refresh_file_list()
            except Exception as e:
                self.log(f"Error deleting file: {e}")
                messagebox.showerror("Error", f"Error deleting file: {e}")
    
    def update_status_loop(self):
        """Update status indicators in a loop"""
        while self.update_thread_running:
            try:
                # Update LSL status
                if self.streaming_active and self.lsl_process and self.lsl_process.poll() is None:
                    self.lsl_status_label.config(text="LSL Streaming: Running", foreground="green")
                else:
                    self.lsl_status_label.config(text="LSL Streaming: Not Running", foreground="red")
                    if self.streaming_active:
                        self.streaming_active = False
                        self.start_lsl_btn.config(state=tk.NORMAL)
                        self.stop_lsl_btn.config(state=tk.DISABLED)
                
                # Update LabRecorder status
                if self.labrecorder_process and self.labrecorder_process.poll() is None:
                    self.labrecorder_status_label.config(text="LabRecorder: Running", foreground="green")
                else:
                    self.labrecorder_status_label.config(text="LabRecorder: Not Running", foreground="red")
                    if self.labrecorder_process:
                        self.labrecorder_process = None
                        self.start_labrecorder_btn.config(state=tk.NORMAL)
                        self.close_labrecorder_btn.config(state=tk.DISABLED)
            
            except Exception as e:
                print(f"Error in status update: {e}")
            
            time.sleep(1)
    
    def log(self, message):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Add to log text widget
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        
        # Also print to console
        print(log_message.strip())
    
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
    
    def save_logs(self):
        """Save logs to a file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Saved", f"Logs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving logs: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        # Save current configuration
        self.save_current_config()
        
        # Stop update thread
        self.update_thread_running = False
        
        # Stop processes
        if self.streaming_active:
            self.stop_lsl_streaming()
        if self.labrecorder_process:
            self.close_labrecorder()
        
        # Close window
        self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = RecordingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
