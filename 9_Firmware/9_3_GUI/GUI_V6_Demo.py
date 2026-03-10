
"""
Radar System GUI - Demo Version
Compatible with FT601 USB 3.0 interface
Includes simulated radar data for demonstration
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import random
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RadarTarget:
    """Represents a detected radar target"""
    id: int
    range: float  # meters
    velocity: float  # m/s
    azimuth: float  # degrees
    elevation: float  # degrees
    snr: float  # dB
    timestamp: float
    track_id: int = -1
    range_bin: int = 0
    doppler_bin: int = 0
    
@dataclass
class RadarSettings:
    """Radar system settings"""
    system_frequency: float = 10e9  # Hz
    chirp_duration_long: float = 30e-6  # seconds
    chirp_duration_short: float = 0.5e-6  # seconds
    chirps_per_frame: int = 32
    range_bins: int = 1024
    doppler_bins: int = 32
    prf: float = 1000  # Hz
    max_range: float = 5000  # meters
    max_velocity: float = 100  # m/s
    cfar_threshold: float = 13.0  # dB
    beam_width: float = 3.0  # degrees
    
@dataclass
class SystemStatus:
    """System status information"""
    uptime: float = 0
    temperature: float = 25.0
    fpga_utilization: float = 0
    usb_data_rate: float = 0
    packets_received: int = 0
    packets_lost: int = 0
    errors: int = 0
    warnings: List[str] = None

# ============================================================================
# SIMULATED RADAR DATA GENERATOR
# ============================================================================

class RadarDataSimulator:
    """Generates simulated radar data for demo mode"""
    
    def __init__(self):
        self.settings = RadarSettings()
        self.targets = []
        self.frame_count = 0
        self.create_test_scenario()
        
    def create_test_scenario(self):
        """Create a test scenario with moving targets"""
        # Target 1: Fast-moving aircraft
        self.targets.append({
            'id': 1,
            'range': 2500,
            'velocity': -80,  # Approaching
            'azimuth': 45,
            'elevation': 5,
            'snr': 25,
            'range_drift': -0.5,  # m per frame
            'azimuth_drift': 0.1,  # deg per frame
            'velocity_drift': 0
        })
        
        # Target 2: Slow-moving vehicle
        self.targets.append({
            'id': 2,
            'range': 800,
            'velocity': 15,  # Receding
            'azimuth': -30,
            'elevation': 0,
            'snr': 18,
            'range_drift': 0.2,
            'azimuth_drift': -0.05,
            'velocity_drift': 0.1
        })
        
        # Target 3: Stationary object
        self.targets.append({
            'id': 3,
            'range': 1500,
            'velocity': 0,
            'azimuth': 10,
            'elevation': 2,
            'snr': 22,
            'range_drift': 0,
            'azimuth_drift': 0,
            'velocity_drift': 0
        })
        
        # Target 4: Helicopter (low speed, high SNR)
        self.targets.append({
            'id': 4,
            'range': 1200,
            'velocity': -5,
            'azimuth': 75,
            'elevation': 15,
            'snr': 30,
            'range_drift': -0.1,
            'azimuth_drift': -0.2,
            'velocity_drift': 0.05
        })
        
        # Target 5: Drone (small, fluctuating)
        self.targets.append({
            'id': 5,
            'range': 300,
            'velocity': 10,
            'azimuth': -60,
            'elevation': 8,
            'snr': 12,
            'range_drift': 0.3,
            'azimuth_drift': 0.4,
            'velocity_drift': -0.2
        })
        
    def generate_range_doppler_map(self):
        """Generate simulated range-Doppler map"""
        map_data = np.zeros((self.settings.range_bins, self.settings.doppler_bins))
        
        # Add noise floor
        noise_floor = 10 * np.random.random(map_data.shape)
        map_data += noise_floor
        
        # Add targets
        for target in self.targets:
            # Update target position
            target['range'] += target['range_drift']
            target['azimuth'] += target['azimuth_drift']
            target['velocity'] += target['velocity_drift']
            
            # Keep within bounds
            target['range'] = max(100, min(5000, target['range']))
            target['azimuth'] = max(-90, min(90, target['azimuth']))
            target['velocity'] = max(-100, min(100, target['velocity']))
            
            # Convert to bin indices
            range_bin = int((target['range'] / self.settings.max_range) * 
                          (self.settings.range_bins - 1))
            doppler_bin = int(((target['velocity'] + self.settings.max_velocity) / 
                              (2 * self.settings.max_velocity)) * 
                              (self.settings.doppler_bins - 1))
            
            # Ensure valid indices
            range_bin = max(0, min(self.settings.range_bins - 1, range_bin))
            doppler_bin = max(0, min(self.settings.doppler_bins - 1, doppler_bin))
            
            # Add target peak with spreading
            for r in range(max(0, range_bin-2), min(self.settings.range_bins, range_bin+3)):
                for d in range(max(0, doppler_bin-2), min(self.settings.doppler_bins, doppler_bin+3)):
                    distance = np.sqrt((r - range_bin)**2 + (d - doppler_bin)**2)
                    if distance < 3:
                        amplitude = target['snr'] * np.exp(-distance/2)
                        map_data[r, d] += amplitude * (0.8 + 0.4 * np.random.random())
        
        # Add some clutter
        clutter_bins = np.random.randint(0, self.settings.range_bins, 20)
        for cb in clutter_bins:
            map_data[cb, :] += 5 * np.random.random(self.settings.doppler_bins)
        
        self.frame_count += 1
        return map_data
    
    def get_detected_targets(self):
        """Get list of detected targets"""
        detected = []
        for i, target in enumerate(self.targets):
            # Random detection probability based on SNR
            if np.random.random() < (target['snr'] / 40):
                detected.append(RadarTarget(
                    id=target['id'],
                    range=target['range'],
                    velocity=target['velocity'],
                    azimuth=target['azimuth'],
                    elevation=target['elevation'],
                    snr=target['snr'] + 2 * np.random.random() - 1,
                    timestamp=time.time(),
                    track_id=target['id'],
                    range_bin=int((target['range'] / self.settings.max_range) * 
                                 (self.settings.range_bins - 1)),
                    doppler_bin=int(((target['velocity'] + self.settings.max_velocity) / 
                                    (2 * self.settings.max_velocity)) * 
                                    (self.settings.doppler_bins - 1))
                ))
        return detected

# ============================================================================
# SIMULATED USB INTERFACE
# ============================================================================

class SimulatedUSBInterface:
    """Simulates FT601 USB interface for demo mode"""
    
    def __init__(self):
        self.is_open = False
        self.data_rate = 0
        self.packet_count = 0
        self.byte_count = 0
        self.start_time = time.time()
        
    def list_devices(self):
        """Return simulated devices"""
        return [
            {'description': 'FT601 USB 3.0 Device (Demo Mode)', 
             'vid': 0x0403, 'pid': 0x6030},
            {'description': 'FT601 USB 3.0 Device - Channel 1', 
             'vid': 0x0403, 'pid': 0x6030}
        ]
    
    def open_device(self, device_info):
        """Simulate opening device"""
        self.is_open = True
        self.start_time = time.time()
        logger.info(f"Demo mode: Opened simulated FT601 device")
        return True
    
    def close(self):
        """Simulate closing device"""
        self.is_open = False
        logger.info("Demo mode: Closed simulated FT601 device")
    
    def read_data(self, size=4096):
        """Simulate reading data (returns None in demo mode)"""
        # In demo mode, we generate data separately
        return None
    
    def write_data(self, data):
        """Simulate writing data"""
        self.packet_count += 1
        self.byte_count += len(data)
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.data_rate = self.byte_count / elapsed / 1024  # KB/s
        return True
    
    def get_stats(self):
        """Get interface statistics"""
        return {
            'packets': self.packet_count,
            'bytes': self.byte_count,
            'data_rate': self.data_rate
        }

# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class RadarGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Radar System GUI - Demo Mode")
        self.root.geometry("1600x900")
        
        # Set icon if available
        try:
            self.root.iconbitmap(default='radar.ico')
        except:
            pass
        
        # Configure dark theme
        self.setup_dark_theme()
        
        # Initialize components
        self.settings = RadarSettings()
        self.status = SystemStatus()
        self.simulator = RadarDataSimulator()
        self.usb_interface = SimulatedUSBInterface()
        
        # Data queues
        self.radar_data_queue = queue.Queue(maxsize=100)
        self.command_queue = queue.Queue()
        
        # Control flags
        self.demo_mode = tk.BooleanVar(value=True)
        self.running = False
        self.recording = False
        self.auto_scan = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)
        self.show_targets = tk.BooleanVar(value=True)
        self.color_map = tk.StringVar(value='hot')
        
        # Data storage
        self.range_doppler_map = np.zeros((1024, 32))
        self.detected_targets = []
        self.target_history = []
        self.recorded_frames = []
        
        # Animation
        self.animation_running = True
        self.frame_count = 0
        self.fps = 0
        self.last_frame_time = time.time()
        
        # Create GUI
        self.create_menu()
        self.create_gui()
        self.create_status_bar()
        
        # Start background threads
        self.start_background_threads()
        
        # Start animation
        self.animate()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        logger.info("Radar GUI initialized in demo mode")
    
    def setup_dark_theme(self):
        """Configure dark theme for the GUI"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Dark theme colors
        dark_bg = '#2b2b2b'
        dark_fg = '#e0e0e0'
        dark_select = '#404040'
        dark_button = '#3c3f41'
        
        # Configure styles
        self.style.configure('.', 
                           background=dark_bg,
                           foreground=dark_fg,
                           fieldbackground=dark_bg,
                           troughcolor=dark_select)
        
        self.style.configure('TLabel', background=dark_bg, foreground=dark_fg)
        self.style.configure('TFrame', background=dark_bg)
        self.style.configure('TLabelframe', background=dark_bg, foreground=dark_fg)
        self.style.configure('TLabelframe.Label', background=dark_bg, foreground=dark_fg)
        
        self.style.configure('TButton', 
                           background=dark_button,
                           foreground=dark_fg,
                           borderwidth=1,
                           focuscolor='none')
        self.style.map('TButton',
                      background=[('active', '#4e5254'),
                                ('pressed', '#2d2f31')])
        
        self.style.configure('TNotebook', background=dark_bg)
        self.style.configure('TNotebook.Tab', 
                           background=dark_select,
                           foreground=dark_fg,
                           padding=[10, 2])
        self.style.map('TNotebook.Tab',
                      background=[('selected', dark_button)])
        
        self.style.configure('Treeview',
                           background=dark_select,
                           foreground=dark_fg,
                           fieldbackground=dark_select)
        self.style.map('Treeview',
                      background=[('selected', '#4e5254')])
        
        self.style.configure('Horizontal.TScale', background=dark_bg)
        self.style.configure('Vertical.TScale', background=dark_bg)
    
    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Connect to Hardware", command=self.connect_hardware)
        file_menu.add_command(label="Demo Mode", command=self.enable_demo_mode)
        file_menu.add_separator()
        file_menu.add_command(label="Start Recording", command=self.start_recording)
        file_menu.add_command(label="Stop Recording", command=self.stop_recording)
        file_menu.add_separator()
        file_menu.add_command(label="Load Configuration", command=self.load_config)
        file_menu.add_command(label="Save Configuration", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Grid", variable=self.show_grid)
        view_menu.add_checkbutton(label="Show Targets", variable=self.show_targets)
        view_menu.add_checkbutton(label="Auto Scan", variable=self.auto_scan)
        view_menu.add_separator()
        
        # Color map submenu
        color_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Color Map", menu=color_menu)
        for cmap in ['hot', 'jet', 'viridis', 'plasma', 'inferno', 'magma']:
            color_menu.add_radiobutton(label=cmap.capitalize(), 
                                      variable=self.color_map, 
                                      value=cmap)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Calibration Wizard", command=self.calibration_wizard)
        tools_menu.add_command(label="Beam Pattern Analysis", command=self.beam_analysis)
        tools_menu.add_command(label="Noise Floor Measurement", command=self.noise_measurement)
        tools_menu.add_separator()
        tools_menu.add_command(label="Diagnostics", command=self.show_diagnostics)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_gui(self):
        """Create main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Control panel (top)
        control_frame = ttk.LabelFrame(main_frame, text="System Control", padding=5)
        control_frame.pack(fill='x', pady=(0, 5))
        
        self.create_control_panel(control_frame)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.create_radar_tab()
        self.create_scope_tab()
        self.create_targets_tab()
        self.create_spectrum_tab()
        self.create_history_tab()
        self.create_settings_tab()
    
    def create_control_panel(self, parent):
        """Create system control panel"""
        # Left side - Mode and connection
        left_frame = ttk.Frame(parent)
        left_frame.pack(side='left', fill='x', expand=True)
        
        ttk.Label(left_frame, text="Mode:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        mode_combo = ttk.Combobox(left_frame, textvariable=tk.StringVar(value="Demo Mode"),
                                 values=["Demo Mode", "Hardware Mode"], 
                                 state="readonly", width=15)
        mode_combo.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        
        ttk.Label(left_frame, text="Device:").grid(row=0, column=2, padx=5, pady=2, sticky='w')
        self.device_combo = ttk.Combobox(left_frame, 
                                        values=["FT601 Demo Device"],
                                        state="readonly", width=25)
        self.device_combo.grid(row=0, column=3, padx=5, pady=2, sticky='w')
        self.device_combo.current(0)
        
        ttk.Button(left_frame, text="Refresh", 
                  command=self.refresh_devices).grid(row=0, column=4, padx=2, pady=2)
        
        ttk.Button(left_frame, text="Connect", 
                  command=self.connect_device).grid(row=0, column=5, padx=2, pady=2)
        
        # Right side - Start/Stop
        right_frame = ttk.Frame(parent)
        right_frame.pack(side='right', padx=10)
        
        self.start_button = ttk.Button(right_frame, text="▶ Start", 
                                      command=self.start_radar, width=10)
        self.start_button.pack(side='left', padx=2)
        
        self.stop_button = ttk.Button(right_frame, text="■ Stop", 
                                     command=self.stop_radar, width=10,
                                     state='disabled')
        self.stop_button.pack(side='left', padx=2)
        
        ttk.Button(right_frame, text="⚙ Settings", 
                  command=lambda: self.notebook.select(5)).pack(side='left', padx=2)
    
    def create_radar_tab(self):
        """Create radar display tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Radar Display")
        
        # Main display area
        display_frame = ttk.Frame(tab)
        display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Range-Doppler map
        map_frame = ttk.LabelFrame(display_frame, text="Range-Doppler Map", padding=5)
        map_frame.pack(side='left', fill='both', expand=True)
        
        # Create matplotlib figure
        self.rd_fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
        self.rd_ax = self.rd_fig.add_subplot(111)
        self.rd_ax.set_facecolor('#1a1a1a')
        
        # Initialize plot
        self.rd_img = self.rd_ax.imshow(
            np.random.rand(1024, 32),
            aspect='auto',
            cmap=self.color_map.get(),
            extent=[-self.settings.max_velocity, 
                   self.settings.max_velocity,
                   self.settings.max_range, 0],
            interpolation='bilinear'
        )
        
        self.rd_ax.set_xlabel('Velocity (m/s)', color='white')
        self.rd_ax.set_ylabel('Range (m)', color='white')
        self.rd_ax.tick_params(colors='white')
        
        # Add colorbar
        self.rd_cbar = self.rd_fig.colorbar(self.rd_img, ax=self.rd_ax)
        self.rd_cbar.ax.yaxis.set_tick_params(color='white')
        self.rd_cbar.ax.set_ylabel('Power (dB)', color='white')
        plt.setp(plt.getp(self.rd_cbar.ax.axes, 'yticklabels'), color='white')
        
        # Embed in tkinter
        self.rd_canvas = FigureCanvasTkAgg(self.rd_fig, map_frame)
        self.rd_canvas.draw()
        self.rd_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.rd_canvas, map_frame)
        toolbar.update()
        
        # Target list panel
        target_frame = ttk.LabelFrame(display_frame, text="Detected Targets", padding=5, width=300)
        target_frame.pack(side='right', fill='y', padx=(5, 0))
        target_frame.pack_propagate(False)
        
        # Treeview for targets
        columns = ('ID', 'Range', 'Velocity', 'Azimuth', 'Elevation', 'SNR')
        self.target_tree = ttk.Treeview(target_frame, columns=columns, show='headings', height=20)
        
        # Define headings
        self.target_tree.heading('ID', text='ID')
        self.target_tree.heading('Range', text='Range (m)')
        self.target_tree.heading('Velocity', text='Vel (m/s)')
        self.target_tree.heading('Azimuth', text='Az (°)')
        self.target_tree.heading('Elevation', text='El (°)')
        self.target_tree.heading('SNR', text='SNR (dB)')
        
        # Set column widths
        self.target_tree.column('ID', width=40)
        self.target_tree.column('Range', width=70)
        self.target_tree.column('Velocity', width=70)
        self.target_tree.column('Azimuth', width=60)
        self.target_tree.column('Elevation', width=60)
        self.target_tree.column('SNR', width=60)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(target_frame, orient='vertical', 
                                  command=self.target_tree.yview)
        self.target_tree.configure(yscrollcommand=scrollbar.set)
        
        self.target_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def create_scope_tab(self):
        """Create A-scope tab (range profile)"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="A-Scope")
        
        # Create figure
        self.scope_fig = Figure(figsize=(10, 6), facecolor='#2b2b2b')
        self.scope_ax = self.scope_fig.add_subplot(111)
        self.scope_ax.set_facecolor('#1a1a1a')
        
        # Initialize plot
        self.scope_line, = self.scope_ax.plot([], [], 'g-', linewidth=1)
        self.scope_ax.set_xlim(0, self.settings.max_range)
        self.scope_ax.set_ylim(0, 100)
        self.scope_ax.set_xlabel('Range (m)', color='white')
        self.scope_ax.set_ylabel('Amplitude (dB)', color='white')
        self.scope_ax.set_title('Range Profile (A-Scope)', color='white')
        self.scope_ax.grid(True, alpha=0.3)
        self.scope_ax.tick_params(colors='white')
        
        self.scope_canvas = FigureCanvasTkAgg(self.scope_fig, tab)
        self.scope_canvas.draw()
        self.scope_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Control panel
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', pady=5)
        
        ttk.Label(control_frame, text="Azimuth:").pack(side='left', padx=5)
        self.scope_azimuth = ttk.Scale(control_frame, from_=-90, to=90, 
                                       orient='horizontal', length=200)
        self.scope_azimuth.pack(side='left', padx=5)
        self.scope_azimuth.set(0)
        
        ttk.Label(control_frame, text="Elevation:").pack(side='left', padx=5)
        self.scope_elevation = ttk.Scale(control_frame, from_=-45, to=45,
                                        orient='horizontal', length=200)
        self.scope_elevation.pack(side='left', padx=5)
        self.scope_elevation.set(0)
    
    def create_targets_tab(self):
        """Create PPI scope tab (plan position indicator)"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="PPI Scope")
        
        # Create figure
        self.ppi_fig = Figure(figsize=(8, 8), facecolor='#2b2b2b')
        self.ppi_ax = self.ppi_fig.add_subplot(111, projection='polar')
        self.ppi_ax.set_facecolor('#1a1a1a')
        
        # Initialize plot
        self.ppi_scatter = self.ppi_ax.scatter([], [], c=[], s=50, alpha=0.8, cmap='hot')
        self.ppi_ax.set_ylim(0, self.settings.max_range)
        self.ppi_ax.grid(True, alpha=0.3)
        self.ppi_ax.tick_params(colors='white')
        
        # Convert azimuth to radar convention (0° = North, clockwise)
        self.ppi_ax.set_theta_zero_location('N')
        self.ppi_ax.set_theta_direction(-1)
        
        self.ppi_canvas = FigureCanvasTkAgg(self.ppi_fig, tab)
        self.ppi_canvas.draw()
        self.ppi_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_spectrum_tab(self):
        """Create Doppler spectrum tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Doppler Spectrum")
        
        # Create figure
        self.spec_fig = Figure(figsize=(10, 6), facecolor='#2b2b2b')
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_facecolor('#1a1a1a')
        
        # Initialize plot
        self.spec_line, = self.spec_ax.plot([], [], 'b-', linewidth=1)
        self.spec_ax.set_xlim(-self.settings.max_velocity, self.settings.max_velocity)
        self.spec_ax.set_ylim(0, 100)
        self.spec_ax.set_xlabel('Velocity (m/s)', color='white')
        self.spec_ax.set_ylabel('Power (dB)', color='white')
        self.spec_ax.set_title('Doppler Spectrum', color='white')
        self.spec_ax.grid(True, alpha=0.3)
        self.spec_ax.tick_params(colors='white')
        
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, tab)
        self.spec_canvas.draw()
        self.spec_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Range selector
        range_frame = ttk.Frame(tab)
        range_frame.pack(fill='x', pady=5)
        
        ttk.Label(range_frame, text="Range Bin:").pack(side='left', padx=5)
        self.spec_range = ttk.Scale(range_frame, from_=0, to=1023,
                                    orient='horizontal', length=400)
        self.spec_range.pack(side='left', padx=5)
        self.spec_range.set(512)
        
        ttk.Label(range_frame, text="512").pack(side='left', padx=5)
    
    def create_history_tab(self):
        """Create target history tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Target History")
        
        # Create figure
        self.hist_fig = Figure(figsize=(10, 6), facecolor='#2b2b2b')
        
        # Range history
        self.hist_ax1 = self.hist_fig.add_subplot(211)
        self.hist_ax1.set_facecolor('#1a1a1a')
        self.hist_ax1.set_xlabel('Time (s)', color='white')
        self.hist_ax1.set_ylabel('Range (m)', color='white')
        self.hist_ax1.set_title('Target Range History', color='white')
        self.hist_ax1.grid(True, alpha=0.3)
        self.hist_ax1.tick_params(colors='white')
        
        # Velocity history
        self.hist_ax2 = self.hist_fig.add_subplot(212)
        self.hist_ax2.set_facecolor('#1a1a1a')
        self.hist_ax2.set_xlabel('Time (s)', color='white')
        self.hist_ax2.set_ylabel('Velocity (m/s)', color='white')
        self.hist_ax2.set_title('Target Velocity History', color='white')
        self.hist_ax2.grid(True, alpha=0.3)
        self.hist_ax2.tick_params(colors='white')
        
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, tab)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Clear button
        ttk.Button(tab, text="Clear History", 
                  command=self.clear_history).pack(pady=5)
    
    def create_settings_tab(self):
        """Create settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Settings")
        
        # Create notebook for settings categories
        settings_notebook = ttk.Notebook(tab)
        settings_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Radar settings
        radar_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(radar_frame, text="Radar")
        
        self.create_radar_settings(radar_frame)
        
        # Display settings
        display_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(display_frame, text="Display")
        
        self.create_display_settings(display_frame)
        
        # Detection settings
        detection_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(detection_frame, text="Detection")
        
        self.create_detection_settings(detection_frame)
        
        # Recording settings
        record_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(record_frame, text="Recording")
        
        self.create_recording_settings(record_frame)
        
        # System settings
        system_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(system_frame, text="System")
        
        self.create_system_settings(system_frame)
    
    def create_radar_settings(self, parent):
        """Create radar settings controls"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Settings variables
        self.settings_vars = {}
        
        settings = [
            ('System Frequency (GHz):', 'freq', 10.0, 1.0, 20.0),
            ('Long Chirp Duration (µs):', 'long_dur', 30.0, 1.0, 100.0),
            ('Short Chirp Duration (µs):', 'short_dur', 0.5, 0.1, 10.0),
            ('Chirps per Frame:', 'chirps', 32, 1, 128),
            ('Range Bins:', 'range_bins', 1024, 64, 2048),
            ('Doppler Bins:', 'doppler_bins', 32, 8, 128),
            ('PRF (Hz):', 'prf', 1000, 100, 10000),
            ('Max Range (m):', 'max_range', 5000, 100, 50000),
            ('Max Velocity (m/s):', 'max_vel', 100, 10, 500),
            ('Beam Width (°):', 'beam_width', 3.0, 0.5, 10.0)
        ]
        
        for i, (label, key, default, minv, maxv) in enumerate(settings):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', padx=10, pady=5)
            
            ttk.Label(frame, text=label, width=25).pack(side='left')
            
            var = tk.DoubleVar(value=default)
            self.settings_vars[key] = var
            
            entry = ttk.Entry(frame, textvariable=var, width=15)
            entry.pack(side='left', padx=5)
            
            ttk.Label(frame, text=f"({minv}-{maxv})").pack(side='left')
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Apply button
        ttk.Button(scrollable_frame, text="Apply Settings", 
                  command=self.apply_settings).pack(pady=10)
    
    def create_display_settings(self, parent):
        """Create display settings controls"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Update rate
        ttk.Label(frame, text="Update Rate (Hz):").grid(row=0, column=0, 
                                                        sticky='w', pady=5)
        self.update_rate = ttk.Scale(frame, from_=1, to=60, 
                                     orient='horizontal', length=200)
        self.update_rate.set(20)
        self.update_rate.grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(frame, text="20").grid(row=0, column=2, sticky='w')
        
        # Color map
        ttk.Label(frame, text="Color Map:").grid(row=1, column=0, 
                                                 sticky='w', pady=5)
        cmap_combo = ttk.Combobox(frame, textvariable=self.color_map,
                                  values=['hot', 'jet', 'viridis', 'plasma'],
                                  state='readonly', width=15)
        cmap_combo.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        # Grid
        ttk.Checkbutton(frame, text="Show Grid", 
                       variable=self.show_grid).grid(row=2, column=0, 
                                                     columnspan=2, sticky='w', pady=5)
        
        # Targets
        ttk.Checkbutton(frame, text="Show Targets", 
                       variable=self.show_targets).grid(row=3, column=0, 
                                                        columnspan=2, sticky='w', pady=5)
        
        # Auto scan
        ttk.Checkbutton(frame, text="Auto Scan", 
                       variable=self.auto_scan).grid(row=4, column=0, 
                                                     columnspan=2, sticky='w', pady=5)
        
        # Background color
        ttk.Label(frame, text="Background:").grid(row=5, column=0, 
                                                  sticky='w', pady=5)
        bg_colors = ['Dark', 'Black', 'White']
        bg_combo = ttk.Combobox(frame, values=bg_colors, state='readonly', width=15)
        bg_combo.set('Dark')
        bg_combo.grid(row=5, column=1, padx=10, pady=5, sticky='w')
    
    def create_detection_settings(self, parent):
        """Create detection settings controls"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # CFAR threshold
        ttk.Label(frame, text="CFAR Threshold (dB):").grid(row=0, column=0,
                                                           sticky='w', pady=5)
        self.cfar_threshold = ttk.Scale(frame, from_=5, to=30,
                                        orient='horizontal', length=200)
        self.cfar_threshold.set(13)
        self.cfar_threshold.grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(frame, text="13.0").grid(row=0, column=2, sticky='w')
        
        # Detection sensitivity
        ttk.Label(frame, text="Sensitivity:").grid(row=1, column=0,
                                                   sticky='w', pady=5)
        self.sensitivity = ttk.Scale(frame, from_=0, to=100,
                                     orient='horizontal', length=200)
        self.sensitivity.set(75)
        self.sensitivity.grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(frame, text="75%").grid(row=1, column=2, sticky='w')
        
        # Min SNR
        ttk.Label(frame, text="Min SNR (dB):").grid(row=2, column=0,
                                                    sticky='w', pady=5)
        self.min_snr = ttk.Scale(frame, from_=0, to=20,
                                 orient='horizontal', length=200)
        self.min_snr.set(10)
        self.min_snr.grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(frame, text="10.0").grid(row=2, column=2, sticky='w')
        
        # Detection mode
        ttk.Label(frame, text="Detection Mode:").grid(row=3, column=0,
                                                      sticky='w', pady=5)
        mode_combo = ttk.Combobox(frame, 
                                  values=['CFAR', 'Threshold', 'Peak'],
                                  state='readonly', width=15)
        mode_combo.set('CFAR')
        mode_combo.grid(row=3, column=1, padx=10, pady=5, sticky='w')
    
    def create_recording_settings(self, parent):
        """Create recording settings controls"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Recording directory
        ttk.Label(frame, text="Save Directory:").grid(row=0, column=0,
                                                      sticky='w', pady=5)
        self.record_dir = tk.StringVar(value="./recordings")
        ttk.Entry(frame, textvariable=self.record_dir, width=40).grid(
            row=0, column=1, padx=10, pady=5, columnspan=2)
        
        # Browse button
        ttk.Button(frame, text="Browse", 
                  command=self.browse_directory).grid(row=0, column=3, padx=5)
        
        # File format
        ttk.Label(frame, text="File Format:").grid(row=1, column=0,
                                                   sticky='w', pady=5)
        format_combo = ttk.Combobox(frame, 
                                    values=['HDF5', 'NPZ', 'CSV', 'MAT'],
                                    state='readonly', width=15)
        format_combo.set('HDF5')
        format_combo.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        # Auto naming
        ttk.Checkbutton(frame, text="Auto-generate filenames",
                       variable=tk.BooleanVar(value=True)).grid(
                           row=2, column=0, columnspan=2, sticky='w', pady=5)
        
        # Max file size
        ttk.Label(frame, text="Max File Size (MB):").grid(row=3, column=0,
                                                          sticky='w', pady=5)
        ttk.Entry(frame, width=15).grid(row=3, column=1, padx=10, pady=5, sticky='w')
    
    def create_system_settings(self, parent):
        """Create system settings controls"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # System info
        info_frame = ttk.LabelFrame(frame, text="System Information", padding=10)
        info_frame.pack(fill='x', pady=10)
        
        ttk.Label(info_frame, text="FPGA Temperature:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Label(info_frame, text="25.5 °C").grid(row=0, column=1, padx=20, sticky='w')
        
        ttk.Label(info_frame, text="FPGA Utilization:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Label(info_frame, text="45%").grid(row=1, column=1, padx=20, sticky='w')
        
        ttk.Label(info_frame, text="USB Data Rate:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Label(info_frame, text="0 KB/s").grid(row=2, column=1, padx=20, sticky='w')
        
        ttk.Label(info_frame, text="Uptime:").grid(row=3, column=0, sticky='w', pady=2)
        ttk.Label(info_frame, text="00:00:00").grid(row=3, column=1, padx=20, sticky='w')
        
        # Control buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Button(button_frame, text="System Diagnostics", 
                  command=self.show_diagnostics).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Reset System", 
                  command=self.reset_system).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Factory Reset", 
                  command=self.factory_reset).pack(side='left', padx=5)
    
    def create_status_bar(self):
        """Create status bar at bottom of window"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side='bottom', fill='x')
        
        # Left side status
        self.status_label = ttk.Label(status_frame, text="Status: Demo Mode", 
                                      relief='sunken', padding=2)
        self.status_label.pack(side='left', fill='x', expand=True)
        
        # Right side indicators
        self.fps_label = ttk.Label(status_frame, text="FPS: 0", 
                                   relief='sunken', width=10)
        self.fps_label.pack(side='right', padx=1)
        
        self.data_label = ttk.Label(status_frame, text="Data: 0 KB/s", 
                                    relief='sunken', width=12)
        self.data_label.pack(side='right', padx=1)
        
        self.targets_label = ttk.Label(status_frame, text="Targets: 0", 
                                       relief='sunken', width=10)
        self.targets_label.pack(side='right', padx=1)
        
        self.time_label = ttk.Label(status_frame, text=time.strftime("%H:%M:%S"),
                                    relief='sunken', width=8)
        self.time_label.pack(side='right', padx=1)
    
    # ============================================================================
    # GUI UPDATE METHODS
    # ============================================================================
    
    def animate(self):
        """Animation loop for updating plots"""
        if not self.animation_running:
            return
            
        try:
            # Update FPS counter
            current_time = time.time()
            dt = current_time - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 / dt
            self.last_frame_time = current_time
            
            # Update displays if running
            if self.running:
                self.update_radar_display()
                self.update_target_list()
                self.update_scopes()
                self.frame_count += 1
            
            # Update status bar
            self.update_status_bar()
            
        except Exception as e:
            logger.error(f"Animation error: {e}")
        
        # Schedule next update
        update_ms = int(1000 / max(1, self.update_rate.get()))
        self.root.after(update_ms, self.animate)
    
    def update_radar_display(self):
        """Update radar display with new data"""
        try:
            # Generate or get new radar data
            if self.demo_mode.get():
                # Generate simulated data
                rd_map = self.simulator.generate_range_doppler_map()
                targets = self.simulator.get_detected_targets()
            else:
                # Get data from queue
                try:
                    data = self.radar_data_queue.get_nowait()
                    rd_map = data['map']
                    targets = data['targets']
                except queue.Empty:
                    return
            
            # Update range-Doppler map
            self.range_doppler_map = rd_map
            log_map = 10 * np.log10(rd_map + 1)
            
            # Update image
            self.rd_img.set_data(log_map)
            self.rd_img.set_cmap(self.color_map.get())
            
            # Update colorbar limits
            vmin = np.percentile(log_map, 5)
            vmax = np.percentile(log_map, 95)
            self.rd_img.set_clim(vmin, vmax)
            
            # Draw targets if enabled
            if self.show_targets.get():
                # Clear previous target markers
                for artist in self.rd_ax.lines + self.rd_ax.texts:
                    if hasattr(artist, 'is_target_marker') and artist.is_target_marker:
                        artist.remove()
                
                # Add new target markers
                for target in targets:
                    x = target.velocity
                    y = target.range
                    self.rd_ax.plot(x, y, 'wo', markersize=8, 
                                   markeredgecolor='red', markeredgewidth=2,
                                   is_target_marker=True)
                    self.rd_ax.text(x, y-100, str(target.id), color='white',
                                   ha='center', va='top', fontsize=8,
                                   is_target_marker=True)
            
            # Update detected targets list
            self.detected_targets = targets
            
            # Add to history
            self.target_history.append({
                'time': current_time,
                'targets': targets
            })
            
            # Limit history size
            if len(self.target_history) > 1000:
                self.target_history = self.target_history[-1000:]
            
            # Update canvas
            self.rd_canvas.draw_idle()
            
        except Exception as e:
            logger.error(f"Error updating radar display: {e}")
    
    def update_target_list(self):
        """Update the targets treeview"""
        # Clear existing items
        for item in self.target_tree.get_children():
            self.target_tree.delete(item)
        
        # Add new targets
        for target in self.detected_targets:
            values = (
                target.id,
                f"{target.range:.1f}",
                f"{target.velocity:.1f}",
                f"{target.azimuth:.1f}",
                f"{target.elevation:.1f}",
                f"{target.snr:.1f}"
            )
            self.target_tree.insert('', 'end', values=values)
    
    def update_scopes(self):
        """Update A-scope and spectrum displays"""
        if not self.detected_targets:
            return
        
        try:
            # A-scope (range profile)
            range_profile = np.mean(self.range_doppler_map, axis=1)
            range_axis = np.linspace(0, self.settings.max_range, len(range_profile))
            
            self.scope_line.set_data(range_axis, 10 * np.log10(range_profile + 1))
            self.scope_ax.relim()
            self.scope_ax.autoscale_view(scalex=False)
            self.scope_canvas.draw_idle()
            
            # Doppler spectrum at selected range
            range_bin = int(self.spec_range.get())
            if range_bin < self.range_doppler_map.shape[0]:
                spectrum = self.range_doppler_map[range_bin, :]
                vel_axis = np.linspace(-self.settings.max_velocity, 
                                       self.settings.max_velocity, 
                                       len(spectrum))
                
                self.spec_line.set_data(vel_axis, 10 * np.log10(spectrum + 1))
                self.spec_ax.relim()
                self.spec_ax.autoscale_view(scalex=False)
                self.spec_canvas.draw_idle()
            
            # PPI scope
            if self.detected_targets:
                theta = [np.radians(90 - t.azimuth) for t in self.detected_targets]
                r = [t.range for t in self.detected_targets]
                colors = [t.snr for t in self.detected_targets]
                
                self.ppi_scatter.set_offsets(np.c_[theta, r])
                self.ppi_scatter.set_array(np.array(colors))
                self.ppi_scatter.set_clim(0, 40)
                self.ppi_canvas.draw_idle()
            
        except Exception as e:
            logger.error(f"Error updating scopes: {e}")
    
    def update_status_bar(self):
        """Update status bar information"""
        # Time
        self.time_label.config(text=time.strftime("%H:%M:%S"))
        
        # Targets count
        self.targets_label.config(text=f"Targets: {len(self.detected_targets)}")
        
        # Data rate
        if self.usb_interface.is_open:
            stats = self.usb_interface.get_stats()
            data_rate = stats['data_rate']
            self.data_label.config(text=f"Data: {data_rate:.1f} KB/s")
        else:
            self.data_label.config(text="Data: 0 KB/s")
        
        # FPS
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        
        # Status text
        if self.running:
            status = f"Status: {'Recording' if self.recording else 'Running'} - Frame {self.frame_count}"
        else:
            status = "Status: Demo Mode - Stopped"
        
        if self.recording:
            status += f" - Recording to {self.record_dir.get()}"
        
        self.status_label.config(text=status)
    
    # ============================================================================
    # BACKGROUND THREADS
    # ============================================================================
    
    def start_background_threads(self):
        """Start background processing threads"""
        self.running = False
        self.background_thread = threading.Thread(target=self.background_worker, 
                                                  daemon=True)
        self.background_thread.start()
    
    def background_worker(self):
        """Background worker thread for data processing"""
        while True:
            if self.running and not self.demo_mode.get():
                # Hardware mode - read from USB
                try:
                    data = self.usb_interface.read_data(4096)
                    if data:
                        # Parse and process data
                        processed = self.process_usb_data(data)
                        if processed:
                            self.radar_data_queue.put(processed, timeout=1)
                except Exception as e:
                    logger.error(f"Background worker error: {e}")
            else:
                # Demo mode - generate data at lower rate
                time.sleep(0.05)  # 20 Hz
    
    def process_usb_data(self, data):
        """Process raw USB data into radar frames"""
        # This would parse actual USB packets
        # For demo, return None
        return None
    
    # ============================================================================
    # COMMAND HANDLERS
    # ============================================================================
    
    def refresh_devices(self):
        """Refresh device list"""
        devices = self.usb_interface.list_devices()
        device_names = [d['description'] for d in devices]
        self.device_combo['values'] = device_names
        if device_names:
            self.device_combo.current(0)
        logger.info(f"Found {len(devices)} devices")
    
    def connect_device(self):
        """Connect to selected device"""
        if self.demo_mode.get():
            device_name = self.device_combo.get()
            if self.usb_interface.open_device({'description': device_name}):
                messagebox.showinfo("Success", f"Connected to {device_name} in demo mode")
                self.status_label.config(text=f"Status: Connected - {device_name}")
            else:
                messagebox.showerror("Error", "Failed to connect to device")
        else:
            messagebox.showinfo("Info", "Hardware mode not available in demo version")
    
    def connect_hardware(self):
        """Switch to hardware mode"""
        self.demo_mode.set(False)
        messagebox.showinfo("Info", "Hardware mode requires actual FT601 device")
    
    def enable_demo_mode(self):
        """Enable demo mode"""
        self.demo_mode.set(True)
        self.status_label.config(text="Status: Demo Mode")
    
    def start_radar(self):
        """Start radar operation"""
        self.running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="Status: Running")
        logger.info("Radar started")
    
    def stop_radar(self):
        """Stop radar operation"""
        self.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Status: Stopped")
        logger.info("Radar stopped")
    
    def start_recording(self):
        """Start recording data"""
        if not self.running:
            messagebox.showwarning("Warning", "Start radar first")
            return
        
        self.recording = True
        logger.info(f"Recording started to {self.record_dir.get()}")
    
    def stop_recording(self):
        """Stop recording data"""
        self.recording = False
        logger.info("Recording stopped")
    
    def apply_settings(self):
        """Apply radar settings"""
        try:
            self.settings.system_frequency = self.settings_vars['freq'].get() * 1e9
            self.settings.chirp_duration_long = self.settings_vars['long_dur'].get() * 1e-6
            self.settings.chirp_duration_short = self.settings_vars['short_dur'].get() * 1e-6
            self.settings.chirps_per_frame = int(self.settings_vars['chirps'].get())
            self.settings.range_bins = int(self.settings_vars['range_bins'].get())
            self.settings.doppler_bins = int(self.settings_vars['doppler_bins'].get())
            self.settings.prf = self.settings_vars['prf'].get()
            self.settings.max_range = self.settings_vars['max_range'].get()
            self.settings.max_velocity = self.settings_vars['max_vel'].get()
            
            messagebox.showinfo("Success", "Settings applied")
            logger.info("Settings updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid settings: {e}")
    
    def load_config(self):
        """Load configuration from file"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                # Apply config...
                messagebox.showinfo("Success", f"Loaded configuration from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config = {
                    'settings': {k: v.get() for k, v in self.settings_vars.items()},
                    'display': {
                        'color_map': self.color_map.get(),
                        'show_grid': self.show_grid.get(),
                        'show_targets': self.show_targets.get(),
                        'auto_scan': self.auto_scan.get()
                    }
                }
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Success", f"Saved configuration to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def export_data(self):
        """Export radar data"""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            title="Export Data",
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )
        if filename:
            try:
                np.savez(filename, 
                         map=self.range_doppler_map,
                         targets=[(t.range, t.velocity, t.azimuth, t.snr) 
                                 for t in self.detected_targets])
                messagebox.showinfo("Success", f"Exported data to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def clear_history(self):
        """Clear target history"""
        self.target_history = []
        self.hist_ax1.clear()
        self.hist_ax2.clear()
        self.hist_canvas.draw()
    
    def browse_directory(self):
        """Browse for directory"""
        from tkinter import filedialog
        directory = filedialog.askdirectory(title="Select Recording Directory")
        if directory:
            self.record_dir.set(directory)
    
    def calibration_wizard(self):
        """Open calibration wizard"""
        messagebox.showinfo("Calibration", "Calibration wizard not available in demo mode")
    
    def beam_analysis(self):
        """Open beam pattern analysis"""
        messagebox.showinfo("Beam Analysis", "Beam analysis not available in demo mode")
    
    def noise_measurement(self):
        """Measure noise floor"""
        messagebox.showinfo("Noise Measurement", "Noise measurement not available in demo mode")
    
    def show_diagnostics(self):
        """Show system diagnostics"""
        import platform
        info = f"""
        System Diagnostics
        =================
        
        Platform: {platform.platform()}
        Python: {platform.python_version()}
        
        Radar System
        ------------
        Mode: {'Demo' if self.demo_mode.get() else 'Hardware'}
        Status: {'Running' if self.running else 'Stopped'}
        Frames: {self.frame_count}
        Targets: {len(self.detected_targets)}
        
        USB Interface
        -------------
        Connected: {self.usb_interface.is_open}
        Packets: {self.usb_interface.packet_count}
        Bytes: {self.usb_interface.byte_count}
        Data Rate: {self.usb_interface.data_rate:.1f} KB/s
        
        Display
        -------
        FPS: {self.fps:.1f}
        Update Rate: {self.update_rate.get()} Hz
        Color Map: {self.color_map.get()}
        """
        
        messagebox.showinfo("Diagnostics", info)
    
    def show_docs(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", 
                           "Radar System GUI Documentation\n\n"
                           "Version 1.0\n\n"
                           "For more information, visit:\n"
                           "https://github.com/radar-system/docs")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        Advanced Radar System GUI
        Version 1.0.0
        
        A comprehensive radar control and visualization
        interface with FT601 USB 3.0 support.
        
        Features:
        • Real-time Range-Doppler display
        • Target detection and tracking
        • A-scope and PPI displays
        • Doppler spectrum analysis
        • Data recording and playback
        
        © 2025 Radar Systems Inc.
        """
        messagebox.showinfo("About", about_text)
    
    def reset_system(self):
        """Reset system"""
        if messagebox.askyesno("Confirm Reset", "Reset all system parameters?"):
            self.frame_count = 0
            self.detected_targets = []
            self.target_history = []
            self.status.uptime = 0
            logger.info("System reset")
    
    def factory_reset(self):
        """Factory reset"""
        if messagebox.askyesno("Confirm Factory Reset", 
                               "This will reset ALL settings to defaults. Continue?"):
            # Reset to defaults
            for key, var in self.settings_vars.items():
                var.set(self.get_default_value(key))
            self.color_map.set('hot')
            self.show_grid.set(True)
            self.show_targets.set(True)
            self.auto_scan.set(True)
            logger.info("Factory reset performed")
    
    def get_default_value(self, key):
        """Get default value for settings key"""
        defaults = {
            'freq': 10.0,
            'long_dur': 30.0,
            'short_dur': 0.5,
            'chirps': 32,
            'range_bins': 1024,
            'doppler_bins': 32,
            'prf': 1000,
            'max_range': 5000,
            'max_vel': 100,
            'beam_width': 3.0
        }
        return defaults.get(key, 0)
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.animation_running = False
            self.running = False
            self.usb_interface.close()
            self.root.destroy()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    try:
        # Create root window
        root = tk.Tk()
        
        # Set application icon if available
        try:
            root.iconbitmap(default='radar.ico')
        except:
            pass
        
        # Create application
        app = RadarGUI(root)
        
        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{e}")

if __name__ == "__main__":
    main()
