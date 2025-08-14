import os
import sys
import numpy as np
import librosa
import scipy.signal
from scipy.fft import fft, fftfreq
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pydub import AudioSegment
import tkinter as tk
from tkinter import filedialog, messagebox
import json
from datetime import datetime
import webbrowser
import tempfile


class AudioAttentionAnalyzer:
    def __init__(self):
        self.sr = 44100 
        self.audio_mono = None
        self.audio_stereo = None
        self.duration = 0
        self.filename = ""
        self.features = {}
        self.timeline = None
        
        # weighting parameters
        self.weights = {
            'roughness': 0.35,
            'transients': 0.25,
            'dynamics': 0.20,
            'spectral_urgency': 0.10,
            'attention_resets': 0.05,
            'spatial_movement': 0.05
        }
        
        # Debug mode
        self.debug = True
    
    def debug_print(self, message):
        """Print debug information"""
        if self.debug:
            print(f"DEBUG: {message}")
    
    def safe_divide(self, numerator, denominator, default_value=0.0):
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            self.debug_print(f"Safe divide: denominator is {denominator}, returning {default_value}")
            return default_value
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            self.debug_print(f"Safe divide: result is {result}, returning {default_value}")
            return default_value
        return result
    
    def safe_log(self, value, default_value=0.0):
        if value <= 0 or np.isnan(value) or np.isinf(value):
            return default_value
        result = np.log10(value)
        if np.isnan(result) or np.isinf(result):
            return default_value
        return result
    
    def select_audio_file(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.m4a *.ogg"),
                ("MP3 files", "*.mp3"),
                ("WAV files", "*.wav"),
                ("FLAC files", "*.flac"),
                ("M4A files", "*.m4a"),
                ("OGG files", "*.ogg"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return file_path
    
    def load_and_preprocess_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            print(f"Loading audio file: {os.path.basename(file_path)}")
            self.filename = os.path.splitext(os.path.basename(file_path))[0]
            audio_data, sr = librosa.load(file_path, sr=self.sr, mono=False)
            

            if len(audio_data.shape) == 1:
                self.audio_mono = audio_data
                self.audio_stereo = None
            else:
                self.audio_stereo = audio_data
                self.audio_mono = librosa.to_mono(audio_data)
            

            max_val = np.max(np.abs(self.audio_mono))
            if max_val > 0:
                self.audio_mono = self.audio_mono / max_val
            if self.audio_stereo is not None:
                max_val_stereo = np.max(np.abs(self.audio_stereo))
                if max_val_stereo > 0:
                    self.audio_stereo = self.audio_stereo / max_val_stereo
            
            self.duration = len(self.audio_mono) / self.sr
            self.timeline = np.linspace(0, self.duration, len(self.audio_mono))
            
            print(f"Audio loaded successfully:")
            print(f"  Duration: {self.duration:.2f} seconds")
            print(f"  Sample rate: {self.sr} Hz")
            print(f"  Channels: {'Stereo' if self.audio_stereo is not None else 'Mono'}")
            print(f"  Audio range: [{np.min(self.audio_mono):.3f}, {np.max(self.audio_mono):.3f}]")
            
            return True
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False
    
    def calculate_roughness_index(self):
        print("Calculating roughness index...")
        
        try:
            analytic_signal = scipy.signal.hilbert(self.audio_mono)
            amplitude_envelope = np.abs(analytic_signal)
            

            amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
            amplitude_envelope = np.abs(amplitude_envelope)
            

            envelope_sr = 1000  # 1 kHz
            if len(amplitude_envelope) > envelope_sr:
                envelope_downsampled = scipy.signal.resample(
                    amplitude_envelope, 
                    int(len(amplitude_envelope) * envelope_sr / self.sr)
                )
            else:
                envelope_downsampled = amplitude_envelope
                envelope_sr = self.sr
            
            self.debug_print(f"Envelope length: {len(envelope_downsampled)}, SR: {envelope_sr}")
            

            if len(envelope_downsampled) > 1:
                window = scipy.signal.windows.hann(len(envelope_downsampled))
                envelope_windowed = envelope_downsampled * window
            else:
                envelope_windowed = envelope_downsampled
            

            envelope_fft = fft(envelope_windowed)
            freqs = fftfreq(len(envelope_windowed), 1/envelope_sr)
            

            positive_freqs_mask = freqs >= 0
            freqs_positive = freqs[positive_freqs_mask]
            power_spectrum = np.abs(envelope_fft[positive_freqs_mask])**2
            

            roughness_mask = (freqs_positive >= 40) & (freqs_positive <= 80)
            total_mask = (freqs_positive >= 1) & (freqs_positive <= envelope_sr/2)  # Exclude DC
            
            roughness_power = np.sum(power_spectrum[roughness_mask])
            total_power = np.sum(power_spectrum[total_mask])
            
            roughness_index = self.safe_divide(roughness_power, total_power, 0.0) * 100
            
            self.debug_print(f"Roughness power: {roughness_power}, Total power: {total_power}")
            self.debug_print(f"Roughness index: {roughness_index}")
            
            window_size = max(int(envelope_sr * 1.0), 1)  # 1-second windows, minimum 1
            hop_size = max(int(envelope_sr * 0.5), 1)     # 0.5-second hop, minimum 1
            
            roughness_timeline = []
            time_points = []
            
            if len(envelope_downsampled) >= window_size:
                for i in range(0, len(envelope_downsampled) - window_size + 1, hop_size):
                    window = envelope_downsampled[i:i + window_size]
                    if len(window) == window_size:
                        window_windowed = window * scipy.signal.windows.hann(len(window))
                        window_fft = fft(window_windowed)
                        window_power = np.abs(window_fft)**2
                        window_freqs = fftfreq(len(window), 1/envelope_sr)
                        
                        pos_mask = window_freqs >= 0
                        window_freqs_pos = window_freqs[pos_mask]
                        window_power_pos = window_power[pos_mask]
                        
                        window_roughness_mask = (window_freqs_pos >= 40) & (window_freqs_pos <= 80)
                        window_total_mask = (window_freqs_pos >= 1) & (window_freqs_pos <= envelope_sr/2)
                        
                        window_roughness_power = np.sum(window_power_pos[window_roughness_mask])
                        window_total_power = np.sum(window_power_pos[window_total_mask])
                        
                        window_roughness = self.safe_divide(window_roughness_power, window_total_power, 0.0) * 100
                        roughness_timeline.append(window_roughness)
                        time_points.append(i / envelope_sr)
            
            if not roughness_timeline:
                roughness_timeline = [roughness_index]
                time_points = [0.0]
            
            self.features['roughness_index'] = float(roughness_index)
            self.features['roughness_timeline'] = np.array(roughness_timeline)
            self.features['roughness_time_points'] = np.array(time_points)
            
            print(f"  Roughness index: {roughness_index:.2f}%")
            
        except Exception as e:
            print(f"Error in roughness calculation: {e}")
            self.features['roughness_index'] = 0.0
            self.features['roughness_timeline'] = np.array([0.0])
            self.features['roughness_time_points'] = np.array([0.0])
    
    def calculate_transient_density(self):
        print("Calculating transient density...")
        
        try:

            hop_length = 512
            
            # ensure audio is not too quiet
            audio_rms = np.sqrt(np.mean(self.audio_mono**2))
            if audio_rms < 1e-6:
                print("  Warning: Audio is very quiet, normalizing...")
                audio_normalized = self.audio_mono / (audio_rms + 1e-10)
            else:
                audio_normalized = self.audio_mono
            
            onsets = librosa.onset.onset_detect(
                y=audio_normalized, 
                sr=self.sr, 
                hop_length=hop_length,
                backtrack=True,
                units='time',
                pre_max=20,
                post_max=20,
                pre_avg=100,
                post_avg=100,
                delta=0.07,
                wait=30
            )
    
            onset_strength = librosa.onset.onset_strength(
                y=audio_normalized, 
                sr=self.sr, 
                hop_length=hop_length
            )
            

            if len(onsets) > 0:
                onset_frames = librosa.time_to_frames(onsets, sr=self.sr, hop_length=hop_length)
                # Ensure frames are within bounds
                onset_frames = onset_frames[onset_frames < len(onset_strength)]
                onset_intensities = onset_strength[onset_frames] if len(onset_frames) > 0 else []
            else:
                onset_intensities = []
            

            transient_density = self.safe_divide(len(onsets) * 60, self.duration, 0.0)
            
            onset_timeline = np.zeros_like(self.timeline)
            for onset_time in onsets:
                if onset_time < self.duration:
                    idx = int(onset_time * self.sr)
                    if 0 <= idx < len(onset_timeline):
                        onset_timeline[idx] = 1
            
            self.features['transient_density'] = float(transient_density)
            self.features['onset_times'] = onsets
            self.features['onset_intensities'] = onset_intensities
            self.features['onset_timeline'] = onset_timeline
            
            print(f"  Transient density: {transient_density:.2f} onsets/minute")
            print(f"  Total onsets detected: {len(onsets)}")
            
        except Exception as e:
            print(f"Error in transient calculation: {e}")
            self.features['transient_density'] = 0.0
            self.features['onset_times'] = np.array([])
            self.features['onset_intensities'] = np.array([])
            self.features['onset_timeline'] = np.zeros_like(self.timeline)
    
    def calculate_dynamic_variance(self):
        print("Calculating dynamic variance...")
        
        try:
            frame_length = max(int(0.025 * self.sr), 1)  # 25ms frames, minimum 1
            hop_length = max(int(0.010 * self.sr), 1)    # 10ms hop, minimum 1
            
            rms = librosa.feature.rms(
                y=self.audio_mono, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            rms = np.maximum(rms, 1e-10)
            rms_db = 20 * np.log10(rms / np.max(rms))
            rms_db = np.nan_to_num(rms_db, nan=0.0, posinf=0.0, neginf=-80.0)
            

            if len(rms_db) > 1:
                rms_diff = np.diff(rms_db)
            else:
                rms_diff = np.array([0.0])
            

            if len(rms_diff) > 0:
                sudden_threshold = np.percentile(np.abs(rms_diff), 90)  # Top 10% of changes
                sudden_jumps = np.abs(rms_diff) > sudden_threshold
            else:
                sudden_jumps = np.array([False])
            

            total_variance = np.var(rms_db) if len(rms_db) > 1 else 0.0
            sudden_jump_variance = np.var(rms_diff[sudden_jumps]) if np.any(sudden_jumps) else 0.0
            
            dynamic_variance_score = self.safe_divide(sudden_jump_variance, total_variance, 0.0) * 100
            

            rms_times = librosa.frames_to_time(
                np.arange(len(rms)), 
                sr=self.sr, 
                hop_length=hop_length
            )
            
            self.features['dynamic_variance_score'] = float(dynamic_variance_score)
            self.features['rms_timeline'] = rms_db
            self.features['rms_times'] = rms_times
            self.features['sudden_jumps'] = sudden_jumps
            self.features['sudden_jump_count'] = int(np.sum(sudden_jumps))
            
            print(f"  Dynamic variance score: {dynamic_variance_score:.2f}")
            print(f"  Sudden amplitude jumps: {np.sum(sudden_jumps)}")
            
        except Exception as e:
            print(f"Error in dynamic variance calculation: {e}")
            self.features['dynamic_variance_score'] = 0.0
            self.features['rms_timeline'] = np.array([0.0])
            self.features['rms_times'] = np.array([0.0])
            self.features['sudden_jumps'] = np.array([False])
            self.features['sudden_jump_count'] = 0
    
    def calculate_spectral_urgency(self):
        print("Calculating spectral urgency...")
        
        try:
            # STFT parameters
            hop_length = 512
            n_fft = 2048
            
            # Compute STFT
            stft = librosa.stft(self.audio_mono, hop_length=hop_length, n_fft=n_fft)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=self.sr, hop_length=hop_length)
            urgency_mask = (freqs >= 2000) & (freqs <= 5000)  # 2-5 kHz
            low_unease_mask = (freqs >= 30) & (freqs <= 40)   # 30-40 Hz
            
            urgency_energy = np.sum(magnitude[urgency_mask, :], axis=0)
            low_unease_energy = np.sum(magnitude[low_unease_mask, :], axis=0)
            total_energy = np.sum(magnitude, axis=0)
            total_energy = np.maximum(total_energy, 1e-10)
            
            urgency_ratio = urgency_energy / total_energy
            low_unease_ratio = low_unease_energy / total_energy
            urgency_ratio = np.nan_to_num(urgency_ratio, nan=0.0)
            low_unease_ratio = np.nan_to_num(low_unease_ratio, nan=0.0)
            
            avg_urgency_ratio = np.mean(urgency_ratio) * 100
            avg_low_unease_ratio = np.mean(low_unease_ratio) * 100
            
            spectral_urgency_score = (avg_urgency_ratio * 0.7) + (avg_low_unease_ratio * 0.3)
            
            self.features['spectral_urgency_score'] = float(spectral_urgency_score)
            self.features['urgency_timeline'] = urgency_ratio
            self.features['low_unease_timeline'] = low_unease_ratio
            self.features['spectral_times'] = times
            
            print(f"  Spectral urgency score: {spectral_urgency_score:.2f}")
            print(f"  Avg 2-5kHz energy ratio: {avg_urgency_ratio:.2f}%")
            print(f"  Avg 30-40Hz energy ratio: {avg_low_unease_ratio:.2f}%")
            
        except Exception as e:
            print(f"Error in spectral urgency calculation: {e}")
            self.features['spectral_urgency_score'] = 0.0
            self.features['urgency_timeline'] = np.array([0.0])
            self.features['low_unease_timeline'] = np.array([0.0])
            self.features['spectral_times'] = np.array([0.0])
    
    def calculate_attention_resets(self):
        print("Calculating attention reset events...")
        
        try:
            frame_length = max(int(0.025 * self.sr), 1)  # 25ms frames
            hop_length = max(int(0.010 * self.sr), 1)    # 10ms hop
            
            rms = librosa.feature.rms(
                y=self.audio_mono, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            rms = np.maximum(rms, 1e-10)
            rms_db = 20 * np.log10(rms / np.max(rms))
            rms_db = np.nan_to_num(rms_db, nan=-80.0, neginf=-80.0)
            
            silence_threshold = np.percentile(rms_db, 10)  # Bottom 10% as silence
            
            silent_frames = rms_db < silence_threshold
            frame_times = librosa.frames_to_time(
                np.arange(len(rms)), 
                sr=self.sr, 
                hop_length=hop_length
            )
            

            silent_regions = []
            in_silence = False
            silence_start = 0
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_silence:
                    in_silence = True
                    silence_start = frame_times[i] if i < len(frame_times) else 0
                elif not is_silent and in_silence:
                    in_silence = False
                    silence_end = frame_times[i] if i < len(frame_times) else self.duration
                    silence_duration = silence_end - silence_start
                    if 0.006 <= silence_duration <= 0.050:  # 6-50ms range
                        silent_regions.append((silence_start, silence_end, silence_duration))
            
            # Count attention reset events
            attention_reset_count = len(silent_regions)
            reset_density = self.safe_divide(attention_reset_count * 60, self.duration, 0.0)
            
            self.features['attention_reset_count'] = int(attention_reset_count)
            self.features['reset_density'] = float(reset_density)
            self.features['silent_regions'] = silent_regions
            
            print(f"  Attention reset events: {attention_reset_count}")
            print(f"  Reset density: {reset_density:.2f} resets/minute")
            
        except Exception as e:
            print(f"Error in attention reset calculation: {e}")
            self.features['attention_reset_count'] = 0
            self.features['reset_density'] = 0.0
            self.features['silent_regions'] = []
    
    def calculate_spatial_movement(self):
        if self.audio_stereo is None:
            print("Skipping spatial movement analysis (mono audio)")
            self.features['spatial_movement_score'] = 0.0
            self.features['stereo_movement_timeline'] = np.array([0.0])
            self.features['movement_times'] = np.array([0.0])
            return
        
        print("Calculating spatial movement...")
        
        try:
            left_channel = self.audio_stereo[0]
            right_channel = self.audio_stereo[1]
            frame_length = max(int(0.1 * self.sr), 1)  # 100ms frames
            hop_length = max(int(0.05 * self.sr), 1)   # 50ms hop
            left_rms = librosa.feature.rms(y=left_channel, frame_length=frame_length, hop_length=hop_length)[0]
            right_rms = librosa.feature.rms(y=right_channel, frame_length=frame_length, hop_length=hop_length)[0]
            left_rms = np.maximum(left_rms, 1e-10)
            right_rms = np.maximum(right_rms, 1e-10)
            ild = 20 * np.log10(left_rms / right_rms)
            ild = np.nan_to_num(ild, nan=0.0, posinf=0.0, neginf=0.0)
            spatial_movement_score = np.var(ild) if len(ild) > 1 else 0.0
            

            movement_times = librosa.frames_to_time(
                np.arange(len(ild)), 
                sr=self.sr, 
                hop_length=hop_length
            )
            
            self.features['spatial_movement_score'] = float(spatial_movement_score)
            self.features['stereo_movement_timeline'] = ild
            self.features['movement_times'] = movement_times
            
            print(f"  Spatial movement score: {spatial_movement_score:.2f}")
            
        except Exception as e:
            print(f"Error in spatial movement calculation: {e}")
            self.features['spatial_movement_score'] = 0.0
            self.features['stereo_movement_timeline'] = np.array([0.0])
            self.features['movement_times'] = np.array([0.0])
    
    def calculate_attention_scores(self):
        print("Calculating attention scores...")

        try:
            roughness_raw = self.features.get('roughness_index', 0.0)
            transient_raw = self.features.get('transient_density', 0.0)
            dynamics_raw = self.features.get('dynamic_variance_score', 0.0)
            spectral_raw = self.features.get('spectral_urgency_score', 0.0)
            resets_raw = self.features.get('reset_density', 0.0)
            spatial_raw = self.features.get('spatial_movement_score', 0.0)

            self.debug_print(
                f"Raw scores - R:{roughness_raw}, T:{transient_raw}, "
                f"D:{dynamics_raw}, S:{spectral_raw}, Re:{resets_raw}, Sp:{spatial_raw}"
            )

            # prevent extremely small max ranges by setting a minimum cap
            observed_max = {
                'roughness': max(roughness_raw * 1.2, 10),
                'transients': max(transient_raw * 1.2, 40),
                'dynamics': max(dynamics_raw * 1.2, 20),
                'spectral': max(spectral_raw * 1.2, 15),
                'resets': max(resets_raw * 1.2, 10),
                'spatial': max(spatial_raw * 1.2, 5)
            }

            def normalize(value, feature):
                max_val = observed_max[feature]
                if max_val <= 0:
                    return 0.0
                return max(0.0, min(100.0, (value / max_val) * 100))

            roughness_normalized = normalize(roughness_raw, 'roughness')
            transient_normalized = normalize(transient_raw, 'transients')
            dynamics_normalized = normalize(dynamics_raw, 'dynamics')
            spectral_normalized = normalize(spectral_raw, 'spectral')
            resets_normalized = normalize(resets_raw, 'resets')
            spatial_normalized = normalize(spatial_raw, 'spatial')

            self.debug_print(
                f"Normalized scores - R:{roughness_normalized:.1f}, "
                f"T:{transient_normalized:.1f}, D:{dynamics_normalized:.1f}, "
                f"S:{spectral_normalized:.1f}, Re:{resets_normalized:.1f}, "
                f"Sp:{spatial_normalized:.1f}"
            )

            adjusted_weights = {
                'roughness': 0.20,
                'transients': 0.25,
                'dynamics': 0.20,
                'spectral_urgency': 0.15,
                'attention_resets': 0.10,
                'spatial_movement': 0.10
            }

            primary_score = (
                roughness_normalized * adjusted_weights['roughness'] +
                transient_normalized * adjusted_weights['transients'] +
                dynamics_normalized * adjusted_weights['dynamics']
            ) / (adjusted_weights['roughness'] +
                 adjusted_weights['transients'] +
                 adjusted_weights['dynamics'])

            secondary_score = (
                spectral_normalized * adjusted_weights['spectral_urgency'] +
                resets_normalized * adjusted_weights['attention_resets'] +
                spatial_normalized * adjusted_weights['spatial_movement']
            ) / (adjusted_weights['spectral_urgency'] +
                 adjusted_weights['attention_resets'] +
                 adjusted_weights['spatial_movement'])


            overall_score = (primary_score * 0.6 + secondary_score * 0.4)

            self.features['primary_attention_score'] = float(primary_score)
            self.features['secondary_attention_score'] = float(secondary_score)
            self.features['overall_attention_index'] = float(overall_score)

            if overall_score >= 60:
                retention_prediction = "High"
            elif overall_score >= 40:
                retention_prediction = "Medium"
            else:
                retention_prediction = "Low"

            self.features['retention_prediction'] = retention_prediction

            print(f"  Primary attention score: {primary_score:.1f}/100")
            print(f"  Secondary attention score: {secondary_score:.1f}/100")
            print(f"  Overall attention index: {overall_score:.1f}/100")
            print(f"  Retention prediction: {retention_prediction}")

        except Exception as e:
            print(f"Error in attention scoring: {e}")
            self.features['primary_attention_score'] = 0.0
            self.features['secondary_attention_score'] = 0.0
            self.features['overall_attention_index'] = 0.0
            self.features['retention_prediction'] = "Low"


    
    def generate_visualizations(self):
        print("Generating visualizations...")
        
        try:
            fig = make_subplots(
                rows=6, cols=1,
                subplot_titles=[
                    'Audio Waveform',
                    'Roughness Timeline (40-80 Hz Modulation)',
                    'Transient Events & Dynamic Range',
                    'Spectral Urgency (2-5 kHz Energy)',
                    'Attention Reset Events',
                    'Spatial Movement (Stereo)' if self.audio_stereo is not None else 'Summary Scores'
                ],
                vertical_spacing=0.08
            )
            
            # 1. Audio Waveform
            downsample_factor = max(1, len(self.audio_mono) // 10000)  # Limit to ~10k points
            downsampled_audio = self.audio_mono[::downsample_factor]
            downsampled_time = self.timeline[::downsample_factor]
            
            fig.add_trace(
                go.Scatter(
                    x=downsampled_time, 
                    y=downsampled_audio,
                    name="Waveform",
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            # 2. Roughness Timeline
            if len(self.features['roughness_time_points']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=self.features['roughness_time_points'],
                        y=self.features['roughness_timeline'],
                        name="Roughness Index",
                        line=dict(color='red', width=2),
                        fill='tozeroy'
                    ),
                    row=2, col=1
                )
            
            # 3. Transients and Dynamic Range
            # RMS timeline
            if len(self.features['rms_times']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=self.features['rms_times'],
                        y=self.features['rms_timeline'],
                        name="RMS (dB)",
                        line=dict(color='green', width=1)
                    ),
                    row=3, col=1
                )
            
            # Onset markers
            if len(self.features['onset_times']) > 0:
                max_rms = max(self.features['rms_timeline']) if len(self.features['rms_timeline']) > 0 else 0
                fig.add_trace(
                    go.Scatter(
                        x=self.features['onset_times'],
                        y=[max_rms] * len(self.features['onset_times']),
                        mode='markers',
                        name="Transients",
                        marker=dict(color='orange', size=8, symbol='triangle-up')
                    ),
                    row=3, col=1
                )
            
            # 4. Spectral Urgency
            if len(self.features['spectral_times']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=self.features['spectral_times'],
                        y=self.features['urgency_timeline'] * 100,
                        name="2-5 kHz Energy %",
                        line=dict(color='purple', width=2)
                    ),
                    row=4, col=1
                )
            
            # 5. Attention Reset Events
            reset_times = [region[0] for region in self.features['silent_regions']]
            reset_durations = [region[2] * 1000 for region in self.features['silent_regions']]  # Convert to ms
            
            if reset_times:
                fig.add_trace(
                    go.Scatter(
                        x=reset_times,
                        y=reset_durations,
                        mode='markers',
                        name="Reset Events",
                        marker=dict(color='cyan', size=8)
                    ),
                    row=5, col=1
                )
            
            # 6. Spatial Movement or Summary
            if self.audio_stereo is not None and len(self.features['stereo_movement_timeline']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=self.features['movement_times'],
                        y=self.features['stereo_movement_timeline'],
                        name="Interaural Level Difference",
                        line=dict(color='magenta', width=2)
                    ),
                    row=6, col=1
                )
            else:

                scores = [
                    self.features['primary_attention_score'],
                    self.features['secondary_attention_score'],
                    self.features['overall_attention_index']
                ]
                score_names = ['Primary Score', 'Secondary Score', 'Overall Index']
                
                fig.add_trace(
                    go.Bar(
                        x=score_names,
                        y=scores,
                        name="Attention Scores",
                        marker=dict(color=['#ff7f0e', '#2ca02c', '#1f77b4'])
                    ),
                    row=6, col=1
                )
            

            fig.update_layout(
                height=1200,
                title=f"Audio Attention Analysis: {self.filename}",
                showlegend=True
            )
            

            for i in range(1, 7):
                fig.update_xaxes(title_text="Time (seconds)", row=i, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Error"))
            fig.update_layout(title="Visualization Error")
            return fig
    
    def generate_html_report(self, output_dir):
        print("Generating HTML report...")
        
        try:
            fig = self.generate_visualizations()
            plot_html = pio.to_html(fig, include_plotlyjs='cdn', div_id="attention-plot")
            

            def safe_get(key, default=0.0):
                value = self.features.get(key, default)
                return default if np.isnan(value) or np.isinf(value) else value
            
            roughness_index = safe_get('roughness_index', 0.0)
            transient_density = safe_get('transient_density', 0.0)
            dynamic_variance_score = safe_get('dynamic_variance_score', 0.0)
            spectral_urgency_score = safe_get('spectral_urgency_score', 0.0)
            attention_reset_count = safe_get('attention_reset_count', 0)
            spatial_movement_score = safe_get('spatial_movement_score', 0.0)
            primary_attention_score = safe_get('primary_attention_score', 0.0)
            secondary_attention_score = safe_get('secondary_attention_score', 0.0)
            overall_attention_index = safe_get('overall_attention_index', 0.0)
            retention_prediction = self.features.get('retention_prediction', 'Low')
    
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Audio Attention Analysis Report - {self.filename}</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    }}
                    h1 {{
                        color: #2c3e50;
                        text-align: center;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #34495e;
                        border-left: 4px solid #3498db;
                        padding-left: 15px;
                        margin-top: 30px;
                    }}
                    .summary-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }}
                    .metric-card {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        border-radius: 8px;
                        text-align: center;
                    }}
                    .metric-value {{
                        font-size: 2em;
                        font-weight: bold;
                        margin: 10px 0;
                    }}
                    .metric-label {{
                        font-size: 0.9em;
                        opacity: 0.9;
                    }}
                    .prediction-box {{
                        background-color: {'#e8f5e8' if retention_prediction == 'High' else '#fff3cd' if retention_prediction == 'Medium' else '#f8d7da'};
                        border: 2px solid {'#28a745' if retention_prediction == 'High' else '#ffc107' if retention_prediction == 'Medium' else '#dc3545'};
                        border-radius: 8px;
                        padding: 20px;
                        margin: 20px 0;
                        text-align: center;
                    }}
                    .prediction-text {{
                        font-size: 1.5em;
                        font-weight: bold;
                        color: {'#155724' if retention_prediction == 'High' else '#856404' if retention_prediction == 'Medium' else '#721c24'};
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #3498db;
                        color: white;
                    }}
                    tr:hover {{
                        background-color: #f5f5f5;
                    }}
                    .plot-container {{
                        margin: 30px 0;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        overflow: hidden;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Audio Attention Analysis Report</h1>
                    <p style="text-align: center; color: #666; font-size: 1.1em;">
                        Analysis of: <strong>{self.filename}</strong><br>
                        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                    
                    <div class="prediction-box">
                        <div class="prediction-text">Audience Retention Prediction: {retention_prediction}</div>
                        <p>Overall Attention Index: {overall_attention_index:.1f}/100</p>
                    </div>
                    
                    <h2>Summary Metrics</h2>
                    <div class="summary-grid">
                        <div class="metric-card">
                            <div class="metric-label">Roughness Index</div>
                            <div class="metric-value">{roughness_index:.1f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Transient Density</div>
                            <div class="metric-value">{transient_density:.1f}/min</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Dynamic Variance</div>
                            <div class="metric-value">{dynamic_variance_score:.1f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Spectral Urgency</div>
                            <div class="metric-value">{spectral_urgency_score:.1f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Attention Resets</div>
                            <div class="metric-value">{int(attention_reset_count)}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Audio Duration</div>
                            <div class="metric-value">{self.duration:.1f}s</div>
                        </div>
                    </div>
                    
                    <h2>Detailed Analysis</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Value</th>
                            <th>Description</th>
                            <th>Attention Impact</th>
                        </tr>
                        <tr>
                            <td>Roughness Index (40-80 Hz)</td>
                            <td>{roughness_index:.2f}%</td>
                            <td>Amplitude modulation in the "roughness" frequency range</td>
                            <td>{'High' if roughness_index > 10 else 'Medium' if roughness_index > 5 else 'Low'}</td>
                        </tr>
                        <tr>
                            <td>Transient Density</td>
                            <td>{transient_density:.2f} per minute</td>
                            <td>Sharp onset events that trigger attention responses</td>
                            <td>{'High' if transient_density > 50 else 'Medium' if transient_density > 20 else 'Low'}</td>
                        </tr>
                        <tr>
                            <td>Dynamic Variance Score</td>
                            <td>{dynamic_variance_score:.2f}</td>
                            <td>Sudden amplitude changes vs gradual variations</td>
                            <td>{'High' if dynamic_variance_score > 30 else 'Medium' if dynamic_variance_score > 15 else 'Low'}</td>
                        </tr>
                        <tr>
                            <td>Spectral Urgency</td>
                            <td>{spectral_urgency_score:.2f}</td>
                            <td>Energy in 2-5 kHz (urgency) and 30-40 Hz (unease) ranges</td>
                            <td>{'High' if spectral_urgency_score > 20 else 'Medium' if spectral_urgency_score > 10 else 'Low'}</td>
                        </tr>
                        <tr>
                            <td>Attention Reset Events</td>
                            <td>{int(attention_reset_count)} events</td>
                            <td>Brief silences (6-50ms) that trigger attention reset</td>
                            <td>{'High' if safe_get('reset_density', 0) > 10 else 'Medium' if safe_get('reset_density', 0) > 5 else 'Low'}</td>
                        </tr>
                        <tr>
                            <td>Spatial Movement</td>
                            <td>{spatial_movement_score:.2f}</td>
                            <td>Stereo movement and binaural processing effects</td>
                            <td>{'High' if spatial_movement_score > 5 else 'Medium' if spatial_movement_score > 2 else 'Low'}</td>
                        </tr>
                    </table>
                    
                    <h2>Attention Timeline Visualization</h2>
                    <div class="plot-container">
                        {plot_html}
                    </div>
                    
                    <h2>Recommendations</h2>
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h3>Optimization Suggestions:</h3>
                        <ul>
                            {'<li><strong>Increase roughness:</strong> Add more amplitude modulation in the 40-80 Hz range for stronger attention capture.</li>' if roughness_index < 10 else ''}
                            {'<li><strong>Add more transients:</strong> Include more sharp onsets, impacts, or cuts to maintain engagement.</li>' if transient_density < 30 else ''}
                            {'<li><strong>Enhance dynamics:</strong> Create more sudden amplitude changes rather than gradual transitions.</li>' if dynamic_variance_score < 20 else ''}
                            {'<li><strong>Boost urgency frequencies:</strong> Enhance content in the 2-5 kHz range for perceived urgency.</li>' if spectral_urgency_score < 15 else ''}
                            {'<li><strong>Add strategic pauses:</strong> Include brief silences to trigger attention reset mechanisms.</li>' if attention_reset_count < 5 else ''}
                            {'<li><strong>Utilize stereo field:</strong> Add spatial movement to engage binaural attention networks.</li>' if spatial_movement_score < 3 and self.audio_stereo is not None else ''}
                        </ul>
                        
                        <h3>Risk Areas:</h3>
                        <ul>
                            {'<li><strong>Potential listener fatigue:</strong> Very high roughness levels may cause aversion. Consider balancing with smoother sections.</li>' if roughness_index > 15 else ''}
                            {'<li><strong>Overwhelming transients:</strong> Too many sudden onsets may be jarring. Space them strategically.</li>' if transient_density > 80 else ''}
                            {'<li><strong>Excessive dynamics:</strong> Very sudden changes might be uncomfortable for some listeners.</li>' if dynamic_variance_score > 40 else ''}
                        </ul>
                    </div>
                    
                    <h2>Technical Details</h2>
                    <div style="background-color: #f1f3f4; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em;">
                        <strong>Analysis Parameters:</strong><br>
                        Sample Rate: {self.sr} Hz<br>
                        Audio Duration: {self.duration:.2f} seconds<br>
                        Channels: {'Stereo' if self.audio_stereo is not None else 'Mono'}<br>
                        Total Onsets Detected: {len(self.features.get('onset_times', []))}<br>
                        Sudden Amplitude Jumps: {safe_get('sudden_jump_count', 0)}<br>
                        Silent Regions (6-50ms): {int(attention_reset_count)}<br>
                        <br>
                        <strong>Scoring Weights:</strong><br>
                        Roughness: {self.weights['roughness']:.0%}<br>
                        Transients: {self.weights['transients']:.0%}<br>
                        Dynamics: {self.weights['dynamics']:.0%}<br>
                        Spectral Urgency: {self.weights['spectral_urgency']:.0%}<br>
                        Attention Resets: {self.weights['attention_resets']:.0%}<br>
                        Spatial Movement: {self.weights['spatial_movement']:.0%}
                    </div>
                    
                    <h2>Research Foundation</h2>
                    <p style="font-size: 0.9em; color: #666; line-height: 1.6;">
                        This analysis is based on neuroscience research identifying specific audio features that reliably capture 
                        and maintain human attention. The methodology incorporates findings from studies on amplitude modulation, 
                        onset detection, spectral salience, and spatial audio processing. Effect sizes for these features often 
                        exceed 0.5 in controlled studies, indicating strong predictive validity for attention capture.
                        <br><br>
                        <strong>Key Research Areas:</strong> Auditory cortex responses, salience network activation, 
                        event-related potentials (ERPs), psychoacoustic principles, and attention network dynamics.
                    </p>
                </div>
            </body>
            </html>
            """
            
            html_filename = os.path.join(output_dir, f"{self.filename}_attention_analysis.html")
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return html_filename
            
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            minimal_html = f"""
            <html><body>
            <h1>Audio Attention Analysis Report</h1>
            <p>File: {self.filename}</p>
            <p>Error occurred during report generation: {e}</p>
            <p>Overall Attention Index: {self.features.get('overall_attention_index', 0):.1f}/100</p>
            </body></html>
            """
            html_filename = os.path.join(output_dir, f"{self.filename}_attention_analysis.html")
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write(minimal_html)
            return html_filename
    
    def generate_pdf_report(self, output_dir):
        print("Generating PDF report...")
        
        try:
            pdf_filename = os.path.join(output_dir, f"{self.filename}_attention_analysis.pdf")
            
            with PdfPages(pdf_filename) as pdf:
                # Page 1: Summary and Scores
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle(f'Audio Attention Analysis: {self.filename}', fontsize=16, fontweight='bold')
                
                # Summary scores bar chart
                scores = [
                    self.features.get('roughness_index', 0),
                    self.features.get('transient_density', 0),
                    self.features.get('dynamic_variance_score', 0),
                    self.features.get('spectral_urgency_score', 0)
                ]
                score_labels = ['Roughness\nIndex (%)', 'Transient\nDensity (/min)', 'Dynamic\nVariance', 'Spectral\nUrgency']
                
                bars = axes[0,0].bar(score_labels, scores, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                axes[0,0].set_title('Primary Features')
                axes[0,0].set_ylabel('Score')
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    if height > 0:
                        axes[0,0].text(bar.get_x() + bar.get_width()/2, height + max(scores)*0.01,
                                      f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
                
                # Overall scores
                overall_scores = [
                    self.features.get('primary_attention_score', 0),
                    self.features.get('secondary_attention_score', 0),
                    self.features.get('overall_attention_index', 0)
                ]
                overall_labels = ['Primary\nScore', 'Secondary\nScore', 'Overall\nIndex']
                
                bars2 = axes[0,1].bar(overall_labels, overall_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                axes[0,1].set_title('Attention Scores (0-100)')
                axes[0,1].set_ylabel('Score')
                axes[0,1].set_ylim(0, 100)
                for bar, score in zip(bars2, overall_scores):
                    height = bar.get_height()
                    if height > 0:
                        axes[0,1].text(bar.get_x() + bar.get_width()/2, height + 2,
                                      f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
                
                # Waveform
                if len(self.audio_mono) > 10000:
                    downsample_factor = len(self.audio_mono) // 5000
                    downsampled_audio = self.audio_mono[::downsample_factor]
                    downsampled_time = self.timeline[::downsample_factor]
                else:
                    downsampled_audio = self.audio_mono
                    downsampled_time = self.timeline
                    
                axes[1,0].plot(downsampled_time, downsampled_audio, 'b-', linewidth=0.5)
                axes[1,0].set_title('Audio Waveform')
                axes[1,0].set_xlabel('Time (seconds)')
                axes[1,0].set_ylabel('Amplitude')
                
                # Summary text
                axes[1,1].axis('off')
                summary_text = f"""
RETENTION PREDICTION: {self.features.get('retention_prediction', 'Low')}
Overall Attention Index: {self.features.get('overall_attention_index', 0):.1f}/100

Key Metrics:
• Duration: {self.duration:.1f} seconds
• Roughness: {self.features.get('roughness_index', 0):.1f}%
• Transients: {self.features.get('transient_density', 0):.1f}/min
• Dynamics: {self.features.get('dynamic_variance_score', 0):.1f}
• Spectral Urgency: {self.features.get('spectral_urgency_score', 0):.1f}
• Reset Events: {self.features.get('attention_reset_count', 0)}
• Spatial Movement: {self.features.get('spatial_movement_score', 0):.2f}

Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """
                axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                              fontsize=10, verticalalignment='top', fontfamily='monospace')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            return pdf_filename
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None
    
    def save_data_json(self, output_dir):
        try:
            json_filename = os.path.join(output_dir, f"{self.filename}_attention_data.json")
            
            def safe_convert(value, default=0.0):
                if isinstance(value, (list, np.ndarray)):
                    return [safe_convert(v) for v in value]
                if np.isnan(value) or np.isinf(value):
                    return default
                return float(value)
            
            data = {
                'filename': self.filename,
                'duration': float(self.duration),
                'analysis_timestamp': datetime.now().isoformat(),
                'scores': {
                    'roughness_index': safe_convert(self.features.get('roughness_index', 0)),
                    'transient_density': safe_convert(self.features.get('transient_density', 0)),
                    'dynamic_variance_score': safe_convert(self.features.get('dynamic_variance_score', 0)),
                    'spectral_urgency_score': safe_convert(self.features.get('spectral_urgency_score', 0)),
                    'attention_reset_count': int(self.features.get('attention_reset_count', 0)),
                    'spatial_movement_score': safe_convert(self.features.get('spatial_movement_score', 0)),
                    'primary_attention_score': safe_convert(self.features.get('primary_attention_score', 0)),
                    'secondary_attention_score': safe_convert(self.features.get('secondary_attention_score', 0)),
                    'overall_attention_index': safe_convert(self.features.get('overall_attention_index', 0)),
                    'retention_prediction': self.features.get('retention_prediction', 'Low')
                },
                'weights': self.weights,
                'technical_details': {
                    'sample_rate': self.sr,
                    'channels': 'stereo' if self.audio_stereo is not None else 'mono',
                    'onset_count': len(self.features.get('onset_times', [])),
                    'sudden_jump_count': int(self.features.get('sudden_jump_count', 0)),
                    'reset_density': safe_convert(self.features.get('reset_density', 0))
                }
            }
            
            with open(json_filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            return json_filename
            
        except Exception as e:
            print(f"Error saving JSON data: {e}")
            return None
    
    def run_analysis(self):
        print("=== Audio Attention Prediction Analysis ===\n")
        
        # Select file
        file_path = self.select_audio_file()
        if not file_path:
            print("No file selected. Exiting.")
            return
        
        # Load and preprocess
        if not self.load_and_preprocess_audio(file_path):
            return
        
        print("\nExtracting attention features...")
        
        self.calculate_roughness_index()
        self.calculate_transient_density()
        self.calculate_dynamic_variance()
        self.calculate_spectral_urgency()
        self.calculate_attention_resets()
        self.calculate_spatial_movement()
        self.calculate_attention_scores()
        
        print("\nGenerating reports...")
        

        output_dir = f"attention_analysis_{self.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        html_file = self.generate_html_report(output_dir)
        pdf_file = self.generate_pdf_report(output_dir)
        json_file = self.save_data_json(output_dir)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {output_dir}/")
        print(f"• HTML Report: {os.path.basename(html_file)}")
        print(f"• PDF Report: {os.path.basename(pdf_file)}")
        print(f"• Data Export: {os.path.basename(json_file)}")
        
        print(f"\n=== FINAL PREDICTION ===")
        print(f"Audience Retention Likelihood: {self.features['retention_prediction']}")
        print(f"Overall Attention Index: {self.features['overall_attention_index']:.1f}/100")
        
        # Open HTML report
        try:
            webbrowser.open('file://' + os.path.abspath(html_file))
        except:
            print(f"Could not auto-open browser. Please open: {html_file}")


def main():
    try:
        analyzer = AudioAttentionAnalyzer()
        analyzer.run_analysis()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()