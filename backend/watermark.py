"""
Ultrasonic Audio Watermarking Module
Nhúng và phát hiện watermark ở tần số siêu âm (18-22 kHz)
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, fftconvolve
from scipy.fftpack import fft, ifft
import hashlib
import os
from typing import Tuple, Optional, Dict, Any
from pydub import AudioSegment
import tempfile


class UltrasonicWatermark:
    """
    Ultrasonic Watermarking using Spread Spectrum technique
    
    - Embeds data at 18-20 kHz (inaudible to humans)
    - Survives MP3 compression (with some degradation)
    - Contains: Producer ID, Track ID, Timestamp
    """
    
    # Frequency range for ultrasonic watermark
    FREQ_MIN = 18000  # 18 kHz
    FREQ_MAX = 20000  # 20 kHz
    CARRIER_FREQ = 19000  # Center frequency
    SAMPLE_RATE = 44100
    
    # Watermark parameters
    CHIP_RATE = 32  # Spread spectrum chip rate
    WATERMARK_STRENGTH = 0.01  # Amplitude (very low to stay inaudible)
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._generate_spreading_code()
    
    def _generate_spreading_code(self, seed: int = 42) -> np.ndarray:
        """Generate pseudo-random spreading code (PN sequence)"""
        np.random.seed(seed)
        # Gold code-like sequence for better correlation
        self.pn_sequence = np.sign(np.random.randn(1024))  # 1024 chips
        return self.pn_sequence
    
    def _text_to_bits(self, text: str) -> np.ndarray:
        """Convert text to binary array"""
        # Add header for synchronization
        header = "WM1:"  # Watermark version 1
        full_text = header + text
        
        bits = []
        for char in full_text:
            byte = ord(char)
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        
        return np.array(bits, dtype=np.float64)
    
    def _bits_to_text(self, bits: np.ndarray) -> str:
        """Convert binary array back to text"""
        # Ensure even number of bits for bytes
        if len(bits) % 8 != 0:
            bits = bits[:-(len(bits) % 8)]
        
        chars = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | int(bits[i + j] > 0.5)
            chars.append(chr(byte))
        
        text = ''.join(chars)
        
        # Find and extract watermark
        if "WM1:" in text:
            start = text.find("WM1:")
            return text[start + 4:]  # Remove header
        
        return text
    
    def _create_watermark_signal(self, data_bits: np.ndarray) -> np.ndarray:
        """
        Create ultrasonic watermark signal using Direct Sequence Spread Spectrum
        """
        # Spread each bit with PN sequence
        samples_per_bit = self.CHIP_RATE
        total_samples = len(data_bits) * samples_per_bit * len(self.pn_sequence)
        
        # Normalize to be much smaller than carrier amplitude
        self.WATERMARK_STRENGTH = 0.005  # Very subtle
        
        watermark = np.zeros(total_samples)
        
        for i, bit in enumerate(data_bits):
            start = i * samples_per_bit * len(self.pn_sequence)
            end = start + samples_per_bit * len(self.pn_sequence)
            
            # Spread spectrum: multiply PN sequence by bit value
            if bit == 1:
                spread = self.pn_sequence
            else:
                spread = -self.pn_sequence
            
            # Repeat to match sample length
            repeat_count = samples_per_bit
            spread_signal = np.repeat(spread, repeat_count)
            
            watermark[start:start + len(spread_signal)] = spread_signal * self.WATERMARK_STRENGTH
        
        # Modulate onto ultrasonic carrier
        t = np.arange(len(watermark)) / self.sample_rate
        carrier = np.sin(2 * np.pi * self.CARRIER_FREQ * t)
        
        modulated = watermark * carrier
        
        return modulated
    
    def embed(self, audio_path: str, producer_id: str, track_id: str = "", 
              output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Embed ultrasonic watermark into audio file
        
        Args:
            audio_path: Path to input audio file
            producer_id: Producer identifier (max 32 chars)
            track_id: Track identifier (max 32 chars)
            output_path: Output file path (optional)
        
        Returns:
            Dict with status, output_path, watermark_info
        """
        # Create watermark data
        timestamp = str(int(np.datetime64('now', 's').astype(np.int64)))
        watermark_data = f"{producer_id[:32]}|{track_id[:32]}|{timestamp}"
        
        print(f"[Watermark] Embedding: {watermark_data}")
        
        # Load audio
        try:
            # Use pydub for format support
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            return {"success": False, "error": f"Cannot load audio: {str(e)}"}
        
        # Convert to mono if needed, and get sample rate
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate
        if audio.frame_rate < 44100:
            audio = audio.set_frame_rate(44100)
        
        self.sample_rate = audio.frame_rate
        
        # Get audio samples
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        samples = samples / np.max(np.abs(samples))  # Normalize
        
        # Create watermark signal
        data_bits = self._text_to_bits(watermark_data)
        watermark_signal = self._create_watermark_signal(data_bits)
        
        # Ensure watermark fits in audio
        if len(watermark_signal) > len(samples):
            # Repeat audio to fit watermark
            repeat_count = int(np.ceil(len(watermark_signal) / len(samples)))
            samples = np.tile(samples, repeat_count)
        
        # Embed watermark
        watermarked = samples.copy()
        watermarked[:len(watermark_signal)] += watermark_signal[:len(samples)]
        
        # Normalize to prevent clipping
        watermarked = watermarked / np.max(np.abs(watermarked)) * 0.95
        
        # Convert back to int16
        watermarked_int = (watermarked * 32767).astype(np.int16)
        
        # Create output audio
        output_audio = audio._spawn(watermarked_int.tobytes())
        
        # Set output path
        if output_path is None:
            base, ext = os.path.splitext(audio_path)
            output_path = f"{base}_watermarked{ext}"
        
        # Export
        output_audio.export(output_path, format=os.path.splitext(output_path)[1][1:])
        
        return {
            "success": True,
            "output_path": output_path,
            "watermark_data": watermark_data,
            "producer_id": producer_id,
            "track_id": track_id,
            "timestamp": timestamp,
            "carrier_freq": self.CARRIER_FREQ,
            "message": f"✅ Watermark embedded at {self.CARRIER_FREQ}Hz"
        }
    
    def detect(self, audio_path: str) -> Dict[str, Any]:
        """
        Detect and extract ultrasonic watermark from audio
        
        Args:
            audio_path: Path to audio file to analyze
        
        Returns:
            Dict with detection status, extracted data, confidence
        """
        print(f"[Watermark] Detecting in: {audio_path}")
        
        # Load audio
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            return {"success": False, "error": f"Cannot load audio: {str(e)}"}
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        self.sample_rate = audio.frame_rate
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        samples = samples / np.max(np.abs(samples))
        
        # Step 1: Check for ultrasonic energy presence
        ultrasonic_energy = self._check_ultrasonic_energy(samples)
        print(f"[Watermark] Ultrasonic energy: {ultrasonic_energy:.4f}")
        
        if ultrasonic_energy < 0.001:
            return {
                "success": False,
                "has_watermark": False,
                "message": "❌ No ultrasonic watermark detected",
                "ultrasonic_energy": ultrasonic_energy
            }
        
        # Step 2: Demodulate and extract
        extracted_bits = self._extract_bits(samples)
        
        # Step 3: Convert to text
        try:
            watermark_text = self._bits_to_text(extracted_bits)
            
            # Parse watermark data
            parts = watermark_text.split("|")
            
            if len(parts) >= 3:
                producer_id = parts[0]
                track_id = parts[1]
                timestamp = parts[2]
                
                # Calculate confidence based on bit error rate
                confidence = self._calculate_confidence(extracted_bits)
                
                return {
                    "success": True,
                    "has_watermark": True,
                    "producer_id": producer_id,
                    "track_id": track_id,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "ultrasonic_energy": ultrasonic_energy,
                    "carrier_freq": self.CARRIER_FREQ,
                    "message": f"✅ Watermark found: Producer={producer_id}"
                }
            else:
                return {
                    "success": True,
                    "has_watermark": True,
                    "raw_data": watermark_text[:64],
                    "confidence": 0.5,
                    "ultrasonic_energy": ultrasonic_energy,
                    "message": "⚠️ Watermark detected but incomplete"
                }
                
        except Exception as e:
            return {
                "success": True,
                "has_watermark": True,
                "ultrasonic_energy": ultrasonic_energy,
                "message": f"⚠️ Watermark detected but unreadable: {str(e)}"
            }
    
    def _check_ultrasonic_energy(self, samples: np.ndarray) -> float:
        """Check for energy in ultrasonic frequency range"""
        # Compute FFT
        n = len(samples)
        freqs = np.fft.fftfreq(n, 1/self.sample_rate)
        fft_vals = np.abs(np.fft.fft(samples))
        
        # Find energy in ultrasonic range
        ultrasonic_mask = (freqs >= self.FREQ_MIN) & (freqs <= self.FREQ_MAX)
        ultrasonic_energy = np.sum(fft_vals[ultrasonic_mask] ** 2)
        total_energy = np.sum(fft_vals ** 2)
        
        return ultrasonic_energy / total_energy if total_energy > 0 else 0
    
    def _extract_bits(self, samples: np.ndarray) -> np.ndarray:
        """Demodulate and extract watermark bits"""
        # Demodulate from carrier
        t = np.arange(len(samples)) / self.sample_rate
        carrier = np.sin(2 * np.pi * self.CARRIER_FREQ * t)
        
        # Multiply by carrier (coherent demodulation)
        demodulated = samples * carrier
        
        # Low-pass filter to get baseband
        demodulated = self._lowpass_filter(demodulated, 5000)  # 5kHz cutoff
        
        # Correlate with PN sequence to despread
        samples_per_bit = self.CHIP_RATE
        pn_len = len(self.pn_sequence)
        chip_samples = samples_per_bit * pn_len
        
        bits = []
        num_bits = len(samples) // chip_samples
        
        for i in range(min(num_bits, 512)):  # Max 512 bits
            start = i * chip_samples
            end = start + chip_samples
            
            if end > len(demodulated):
                break
            
            segment = demodulated[start:end]
            
            # Downsample to match PN sequence length
            downsampled = np.mean(segment.reshape(-1, samples_per_bit), axis=1)
            
            # Correlate with PN sequence
            correlation_pos = np.abs(np.corrcoef(downsampled[:pn_len], self.pn_sequence)[0, 1])
            correlation_neg = np.abs(np.corrcoef(downsampled[:pn_len], -self.pn_sequence)[0, 1])
            
            # Determine bit based on correlation
            if not np.isnan(correlation_pos) and not np.isnan(correlation_neg):
                bits.append(1 if correlation_pos > correlation_neg else 0)
            else:
                bits.append(0.5)  # Unknown
        
        return np.array(bits)
    
    def _lowpass_filter(self, signal: np.ndarray, cutoff: int) -> np.ndarray:
        """Apply low-pass Butterworth filter"""
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype='low')
        return filtfilt(b, a, signal)
    
    def _calculate_confidence(self, bits: np.ndarray) -> float:
        """Calculate confidence score based on bit consistency"""
        # Check for valid bit patterns (no 0.5 values)
        valid_bits = bits[bits != 0.5]
        if len(valid_bits) == 0:
            return 0.0
        
        # Check header presence
        header_bits = self._text_to_bits("WM1:")
        if len(valid_bits) >= len(header_bits):
            header_match = np.mean(valid_bits[:len(header_bits)] == header_bits)
            return float(header_match)
        
        return 0.5
    
    def analyze_spectrum(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio spectrum and return ultrasonic region data
        Useful for visualization
        """
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        
        # Compute FFT
        n = min(len(samples), 65536)  # Limit for visualization
        samples = samples[:n]
        
        freqs = np.fft.fftfreq(n, 1/audio.frame_rate)[:n//2]
        fft_vals = np.abs(np.fft.fft(samples))[:n//2]
        
        # Normalize
        fft_vals = fft_vals / np.max(fft_vals) * 100
        
        # Find ultrasonic region
        ultra_mask = freqs >= 1000  # From 1kHz
        
        return {
            "success": True,
            "freqs": freqs[ultra_mask].tolist()[-500:],  # Last 500 points
            "magnitudes": fft_vals[ultra_mask].tolist()[-500:],
            "ultrasonic_energy": self._check_ultrasonic_energy(samples),
            "peak_freq": float(freqs[np.argmax(fft_vals)])
        }


# Singleton instance
_watermark_instance = None

def get_watermark() -> UltrasonicWatermark:
    global _watermark_instance
    if _watermark_instance is None:
        _watermark_instance = UltrasonicWatermark()
    return _watermark_instance


# API functions for FastAPI integration
def embed_watermark(audio_path: str, producer_id: str, track_id: str = "", 
                    output_path: Optional[str] = None) -> Dict[str, Any]:
    """API wrapper for embedding watermark"""
    wm = get_watermark()
    return wm.embed(audio_path, producer_id, track_id, output_path)


def detect_watermark(audio_path: str) -> Dict[str, Any]:
    """API wrapper for detecting watermark"""
    wm = get_watermark()
    return wm.detect(audio_path)


def analyze_spectrum(audio_path: str) -> Dict[str, Any]:
    """API wrapper for spectrum analysis"""
    wm = get_watermark()
    return wm.analyze_spectrum(audio_path)


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python watermark.py <audio_file> [producer_id] [track_id]")
        print("       python watermark.py --detect <audio_file>")
        sys.exit(1)
    
    if sys.argv[1] == "--detect":
        result = detect_watermark(sys.argv[2])
        print("\n=== Detection Result ===")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        audio_file = sys.argv[1]
        producer = sys.argv[2] if len(sys.argv) > 2 else "PRODUCER_001"
        track = sys.argv[3] if len(sys.argv) > 3 else "TRACK_001"
        
        result = embed_watermark(audio_file, producer, track)
        print("\n=== Embed Result ===")
        for key, value in result.items():
            print(f"  {key}: {value}")