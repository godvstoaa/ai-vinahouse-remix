"""
Music Theory Engine - Advanced music theory for AI remixing
Implements professional-grade music theory concepts
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScaleType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    LOCRIAN = "locrian"
    HARMONIC_MINOR = "harmonic_minor"
    MELODIC_MINOR = "melodic_minor"
    PENTATONIC_MAJOR = "pentatonic_major"
    PENTATONIC_MINOR = "pentatonic_minor"
    BLUES = "blues"


class ChordQuality(Enum):
    MAJOR = "maj"
    MINOR = "min"
    DIMINISHED = "dim"
    AUGMENTED = "aug"
    DOMINANT = "7"
    MAJOR_7 = "maj7"
    MINOR_7 = "m7"
    SUS2 = "sus2"
    SUS4 = "sus4"
    ADD9 = "add9"


@dataclass
class Chord:
    """Represents a musical chord"""
    root: str  # Note name (C, C#, D, etc.)
    quality: ChordQuality
    notes: List[str]  # Notes in the chord
    bass_note: Optional[str] = None  # For inversions/slash chords
    
    def __str__(self):
        bass = f"/{self.bass_note}" if self.bass_note else ""
        return f"{self.root}{self.quality.value}{bass}"


@dataclass
class KeySignature:
    """Represents a musical key"""
    tonic: str
    scale_type: ScaleType
    accidentals: int  # Number of sharps (+) or flats (-)
    
    def __str__(self):
        return f"{self.tonic} {self.scale_type.value.replace('_', ' ')}"


# Circle of Fifths for key relationships
CIRCLE_OF_FIFTHS = [
    'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#',  # Sharp side
    'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F'  # Flat side (overlapping at F#/Gb)
]

# Key relationships (how well keys mix together)
KEY_COMPATIBILITY = {
    # Same key = 1.0, Perfect 4th/5th = 0.9, Relative major/minor = 0.85
    # Parallel major/minor = 0.7, Tritone = 0.3
    'same': 1.0,
    'perfect_fourth': 0.9,
    'perfect_fifth': 0.9,
    'relative': 0.85,
    'parallel': 0.7,
    'tritone': 0.3,
}

# Scale intervals (semitones from root)
SCALE_INTERVALS = {
    ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
    ScaleType.MINOR: [0, 2, 3, 5, 7, 8, 10],
    ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
    ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
    ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    ScaleType.LOCRIAN: [0, 1, 3, 5, 6, 8, 10],
    ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
    ScaleType.MELODIC_MINOR: [0, 2, 3, 5, 7, 9, 11],
    ScaleType.PENTATONIC_MAJOR: [0, 2, 4, 7, 9],
    ScaleType.PENTATONIC_MINOR: [0, 3, 5, 7, 10],
    ScaleType.BLUES: [0, 3, 5, 6, 7, 10],
}

# Note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Chord progressions by genre and mood
CHORD_PROGRESSIONS = {
    'pop': {
        'happy': [
            ['I', 'V', 'vi', 'IV'],  # Axis progression
            ['I', 'IV', 'V', 'IV'],
            ['vi', 'IV', 'I', 'V'],
        ],
        'sad': [
            ['vi', 'IV', 'I', 'V'],
            ['i', 'VI', 'III', 'VII'],
        ],
    },
    'edm': {
        'energetic': [
            ['i', 'VI', 'III', 'VII'],  # Natural minor
            ['i', 'iv', 'VII', 'V'],
            ['I', 'V', 'vi', 'IV'],
        ],
        'melancholic': [
            ['i', 'VII', 'VI', 'VII'],
            ['i', 'VI', 'i', 'VI'],
        ],
    },
    'hiphop': {
        'dark': [
            ['i', 'VII', 'VI', 'VII'],
            ['i', 'iv', 'i', 'VII'],
        ],
        'smooth': [
            ['ii7', 'V7', 'Imaj7', 'vi7'],  # Jazz-influenced
        ],
    },
    'house': {
        'uplifting': [
            ['I', 'V', 'vi', 'IV'],
            ['vi', 'I', 'V', 'IV'],
        ],
        'deep': [
            ['i7', 'iv7', 'VII7', 'VII7'],
            ['i', 'VI', 'VII', 'VII'],
        ],
    },
    'lofi': {
        'chill': [
            ['ii7', 'V7', 'Imaj7', 'vi7'],
            ['iii7', 'vi7', 'ii7', 'V7'],
            ['Imaj7', 'ii7', 'iii7', 'IVmaj7'],
        ],
        'nostalgic': [
            ['I', 'III', 'IV', 'iv'],  # With minor iv for nostalgia
        ],
    },
    'trap': {
        'dark': [
            ['i', 'VI', 'III', 'VII'],
            ['i', 'i', 'VI', 'VII'],
        ],
        'melodic': [
            ['vi', 'IV', 'I', 'V'],
        ],
    },
}

# Common song structures by genre
SONG_STRUCTURES = {
    'pop': ['intro', 'verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus', 'outro'],
    'edm': ['intro', 'buildup', 'drop', 'breakdown', 'buildup', 'drop', 'outro'],
    'house': ['intro', 'groove', 'build', 'drop', 'breakdown', 'drop', 'outro'],
    'hiphop': ['intro', 'verse', 'hook', 'verse', 'hook', 'bridge', 'hook', 'outro'],
    'lofi': ['intro', 'loop', 'variation', 'loop', 'bridge', 'loop', 'outro'],
    'trap': ['intro', 'verse', 'hook', 'verse', 'hook', 'bridge', 'hook', 'outro'],
    'techno': ['intro', 'groove', 'develop', 'climax', 'breakdown', 'groove', 'outro'],
}


class MusicTheoryEngine:
    """
    Advanced music theory engine for AI remixing.
    Handles key detection, chord analysis, harmonic mixing, and more.
    """
    
    def __init__(self):
        self.sample_rate = 22050
        
    def analyze_key(self, audio: np.ndarray, sr: int = None) -> KeySignature:
        """
        Detect the musical key of an audio file using Krumhansl-Schmuckler algorithm.
        
        Returns detailed key signature with scale type.
        """
        sr = sr or self.sample_rate
        
        # Get chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Correlate with all 12 possible roots
        best_correlation = -1
        best_key = 'C'
        best_is_major = True
        
        for shift in range(12):
            # Shift profiles
            major_shifted = np.roll(major_profile, shift)
            minor_shifted = np.roll(minor_profile, shift)
            
            # Calculate correlations
            major_corr = np.corrcoef(chroma_mean, major_shifted)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_shifted)[0, 1]
            
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = NOTE_NAMES[shift]
                best_is_major = True
                
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = NOTE_NAMES[shift]
                best_is_major = False
        
        # Determine scale type and accidentals
        if best_is_major:
            scale_type = ScaleType.MAJOR
        else:
            scale_type = ScaleType.MINOR
            
        accidentals = self._get_accidentals(best_key, scale_type)
        
        return KeySignature(
            tonic=best_key,
            scale_type=scale_type,
            accidentals=accidentals
        )
    
    def analyze_chords(self, audio: np.ndarray, sr: int = None, 
                       beats: np.ndarray = None) -> List[Tuple[float, Chord]]:
        """
        Detect chord progression throughout the track.
        
        Returns list of (time, Chord) tuples.
        """
        sr = sr or self.sample_rate
        
        if beats is None:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Get chromagram
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        chords = []
        
        for i, beat_time in enumerate(beat_times):
            frame = int(beat_time * sr / 512)  # hop_length = 512
            if frame < chroma.shape[1]:
                chroma_slice = chroma[:, max(0, frame-2):min(chroma.shape[1], frame+3)]
                chroma_mean = np.mean(chroma_slice, axis=1)
                
                chord = self._detect_chord_from_chroma(chroma_mean)
                chords.append((beat_time, chord))
        
        return chords
    
    def _detect_chord_from_chroma(self, chroma: np.ndarray) -> Chord:
        """Detect chord from chroma vector"""
        # Chord templates
        major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])  # 1, 3, 5
        minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])  # 1, b3, 5
        dim_template = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])   # 1, b3, b5
        aug_template = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])   # 1, 3, #5
        
        best_score = -1
        best_root = 0
        best_quality = ChordQuality.MAJOR
        
        templates = [
            (major_template, ChordQuality.MAJOR),
            (minor_template, ChordQuality.MINOR),
            (dim_template, ChordQuality.DIMINISHED),
            (aug_template, ChordQuality.AUGMENTED),
        ]
        
        for root in range(12):
            for template, quality in templates:
                shifted = np.roll(template, root)
                score = np.sum(chroma * shifted)
                
                if score > best_score:
                    best_score = score
                    best_root = root
                    best_quality = quality
        
        root_note = NOTE_NAMES[best_root]
        notes = self._get_chord_notes(root_note, best_quality)
        
        return Chord(root=root_note, quality=best_quality, notes=notes)
    
    def _get_chord_notes(self, root: str, quality: ChordQuality) -> List[str]:
        """Get notes in a chord"""
        root_idx = NOTE_NAMES.index(root)
        
        if quality == ChordQuality.MAJOR:
            intervals = [0, 4, 7]
        elif quality == ChordQuality.MINOR:
            intervals = [0, 3, 7]
        elif quality == ChordQuality.DIMINISHED:
            intervals = [0, 3, 6]
        elif quality == ChordQuality.AUGMENTED:
            intervals = [0, 4, 8]
        elif quality == ChordQuality.DOMINANT:
            intervals = [0, 4, 7, 10]
        elif quality == ChordQuality.MAJOR_7:
            intervals = [0, 4, 7, 11]
        elif quality == ChordQuality.MINOR_7:
            intervals = [0, 3, 7, 10]
        elif quality == ChordQuality.SUS2:
            intervals = [0, 2, 7]
        elif quality == ChordQuality.SUS4:
            intervals = [0, 5, 7]
        elif quality == ChordQuality.ADD9:
            intervals = [0, 4, 7, 14]
        else:
            intervals = [0, 4, 7]
        
        return [NOTE_NAMES[(root_idx + i) % 12] for i in intervals]
    
    def _get_accidentals(self, key: str, scale_type: ScaleType) -> int:
        """Get number of sharps (+) or flats (-) in key signature"""
        # Position in circle of fifths
        sharp_order = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
        flat_order = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
        
        if scale_type == ScaleType.MAJOR:
            major_keys_sharps = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#']
            major_keys_flats = ['C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb']
            
            if key in major_keys_sharps:
                return major_keys_sharps.index(key)
            elif key in major_keys_flats:
                return -major_keys_flats.index(key)
        else:  # Minor
            minor_keys_sharps = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#']
            minor_keys_flats = ['A', 'D', 'G', 'C', 'F', 'Bb', 'Eb']
            
            if key in minor_keys_sharps:
                return minor_keys_sharps.index(key)
            elif key in minor_keys_flats:
                return -minor_keys_flats.index(key)
        
        return 0
    
    def get_compatible_keys(self, key: KeySignature) -> List[Tuple[KeySignature, float]]:
        """
        Get keys that mix well with the given key for harmonic mixing.
        
        Returns list of (key, compatibility_score) tuples.
        """
        compatible = [(key, KEY_COMPATIBILITY['same'])]
        
        root_idx = NOTE_NAMES.index(key.tonic)
        
        # Perfect fifth up
        fifth_up = NOTE_NAMES[(root_idx + 7) % 12]
        compatible.append((KeySignature(fifth_up, key.scale_type, 0), KEY_COMPATIBILITY['perfect_fifth']))
        
        # Perfect fourth up
        fourth_up = NOTE_NAMES[(root_idx + 5) % 12]
        compatible.append((KeySignature(fourth_up, key.scale_type, 0), KEY_COMPATIBILITY['perfect_fourth']))
        
        # Relative major/minor
        if key.scale_type == ScaleType.MAJOR:
            # Relative minor is 3 semitones down
            relative_minor = NOTE_NAMES[(root_idx - 3) % 12]
            compatible.append((KeySignature(relative_minor, ScaleType.MINOR, 0), KEY_COMPATIBILITY['relative']))
        else:
            # Relative major is 3 semitones up
            relative_major = NOTE_NAMES[(root_idx + 3) % 12]
            compatible.append((KeySignature(relative_major, ScaleType.MAJOR, 0), KEY_COMPATIBILITY['relative']))
        
        # Parallel major/minor
        if key.scale_type == ScaleType.MAJOR:
            compatible.append((KeySignature(key.tonic, ScaleType.MINOR, 0), KEY_COMPATIBILITY['parallel']))
        else:
            compatible.append((KeySignature(key.tonic, ScaleType.MAJOR, 0), KEY_COMPATIBILITY['parallel']))
        
        return compatible
    
    def suggest_chord_progression(self, key: KeySignature, genre: str, mood: str = 'happy') -> List[Chord]:
        """
        Suggest a chord progression based on key, genre and mood.
        
        Returns list of Chord objects.
        """
        # Get roman numeral progression
        genre_progressions = CHORD_PROGRESSIONS.get(genre, CHORD_PROGRESSIONS['pop'])
        mood_progressions = genre_progressions.get(mood, list(genre_progressions.values())[0])
        
        roman_numerals = mood_progressions[0]  # Use first progression
        
        # Convert roman numerals to actual chords
        chords = []
        scale_degrees = self._get_scale_degrees(key)
        
        for numeral in roman_numerals:
            chord = self._roman_numeral_to_chord(numeral, key, scale_degrees)
            chords.append(chord)
        
        return chords
    
    def _get_scale_degrees(self, key: KeySignature) -> Dict[str, int]:
        """Get scale degree to semitone mapping"""
        intervals = SCALE_INTERVALS.get(key.scale_type, SCALE_INTERVALS[ScaleType.MAJOR])
        root_idx = NOTE_NAMES.index(key.tonic)
        
        degrees = {}
        degree_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
        
        for i, name in enumerate(degree_names):
            degrees[name] = (root_idx + intervals[i]) % 12
        
        return degrees
    
    def _roman_numeral_to_chord(self, numeral: str, key: KeySignature, 
                                scale_degrees: Dict[str, int]) -> Chord:
        """Convert roman numeral to chord"""
        # Determine if major or minor
        is_minor = numeral.islower() or 'i' in numeral.lower() and numeral[0].islower()
        
        # Clean numeral
        clean_numeral = numeral.replace('7', '').replace('maj', '').replace('min', '')
        clean_numeral = clean_numeral.upper()
        
        # Get root
        root_idx = scale_degrees.get(clean_numeral, 0)
        root_note = NOTE_NAMES[root_idx]
        
        # Determine quality
        if is_minor:
            quality = ChordQuality.MINOR
        else:
            quality = ChordQuality.MAJOR
        
        notes = self._get_chord_notes(root_note, quality)
        
        return Chord(root=root_note, quality=quality, notes=notes)
    
    def get_song_structure(self, genre: str, duration_seconds: float) -> Dict[str, Tuple[float, float]]:
        """
        Suggest song structure based on genre.
        
        Returns dict of {section_name: (start_time, end_time)}
        """
        structure = SONG_STRUCTURES.get(genre, SONG_STRUCTURES['pop'])
        
        section_duration = duration_seconds / len(structure)
        sections = {}
        
        for i, section in enumerate(structure):
            start = i * section_duration
            end = (i + 1) * section_duration
            sections[section] = (start, end)
        
        return sections
    
    def analyze_rhythm_complexity(self, audio: np.ndarray, sr: int = None) -> Dict[str, float]:
        """
        Analyze rhythmic complexity and groove.
        
        Returns metrics for syncopation, density, and swing.
        """
        sr = sr or self.sample_rate
        
        # Get onset envelope
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Get tempo and beats
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        
        # Calculate syncopation (notes off the beat)
        beat_frames = librosa.frames_to_time(beats, sr=sr)
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Measure distance from nearest beat
        syncopation_scores = []
        for onset_time in onset_times:
            nearest_beat = min(beat_frames, key=lambda b: abs(b - onset_time))
            distance = abs(onset_time - nearest_beat)
            syncopation_scores.append(distance)
        
        avg_syncopation = np.mean(syncopation_scores) if syncopation_scores else 0
        
        # Calculate density (notes per second)
        duration = len(audio) / sr
        density = len(onset_times) / duration if duration > 0 else 0
        
        # Estimate swing (uneven beat subdivisions)
        # Look at the spectral flux rhythm
        spectral_flux = np.diff(np.abs(librosa.stft(audio)), axis=1)
        flux_sum = np.sum(spectral_flux, axis=0)
        
        # Look for swing pattern (long-short-long-short)
        swing_measure = np.std(np.diff(flux_sum)) / (np.mean(flux_sum) + 1e-10)
        
        return {
            'syncopation': min(avg_syncopation * 10, 1.0),  # Normalized 0-1
            'density': min(density / 10, 1.0),  # Normalized 0-1
            'swing': min(swing_measure, 1.0),  # Normalized 0-1
            'tempo': float(tempo),
        }
    
    def calculate_harmonic_tension(self, chord_progression: List[Chord]) -> List[float]:
        """
        Calculate harmonic tension for each chord in progression.
        
        Higher tension = more desire to resolve.
        """
        tensions = []
        
        # Tension based on chord function
        # Tonic = low tension, Dominant = high tension, Subdominant = medium
        tension_values = {
            ChordQuality.MAJOR: 0.3,  # Can be tonic or dominant
            ChordQuality.MINOR: 0.4,  # Usually tonic or subdominant
            ChordQuality.DOMINANT: 0.8,  # High tension, wants to resolve
            ChordQuality.DIMINISHED: 0.9,  # Very high tension
            ChordQuality.AUGMENTED: 0.85,
            ChordQuality.MAJOR_7: 0.5,
            ChordQuality.MINOR_7: 0.4,
        }
        
        for i, chord in enumerate(chord_progression):
            base_tension = tension_values.get(chord.quality, 0.5)
            
            # Adjust for position in progression
            position_factor = 1.0
            if i == len(chord_progression) - 1:
                # Last chord usually resolves
                position_factor = 0.3
            
            tensions.append(base_tension * position_factor)
        
        return tensions
    
    def suggest_tempo_adjustment(self, original_bpm: float, target_genre: str) -> float:
        """
        Suggest optimal tempo for a genre transition.
        
        Returns suggested BPM that stays within genre norms.
        """
        # Typical BPM ranges by genre
        genre_bpm_ranges = {
            'edm': (128, 130),
            'house': (120, 130),
            'deep_house': (118, 125),
            'techno': (130, 150),
            'trance': (130, 145),
            'hiphop': (85, 95),
            'trap': (140, 150),
            'lofi': (70, 90),
            'dubstep': (140, 150),
            'drum_and_bass': (170, 180),
            'pop': (110, 130),
            'rock': (110, 140),
        }
        
        target_range = genre_bpm_ranges.get(target_genre, (120, 130))
        min_bpm, max_bpm = target_range
        
        # Find the best adjustment
        if original_bpm < min_bpm:
            # Double the tempo if too slow
            if original_bpm * 2 <= max_bpm:
                return original_bpm * 2
            return min_bpm
        elif original_bpm > max_bpm:
            # Half the tempo if too fast
            if original_bpm / 2 >= min_bpm:
                return original_bpm / 2
            return max_bpm
        else:
            # Already in range
            return original_bpm
    
    def analyze_energy_contour(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Analyze energy contour over time.
        
        Returns array of energy values normalized 0-1.
        """
        sr = sr or self.sample_rate
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Normalize
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
        
        return rms_normalized
    
    def detect_drops_and_builds(self, audio: np.ndarray, sr: int = None) -> List[Tuple[float, str]]:
        """
        Detect drops and buildups in EDM/club music.
        
        Returns list of (time, event_type) tuples.
        """
        sr = sr or self.sample_rate
        
        # Get energy contour
        energy = self.analyze_energy_contour(audio, sr)
        
        # Smooth the energy
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(energy, sigma=10)
        
        events = []
        times = librosa.frames_to_time(range(len(smoothed)), sr=sr)
        
        # Find rapid energy changes
        diff = np.diff(smoothed)
        threshold = np.std(diff) * 2
        
        for i, d in enumerate(diff):
            if d > threshold:
                # Rapid increase = build
                events.append((times[i], 'build'))
            elif d < -threshold:
                # Rapid decrease = drop
                events.append((times[i], 'drop'))
        
        return events


# Singleton instance
music_theory_engine = MusicTheoryEngine()