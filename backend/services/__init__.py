# Services package
from .source_separator import SourceSeparator
from .genre_detector import GenreDetector
from .audio_analyzer import ProfessionalAudioAnalyzer, audio_analyzer
from .remix_generator import RemixGenerator, remix_generator
from .demucs_separator import DemucsSeparator, demucs_separator
from .pedalboard_effects import professional_effects, HAS_PEDALBOARD

# Backward compatibility alias
AudioAnalyzer = ProfessionalAudioAnalyzer

__all__ = [
    'SourceSeparator', 
    'GenreDetector', 
    'AudioAnalyzer',
    'ProfessionalAudioAnalyzer',
    'audio_analyzer',
    'RemixGenerator', 
    'remix_generator',
    'DemucsSeparator',
    'demucs_separator',
    'professional_effects',
    'HAS_PEDALBOARD'
]