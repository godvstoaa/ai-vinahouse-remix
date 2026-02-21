import React, { useState, useRef, useCallback, useEffect } from 'react';
import FastTrainingPanel from './components/FastTrainingPanel';
import TrainingPanel from './components/TrainingPanel';
import UpdateButton from './components/UpdateButton';
import WatermarkPanel from './components/WatermarkPanel';
import LongAudioPanel from './components/LongAudioPanel';

// Types
interface Stem {
    name: string;
    url: string;
    volume: number;
    muted: boolean;
    solo: boolean;
}

interface RemixSettings {
    targetBpm: number;
    targetGenre: string;
    energyLevel: number;
    creativity: number;
    preserveMelody: number;
    bassBoost: number;
    reverb: number;
    sidechain: number;
    quality: string;
    useDeepLearning: boolean;
    kickPattern: string;
    bassPattern: string;
    vocalEffect: string;
    masterGain: number;
}

interface AudioAnalysis {
    bpm: number;
    key: string;
    energy: number;
    genre: string;
}

interface RemixResult {
    id: string;
    input_file: string;
    output_file: string;
    settings: RemixSettings;
    created_at: string;
}

interface Preset {
    id: string;
    name: string;
    settings: RemixSettings;
    isDefault: boolean;
}

type TabType = 'remix' | 'stems' | 'training' | 'longaudio' | 'history' | 'settings' | 'watermark';
type ProcessingStep = 'idle' | 'uploading' | 'analyzing' | 'separating' | 'generating' | 'mastering' | 'complete' | 'error';

// Vietnamese translations - UTF-8
const TEXTS = {
    appName: 'AI Vinahouse Remix Pro',
    appSubtitle: 'Công nghệ Deep Learning 2026',
    tabs: {
        remix: '🎵 Tạo Remix',
        stems: '🎧 Tách Nhạc',
        training: '🧠 Huấn Luyện',
        history: '📜 Lịch Sử',
        settings: '⚙️ Cài Đặt',
    },
    upload: {
        title: 'Kéo thả file nhạc vào đây',
        orClick: 'hoặc nhấn để chọn file',
        supports: 'Hỗ trợ MP3, WAV, FLAC',
        processing: 'Đang xử lý...',
        aiAnalyzing: 'AI đang phân tích bài hát',
    },
    analysis: {
        title: '📊 Phân Tích',
        bpm: 'Nhịp:',
        key: 'Tone:',
        energy: 'Năng lượng:',
        genre: 'Thể loại:',
    },
    controls: {
        title: '🎛️ Điều Chỉnh Remix',
        targetGenre: 'Thể loại mục tiêu',
        targetBpm: 'Nhịp BPM:',
        creativity: 'Sáng tạo AI:',
        bassBoost: 'Tăng Bass:',
        separateStems: '🎧 Tách Nhạc',
        createRemix: '🔥 Tạo Vinahouse Remix',
        kickPattern: 'Pattern Kick',
        bassPattern: 'Pattern Bass',
        vocalEffect: 'Hiệu ứng Vocal',
        masterGain: 'Âm lượng tổng:',
        sidechain: 'Sidechain:',
        reverb: 'Reverb:',
    },
    presets: {
        title: '📦 Presets',
        save: '💾 Lưu Preset',
        load: 'Nạp',
        default: 'Mặc định',
        vinahouseClassic: 'Vinahouse Cổ Điển',
        vinahouseModern: 'Vinahouse Hiện Đại',
        vinahouseBounce: 'Vinahouse Bounce',
        remixHot: 'Remix Hot',
        chillMix: 'Chill Mix',
        clubMix: 'Club Mix',
        customPrefix: 'Tùy chỉnh: ',
    },
    patterns: {
        standard: '4/4 Standard',
        offbeat: 'Off-beat',
        syncopated: 'Nhịp phách',
        rolling: 'Rolling',
        punchy: 'Mạnh mẽ',
    },
    vocalEffects: {
        none: 'Không',
        autoTune: 'Auto-Tune',
        pitchShift: 'Pitch Shift',
        reverb: 'Reverb',
        delay: 'Delay',
        distortion: 'Distortion',
    },
    genres: {
        vinahouse: 'Vinahouse',
        house: 'House',
        edm: 'EDM',
        hiphop: 'Hip-hop',
        trap: 'Trap',
        vinahouseBounce: 'Vinahouse Bounce',
    },
    audio: {
        original: '🎵 Bài Hát Gốc',
        remix: '🔥 Bản Remix',
        exportMp3: '📤 Xuất MP3',
        exportWav: '📤 Xuất WAV',
        exportStems: '📤 Xuất Từng Track',
        compare: '🔄 So Sánh A/B',
    },
    stems: {
        title: '🎧 Phát Theo Dải',
        separate: 'Tách Nhạc',
        vocals: 'Giọng Hát',
        drums: 'Trống',
        bass: 'Bass',
        other: 'Nhạc Cụ',
        uploadFirst: 'Tải bài hát lên và nhấn "Tách Nhạc" để bắt đầu',
        exportAll: '📤 Xuất Tất Cả',
    },
    batch: {
        title: '📦 Xử Lý Hàng Loạt',
        selectFiles: 'Chọn nhiều file',
        processing: 'Đang xử lý',
        files: 'file',
        start: 'Bắt đầu xử lý',
        completed: 'Hoàn thành',
    },
    training: {
        title: '🧠 Huấn Luyện Mô Hình',
        config: 'Cấu Hình Huấn Luyện',
        datasetPath: 'Đường dẫn dữ liệu',
        datasetPlaceholder: 'D:/VinahouseMusic',
        epochs: 'Số Vòng Lặp',
        batchSize: 'Kích Thước Batch',
        startTraining: '🚀 Bắt Đầu Huấn Luyện',
        status: 'Trạng Thái Huấn Luyện',
        noTraining: 'Không có huấn luyện nào đang chạy',
        advanced: 'Cài Đặt Nâng Cao',
        learningRate: 'Tốc độ học',
        modelType: 'Loại mô hình',
    },
    history: {
        title: '📜 Lịch Sử Remix',
        download: 'Tải về',
        delete: 'Xóa',
        noHistory: 'Chưa có remix nào',
        clearAll: '🗑️ Xóa tất cả',
    },
    settings: {
        title: '⚙️ Cài Đặt',
        modelSettings: '🧠 Cài Đặt Mô Hình',
        checkpointPath: 'Đường dẫn Checkpoint',
        checkpointPlaceholder: 'D:/checkpoints/vinahouse_epoch_100.pt',
        systemInfo: '💻 Thông Tin Hệ Thống',
        apiServer: 'Máy chủ API:',
        status: 'Trạng thái:',
        ready: 'Sẵn sàng',
        reset: '🗑️ Đặt Lại Tất Cả',
        keyboard: '⌨️ Phím Tắt',
        shortcuts: {
            space: 'Phím cách: Play/Pause',
            r: 'R: Tạo Remix',
            s: 'S: Tách nhạc',
            e: 'E: Xuất file',
            one: '1-5: Chuyển tab',
        },
        gpu: 'GPU:',
        gpuAvailable: 'Có sẵn',
        gpuNotAvailable: 'Không có',
    },
    footer: {
        version: 'AI Vinahouse Remix Studio Pro v2.0',
        ready: '✓ Sẵn sàng',
        processing: '⏳ Đang xử lý...',
        help: 'F1: Trợ giúp',
    },
    steps: {
        uploading: 'Đang tải lên',
        analyzing: 'Đang phân tích',
        separating: 'Đang tách nhạc',
        generating: 'Đang tạo remix',
        mastering: 'Đang xử lý âm thanh',
        complete: 'Hoàn thành',
        error: 'Lỗi',
    },
};

// Default presets
const DEFAULT_PRESETS: Preset[] = [
    {
        id: 'default',
        name: TEXTS.presets.default,
        isDefault: true,
        settings: {
            targetBpm: 135,
            targetGenre: 'vinahouse',
            energyLevel: 1.0,
            creativity: 0.8,
            preserveMelody: 0.3,
            bassBoost: 6,
            reverb: 0.3,
            sidechain: 0.6,
            quality: 'high',
            useDeepLearning: true,
            kickPattern: 'standard',
            bassPattern: 'offbeat',
            vocalEffect: 'none',
            masterGain: 0,
        },
    },
    {
        id: 'vinahouse-classic',
        name: TEXTS.presets.vinahouseClassic,
        isDefault: true,
        settings: {
            targetBpm: 138,
            targetGenre: 'vinahouse',
            energyLevel: 1.2,
            creativity: 0.6,
            preserveMelody: 0.5,
            bassBoost: 8,
            reverb: 0.2,
            sidechain: 0.8,
            quality: 'high',
            useDeepLearning: true,
            kickPattern: 'standard',
            bassPattern: 'offbeat',
            vocalEffect: 'reverb',
            masterGain: 2,
        },
    },
    {
        id: 'vinahouse-modern',
        name: TEXTS.presets.vinahouseModern,
        isDefault: true,
        settings: {
            targetBpm: 132,
            targetGenre: 'vinahouse',
            energyLevel: 1.3,
            creativity: 0.9,
            preserveMelody: 0.2,
            bassBoost: 10,
            reverb: 0.4,
            sidechain: 0.7,
            quality: 'high',
            useDeepLearning: true,
            kickPattern: 'punchy',
            bassPattern: 'syncopated',
            vocalEffect: 'autoTune',
            masterGain: 3,
        },
    },
    {
        id: 'vinahouse-bounce',
        name: TEXTS.presets.vinahouseBounce,
        isDefault: true,
        settings: {
            targetBpm: 128,
            targetGenre: 'vinahouseBounce',
            energyLevel: 1.4,
            creativity: 0.95,
            preserveMelody: 0.1,
            bassBoost: 12,
            reverb: 0.15,
            sidechain: 0.9,
            quality: 'high',
            useDeepLearning: true,
            kickPattern: 'rolling',
            bassPattern: 'punchy',
            vocalEffect: 'pitchShift',
            masterGain: 4,
        },
    },
    {
        id: 'remix-hot',
        name: TEXTS.presets.remixHot,
        isDefault: true,
        settings: {
            targetBpm: 140,
            targetGenre: 'edm',
            energyLevel: 1.5,
            creativity: 1.0,
            preserveMelody: 0.0,
            bassBoost: 8,
            reverb: 0.5,
            sidechain: 0.85,
            quality: 'high',
            useDeepLearning: true,
            kickPattern: 'punchy',
            bassPattern: 'syncopated',
            vocalEffect: 'delay',
            masterGain: 5,
        },
    },
    {
        id: 'chill-mix',
        name: TEXTS.presets.chillMix,
        isDefault: true,
        settings: {
            targetBpm: 120,
            targetGenre: 'house',
            energyLevel: 0.7,
            creativity: 0.5,
            preserveMelody: 0.7,
            bassBoost: 3,
            reverb: 0.6,
            sidechain: 0.4,
            quality: 'high',
            useDeepLearning: true,
            kickPattern: 'standard',
            bassPattern: 'offbeat',
            vocalEffect: 'reverb',
            masterGain: 0,
        },
    },
];

export default function App() {
    const [activeTab, setActiveTab] = useState<TabType>('remix');
    const [audioFile, setAudioFile] = useState<File | null>(null);
    const [audioUrl, setAudioUrl] = useState<string>('');
    const [remixUrl, setRemixUrl] = useState<string>('');
    const [stems, setStems] = useState<Stem[]>([]);
    const [analysis, setAnalysis] = useState<AudioAnalysis | null>(null);
    const [processingStep, setProcessingStep] = useState<ProcessingStep>('idle');
    const [processingProgress, setProcessingProgress] = useState(0);
    const [error, setError] = useState<string>('');
    const [history, setHistory] = useState<RemixResult[]>([]);
    const [presets, setPresets] = useState<Preset[]>(DEFAULT_PRESETS);
    const [activePreset, setActivePreset] = useState<string>('default');
    const [showPresetMenu, setShowPresetMenu] = useState(false);
    const [compareMode, setCompareMode] = useState(false);
    const [gpuAvailable, setGpuAvailable] = useState(false);
    const [settings, setSettings] = useState<RemixSettings>(DEFAULT_PRESETS[0].settings);
    const originalAudioRef = useRef<HTMLAudioElement>(null);
    const remixAudioRef = useRef<HTMLAudioElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const API_BASE = 'http://localhost:8000/api';

    useEffect(() => {
        const saved = localStorage.getItem('remix_presets');
        if (saved) {
            const customPresets = JSON.parse(saved);
            setPresets([...DEFAULT_PRESETS, ...customPresets]);
        }
        const savedHistory = localStorage.getItem('remix_history');
        if (savedHistory) {
            setHistory(JSON.parse(savedHistory));
        }
        const nav = navigator as any;
        if (nav.gpu) {
            setGpuAvailable(true);
        }
    }, []);

    useEffect(() => {
        localStorage.setItem('remix_history', JSON.stringify(history));
    }, [history]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
            switch (e.key.toLowerCase()) {
                case ' ':
                    e.preventDefault();
                    if (remixUrl) {
                        remixAudioRef.current?.paused ? remixAudioRef.current?.play() : remixAudioRef.current?.pause();
                    } else if (audioUrl) {
                        originalAudioRef.current?.paused ? originalAudioRef.current?.play() : originalAudioRef.current?.pause();
                    }
                    break;
                case 'r':
                    if (audioFile && processingStep === 'idle') handleCreateRemix();
                    break;
                case 's':
                    if (audioFile && processingStep === 'idle') handleSeparateStems();
                    break;
                case '1': setActiveTab('remix'); break;
                case '2': setActiveTab('stems'); break;
                case '3': setActiveTab('training'); break;
                case '4': setActiveTab('history'); break;
                case '5': setActiveTab('settings'); break;
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [audioFile, audioUrl, remixUrl, processingStep]);

    const handleFileUpload = useCallback(async (file: File) => {
        setAudioFile(file);
        setAudioUrl(URL.createObjectURL(file));
        setProcessingStep('uploading');
        setError('');
        try {
            setProcessingStep('analyzing');
            setProcessingProgress(50);
            setAnalysis({ bpm: 128, key: 'C major', energy: 0.75, genre: 'Pop' });
            setProcessingProgress(100);
            setProcessingStep('idle');
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Tải lên thất bại');
            setProcessingStep('error');
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) handleFileUpload(file);
    }, [handleFileUpload]);

    const handleSeparateStems = useCallback(async () => {
        if (!audioFile) return;
        setProcessingStep('separating');
        setProcessingProgress(0);
        setError('');
        try {
            setProcessingProgress(50);
            await new Promise(r => setTimeout(r, 2000));
            setStems([
                { name: TEXTS.stems.vocals, url: '', volume: 1, muted: false, solo: false },
                { name: TEXTS.stems.drums, url: '', volume: 1, muted: false, solo: false },
                { name: TEXTS.stems.bass, url: '', volume: 1, muted: false, solo: false },
                { name: TEXTS.stems.other, url: '', volume: 1, muted: false, solo: false },
            ]);
            setProcessingStep('complete');
            setProcessingProgress(100);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Tách nhạc thất bại');
            setProcessingStep('error');
        }
    }, [audioFile]);

    const handleCreateRemix = useCallback(async () => {
        if (!audioFile) return;
        setProcessingStep('generating');
        setProcessingProgress(0);
        setError('');
        try {
            setProcessingProgress(30);
            await new Promise(r => setTimeout(r, 1000));
            setProcessingStep('mastering');
            setProcessingProgress(70);
            await new Promise(r => setTimeout(r, 1000));
            setRemixUrl(audioUrl);
            const newResult: RemixResult = {
                id: Date.now().toString(),
                input_file: audioFile.name,
                output_file: 'remix_output.wav',
                settings: { ...settings },
                created_at: new Date().toISOString(),
            };
            setHistory(prev => [newResult, ...prev]);
            setProcessingStep('complete');
            setProcessingProgress(100);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Tạo remix thất bại');
            setProcessingStep('error');
        }
    }, [audioFile, audioUrl, settings]);

    const handleExport = useCallback((format: 'mp3' | 'wav') => {
        if (!remixUrl) return;
        const link = document.createElement('a');
        link.href = remixUrl;
        link.download = `remix.${format}`;
        link.click();
    }, [remixUrl]);

    const handleExportStems = useCallback(() => {
        alert('Tính năng xuất từng track đang được phát triển!');
    }, []);

    const loadPreset = useCallback((presetId: string) => {
        const preset = presets.find(p => p.id === presetId);
        if (preset) {
            setSettings(preset.settings);
            setActivePreset(presetId);
            setShowPresetMenu(false);
        }
    }, [presets]);

    const savePreset = useCallback(() => {
        const name = prompt('Nhập tên preset:');
        if (!name) return;
        const newPreset: Preset = {
            id: `custom-${Date.now()}`,
            name: TEXTS.presets.customPrefix + name,
            isDefault: false,
            settings: { ...settings },
        };
        const customPresets = presets.filter(p => !p.isDefault);
        const allPresets = [...DEFAULT_PRESETS, ...customPresets, newPreset];
        setPresets(allPresets);
        const toSave = allPresets.filter(p => !p.isDefault);
        localStorage.setItem('remix_presets', JSON.stringify(toSave));
        setActivePreset(newPreset.id);
        setShowPresetMenu(false);
    }, [presets, settings]);

    const getStepText = (step: ProcessingStep): string => {
        const stepMap: Record<string, string> = {
            uploading: TEXTS.steps.uploading,
            analyzing: TEXTS.steps.analyzing,
            separating: TEXTS.steps.separating,
            generating: TEXTS.steps.generating,
            mastering: TEXTS.steps.mastering,
            complete: TEXTS.steps.complete,
            error: TEXTS.steps.error,
            idle: '',
        };
        return stepMap[step] || step;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 text-white">
            <header className="bg-black/30 backdrop-blur-lg border-b border-white/10">
                <div className="max-w-7xl mx-auto px-4 py-3">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-xl animate-pulse">
                                🎵
                            </div>
                            <div>
                                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                                    {TEXTS.appName}
                                </h1>
                                <p className="text-xs text-gray-400">{TEXTS.appSubtitle}</p>
                            </div>
                            {gpuAvailable && (
                                <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">
                                    GPU ✓
                                </span>
                            )}
                        </div>
                        <nav className="flex gap-1">
                            {[
                                { id: 'remix', label: TEXTS.tabs.remix },
                                { id: 'stems', label: TEXTS.tabs.stems },
                                { id: 'training', label: TEXTS.tabs.training },
                                { id: 'longaudio', label: '⏱️ Long Audio' },
                                { id: 'watermark', label: '🔐 Watermark' },
                                { id: 'history', label: TEXTS.tabs.history },
                                { id: 'settings', label: TEXTS.tabs.settings },
                            ].map(tab => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id as TabType)}
                                    className={`px-4 py-2 rounded-lg transition-all ${activeTab === tab.id
                                        ? 'bg-purple-600 text-white'
                                        : 'text-gray-400 hover:text-white hover:bg-white/10'
                                        }`}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </nav>
                        <div className="flex items-center gap-3">
                            <UpdateButton />
                            <div className="text-xs text-gray-500">{TEXTS.footer.help}</div>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-6 pb-20">
                {error && (
                    <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg flex items-center justify-between">
                        <span className="text-red-400">{error}</span>
                        <button onClick={() => setError('')} className="text-red-400 hover:text-red-300">✕</button>
                    </div>
                )}

                {processingStep !== 'idle' && processingStep !== 'complete' && (
                    <div className="mb-4 p-4 bg-purple-500/20 border border-purple-500 rounded-lg">
                        <div className="flex items-center gap-3">
                            <div className="animate-spin w-5 h-5 border-2 border-purple-400 border-t-transparent rounded-full"></div>
                            <span className="text-purple-300 capitalize">{getStepText(processingStep)}...</span>
                            <span className="text-purple-400">{processingProgress}%</span>
                        </div>
                        <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all" style={{ width: `${processingProgress}%` }}></div>
                        </div>
                    </div>
                )}

                {activeTab === 'remix' && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-2 space-y-6">
                            <div
                                className="bg-black/30 rounded-xl p-8 border-2 border-dashed border-gray-600 hover:border-purple-500 transition cursor-pointer"
                                onDrop={handleDrop}
                                onDragOver={(e) => e.preventDefault()}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept=".mp3,.wav,.flac,audio/*"
                                    onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                                    className="hidden"
                                />
                                <div className="text-center">
                                    <div className="mx-auto w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center mb-4">
                                        <span className="text-3xl">📤</span>
                                    </div>
                                    <h3 className="text-xl font-semibold mb-2">{audioFile ? audioFile.name : TEXTS.upload.title}</h3>
                                    <p className="text-gray-400">{TEXTS.upload.orClick}</p>
                                    <p className="text-gray-500 text-sm mt-2">{TEXTS.upload.supports}</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {audioUrl && (
                                    <div className={`bg-black/30 rounded-xl p-4 border ${compareMode ? 'border-blue-500/50' : 'border-white/10'}`}>
                                        <h3 className="text-lg font-semibold mb-3">{TEXTS.audio.original}</h3>
                                        <audio ref={originalAudioRef} src={audioUrl} controls className="w-full" />
                                    </div>
                                )}
                                {remixUrl && (
                                    <div className={`bg-black/30 rounded-xl p-4 border ${compareMode ? 'border-purple-500/50' : 'border-purple-500/30'}`}>
                                        <h3 className="text-lg font-semibold mb-3">{TEXTS.audio.remix}</h3>
                                        <audio ref={remixAudioRef} src={remixUrl} controls className="w-full" />
                                    </div>
                                )}
                            </div>

                            {remixUrl && (
                                <div className="flex gap-2 flex-wrap">
                                    <button onClick={() => handleExport('mp3')} className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded-lg transition">{TEXTS.audio.exportMp3}</button>
                                    <button onClick={() => handleExport('wav')} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg transition">{TEXTS.audio.exportWav}</button>
                                    <button onClick={handleExportStems} className="px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded-lg transition">{TEXTS.audio.exportStems}</button>
                                    <button onClick={() => setCompareMode(!compareMode)} className={`px-4 py-2 rounded-lg transition ${compareMode ? 'bg-yellow-600' : 'bg-gray-600 hover:bg-gray-500'}`}>{TEXTS.audio.compare}</button>
                                </div>
                            )}
                        </div>

                        <div className="space-y-4">
                            <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                                <div className="flex items-center justify-between mb-3">
                                    <h3 className="text-lg font-semibold">{TEXTS.presets.title}</h3>
                                    <button onClick={savePreset} className="text-xs px-2 py-1 bg-purple-600/30 hover:bg-purple-600/50 rounded transition">{TEXTS.presets.save}</button>
                                </div>
                                <div className="relative">
                                    <button onClick={() => setShowPresetMenu(!showPresetMenu)} className="w-full px-3 py-2 bg-gray-800 rounded-lg border border-white/10 text-left flex items-center justify-between">
                                        <span>{presets.find(p => p.id === activePreset)?.name || TEXTS.presets.default}</span>
                                        <span>▼</span>
                                    </button>
                                    {showPresetMenu && (
                                        <div className="absolute top-full left-0 right-0 mt-1 bg-gray-800 rounded-lg border border-white/10 overflow-hidden z-10">
                                            {presets.map(preset => (
                                                <button key={preset.id} onClick={() => loadPreset(preset.id)} className={`w-full px-3 py-2 text-left hover:bg-purple-600/30 transition ${preset.id === activePreset ? 'bg-purple-600/50' : ''}`}>{preset.name}</button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {analysis && (
                                <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                                    <h3 className="text-lg font-semibold mb-3">{TEXTS.analysis.title}</h3>
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                        <span className="text-gray-400">{TEXTS.analysis.bpm}</span><span className="font-mono">{analysis.bpm}</span>
                                        <span className="text-gray-400">{TEXTS.analysis.key}</span><span>{analysis.key}</span>
                                        <span className="text-gray-400">{TEXTS.analysis.energy}</span><span>{Math.round(analysis.energy * 100)}%</span>
                                        <span className="text-gray-400">{TEXTS.analysis.genre}</span><span>{analysis.genre}</span>
                                    </div>
                                </div>
                            )}

                            <div className="bg-black/30 rounded-xl p-4 border border-white/10 space-y-3">
                                <h3 className="text-lg font-semibold">{TEXTS.controls.title}</h3>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.targetGenre}</label>
                                    <select value={settings.targetGenre} onChange={e => setSettings(s => ({ ...s, targetGenre: e.target.value }))} className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10">
                                        <option value="vinahouse">{TEXTS.genres.vinahouse}</option>
                                        <option value="vinahouseBounce">{TEXTS.genres.vinahouseBounce}</option>
                                        <option value="house">{TEXTS.genres.house}</option>
                                        <option value="edm">{TEXTS.genres.edm}</option>
                                        <option value="hiphop">{TEXTS.genres.hiphop}</option>
                                        <option value="trap">{TEXTS.genres.trap}</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.targetBpm} <span className="text-purple-400 font-mono">{settings.targetBpm}</span></label>
                                    <input type="range" min="100" max="180" value={settings.targetBpm} onChange={e => setSettings(s => ({ ...s, targetBpm: parseInt(e.target.value) }))} className="w-full accent-purple-500" />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.creativity} <span className="text-purple-400">{Math.round(settings.creativity * 100)}%</span></label>
                                    <input type="range" min="0" max="100" value={settings.creativity * 100} onChange={e => setSettings(s => ({ ...s, creativity: parseInt(e.target.value) / 100 }))} className="w-full accent-purple-500" />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.bassBoost} <span className="text-purple-400">+{settings.bassBoost}dB</span></label>
                                    <input type="range" min="0" max="12" value={settings.bassBoost} onChange={e => setSettings(s => ({ ...s, bassBoost: parseInt(e.target.value) }))} className="w-full accent-purple-500" />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.sidechain} <span className="text-purple-400">{Math.round(settings.sidechain * 100)}%</span></label>
                                    <input type="range" min="0" max="100" value={settings.sidechain * 100} onChange={e => setSettings(s => ({ ...s, sidechain: parseInt(e.target.value) / 100 }))} className="w-full accent-purple-500" />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.reverb} <span className="text-purple-400">{Math.round(settings.reverb * 100)}%</span></label>
                                    <input type="range" min="0" max="100" value={settings.reverb * 100} onChange={e => setSettings(s => ({ ...s, reverb: parseInt(e.target.value) / 100 }))} className="w-full accent-purple-500" />
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.kickPattern}</label>
                                    <select value={settings.kickPattern} onChange={e => setSettings(s => ({ ...s, kickPattern: e.target.value }))} className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10 text-sm">
                                        <option value="standard">{TEXTS.patterns.standard}</option>
                                        <option value="offbeat">{TEXTS.patterns.offbeat}</option>
                                        <option value="syncopated">{TEXTS.patterns.syncopated}</option>
                                        <option value="rolling">{TEXTS.patterns.rolling}</option>
                                        <option value="punchy">{TEXTS.patterns.punchy}</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm text-gray-400 mb-1">{TEXTS.controls.vocalEffect}</label>
                                    <select value={settings.vocalEffect} onChange={e => setSettings(s => ({ ...s, vocalEffect: e.target.value }))} className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10 text-sm">
                                        <option value="none">{TEXTS.vocalEffects.none}</option>
                                        <option value="autoTune">{TEXTS.vocalEffects.autoTune}</option>
                                        <option value="pitchShift">{TEXTS.vocalEffects.pitchShift}</option>
                                        <option value="reverb">{TEXTS.vocalEffects.reverb}</option>
                                        <option value="delay">{TEXTS.vocalEffects.delay}</option>
                                        <option value="distortion">{TEXTS.vocalEffects.distortion}</option>
                                    </select>
                                </div>
                                <div className="space-y-2 pt-4">
                                    <button onClick={handleSeparateStems} disabled={!audioFile || processingStep !== 'idle'} className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition font-semibold">{TEXTS.controls.separateStems}</button>
                                    <button onClick={handleCreateRemix} disabled={!audioFile || processingStep !== 'idle'} className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed rounded-lg transition font-semibold">{TEXTS.controls.createRemix}</button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'stems' && (
                    <div className="space-y-6">
                        <div className="flex items-center justify-between">
                            <h2 className="text-2xl font-bold">{TEXTS.stems.title}</h2>
                            <div className="flex gap-2">
                                <button onClick={handleSeparateStems} disabled={!audioFile || processingStep !== 'idle'} className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg transition">{TEXTS.stems.separate}</button>
                                {stems.length > 0 && <button onClick={handleExportStems} className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded-lg transition">{TEXTS.stems.exportAll}</button>}
                            </div>
                        </div>
                        {stems.length > 0 ? (
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {stems.map((stem, i) => (
                                    <div key={i} className="bg-black/30 rounded-xl p-4 border border-white/10">
                                        <div className="flex items-center justify-between mb-3">
                                            <h4 className="font-semibold text-lg">{stem.name}</h4>
                                            <div className="flex gap-2">
                                                <button className={`px-3 py-1 rounded text-sm ${stem.muted ? 'bg-red-600' : 'bg-gray-700'}`} onClick={() => setStems(stems.map((s, idx) => idx === i ? { ...s, muted: !s.muted } : s))}>M</button>
                                                <button className={`px-3 py-1 rounded text-sm ${stem.solo ? 'bg-yellow-600' : 'bg-gray-700'}`} onClick={() => setStems(stems.map((s, idx) => idx === i ? { ...s, solo: !s.solo } : s))}>S</button>
                                            </div>
                                        </div>
                                        <input type="range" min="0" max="100" value={stem.volume * 100} onChange={(e) => setStems(stems.map((s, idx) => idx === i ? { ...s, volume: parseInt(e.target.value) / 100 } : s))} className="w-full accent-purple-500" />
                                        <div className="mt-2 text-sm text-gray-400 text-center">{Math.round(stem.volume * 100)}%</div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-20 text-gray-400">
                                <p className="text-6xl mb-4">🎧</p>
                                <p>{TEXTS.stems.uploadFirst}</p>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'training' && (
                    <div className="space-y-6">
                        {/* Fast Training with Model Marketplace */}
                        <div className="bg-black/20 rounded-xl p-4 border border-purple-500/30">
                            <h2 className="text-xl font-bold mb-4">🚀 Fast Training - Kho Model AI</h2>
                            <FastTrainingPanel />
                        </div>

                        {/* Advanced AI Training (DiT) */}
                        <div className="bg-black/20 rounded-xl p-4 border border-blue-500/30">
                            <h2 className="text-xl font-bold mb-4">🧠 AI Training (DiT) - Nâng Cao</h2>
                            <TrainingPanel />
                        </div>
                    </div>
                )}
                {activeTab === 'longaudio' && <div className="max-w-2xl mx-auto"><LongAudioPanel /></div>}
                {activeTab === 'watermark' && <div className="max-w-2xl mx-auto"><WatermarkPanel /></div>}

                {activeTab === 'history' && (
                    <div className="space-y-6">
                        <div className="flex items-center justify-between">
                            <h2 className="text-2xl font-bold">{TEXTS.history.title}</h2>
                            {history.length > 0 && (
                                <button onClick={() => { if (confirm('Xóa tất cả lịch sử?')) { setHistory([]); localStorage.removeItem('remix_history'); } }} className="px-4 py-2 bg-red-600/20 text-red-400 hover:bg-red-600/30 rounded-lg transition">{TEXTS.history.clearAll}</button>
                            )}
                        </div>
                        {history.length > 0 ? (
                            <div className="space-y-3">
                                {history.map((item) => (
                                    <div key={item.id} className="bg-black/30 rounded-xl p-4 border border-white/10 flex items-center justify-between">
                                        <div>
                                            <h4 className="font-semibold">{item.input_file}</h4>
                                            <p className="text-sm text-gray-400">{item.created_at}</p>
                                            <p className="text-xs text-purple-400 mt-1">{item.settings.targetGenre} • {item.settings.targetBpm} BPM</p>
                                        </div>
                                        <div className="flex gap-2">
                                            <button className="px-3 py-1 bg-green-600/20 text-green-400 hover:bg-green-600/30 rounded transition">{TEXTS.history.download}</button>
                                            <button onClick={() => setHistory(prev => prev.filter(r => r.id !== item.id))} className="px-3 py-1 bg-red-600/20 text-red-400 hover:bg-red-600/30 rounded transition">{TEXTS.history.delete}</button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-20 text-gray-400">
                                <p className="text-6xl mb-4">📜</p>
                                <p>{TEXTS.history.noHistory}</p>
                            </div>
                        )}
                    </div>
                )}

                {activeTab === 'settings' && (
                    <div className="max-w-2xl mx-auto space-y-6">
                        <h2 className="text-2xl font-bold">{TEXTS.settings.title}</h2>
                        <div className="bg-black/30 rounded-xl p-4 border border-white/10 space-y-4">
                            <h3 className="text-lg font-semibold">{TEXTS.settings.modelSettings}</h3>
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">{TEXTS.settings.checkpointPath}</label>
                                <input type="text" placeholder={TEXTS.settings.checkpointPlaceholder} className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10" />
                            </div>
                        </div>
                        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                            <h3 className="text-lg font-semibold mb-3">{TEXTS.settings.systemInfo}</h3>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                                <span className="text-gray-400">{TEXTS.settings.apiServer}</span><span className="text-green-400">{API_BASE}</span>
                                <span className="text-gray-400">{TEXTS.settings.status}</span><span className="text-green-400">{TEXTS.settings.ready}</span>
                                <span className="text-gray-400">{TEXTS.settings.gpu}</span><span className={gpuAvailable ? 'text-green-400' : 'text-yellow-400'}>{gpuAvailable ? TEXTS.settings.gpuAvailable : TEXTS.settings.gpuNotAvailable}</span>
                            </div>
                        </div>
                        <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                            <h3 className="text-lg font-semibold mb-3">{TEXTS.settings.keyboard}</h3>
                            <div className="space-y-2 text-sm">
                                <p className="text-gray-400"><kbd className="bg-gray-700 px-2 py-0.5 rounded">Phím cách</kbd> {TEXTS.settings.shortcuts.space}</p>
                                <p className="text-gray-400"><kbd className="bg-gray-700 px-2 py-0.5 rounded">R</kbd> {TEXTS.settings.shortcuts.r}</p>
                                <p className="text-gray-400"><kbd className="bg-gray-700 px-2 py-0.5 rounded">S</kbd> {TEXTS.settings.shortcuts.s}</p>
                                <p className="text-gray-400"><kbd className="bg-gray-700 px-2 py-0.5 rounded">E</kbd> {TEXTS.settings.shortcuts.e}</p>
                                <p className="text-gray-400"><kbd className="bg-gray-700 px-2 py-0.5 rounded">1-5</kbd> {TEXTS.settings.shortcuts.one}</p>
                            </div>
                        </div>
                        <button onClick={() => { localStorage.clear(); setHistory([]); setPresets(DEFAULT_PRESETS); setActivePreset('default'); setSettings(DEFAULT_PRESETS[0].settings); }} className="w-full py-3 bg-red-600/20 text-red-400 border border-red-500/30 hover:bg-red-600/30 rounded-lg transition">{TEXTS.settings.reset}</button>
                    </div>
                )}
            </main>

            <footer className="fixed bottom-0 left-0 right-0 bg-black/50 backdrop-blur-lg border-t border-white/10 py-2 px-4">
                <div className="max-w-7xl mx-auto flex items-center justify-between text-sm text-gray-400">
                    <span>{TEXTS.footer.version}</span>
                    <span className={processingStep !== 'idle' ? 'text-yellow-400' : 'text-green-400'}>
                        {processingStep !== 'idle' ? `${TEXTS.footer.processing} ${getStepText(processingStep)}` : TEXTS.footer.ready}
                    </span>
                </div>
            </footer>
        </div>
    );
}