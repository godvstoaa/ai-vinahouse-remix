import React, { useState, useEffect, useRef, useCallback } from 'react';

interface TrackInfo {
    start: number;
    end: number;
    duration: number;
    key: string;
    segment_type: string;
    confidence: number;
}

interface ChunkInfo {
    chunk_id: string;
    source_file: string;
    start_time: number;
    end_time: number;
    duration: number;
    bpm: number;
    key: string;
    energy: number;
    track_index: number;
}

interface FileProgress {
    file_path: string;
    file_name: string;
    status: string;
    progress: number;
    message: string;
    chunks_created: number;
    error?: string;
}

interface FolderProgress {
    total_files: number;
    completed_files: number;
    current_file: string | null;
    files: FileProgress[];
    total_chunks: number;
    elapsed_seconds: number;
}

interface ProcessedMetadata {
    file_name: string;
    duration: number;
    num_tracks: number;
    num_chunks: number;
    tracks: TrackInfo[];
    chunks: ChunkInfo[];
}

const LongAudioPanel: React.FC = () => {
    const [folderPath, setFolderPath] = useState('D:\\Music\\Vinahouse');
    const [files, setFiles] = useState<any[]>([]);
    const [processing, setProcessing] = useState(false);
    const [folderProgress, setFolderProgress] = useState<FolderProgress | null>(null);
    const [selectedFile, setSelectedFile] = useState<ProcessedMetadata | null>(null);
    const [activeTab, setActiveTab] = useState<'files' | 'tracks' | 'chunks'>('files');
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const waveformRef = useRef<HTMLCanvasElement>(null);
    const progressInterval = useRef<NodeJS.Timeout | null>(null);

    // Fetch folder progress
    const fetchProgress = useCallback(async () => {
        try {
            const res = await fetch('http://localhost:8002/api/long-audio/folder-progress');
            const data = await res.json();
            if (data.progress) {
                setFolderProgress(data.progress);
            }
        } catch (e) {
            console.error('Failed to fetch progress:', e);
        }
    }, []);

    // Start progress polling when processing
    useEffect(() => {
        if (processing) {
            progressInterval.current = setInterval(fetchProgress, 500);
        } else {
            if (progressInterval.current) {
                clearInterval(progressInterval.current);
            }
        }
        return () => {
            if (progressInterval.current) {
                clearInterval(progressInterval.current);
            }
        };
    }, [processing, fetchProgress]);

    // Scan folder
    const scanFolder = async () => {
        setError(null);
        setLoading(true);
        try {
            const encodedPath = encodeURIComponent(folderPath);
            const res = await fetch(`http://localhost:8002/api/long-audio/scan-folder?folder_path=${encodedPath}&recursive=true`);
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || `Server error: ${res.status}`);
            }
            setFiles(data.files || []);
            if (data.files?.length === 0) {
                setError('No audio files found in this folder');
            }
        } catch (e: any) {
            console.error('Scan failed:', e);
            setError(`Scan failed: ${e.message || 'Cannot connect to server'}`);
        } finally {
            setLoading(false);
        }
    };

    // Process folder
    const processFolder = async () => {
        setError(null);
        setProcessing(true);
        setFolderProgress(null);
        try {
            const res = await fetch('http://localhost:8002/api/long-audio/process-folder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    folder_path: folderPath,
                    recursive: true,
                    force: false,
                    skip_cached: true
                })
            });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || `Server error: ${res.status}`);
            }
            console.log('Process result:', data);
        } catch (e: any) {
            console.error('Process failed:', e);
            setError(`Process failed: ${e.message || 'Cannot connect to server'}`);
            setProcessing(false);
        }
    };

    // Load metadata for visualization
    const loadFileMetadata = async (filePath: string) => {
        try {
            const res = await fetch(`http://localhost:8002/api/long-audio/metadata?file_path=${encodeURIComponent(filePath)}`);
            const data = await res.json();
            setSelectedFile(data);
            drawWaveform(data);
        } catch (e) {
            console.error('Failed to load metadata:', e);
        }
    };

    // Draw waveform with track markers
    const drawWaveform = (meta: ProcessedMetadata) => {
        const canvas = waveformRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;
        const duration = meta.duration;

        // Clear
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);

        // Draw track segments
        const colors = ['#e94560', '#0f3460', '#16213e', '#533483', '#2c003e', '#004e92'];

        meta.tracks.forEach((track, idx) => {
            const startX = (track.start / duration) * width;
            const endX = (track.end / duration) * width;
            const segWidth = endX - startX;

            // Segment background
            ctx.fillStyle = colors[idx % colors.length] + '40';
            ctx.fillRect(startX, 0, segWidth, height);

            // Segment border
            ctx.strokeStyle = colors[idx % colors.length];
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(startX, 0);
            ctx.lineTo(startX, height);
            ctx.stroke();

            // Track label
            ctx.fillStyle = '#fff';
            ctx.font = '10px Arial';
            const label = `T${idx + 1} [${track.key}] ${track.segment_type}`;
            ctx.fillText(label, startX + 4, 15);

            // Time label
            ctx.fillStyle = '#888';
            ctx.font = '9px Arial';
            ctx.fillText(`${formatTime(track.start)}`, startX + 4, height - 5);
        });

        // Draw end line
        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(width - 1, 0);
        ctx.lineTo(width - 1, height);
        ctx.stroke();
    };

    // Cancel processing
    const cancelProcessing = async () => {
        try {
            await fetch('http://localhost:8002/api/long-audio/cancel', { method: 'POST' });
        } catch (e) {
            console.error('Cancel failed:', e);
        }
    };

    // Format time
    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Get status color
    const getStatusColor = (status: string): string => {
        switch (status) {
            case 'completed': return '#22c55e';
            case 'processing': return '#eab308';
            case 'error': return '#ef4444';
            case 'cached': return '#3b82f6';
            default: return '#6b7280';
        }
    };

    return (
        <div className="long-audio-panel" style={{ padding: '20px', color: '#fff' }}>
            <h2 style={{ marginBottom: '20px', color: '#e94560' }}>
                🎵 Smart DJ Set Processor
                <span style={{ fontSize: '14px', color: '#888', marginLeft: '10px' }}>
                    Content-Aware Track Detection
                </span>
            </h2>

            {/* Error Display */}
            {error && (
                <div style={{
                    background: '#ef4444',
                    color: '#fff',
                    padding: '15px',
                    borderRadius: '8px',
                    marginBottom: '20px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                }}>
                    <span>⚠️ {error}</span>
                    <button
                        onClick={() => setError(null)}
                        style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '18px' }}
                    >✕</button>
                </div>
            )}

            {/* Folder Input */}
            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                <input
                    type="text"
                    value={folderPath}
                    onChange={(e) => setFolderPath(e.target.value)}
                    placeholder="Folder path..."
                    style={{
                        flex: 1,
                        padding: '10px',
                        background: '#16213e',
                        border: '1px solid #0f3460',
                        borderRadius: '4px',
                        color: '#fff'
                    }}
                />
                <button
                    onClick={scanFolder}
                    disabled={processing || loading}
                    style={{
                        padding: '10px 20px',
                        background: loading ? '#333' : '#0f3460',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#fff',
                        cursor: (processing || loading) ? 'not-allowed' : 'pointer'
                    }}
                >
                    {loading ? '⏳ Scanning...' : '📂 Scan'}
                </button>
                <button
                    onClick={processFolder}
                    disabled={processing || files.length === 0}
                    style={{
                        padding: '10px 20px',
                        background: processing ? '#333' : '#e94560',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#fff',
                        cursor: processing ? 'not-allowed' : 'pointer'
                    }}
                >
                    {processing ? '⏳ Processing...' : '🚀 Process All'}
                </button>
                {processing && (
                    <button
                        onClick={cancelProcessing}
                        style={{
                            padding: '10px 20px',
                            background: '#ef4444',
                            border: 'none',
                            borderRadius: '4px',
                            color: '#fff',
                            cursor: 'pointer'
                        }}
                    >
                        ✋ Cancel
                    </button>
                )}
            </div>

            {/* Progress Bar */}
            {folderProgress && (
                <div style={{
                    background: '#16213e',
                    padding: '15px',
                    borderRadius: '8px',
                    marginBottom: '20px'
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                        <span>Progress: {folderProgress.completed_files}/{folderProgress.total_files} files</span>
                        <span>Chunks: {folderProgress.total_chunks}</span>
                        <span>Time: {Math.floor(folderProgress.elapsed_seconds)}s</span>
                    </div>
                    <div style={{
                        background: '#0f3460',
                        borderRadius: '4px',
                        height: '8px',
                        overflow: 'hidden'
                    }}>
                        <div style={{
                            background: '#e94560',
                            height: '100%',
                            width: `${(folderProgress.completed_files / folderProgress.total_files) * 100}%`,
                            transition: 'width 0.3s'
                        }} />
                    </div>
                </div>
            )}

            {/* Main Content */}
            <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px' }}>
                {/* File List */}
                <div style={{
                    background: '#16213e',
                    borderRadius: '8px',
                    padding: '15px',
                    maxHeight: '400px',
                    overflowY: 'auto'
                }}>
                    <h3 style={{ marginBottom: '10px' }}>Files ({files.length})</h3>
                    {files.map((file, idx) => (
                        <div
                            key={idx}
                            onClick={() => !processing && loadFileMetadata(file.path)}
                            style={{
                                padding: '8px',
                                background: file.cached ? '#0f346040' : '#1a1a2e',
                                borderRadius: '4px',
                                marginBottom: '5px',
                                cursor: processing ? 'default' : 'pointer',
                                borderLeft: `3px solid ${getStatusColor(file.cached ? 'cached' : 'pending')}`
                            }}
                        >
                            <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{file.name}</div>
                            <div style={{ fontSize: '10px', color: '#888' }}>
                                {file.size_mb}MB • {file.estimated_duration_min}min
                                {file.cached && ' ✓'}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Visualization & Details */}
                <div style={{ background: '#16213e', borderRadius: '8px', padding: '15px' }}>
                    {selectedFile ? (
                        <>
                            <h3 style={{ marginBottom: '10px' }}>
                                {selectedFile.file_name}
                                <span style={{ fontSize: '12px', color: '#888', marginLeft: '10px' }}>
                                    {formatTime(selectedFile.duration)} • {selectedFile.num_tracks} tracks • {selectedFile.num_chunks} chunks
                                </span>
                            </h3>

                            {/* Waveform with Track Markers */}
                            <canvas
                                ref={waveformRef}
                                width={800}
                                height={100}
                                style={{
                                    width: '100%',
                                    height: '100px',
                                    background: '#1a1a2e',
                                    borderRadius: '4px',
                                    marginBottom: '15px'
                                }}
                            />

                            {/* Tabs */}
                            <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
                                {(['files', 'tracks', 'chunks'] as const).map(tab => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        style={{
                                            padding: '8px 16px',
                                            background: activeTab === tab ? '#e94560' : '#0f3460',
                                            border: 'none',
                                            borderRadius: '4px',
                                            color: '#fff',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                        {tab === 'tracks' && ` (${selectedFile.num_tracks})`}
                                        {tab === 'chunks' && ` (${selectedFile.num_chunks})`}
                                    </button>
                                ))}
                            </div>

                            {/* Track List */}
                            {activeTab === 'tracks' && (
                                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                    <table style={{ width: '100%', fontSize: '12px' }}>
                                        <thead>
                                            <tr style={{ color: '#888', textAlign: 'left' }}>
                                                <th>#</th>
                                                <th>Start</th>
                                                <th>End</th>
                                                <th>Duration</th>
                                                <th>Key</th>
                                                <th>Type</th>
                                                <th>Confidence</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {selectedFile.tracks.map((track, idx) => (
                                                <tr key={idx} style={{ borderBottom: '1px solid #0f3460' }}>
                                                    <td>{idx + 1}</td>
                                                    <td>{formatTime(track.start)}</td>
                                                    <td>{formatTime(track.end)}</td>
                                                    <td>{formatTime(track.duration)}</td>
                                                    <td style={{ color: '#e94560' }}>{track.key}</td>
                                                    <td style={{
                                                        color: track.segment_type === 'drop' ? '#22c55e' :
                                                            track.segment_type === 'buildup' ? '#eab308' : '#fff'
                                                    }}>
                                                        {track.segment_type}
                                                    </td>
                                                    <td>{(track.confidence * 100).toFixed(0)}%</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}

                            {/* Chunk List */}
                            {activeTab === 'chunks' && (
                                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '5px' }}>
                                        {selectedFile.chunks.slice(0, 50).map((chunk, idx) => (
                                            <div
                                                key={idx}
                                                style={{
                                                    background: '#0f3460',
                                                    padding: '8px',
                                                    borderRadius: '4px',
                                                    fontSize: '10px'
                                                }}
                                            >
                                                <div style={{ fontWeight: 'bold' }}>T{chunk.track_index + 1} • {chunk.bpm.toFixed(0)} BPM</div>
                                                <div style={{ color: '#888' }}>{formatTime(chunk.start_time)} - {formatTime(chunk.end_time)}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </>
                    ) : (
                        <div style={{ textAlign: 'center', color: '#888', padding: '50px' }}>
                            <p>Select a file to view track segmentation</p>
                            <p style={{ fontSize: '12px' }}>Smart detection uses: Energy + Spectral Flux + Key Change</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LongAudioPanel;