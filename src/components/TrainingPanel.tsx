import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface TrainingStatus {
    is_training: boolean;
    progress: number;
    total_songs: number;
    processed_songs: number;
    current_song: string | null;
    results: any;
}

export const TrainingPanel: React.FC = () => {
    const [datasetPath, setDatasetPath] = useState('');
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [isStarting, setIsStarting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Poll training status
    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const response = await fetch('/api/train/status');
                const data = await response.json();
                setStatus(data);
            } catch (e) {
                console.error('Failed to fetch training status:', e);
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 2000);
        return () => clearInterval(interval);
    }, []);

    const handleStartTraining = async () => {
        if (!datasetPath.trim()) {
            setError('Please enter dataset path');
            return;
        }

        setIsStarting(true);
        setError(null);

        try {
            const response = await fetch('/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_path: datasetPath,
                    output_model_path: 'models',
                    segment_duration: 30.0
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to start training');
            }

            console.log('Training started:', data);
        } catch (e: any) {
            setError(e.message);
        } finally {
            setIsStarting(false);
        }
    };

    const handleStopTraining = async () => {
        try {
            await fetch('/api/train/stop', { method: 'POST' });
        } catch (e) {
            console.error('Failed to stop training:', e);
        }
    };

    return (
        <div className="bg-gradient-to-br from-purple-900/50 to-indigo-900/50 rounded-2xl p-6 border border-purple-500/30">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                <span className="text-3xl">🎓</span>
                Model Training
            </h2>

            {/* Dataset Path Input */}
            <div className="mb-6">
                <label className="block text-purple-200 text-sm mb-2">
                    Music Library Path
                </label>
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={datasetPath}
                        onChange={(e) => setDatasetPath(e.target.value)}
                        placeholder="e.g., C:\Music or /home/user/music"
                        className="flex-1 bg-black/30 border border-purple-500/50 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-400"
                        disabled={status?.is_training}
                    />
                    <button
                        onClick={handleStartTraining}
                        disabled={status?.is_training || isStarting}
                        className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold rounded-lg hover:from-purple-500 hover:to-pink-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isStarting ? 'Starting...' : 'Start Training'}
                    </button>
                </div>
                {error && (
                    <p className="text-red-400 text-sm mt-2">{error}</p>
                )}
            </div>

            {/* Training Progress */}
            <AnimatePresence>
                {status && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="space-y-4"
                    >
                        {/* Status Badge */}
                        <div className="flex items-center justify-between">
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${status.is_training
                                ? 'bg-green-500/20 text-green-400 animate-pulse'
                                : status.results?.status === 'completed'
                                    ? 'bg-blue-500/20 text-blue-400'
                                    : 'bg-gray-500/20 text-gray-400'
                                }`}>
                                {status.is_training
                                    ? '🔄 Training...'
                                    : status.results?.status === 'completed'
                                        ? '✅ Completed'
                                        : '⏸️ Idle'}
                            </span>

                            {status.is_training && (
                                <button
                                    onClick={handleStopTraining}
                                    className="px-4 py-1 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors text-sm"
                                >
                                    Stop
                                </button>
                            )}
                        </div>

                        {/* Progress Bar */}
                        {status.is_training && (
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm text-purple-200">
                                    <span>Progress</span>
                                    <span>{status.processed_songs} / {status.total_songs} songs</span>
                                </div>
                                <div className="h-3 bg-black/30 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${status.progress}%` }}
                                        transition={{ duration: 0.5 }}
                                    />
                                </div>
                                <div className="text-center text-purple-300 font-bold">
                                    {status.progress.toFixed(1)}%
                                </div>
                            </div>
                        )}

                        {/* Current Song */}
                        {status.is_training && status.current_song && (
                            <div className="bg-black/20 rounded-lg p-3">
                                <p className="text-xs text-purple-400 mb-1">Currently Processing:</p>
                                <p className="text-white text-sm truncate">{status.current_song}</p>
                            </div>
                        )}

                        {/* Results */}
                        {status.results && (
                            <div className="bg-black/20 rounded-lg p-4 space-y-3">
                                <h3 className="text-lg font-bold text-white">Training Results</h3>

                                {status.results.status === 'completed' ? (
                                    <div className="space-y-2">
                                        <div className="grid grid-cols-2 gap-4">
                                            <div className="bg-purple-500/20 rounded-lg p-3">
                                                <p className="text-purple-300 text-xs">Songs Processed</p>
                                                <p className="text-2xl font-bold text-white">{status.results.songs_processed}</p>
                                            </div>
                                            <div className="bg-pink-500/20 rounded-lg p-3">
                                                <p className="text-pink-300 text-xs">Genres Learned</p>
                                                <p className="text-2xl font-bold text-white">{status.results.genres_learned}</p>
                                            </div>
                                        </div>

                                        {status.results.genre_profiles && (
                                            <div className="mt-4">
                                                <p className="text-purple-300 text-sm mb-2">Learned Genres:</p>
                                                <div className="flex flex-wrap gap-2">
                                                    {status.results.genre_profiles.map((genre: string) => (
                                                        <span
                                                            key={genre}
                                                            className="px-3 py-1 bg-gradient-to-r from-purple-600/50 to-pink-600/50 rounded-full text-white text-sm"
                                                        >
                                                            {genre}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        <div className="mt-4 text-sm text-purple-300">
                                            <p>Model saved to: <code className="text-pink-400">{status.results.model_path}</code></p>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-red-400">
                                        <p>Training failed: {status.results.error}</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Instructions */}
            <div className="mt-6 p-4 bg-black/20 rounded-lg">
                <h3 className="text-white font-bold mb-2">📋 How to Train</h3>
                <ol className="text-purple-200 text-sm space-y-1 list-decimal list-inside">
                    <li>Organize your music by genre in folders (e.g., Music/EDM/, Music/HipHop/)</li>
                    <li>Enter the root path to your music library</li>
                    <li>Click "Start Training" to begin</li>
                    <li>Wait for training to complete</li>
                    <li>The AI will learn genre styles automatically!</li>
                </ol>
                <p className="text-purple-400 text-xs mt-3">
                    💡 Recommended: 100+ songs per genre for best results
                </p>
            </div>
        </div>
    );
};