/**
 * TrainingPanel - UI để chạy training DiT model
 */

import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../lib/api';

interface TrainingStatus {
    is_running: boolean;
    current_step: number;
    total_steps: number;
    current_epoch: number;
    loss: number;
    learning_rate: number;
    gpu_memory_gb: number;
    eta_hours: number;
    status_message: string;
}

interface DataStatus {
    total_files: number;
    processed_files: number;
    total_segments: number;
    is_processing: boolean;
}

const TrainingPanel: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'data' | 'train' | 'monitor'>('data');

    // Data prep state
    const [dataPath, setDataPath] = useState('D:/Music/Vinahouse/');
    const [nonstopPath, setNonstopPath] = useState('D:/Music/Nonstop/');
    const [workers, setWorkers] = useState(4);
    const [dataStatus, setDataStatus] = useState<DataStatus | null>(null);

    // Training state
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
    const [config, setConfig] = useState({
        batch_size: 2,
        learning_rate: 0.0001,
        gradient_accumulation_steps: 16,
        max_steps: 100000,
        mixed_precision: 'fp16'
    });

    // Poll training status
    useEffect(() => {
        const interval = setInterval(async () => {
            try {
                const response = await api.get('/training/status');
                setTrainingStatus(response.data);
            } catch (e) {
                // Training endpoint not available yet
            }
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    const startDataPrep = async () => {
        try {
            await api.post('/training/data-prep', {
                input_path: dataPath,
                workers: workers
            });
            alert('Data preparation started!');
        } catch (error: any) {
            alert(`Error: ${error.response?.data?.detail || error.message}`);
        }
    };

    const startEmotionalTagging = async () => {
        try {
            await api.post('/training/emotional-tagger', {
                input_path: nonstopPath
            });
            alert('Emotional tagging started!');
        } catch (error: any) {
            alert(`Error: ${error.response?.data?.detail || error.message}`);
        }
    };

    const startTraining = async () => {
        try {
            await api.post('/training/start', config);
            alert('Training started!');
        } catch (error: any) {
            alert(`Error: ${error.response?.data?.detail || error.message}`);
        }
    };

    const stopTraining = async () => {
        try {
            await api.post('/training/stop');
            alert('Training stopped!');
        } catch (error: any) {
            alert(`Error: ${error.response?.data?.detail || error.message}`);
        }
    };

    return (
        <div className="bg-gray-900 rounded-xl p-6 text-white">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
                <span className="text-3xl">🧠</span>
                AI Training Module
            </h2>

            {/* Tabs */}
            <div className="flex gap-2 mb-6">
                {[
                    { id: 'data', label: '📁 Chuẩn bị Data', icon: '📁' },
                    { id: 'train', label: '⚙️ Cấu hình Training', icon: '⚙️' },
                    { id: 'monitor', label: '📊 Theo dõi', icon: '📊' }
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id as any)}
                        className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === tab.id
                                ? 'bg-purple-600 text-white'
                                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                            }`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {activeTab === 'data' && (
                <div className="space-y-6">
                    {/* Data Preparation */}
                    <div className="bg-gray-800 rounded-lg p-5">
                        <h3 className="text-lg font-semibold mb-4">🎵 Chuẩn bị Data Vinahouse (10.000 bài)</h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Đường dẫn thư mục nhạc</label>
                                <input
                                    type="text"
                                    value={dataPath}
                                    onChange={(e) => setDataPath(e.target.value)}
                                    className="w-full bg-gray-700 rounded px-3 py-2 text-white"
                                    placeholder="D:/Music/Vinahouse/"
                                />
                            </div>

                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Số workers (CPU cores)</label>
                                <input
                                    type="number"
                                    value={workers}
                                    onChange={(e) => setWorkers(parseInt(e.target.value))}
                                    className="w-32 bg-gray-700 rounded px-3 py-2 text-white"
                                    min="1"
                                    max="16"
                                />
                            </div>

                            <button
                                onClick={startDataPrep}
                                className="bg-green-600 hover:bg-green-700 px-6 py-2 rounded-lg font-medium transition-all"
                            >
                                ▶️ Bắt đầu Cắt Data
                            </button>
                        </div>
                    </div>

                    {/* Emotional Tagger */}
                    <div className="bg-gray-800 rounded-lg p-5">
                        <h3 className="text-lg font-semibold mb-4">🎭 Phân tích Nonstop (2.000 bài)</h3>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Đường dẫn thư mục Nonstop</label>
                                <input
                                    type="text"
                                    value={nonstopPath}
                                    onChange={(e) => setNonstopPath(e.target.value)}
                                    className="w-full bg-gray-700 rounded px-3 py-2 text-white"
                                    placeholder="D:/Music/Nonstop/"
                                />
                            </div>

                            <button
                                onClick={startEmotionalTagging}
                                className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg font-medium transition-all"
                            >
                                ▶️ Bắt đầu Phân tích Cảm xúc
                            </button>
                        </div>
                    </div>

                    {/* Data Status */}
                    {dataStatus && (
                        <div className="bg-gray-800 rounded-lg p-5">
                            <h3 className="text-lg font-semibold mb-4">📈 Trạng thái Data</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <span className="text-gray-400">Files đã xử lý:</span>
                                    <span className="ml-2 text-white">{dataStatus.processed_files}/{dataStatus.total_files}</span>
                                </div>
                                <div>
                                    <span className="text-gray-400">Segments tạo ra:</span>
                                    <span className="ml-2 text-white">{dataStatus.total_segments.toLocaleString()}</span>
                                </div>
                            </div>

                            {dataStatus.is_processing && (
                                <div className="mt-4">
                                    <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                                        <div
                                            className="bg-purple-500 h-full transition-all"
                                            style={{ width: `${(dataStatus.processed_files / dataStatus.total_files) * 100}%` }}
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'train' && (
                <div className="space-y-6">
                    <div className="bg-gray-800 rounded-lg p-5">
                        <h3 className="text-lg font-semibold mb-4">⚙️ Cấu hình Training</h3>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
                                <input
                                    type="number"
                                    value={config.batch_size}
                                    onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                                    className="w-full bg-gray-700 rounded px-3 py-2 text-white"
                                    min="1"
                                    max="8"
                                />
                                <span className="text-xs text-gray-500">Giữ ở 2 cho 8GB VRAM</span>
                            </div>

                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Learning Rate</label>
                                <input
                                    type="number"
                                    value={config.learning_rate}
                                    onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
                                    className="w-full bg-gray-700 rounded px-3 py-2 text-white"
                                    step="0.00001"
                                />
                            </div>

                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Gradient Accumulation</label>
                                <input
                                    type="number"
                                    value={config.gradient_accumulation_steps}
                                    onChange={(e) => setConfig({ ...config, gradient_accumulation_steps: parseInt(e.target.value) })}
                                    className="w-full bg-gray-700 rounded px-3 py-2 text-white"
                                    min="1"
                                    max="64"
                                />
                                <span className="text-xs text-gray-500">16 = effective batch size 32</span>
                            </div>

                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Max Steps</label>
                                <input
                                    type="number"
                                    value={config.max_steps}
                                    onChange={(e) => setConfig({ ...config, max_steps: parseInt(e.target.value) })}
                                    className="w-full bg-gray-700 rounded px-3 py-2 text-white"
                                    step="1000"
                                />
                            </div>

                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Mixed Precision</label>
                                <select
                                    value={config.mixed_precision}
                                    onChange={(e) => setConfig({ ...config, mixed_precision: e.target.value })}
                                    className="w-full bg-gray-700 rounded px-3 py-2 text-white"
                                >
                                    <option value="fp16">FP16 (Recommended)</option>
                                    <option value="bf16">BF16 (RTX 30/40 series)</option>
                                    <option value="no">FP32 (Slow)</option>
                                </select>
                            </div>
                        </div>

                        <div className="mt-6 flex gap-3">
                            <button
                                onClick={startTraining}
                                disabled={trainingStatus?.is_running}
                                className={`px-6 py-3 rounded-lg font-medium transition-all ${trainingStatus?.is_running
                                        ? 'bg-gray-600 cursor-not-allowed'
                                        : 'bg-green-600 hover:bg-green-700'
                                    }`}
                            >
                                ▶️ Bắt đầu Training
                            </button>

                            <button
                                onClick={stopTraining}
                                disabled={!trainingStatus?.is_running}
                                className={`px-6 py-3 rounded-lg font-medium transition-all ${!trainingStatus?.is_running
                                        ? 'bg-gray-600 cursor-not-allowed'
                                        : 'bg-red-600 hover:bg-red-700'
                                    }`}
                            >
                                ⏹️ Dừng Training
                            </button>
                        </div>
                    </div>

                    {/* Hardware Requirements */}
                    <div className="bg-gray-800 rounded-lg p-5">
                        <h3 className="text-lg font-semibold mb-4">💻 Yêu cầu phần cứng</h3>
                        <div className="grid grid-cols-3 gap-4 text-sm">
                            <div className="bg-gray-700 rounded p-3">
                                <div className="text-gray-400">GPU VRAM</div>
                                <div className="text-xl font-bold text-green-400">8GB+</div>
                                <div className="text-gray-500">RTX 4070</div>
                            </div>
                            <div className="bg-gray-700 rounded p-3">
                                <div className="text-gray-400">RAM</div>
                                <div className="text-xl font-bold text-blue-400">32GB+</div>
                                <div className="text-gray-500">System Memory</div>
                            </div>
                            <div className="bg-gray-700 rounded p-3">
                                <div className="text-gray-400">Thời gian</div>
                                <div className="text-xl font-bold text-yellow-400">7-14 ngày</div>
                                <div className="text-gray-500">24/7 training</div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'monitor' && (
                <div className="space-y-6">
                    {trainingStatus ? (
                        <>
                            {/* Training Progress */}
                            <div className="bg-gray-800 rounded-lg p-5">
                                <h3 className="text-lg font-semibold mb-4">
                                    Training Progress
                                    {trainingStatus.is_running && (
                                        <span className="ml-2 text-green-400 animate-pulse">● Running</span>
                                    )}
                                </h3>

                                {/* Progress Bar */}
                                <div className="mb-4">
                                    <div className="flex justify-between text-sm text-gray-400 mb-1">
                                        <span>Step {trainingStatus.current_step.toLocaleString()} / {trainingStatus.total_steps.toLocaleString()}</span>
                                        <span>{((trainingStatus.current_step / trainingStatus.total_steps) * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="bg-gray-700 rounded-full h-3 overflow-hidden">
                                        <div
                                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all"
                                            style={{ width: `${(trainingStatus.current_step / trainingStatus.total_steps) * 100}%` }}
                                        />
                                    </div>
                                </div>

                                {/* Metrics Grid */}
                                <div className="grid grid-cols-4 gap-4">
                                    <div className="bg-gray-700 rounded p-3 text-center">
                                        <div className="text-gray-400 text-sm">Loss</div>
                                        <div className="text-xl font-bold text-red-400">{trainingStatus.loss.toFixed(4)}</div>
                                    </div>
                                    <div className="bg-gray-700 rounded p-3 text-center">
                                        <div className="text-gray-400 text-sm">LR</div>
                                        <div className="text-xl font-bold text-blue-400">{trainingStatus.learning_rate.toExponential(2)}</div>
                                    </div>
                                    <div className="bg-gray-700 rounded p-3 text-center">
                                        <div className="text-gray-400 text-sm">GPU Memory</div>
                                        <div className="text-xl font-bold text-yellow-400">{trainingStatus.gpu_memory_gb.toFixed(1)} GB</div>
                                    </div>
                                    <div className="bg-gray-700 rounded p-3 text-center">
                                        <div className="text-gray-400 text-sm">ETA</div>
                                        <div className="text-xl font-bold text-green-400">{trainingStatus.eta_hours.toFixed(1)}h</div>
                                    </div>
                                </div>

                                {/* Status Message */}
                                <div className="mt-4 bg-gray-700 rounded p-3">
                                    <span className="text-gray-400">Status:</span>
                                    <span className="ml-2 text-white">{trainingStatus.status_message}</span>
                                </div>
                            </div>

                            {/* TensorBoard Link */}
                            <div className="bg-gray-800 rounded-lg p-5">
                                <h3 className="text-lg font-semibold mb-4">📊 TensorBoard</h3>
                                <p className="text-gray-400 mb-3">
                                    Xem loss curve chi tiết và metrics khác trên TensorBoard:
                                </p>
                                <code className="block bg-gray-700 rounded p-3 text-green-400">
                                    tensorboard --logdir checkpoints/logs/
                                </code>
                                <a
                                    href="http://localhost:6006"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-block mt-3 text-purple-400 hover:text-purple-300"
                                >
                                    → Mở http://localhost:6006
                                </a>
                            </div>
                        </>
                    ) : (
                        <div className="bg-gray-800 rounded-lg p-8 text-center">
                            <div className="text-6xl mb-4">💤</div>
                            <h3 className="text-xl font-semibold mb-2">Chưa có training nào chạy</h3>
                            <p className="text-gray-400">
                                Chuẩn bị data và bắt đầu training từ tab "Cấu hình Training"
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default TrainingPanel;