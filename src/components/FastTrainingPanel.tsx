import React, { useState, useEffect } from 'react';

interface ModelInfo {
    id: string;
    name: string;
    size: string;
    vram: string;
    description: string;
}

interface TrainingEstimate {
    estimated_hours: number;
    estimated_minutes: number;
    method: string;
    recommended_batch_size: number;
}

const FastTrainingPanel: React.FC = () => {
    // Training config
    const [method, setMethod] = useState<string>('lora');
    const [pretrainedModel, setPretrainedModel] = useState<string>('demucs_ht');
    const [datasetPath, setDatasetPath] = useState<string>('');
    const [epochs, setEpochs] = useState<number>(10);
    const [batchSize, setBatchSize] = useState<number>(4);
    const [learningRate, setLearningRate] = useState<string>('0.0001');

    // LoRA params
    const [loraRank, setLoraRank] = useState<number>(8);
    const [loraAlpha, setLoraAlpha] = useState<number>(16);

    // System
    const [gpuVram, setGpuVram] = useState<number>(8);
    const [datasetSize, setDatasetSize] = useState<number>(1000);

    // Training state
    const [isTraining, setIsTraining] = useState<boolean>(false);
    const [trainingProgress, setTrainingProgress] = useState<number>(0);
    const [currentEpoch, setCurrentEpoch] = useState<number>(0);
    const [currentLoss, setCurrentLoss] = useState<number>(0);

    // Available models
    const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
    const [estimate, setEstimate] = useState<TrainingEstimate | null>(null);

    // Methods info
    const methods = [
        {
            id: 'qlora',
            name: 'QLoRA',
            speed: '⚡⚡⚡⚡⚡',
            vram: '🟢 4GB',
            desc: 'Nhanh nhất, ít VRAM nhất (4-bit)',
            epochs: 10
        },
        {
            id: 'lora',
            name: 'LoRA',
            speed: '⚡⚡⚡⚡',
            vram: '🟡 6GB',
            desc: 'Rất nhanh, chỉ train adapter layers',
            epochs: 10
        },
        {
            id: 'transfer',
            name: 'Transfer Learning',
            speed: '⚡⚡⚡',
            vram: '🟠 8GB',
            desc: 'Freeze encoder, train decoder',
            epochs: 20
        },
        {
            id: 'fine_tune',
            name: 'Fine-tune',
            speed: '⚡⚡',
            vram: '🔴 10GB',
            desc: 'Fine-tune nhẹ toàn bộ model',
            epochs: 15
        },
        {
            id: 'full',
            name: 'Full Training',
            speed: '⚡',
            vram: '🔴 16GB+',
            desc: 'Training từ đầu (chậm nhất)',
            epochs: 50
        },
    ];

    // Load available models
    useEffect(() => {
        // Mock data - in real app would fetch from API
        setAvailableModels([
            {
                id: 'demucs_ht',
                name: 'HT-Demucs v4',
                size: '150MB',
                vram: '4GB',
                description: 'Meta AI - Tốt nhất cho tách nhạc'
            },
            {
                id: 'demucs_md',
                name: 'Demucs MDX',
                size: '100MB',
                vram: '3GB',
                description: 'Cân bằng tốc độ và chất lượng'
            },
            {
                id: 'rave_vinahouse',
                name: 'RAVE Vinahouse',
                size: '50MB',
                vram: '2GB',
                description: 'Đã train sẵn trên Vinahouse!'
            },
            {
                id: 'musicgen_small',
                name: 'MusicGen Small',
                size: '300MB',
                vram: '6GB',
                description: 'Text-to-Music generation'
            },
            {
                id: 'svc_rvc',
                name: 'RVC Base',
                size: '190MB',
                vram: '4GB',
                description: 'Voice Conversion cho vocals'
            },
        ]);
    }, []);

    // Calculate estimate when params change
    useEffect(() => {
        calculateEstimate();
    }, [method, datasetSize, gpuVram]);

    const calculateEstimate = () => {
        const baseTimes: Record<string, number> = {
            full: 10.0,
            transfer: 2.0,
            lora: 0.5,
            qlora: 0.3,
            fine_tune: 1.0
        };

        const vramFactor = Math.max(0.5, 8 / gpuVram);
        const estimatedHours = (datasetSize / 1000) * (baseTimes[method] || 1) * vramFactor;

        setEstimate({
            estimated_hours: estimatedHours,
            estimated_minutes: estimatedHours * 60,
            method: method,
            recommended_batch_size: Math.min(8, Math.max(1, Math.floor(gpuVram / 2)))
        });
    };

    const handleMethodChange = (newMethod: string) => {
        setMethod(newMethod);
        const methodInfo = methods.find(m => m.id === newMethod);
        if (methodInfo) {
            setEpochs(methodInfo.epochs);
        }
    };

    const startTraining = async () => {
        setIsTraining(true);
        setTrainingProgress(0);
        setCurrentEpoch(0);

        // Simulate training progress
        for (let i = 0; i < epochs; i++) {
            await new Promise(r => setTimeout(r, 500));
            setCurrentEpoch(i + 1);
            setTrainingProgress(((i + 1) / epochs) * 100);
            setCurrentLoss(Math.random() * 0.5 + 0.1);
        }

        setIsTraining(false);
        alert('Huấn luyện hoàn thành!');
    };

    const stopTraining = () => {
        setIsTraining(false);
        alert('Đã dừng huấn luyện');
    };

    const recommendSettings = () => {
        // Auto-recommend based on hardware
        if (gpuVram < 6) {
            setMethod('qlora');
            setBatchSize(2);
            setLoraRank(4);
        } else if (gpuVram < 12) {
            setMethod('lora');
            setBatchSize(4);
            setLoraRank(8);
        } else {
            setMethod('transfer');
            setBatchSize(8);
            setLoraRank(16);
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-2xl font-bold">🚀 Fast Training</h2>
                    <p className="text-gray-400 text-sm">LoRA, QLoRA, Transfer Learning - Training nhanh x10</p>
                </div>
                <button
                    onClick={recommendSettings}
                    className="px-4 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-sm"
                >
                    🎯 Tự động tối ưu
                </button>
            </div>

            {/* Method Selection */}
            <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                <h3 className="text-lg font-semibold mb-3">📋 Phương Pháp Training</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {methods.map(m => (
                        <button
                            key={m.id}
                            onClick={() => handleMethodChange(m.id)}
                            disabled={isTraining}
                            className={`p-3 rounded-lg border text-left transition ${method === m.id
                                    ? 'bg-purple-600/30 border-purple-500'
                                    : 'bg-gray-800/50 border-gray-700 hover:border-purple-500/50'
                                } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-semibold">{m.name}</span>
                                <span className="text-xs">{m.speed}</span>
                            </div>
                            <div className="text-xs text-gray-400 mb-1">{m.vram}</div>
                            <div className="text-xs text-gray-500">{m.desc}</div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Pre-trained Model */}
            <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                <h3 className="text-lg font-semibold mb-3">🧠 Model Pre-trained</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {availableModels.map(model => (
                        <button
                            key={model.id}
                            onClick={() => setPretrainedModel(model.id)}
                            disabled={isTraining}
                            className={`p-3 rounded-lg border text-left transition ${pretrainedModel === model.id
                                    ? 'bg-green-600/30 border-green-500'
                                    : 'bg-gray-800/50 border-gray-700 hover:border-green-500/50'
                                } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-semibold">{model.name}</span>
                                <span className="text-xs bg-blue-500/20 px-2 py-0.5 rounded">{model.size}</span>
                            </div>
                            <div className="text-xs text-gray-400 mb-1">VRAM: {model.vram}</div>
                            <div className="text-xs text-gray-500">{model.description}</div>
                            {model.id === 'rave_vinahouse' && (
                                <div className="mt-2 text-xs text-green-400">✨ Recommended for Vinahouse!</div>
                            )}
                        </button>
                    ))}
                </div>
            </div>

            {/* Training Config */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Dataset */}
                <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                    <h3 className="text-lg font-semibold mb-3">📁 Dataset</h3>

                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Đường dẫn dataset</label>
                            <input
                                type="text"
                                value={datasetPath}
                                onChange={(e) => setDatasetPath(e.target.value)}
                                placeholder="D:/VinahouseMusic"
                                className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10"
                                disabled={isTraining}
                            />
                        </div>

                        <div>
                            <label className="block text-sm text-gray-400 mb-1">
                                Số lượng bài: <span className="text-purple-400">{datasetSize}</span>
                            </label>
                            <input
                                type="range"
                                min="100"
                                max="10000"
                                step="100"
                                value={datasetSize}
                                onChange={(e) => setDatasetSize(parseInt(e.target.value))}
                                className="w-full accent-purple-500"
                                disabled={isTraining}
                            />
                            <div className="flex justify-between text-xs text-gray-500">
                                <span>100</span>
                                <span>10,000</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* System */}
                <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                    <h3 className="text-lg font-semibold mb-3">💻 Hệ Thống</h3>

                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">
                                GPU VRAM: <span className="text-green-400">{gpuVram}GB</span>
                            </label>
                            <input
                                type="range"
                                min="4"
                                max="24"
                                step="2"
                                value={gpuVram}
                                onChange={(e) => setGpuVram(parseInt(e.target.value))}
                                className="w-full accent-green-500"
                                disabled={isTraining}
                            />
                            <div className="flex justify-between text-xs text-gray-500">
                                <span>4GB</span>
                                <span>24GB</span>
                            </div>
                        </div>

                        <div className="p-2 bg-gray-800/50 rounded text-sm">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Batch size đề xuất:</span>
                                <span className="text-yellow-400">{estimate?.recommended_batch_size || 4}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* LoRA/QLoRA Settings */}
            {(method === 'lora' || method === 'qlora') && (
                <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                    <h3 className="text-lg font-semibold mb-3">🔧 LoRA Settings</h3>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">
                                Rank: <span className="text-purple-400">{loraRank}</span>
                            </label>
                            <input
                                type="range"
                                min="2"
                                max="32"
                                step="2"
                                value={loraRank}
                                onChange={(e) => setLoraRank(parseInt(e.target.value))}
                                className="w-full accent-purple-500"
                                disabled={isTraining}
                            />
                            <div className="text-xs text-gray-500 mt-1">
                                Càng thấp càng nhanh, thấp hơn = ít chi tiết
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm text-gray-400 mb-1">
                                Alpha: <span className="text-purple-400">{loraAlpha}</span>
                            </label>
                            <input
                                type="range"
                                min="8"
                                max="64"
                                step="8"
                                value={loraAlpha}
                                onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
                                className="w-full accent-purple-500"
                                disabled={isTraining}
                            />
                            <div className="text-xs text-gray-500 mt-1">
                                Thường set = 2x Rank
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Basic Training Params */}
            <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                <h3 className="text-lg font-semibold mb-3">⚙️ Tham Số Training</h3>

                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Epochs</label>
                        <input
                            type="number"
                            value={epochs}
                            onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                            className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10"
                            disabled={isTraining}
                        />
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
                        <input
                            type="number"
                            value={batchSize}
                            onChange={(e) => setBatchSize(parseInt(e.target.value) || 4)}
                            className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10"
                            disabled={isTraining}
                        />
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Learning Rate</label>
                        <input
                            type="text"
                            value={learningRate}
                            onChange={(e) => setLearningRate(e.target.value)}
                            className="w-full bg-gray-800 rounded-lg px-3 py-2 border border-white/10"
                            disabled={isTraining}
                        />
                    </div>
                </div>
            </div>

            {/* Estimate */}
            {estimate && (
                <div className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-xl p-4 border border-purple-500/30">
                    <h3 className="text-lg font-semibold mb-3">⏱️ Ước Tính Training</h3>

                    <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                            <div className="text-3xl font-bold text-purple-400">
                                {estimate.estimated_hours.toFixed(1)}h
                            </div>
                            <div className="text-sm text-gray-400">Thời gian</div>
                        </div>
                        <div>
                            <div className="text-3xl font-bold text-pink-400">
                                {epochs}
                            </div>
                            <div className="text-sm text-gray-400">Epochs</div>
                        </div>
                        <div>
                            <div className="text-3xl font-bold text-blue-400">
                                {method.toUpperCase()}
                            </div>
                            <div className="text-sm text-gray-400">Method</div>
                        </div>
                    </div>

                    <div className="mt-3 text-center text-sm text-gray-400">
                        So với Full Training: <span className="text-green-400">
                            {method === 'qlora' ? '33x' : method === 'lora' ? '20x' : method === 'transfer' ? '5x' : '2x'} nhanh hơn
                        </span>
                    </div>
                </div>
            )}

            {/* Training Progress */}
            {isTraining && (
                <div className="bg-black/30 rounded-xl p-4 border border-yellow-500/30">
                    <h3 className="text-lg font-semibold mb-3">🔥 Đang Training...</h3>

                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <span>Epoch: {currentEpoch}/{epochs}</span>
                            <span>Loss: {currentLoss.toFixed(4)}</span>
                        </div>

                        <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all"
                                style={{ width: `${trainingProgress}%` }}
                            ></div>
                        </div>

                        <div className="text-center text-sm text-gray-400">
                            {trainingProgress.toFixed(1)}% hoàn thành
                        </div>
                    </div>
                </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-4">
                {!isTraining ? (
                    <button
                        onClick={startTraining}
                        disabled={!datasetPath}
                        className="flex-1 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed rounded-xl font-bold text-lg transition"
                    >
                        🚀 Bắt Đầu Training
                    </button>
                ) : (
                    <button
                        onClick={stopTraining}
                        className="flex-1 py-4 bg-red-600 hover:bg-red-500 rounded-xl font-bold text-lg transition"
                    >
                        ⏹️ Dừng Training
                    </button>
                )}
            </div>

            {/* Tips */}
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
                <h4 className="font-semibold text-blue-400 mb-2">💡 Mẹo Training Nhanh</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                    <li>• <span className="text-green-400">QLoRA</span>: Nhanh nhất, chỉ cần 4GB VRAM</li>
                    <li>• <span className="text-yellow-400">RAVE Vinahouse</span>: Đã train sẵn, chỉ cần fine-tune nhẹ</li>
                    <li>• <span className="text-purple-400">LoRA Rank 4-8</span>: Đủ tốt cho Vinahouse</li>
                    <li>• <span className="text-blue-400">100-500 bài</span>: Đủ để học style Vinahouse</li>
                    <li>• Tăng <span className="text-pink-400">batch_size</span> nếu có nhiều VRAM</li>
                </ul>
            </div>
        </div>
    );
};

export default FastTrainingPanel;