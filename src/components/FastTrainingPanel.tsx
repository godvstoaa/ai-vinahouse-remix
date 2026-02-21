import React, { useState, useEffect } from 'react';

interface ModelInfo {
    id: string;
    name: string;
    name_cn: string;
    size: string;
    vram: string;
    description: string;
    author: string;
    stars: number;
    downloads: string;
    tags: string[];
    version: string;
    recommended?: boolean;
    new?: boolean;
    hot?: boolean;
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
    const [currentStep, setCurrentStep] = useState<number>(0);
    const [totalSteps, setTotalSteps] = useState<number>(0);
    const [statusMessage, setStatusMessage] = useState<string>('');

    // Available models
    const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
    const [estimate, setEstimate] = useState<TrainingEstimate | null>(null);

    // API Base URLs
    const API_BASE = 'http://localhost:8002/api/training';
    const FAST_TRAIN_API = 'http://localhost:8002/api/fast-train';

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

    // Load available models - Kho model theo phong cách Trung Quốc
    useEffect(() => {
        setAvailableModels([
            // === Source Separation Models ===
            {
                id: 'demucs_ht',
                name: 'HT-Demucs v4',
                name_cn: 'HT-Demucs v4',
                size: '150MB',
                vram: '4GB',
                description: 'Meta AI - Tách nhạc chất lượng cao nhất',
                author: 'Meta AI',
                stars: 5,
                downloads: '1.2M',
                tags: ['separation', 'official', 'stable'],
                version: '4.0',
                recommended: true,
                hot: true
            },
            {
                id: 'demucs_mdx',
                name: 'Demucs MDX23C',
                name_cn: 'Demucs MDX23C',
                size: '180MB',
                vram: '5GB',
                description: 'MDX Challenge winner - Cân bằng nhất',
                author: 'Kuielabs',
                stars: 5,
                downloads: '890K',
                tags: ['separation', 'competition', 'quality'],
                version: '23C',
                hot: true
            },
            {
                id: 'uvr5_mdx',
                name: 'UVR5-MDX-VIP',
                name_cn: 'UVR5-MDX-VIP',
                size: '250MB',
                vram: '6GB',
                description: 'Ultra VR - Tách vocal tốt nhất',
                author: 'Anjok07',
                stars: 5,
                downloads: '2.1M',
                tags: ['separation', 'vocal', 'vip'],
                version: '5.6',
                new: true
            },
            // === RVC Voice Models ===
            {
                id: 'rvc_v2',
                name: 'RVC v2 Base',
                name_cn: 'RVC v2 Base',
                size: '190MB',
                vram: '4GB',
                description: 'Chuyển giọng hát AI phổ biến nhất',
                author: 'liusongxiang',
                stars: 5,
                downloads: '3.5M',
                tags: ['voice', 'rvc', 'popular'],
                version: '2.0',
                hot: true
            },
            {
                id: 'rvc_v2_finetune',
                name: 'RVC v2 Vinahouse',
                name_cn: 'RVC v2 Vinahouse',
                size: '200MB',
                vram: '4GB',
                description: 'Fine-tune riêng cho Vinahouse vocals',
                author: 'Community',
                stars: 4,
                downloads: '150K',
                tags: ['voice', 'vinahouse', 'finetuned'],
                version: '2.1',
                recommended: true
            },
            // === RAVE Music Generation ===
            {
                id: 'rave_vinahouse',
                name: 'RAVE Vinahouse',
                name_cn: 'RAVE Vinahouse',
                size: '50MB',
                vram: '2GB',
                description: 'Đã train sẵn trên Vinahouse - Chất lượng tốt!',
                author: 'ACIDS',
                stars: 5,
                downloads: '500K',
                tags: ['generation', 'vinahouse', 'fast'],
                version: '2.0',
                recommended: true
            },
            {
                id: 'rave_v2',
                name: 'RAVE v2',
                name_cn: 'RAVE v2',
                size: '60MB',
                vram: '3GB',
                description: 'Real-time Audio Variational autoEncoder',
                author: 'ACIDS',
                stars: 5,
                downloads: '800K',
                tags: ['generation', 'realtime', 'quality'],
                version: '2.0',
                new: true
            },
            // === MusicGen ===
            {
                id: 'musicgen_small',
                name: 'MusicGen Small',
                name_cn: 'MusicGen Small',
                size: '300MB',
                vram: '6GB',
                description: 'Text-to-Music generation (300M params)',
                author: 'Meta AI',
                stars: 5,
                downloads: '2.8M',
                tags: ['generation', 'text-to-music', 'official'],
                version: '1.0'
            },
            {
                id: 'musicgen_medium',
                name: 'MusicGen Medium',
                name_cn: 'MusicGen Medium',
                size: '1.2GB',
                vram: '10GB',
                description: 'Text-to-Music cao cấp (1.5B params)',
                author: 'Meta AI',
                stars: 5,
                downloads: '1.5M',
                tags: ['generation', 'text-to-music', 'quality'],
                version: '1.0',
                new: true
            },
            // === So-VITS-SVC ===
            {
                id: 'sovits_svc',
                name: 'So-VITS-SVC 4.1',
                name_cn: 'So-VITS-SVC 4.1',
                size: '220MB',
                vram: '6GB',
                description: 'Singing Voice Conversion nâng cao',
                author: 'svc-develop-team',
                stars: 5,
                downloads: '2.2M',
                tags: ['voice', 'singing', 'advanced'],
                version: '4.1',
                hot: true
            },
            // === AudioCraft ===
            {
                id: 'audiocraft_enhance',
                name: 'AudioCraft Enhance',
                name_cn: 'AudioCraft Enhance',
                size: '500MB',
                vram: '8GB',
                description: 'Nâng cao chất lượng âm thanh AI',
                author: 'Meta AI',
                stars: 4,
                downloads: '600K',
                tags: ['enhancement', 'quality', 'official'],
                version: '1.0',
                new: true
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

    // Poll training status from backend
    useEffect(() => {
        if (!isTraining) return;

        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`${FAST_TRAIN_API}/status`);
                const data = await response.json();

                if (data.is_training) {
                    setCurrentEpoch(data.epoch || 0);
                    setTotalSteps(data.total_epochs || epochs);
                    setCurrentLoss(data.loss || 0);
                    setStatusMessage(`Training epoch ${data.epoch || 0}/${data.total_epochs || epochs}...`);
                    setTrainingProgress(data.progress || 0);
                } else {
                    setIsTraining(false);
                    if (data.progress === 100) {
                        alert('🎉 Huấn luyện hoàn thành!');
                        setStatusMessage('Hoàn thành!');
                    }
                }
            } catch (error) {
                console.error('Failed to poll training status:', error);
            }
        }, 2000);

        return () => clearInterval(pollInterval);
    }, [isTraining]);

    const startTraining = async () => {
        if (!datasetPath) {
            alert('Vui lòng nhập đường dẫn dataset!');
            return;
        }

        setIsTraining(true);
        setTrainingProgress(0);
        setCurrentEpoch(0);
        setCurrentStep(0);
        setStatusMessage('Đang kết nối server...');

        try {
            const response = await fetch(`${FAST_TRAIN_API}/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    batch_size: batchSize,
                    learning_rate: parseFloat(learningRate),
                    epochs: epochs,
                    method: method,
                    pretrained_model: pretrainedModel,
                    dataset_path: datasetPath,
                    lora_rank: loraRank,
                    lora_alpha: loraAlpha
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Không thể bắt đầu training');
            }

            setStatusMessage('Training đã bắt đầu!');
            console.log('Training started:', data);

        } catch (error: any) {
            console.error('Training error:', error);
            setStatusMessage(`Lỗi: ${error.message}`);
            setIsTraining(false);
        }
    };

    const stopTraining = async () => {
        try {
            const response = await fetch(`${FAST_TRAIN_API}/stop`, {
                method: 'POST'
            });

            const data = await response.json();
            console.log('Training stopped:', data);
        } catch (error) {
            console.error('Stop training error:', error);
        }

        setIsTraining(false);
        setStatusMessage('Đã dừng huấn luyện');
    };

    const startDataPrep = async () => {
        if (!datasetPath) {
            alert('Vui lòng nhập đường dẫn dataset!');
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/data-prep`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input_path: datasetPath,
                    workers: 4
                })
            });

            const data = await response.json();
            alert('📁 Bắt đầu chuẩn bị dữ liệu: ' + data.message);
        } catch (error: any) {
            console.error('Data prep error:', error);
            alert('Lỗi chuẩn bị dữ liệu: ' + error.message);
        }
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

            {/* Pre-trained Model Marketplace */}
            <div className="bg-black/30 rounded-xl p-4 border border-white/10">
                <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold">🏪 Kho Model AI</h3>
                    <div className="flex gap-2">
                        <span className="text-xs bg-red-500/20 text-red-400 px-2 py-1 rounded">🔥 Hot</span>
                        <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">✨ Khuyên dùng</span>
                        <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">🆕 Mới</span>
                    </div>
                </div>

                {/* Model Filter */}
                <div className="flex gap-2 mb-4 flex-wrap">
                    {['all', 'separation', 'voice', 'generation', 'enhancement'].map(filter => (
                        <button
                            key={filter}
                            className="text-xs px-3 py-1 rounded-full bg-gray-700 hover:bg-purple-600/50 transition"
                        >
                            {filter === 'all' ? 'Tất cả' :
                                filter === 'separation' ? '🎧 Tách nhạc' :
                                    filter === 'voice' ? '🎤 Giọng hát' :
                                        filter === 'generation' ? '🎵 Tạo nhạc' : '🔊 Xử lý âm thanh'}
                        </button>
                    ))}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {availableModels.map(model => (
                        <button
                            key={model.id}
                            onClick={() => setPretrainedModel(model.id)}
                            disabled={isTraining}
                            className={`p-4 rounded-xl border text-left transition-all hover:scale-[1.02] ${pretrainedModel === model.id
                                ? 'bg-gradient-to-br from-green-600/30 to-emerald-600/20 border-green-500 shadow-lg shadow-green-500/20'
                                : 'bg-gray-800/50 border-gray-700 hover:border-purple-500/50'
                                } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            {/* Header with badges */}
                            <div className="flex items-start justify-between mb-2">
                                <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="font-bold text-white">{model.name}</span>
                                    </div>
                                    <div className="text-xs text-gray-500">v{model.version}</div>
                                </div>
                                <div className="flex gap-1 flex-wrap justify-end">
                                    {model.hot && (
                                        <span className="text-xs bg-red-500/30 text-red-400 px-2 py-0.5 rounded-full">🔥 Hot</span>
                                    )}
                                    {model.recommended && (
                                        <span className="text-xs bg-green-500/30 text-green-400 px-2 py-0.5 rounded-full">✨</span>
                                    )}
                                    {model.new && (
                                        <span className="text-xs bg-blue-500/30 text-blue-400 px-2 py-0.5 rounded-full">🆕</span>
                                    )}
                                </div>
                            </div>

                            {/* Description */}
                            <p className="text-xs text-gray-400 mb-3 line-clamp-2">{model.description}</p>

                            {/* Stats */}
                            <div className="flex items-center gap-4 mb-3 text-xs">
                                <div className="flex items-center gap-1">
                                    <span className="text-yellow-400">{'⭐'.repeat(model.stars)}</span>
                                </div>
                                <div className="flex items-center gap-1 text-gray-500">
                                    <span>📥</span>
                                    <span>{model.downloads}</span>
                                </div>
                            </div>

                            {/* Tags */}
                            <div className="flex gap-1 flex-wrap mb-3">
                                {model.tags.slice(0, 3).map(tag => (
                                    <span key={tag} className="text-xs bg-gray-700/50 text-gray-400 px-2 py-0.5 rounded">
                                        {tag}
                                    </span>
                                ))}
                            </div>

                            {/* Footer */}
                            <div className="flex items-center justify-between pt-2 border-t border-gray-700/50">
                                <div className="flex items-center gap-3 text-xs">
                                    <span className="text-purple-400">💾 {model.size}</span>
                                    <span className="text-blue-400">🎮 {model.vram} VRAM</span>
                                </div>
                                <span className="text-xs text-gray-500">by {model.author}</span>
                            </div>

                            {/* Selected indicator */}
                            {pretrainedModel === model.id && (
                                <div className="mt-2 pt-2 border-t border-green-500/30">
                                    <span className="text-xs text-green-400 font-medium">✓ Đã chọn model này</span>
                                </div>
                            )}
                        </button>
                    ))}
                </div>

                {/* Model count */}
                <div className="mt-4 text-center text-xs text-gray-500">
                    Hiển thị {availableModels.length} models AI có sẵn
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
                    <>
                        <button
                            onClick={startDataPrep}
                            disabled={!datasetPath}
                            className="py-4 px-6 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-xl font-bold transition"
                        >
                            📁 Chuẩn bị Data
                        </button>
                        <button
                            onClick={startTraining}
                            disabled={!datasetPath}
                            className="flex-1 py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed rounded-xl font-bold text-lg transition"
                        >
                            🚀 Bắt Đầu Training
                        </button>
                    </>
                ) : (
                    <button
                        onClick={stopTraining}
                        className="flex-1 py-4 bg-red-600 hover:bg-red-500 rounded-xl font-bold text-lg transition"
                    >
                        ⏹️ Dừng Training
                    </button>
                )}
            </div>

            {/* Status Message */}
            {statusMessage && (
                <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                    <div className="flex items-center gap-2">
                        {isTraining && <div className="animate-spin w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full"></div>}
                        <span className="text-sm text-gray-300">{statusMessage}</span>
                    </div>
                    {isTraining && currentStep > 0 && (
                        <div className="mt-2 text-xs text-gray-500">
                            Step: {currentStep.toLocaleString()} / {totalSteps.toLocaleString()}
                        </div>
                    )}
                </div>
            )}

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