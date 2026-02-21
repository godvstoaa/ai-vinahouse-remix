import React from 'react';

export interface RemixSettings {
    genre: string;
    bpm: number;
    energyLevel: number;
    reverb: number;
    delay: number;
    sidechain: boolean;
    filter: number;
}

interface RemixControlsProps {
    settings: RemixSettings;
    onSettingsChange: (settings: RemixSettings) => void;
    onGenerate: () => void;
    isGenerating: boolean;
    hasStems: boolean;
    originalBpm?: number;
    genres?: string[];
}

const DEFAULT_GENRES = [
    { id: 'vinahouse', name: 'Vinahouse', bpm: 135, icon: '🎉' },
    { id: 'edm', name: 'EDM', bpm: 128, icon: '🎹' },
    { id: 'house', name: 'House', bpm: 124, icon: '🏠' },
    { id: 'techno', name: 'Techno', bpm: 132, icon: '🎛️' },
    { id: 'hardstyle', name: 'Hardstyle', bpm: 150, icon: '💪' },
];

export const RemixControls: React.FC<RemixControlsProps> = ({
    settings,
    onSettingsChange,
    onGenerate,
    isGenerating,
    hasStems,
    originalBpm,
    genres
}) => {
    const genreList = genres?.map(g => typeof g === 'string' ? { id: g, name: g, bpm: 128, icon: '🎵' } : g) || DEFAULT_GENRES;

    const handleGenreSelect = (genreId: string, targetBpm: number) => {
        onSettingsChange({
            ...settings,
            genre: genreId,
            bpm: targetBpm
        });
    };

    const handleSliderChange = (key: keyof RemixSettings, value: number) => {
        onSettingsChange({
            ...settings,
            [key]: value
        });
    };

    return (
        <div className="bg-gray-800/80 backdrop-blur border border-gray-700 rounded-lg p-5 space-y-5">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <span>🎚️</span>
                    Remix Controls
                </h3>
                {originalBpm && (
                    <span className="text-xs text-gray-400 bg-gray-700 px-2 py-1 rounded">
                        Original: {originalBpm} BPM
                    </span>
                )}
            </div>

            {/* Genre Selection */}
            <div>
                <label className="text-sm text-gray-400 mb-2 block">Genre</label>
                <div className="grid grid-cols-5 gap-2">
                    {genreList.map((genre) => (
                        <button
                            key={genre.id}
                            onClick={() => handleGenreSelect(genre.id, genre.bpm)}
                            className={`
                p-3 rounded-lg border transition-all
                ${settings.genre === genre.id
                                    ? 'bg-gradient-to-br from-pink-600 to-purple-600 border-pink-400 text-white'
                                    : 'bg-gray-700/50 border-gray-600 text-gray-300 hover:border-gray-500'
                                }
              `}
                        >
                            <div className="text-xl mb-1">{genre.icon}</div>
                            <div className="text-xs font-medium">{genre.name}</div>
                            <div className="text-[10px] text-gray-400">{genre.bpm} BPM</div>
                        </button>
                    ))}
                </div>
            </div>

            {/* BPM Slider */}
            <div>
                <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-gray-400">Target BPM</label>
                    <span className="text-pink-400 font-mono font-bold">{settings.bpm}</span>
                </div>
                <input
                    type="range"
                    min={100}
                    max={180}
                    value={settings.bpm}
                    onChange={(e) => handleSliderChange('bpm', parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                />
                <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                    <span>100</span>
                    <span>140 (Vinahouse)</span>
                    <span>180</span>
                </div>
            </div>

            {/* Energy Level */}
            <div>
                <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-gray-400">Energy Level</label>
                    <span className="text-orange-400 font-mono font-bold">{(settings.energyLevel * 100).toFixed(0)}%</span>
                </div>
                <input
                    type="range"
                    min={0}
                    max={100}
                    value={settings.energyLevel * 100}
                    onChange={(e) => handleSliderChange('energyLevel', parseInt(e.target.value) / 100)}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                />
                <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                    <span>😴 Chill</span>
                    <span>🔥 Club</span>
                    <span>💥 Festival</span>
                </div>
            </div>

            {/* Effects Row */}
            <div className="grid grid-cols-2 gap-4">
                {/* Reverb */}
                <div>
                    <div className="flex justify-between items-center mb-2">
                        <label className="text-sm text-gray-400">🌫️ Reverb</label>
                        <span className="text-cyan-400 font-mono text-sm">{(settings.reverb * 100).toFixed(0)}%</span>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={100}
                        value={settings.reverb * 100}
                        onChange={(e) => handleSliderChange('reverb', parseInt(e.target.value) / 100)}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                    />
                </div>

                {/* Delay */}
                <div>
                    <div className="flex justify-between items-center mb-2">
                        <label className="text-sm text-gray-400">📢 Delay</label>
                        <span className="text-green-400 font-mono text-sm">{(settings.delay * 100).toFixed(0)}%</span>
                    </div>
                    <input
                        type="range"
                        min={0}
                        max={100}
                        value={settings.delay * 100}
                        onChange={(e) => handleSliderChange('delay', parseInt(e.target.value) / 100)}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-500"
                    />
                </div>
            </div>

            {/* Sidechain Toggle */}
            <div className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                <div>
                    <div className="text-white font-medium">🎹 Sidechain Compression</div>
                    <div className="text-xs text-gray-400">Pumping effect on kick (Vinahouse signature)</div>
                </div>
                <button
                    onClick={() => onSettingsChange({ ...settings, sidechain: !settings.sidechain })}
                    className={`w-14 h-7 rounded-full transition-colors ${settings.sidechain ? 'bg-pink-600' : 'bg-gray-600'
                        }`}
                >
                    <div
                        className={`w-5 h-5 bg-white rounded-full shadow transition-transform ${settings.sidechain ? 'translate-x-8' : 'translate-x-1'
                            }`}
                    />
                </button>
            </div>

            {/* Filter */}
            <div>
                <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-gray-400">🔽 Filter Sweep</label>
                    <span className="text-yellow-400 font-mono text-sm">{(settings.filter * 100).toFixed(0)}%</span>
                </div>
                <input
                    type="range"
                    min={0}
                    max={100}
                    value={settings.filter * 100}
                    onChange={(e) => handleSliderChange('filter', parseInt(e.target.value) / 100)}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-yellow-500"
                />
            </div>

            {/* Generate Button */}
            <button
                onClick={onGenerate}
                disabled={!hasStems || isGenerating}
                className={`
          w-full py-4 rounded-lg font-bold text-lg transition-all
          ${!hasStems
                        ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        : isGenerating
                            ? 'bg-purple-600 text-white animate-pulse cursor-wait'
                            : 'bg-gradient-to-r from-pink-600 to-purple-600 text-white hover:from-pink-500 hover:to-purple-500 shadow-lg hover:shadow-pink-500/25'
                    }
        `}
            >
                {!hasStems
                    ? '⚠️ Separate stems first'
                    : isGenerating
                        ? '🎵 Generating Remix...'
                        : '🎵 Generate Remix'
                }
            </button>

            {/* Info */}
            <div className="text-[10px] text-gray-500 text-center">
                Using Pedalboard VST-grade effects • Club mastering at -8 LUFS
            </div>
        </div>
    );
};

export default RemixControls;