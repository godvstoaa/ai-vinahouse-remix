import { motion } from 'framer-motion'
import { Sliders, Sparkles, Download, Loader2, Play, Pause, AlertCircle } from 'lucide-react'
import { useRemixStore } from '../store/remixStore'
import { useState, useRef } from 'react'

const GENRES = [
    { id: 'edm', name: 'EDM', icon: '🎵' },
    { id: 'house', name: 'House', icon: '🏡' },
    { id: 'hiphop', name: 'Hip-Hop', icon: '🎤' },
    { id: 'pop', name: 'Pop', icon: '🎸' },
    { id: 'trap', name: 'Trap', icon: '💥' },
    { id: 'lofi', name: 'Lo-Fi', icon: '🌙' },
    { id: 'techno', name: 'Techno', icon: '🎹' },
    { id: 'dubstep', name: 'Dubstep', icon: '🔊' },
]

const STYLES = [
    { id: 'club', name: 'Club Mix' },
    { id: 'radio', name: 'Radio Edit' },
    { id: 'extended', name: 'Extended Mix' },
    { id: 'acoustic', name: 'Acoustic Version' },
    { id: 'mashup', name: 'Mashup Style' },
]

function RemixControls() {
    const {
        remixSettings,
        updateRemixSettings,
        isRemixing,
        remixProgress,
        remixUrl,
        createRemix,
        jobId,
        isProcessing,
        audioAnalysis
    } = useRemixStore()

    const [isPlaying, setIsPlaying] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const audioRef = useRef<HTMLAudioElement | null>(null)

    const canRemix = jobId && !isProcessing && audioAnalysis

    const handleGenerateRemix = async () => {
        if (!canRemix) {
            setError('Please upload and process a file first')
            return
        }

        setError(null)

        try {
            await createRemix()
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Remix generation failed')
        }
    }

    const handlePlayPreview = () => {
        if (!remixUrl || !audioRef.current) return

        if (isPlaying) {
            audioRef.current.pause()
        } else {
            audioRef.current.play()
        }
        setIsPlaying(!isPlaying)
    }

    const handleExport = (format: 'wav' | 'mp3' | 'stem') => {
        if (remixUrl) {
            const a = document.createElement('a')
            a.href = remixUrl
            a.download = `remix.${format}`
            a.click()
        }
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-6"
        >
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Sliders className="w-5 h-5 text-primary-400" />
                Remix Controls
            </h3>

            {/* Error Message */}
            {error && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg flex items-center gap-2 text-red-400"
                >
                    <AlertCircle className="w-5 h-5" />
                    <span>{error}</span>
                </motion.div>
            )}

            {/* Genre Selection */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-slate-300 mb-2">
                    Target Genre
                </label>
                <div className="grid grid-cols-4 gap-2">
                    {GENRES.map(genre => (
                        <button
                            key={genre.id}
                            onClick={() => updateRemixSettings({ genre: genre.id })}
                            className={`p-3 rounded-lg text-center transition-all ${remixSettings.genre === genre.id
                                    ? 'bg-primary-500/20 border-primary-500 border'
                                    : 'bg-slate-800 border border-slate-700 hover:border-slate-500'
                                }`}
                        >
                            <span className="text-xl block mb-1">{genre.icon}</span>
                            <span className="text-xs text-slate-300">{genre.name}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Style Selection */}
            <div className="mb-6">
                <label className="block text-sm font-medium text-slate-300 mb-2">
                    Remix Style
                </label>
                <div className="flex flex-wrap gap-2">
                    {STYLES.map(style => (
                        <button
                            key={style.id}
                            onClick={() => updateRemixSettings({ style: style.id })}
                            className={`px-4 py-2 rounded-lg text-sm transition-all ${remixSettings.style === style.id
                                    ? 'bg-accent-500/20 border-accent-500 border text-accent-300'
                                    : 'bg-slate-800 border border-slate-700 text-slate-300 hover:border-slate-500'
                                }`}
                        >
                            {style.name}
                        </button>
                    ))}
                </div>
            </div>

            {/* BPM Control */}
            <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                    <label className="text-sm font-medium text-slate-300">BPM</label>
                    <span className="text-sm text-primary-400">{remixSettings.bpm}</span>
                </div>
                <input
                    type="range"
                    min="60"
                    max="200"
                    value={remixSettings.bpm}
                    onChange={(e) => updateRemixSettings({ bpm: parseInt(e.target.value) })}
                    className="w-full accent-primary-500"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>60</span>
                    <span>Original: {remixSettings.originalBpm}</span>
                    <span>200</span>
                </div>
            </div>

            {/* Energy Level */}
            <div className="mb-6">
                <div className="flex justify-between items-center mb-2">
                    <label className="text-sm font-medium text-slate-300">Energy Level</label>
                    <span className="text-sm text-primary-400">{Math.round(remixSettings.energyLevel * 100)}%</span>
                </div>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={remixSettings.energyLevel}
                    onChange={(e) => updateRemixSettings({ energyLevel: parseFloat(e.target.value) })}
                    className="w-full accent-primary-500"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Chill</span>
                    <span>High Energy</span>
                </div>
            </div>

            {/* Effects */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div>
                    <label className="block text-xs text-slate-400 mb-1">Reverb</label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={remixSettings.reverb}
                        onChange={(e) => updateRemixSettings({ reverb: parseFloat(e.target.value) })}
                        className="w-full accent-primary-500"
                    />
                </div>
                <div>
                    <label className="block text-xs text-slate-400 mb-1">Delay</label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={remixSettings.delay}
                        onChange={(e) => updateRemixSettings({ delay: parseFloat(e.target.value) })}
                        className="w-full accent-primary-500"
                    />
                </div>
                <div>
                    <label className="block text-xs text-slate-400 mb-1">Filter</label>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={remixSettings.filter}
                        onChange={(e) => updateRemixSettings({ filter: parseFloat(e.target.value) })}
                        className="w-full accent-primary-500"
                    />
                </div>
            </div>

            {/* Progress Bar */}
            {isRemixing && (
                <div className="mb-4">
                    <div className="flex justify-between text-sm text-slate-400 mb-1">
                        <span>Generating remix...</span>
                        <span>{remixProgress}%</span>
                    </div>
                    <div className="w-full bg-slate-700 rounded-full h-2">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${remixProgress}%` }}
                            className="h-2 rounded-full bg-gradient-to-r from-primary-500 to-accent-500"
                        />
                    </div>
                </div>
            )}

            {/* Generate Button */}
            <motion.button
                whileHover={{ scale: canRemix ? 1.02 : 1 }}
                whileTap={{ scale: canRemix ? 0.98 : 1 }}
                onClick={handleGenerateRemix}
                disabled={isRemixing || !canRemix}
                className={`w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${canRemix
                        ? 'bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-400 hover:to-accent-400 text-white'
                        : 'bg-slate-700 text-slate-400 cursor-not-allowed'
                    }`}
            >
                {isRemixing ? (
                    <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Generating... {remixProgress}%</span>
                    </>
                ) : (
                    <>
                        <Sparkles className="w-5 h-5" />
                        <span>{canRemix ? 'Generate Remix' : 'Upload a file first'}</span>
                    </>
                )}
            </motion.button>

            {/* Preview & Export */}
            {remixUrl && !isRemixing && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-4 space-y-3"
                >
                    {/* Audio Element */}
                    <audio
                        ref={audioRef}
                        src={remixUrl}
                        onEnded={() => setIsPlaying(false)}
                    />

                    {/* Preview Button */}
                    <button
                        onClick={handlePlayPreview}
                        className="w-full py-3 rounded-xl bg-slate-700 hover:bg-slate-600 text-white font-medium flex items-center justify-center gap-2"
                    >
                        {isPlaying ? (
                            <>
                                <Pause className="w-5 h-5" />
                                <span>Pause Preview</span>
                            </>
                        ) : (
                            <>
                                <Play className="w-5 h-5" />
                                <span>Preview Remix</span>
                            </>
                        )}
                    </button>

                    {/* Export Options */}
                    <div className="flex gap-2">
                        <button
                            onClick={() => handleExport('wav')}
                            className="flex-1 py-3 rounded-xl bg-green-600 hover:bg-green-500 text-white font-medium flex items-center justify-center gap-2"
                        >
                            <Download className="w-5 h-5" />
                            WAV
                        </button>
                        <button
                            onClick={() => handleExport('mp3')}
                            className="flex-1 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium flex items-center justify-center gap-2"
                        >
                            <Download className="w-5 h-5" />
                            MP3
                        </button>
                        <button
                            onClick={() => handleExport('stem')}
                            className="py-3 px-4 rounded-xl bg-slate-700 hover:bg-slate-600 text-white font-medium"
                        >
                            STEM
                        </button>
                    </div>
                </motion.div>
            )}
        </motion.div>
    )
}

export default RemixControls