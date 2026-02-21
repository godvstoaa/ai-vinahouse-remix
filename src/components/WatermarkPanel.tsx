/**
 * Ultrasonic Watermark Panel
 * Nhúng và phát hiện watermark ở tần số siêu âm (18-20 kHz)
 */

import { useState, useCallback } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import {
    Shield,
    Scan,
    AudioWaveform,
    CheckCircle,
    XCircle,
    AlertTriangle,
    Loader2,
    Upload,
    Fingerprint,
    BarChart3
} from 'lucide-react'

const API_BASE = 'http://localhost:8000'

interface WatermarkResult {
    success: boolean
    has_watermark?: boolean
    producer_id?: string
    track_id?: string
    timestamp?: string
    confidence?: number
    ultrasonic_energy?: number
    message?: string
    output_path?: string
}

interface SpectrumData {
    success: boolean
    freqs: number[]
    magnitudes: number[]
    ultrasonic_energy: number
    peak_freq: number
}

export default function WatermarkPanel() {
    const [mode, setMode] = useState<'embed' | 'detect'>('embed')
    const [audioPath, setAudioPath] = useState('')
    const [producerId, setProducerId] = useState('')
    const [trackId, setTrackId] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [result, setResult] = useState<WatermarkResult | null>(null)
    const [spectrum, setSpectrum] = useState<SpectrumData | null>(null)

    // Select audio file
    const handleSelectFile = useCallback(async () => {
        if (window.electronAPI?.selectFile) {
            const path = await window.electronAPI.selectFile()
            if (path) {
                setAudioPath(path)
                setResult(null)
                setSpectrum(null)
            }
        } else {
            // Fallback for web
            const input = document.createElement('input')
            input.type = 'file'
            input.accept = 'audio/*'
            input.onchange = (e) => {
                const file = (e.target as HTMLInputElement).files?.[0]
                if (file) {
                    setAudioPath(file.name)
                }
            }
            input.click()
        }
    }, [])

    // Embed watermark
    const handleEmbed = useCallback(async () => {
        if (!audioPath || !producerId) return

        setIsLoading(true)
        setResult(null)

        try {
            const response = await axios.post(`${API_BASE}/api/watermark/embed`, {
                audio_path: audioPath,
                producer_id: producerId,
                track_id: trackId
            })
            setResult(response.data)
        } catch (error: any) {
            setResult({
                success: false,
                message: error.response?.data?.detail || 'Failed to embed watermark'
            })
        } finally {
            setIsLoading(false)
        }
    }, [audioPath, producerId, trackId])

    // Detect watermark
    const handleDetect = useCallback(async () => {
        if (!audioPath) return

        setIsLoading(true)
        setResult(null)
        setSpectrum(null)

        try {
            // Run both detect and analyze
            const [detectRes, spectrumRes] = await Promise.all([
                axios.post(`${API_BASE}/api/watermark/detect`, {
                    audio_path: audioPath
                }),
                axios.post(`${API_BASE}/api/watermark/analyze`, {
                    audio_path: audioPath
                })
            ])

            setResult(detectRes.data)
            setSpectrum(spectrumRes.data)
        } catch (error: any) {
            setResult({
                success: false,
                message: error.response?.data?.detail || 'Failed to detect watermark'
            })
        } finally {
            setIsLoading(false)
        }
    }, [audioPath])

    // Render spectrum visualization
    const renderSpectrum = () => {
        if (!spectrum || !spectrum.freqs || spectrum.freqs.length === 0) return null

        // Find ultrasonic region (18-20kHz)
        const ultraMin = 18000
        const ultraMax = 20000

        return (
            <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                    <BarChart3 className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-gray-300">Phổ tần số (Frequency Spectrum)</span>
                </div>

                {/* Simple bar representation */}
                <div className="h-24 flex items-end gap-px overflow-hidden">
                    {spectrum.magnitudes.slice(-100).map((mag, i) => {
                        const freq = spectrum.freqs[spectrum.freqs.length - 100 + i] || 0
                        const isUltrasonic = freq >= ultraMin && freq <= ultraMax
                        const height = Math.min(mag, 100)

                        return (
                            <div
                                key={i}
                                className={`w-1 transition-all ${isUltrasonic ? 'bg-purple-500' : 'bg-green-500/50'
                                    }`}
                                style={{ height: `${height}%` }}
                                title={`${Math.round(freq)}Hz: ${mag.toFixed(1)}dB`}
                            />
                        )
                    })}
                </div>

                <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span>1kHz</span>
                    <span className="text-purple-400">18kHz ← Ultrasonic → 20kHz</span>
                    <span>22kHz</span>
                </div>

                {/* Energy meter */}
                <div className="mt-4">
                    <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">Ultrasonic Energy</span>
                        <span className="text-purple-400">{(spectrum.ultrasonic_energy * 100).toFixed(2)}%</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-purple-600 to-pink-500 transition-all"
                            style={{ width: `${Math.min(spectrum.ultrasonic_energy * 1000, 100)}%` }}
                        />
                    </div>
                </div>
            </div>
        )
    }

    // Render result
    const renderResult = () => {
        if (!result) return null

        if (result.success && result.has_watermark) {
            return (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-green-900/30 border border-green-500/30 rounded-lg"
                >
                    <div className="flex items-center gap-2 mb-3">
                        <CheckCircle className="w-5 h-5 text-green-400" />
                        <span className="font-medium text-green-400">Đã tìm thấy Watermark!</span>
                    </div>

                    <div className="space-y-2 text-sm">
                        {result.producer_id && (
                            <div className="flex justify-between">
                                <span className="text-gray-400">Producer ID:</span>
                                <span className="text-white font-mono">{result.producer_id}</span>
                            </div>
                        )}
                        {result.track_id && (
                            <div className="flex justify-between">
                                <span className="text-gray-400">Track ID:</span>
                                <span className="text-white font-mono">{result.track_id}</span>
                            </div>
                        )}
                        {result.timestamp && (
                            <div className="flex justify-between">
                                <span className="text-gray-400">Timestamp:</span>
                                <span className="text-white font-mono">{result.timestamp}</span>
                            </div>
                        )}
                        {result.confidence !== undefined && (
                            <div className="flex justify-between">
                                <span className="text-gray-400">Confidence:</span>
                                <span className="text-white">{(result.confidence * 100).toFixed(1)}%</span>
                            </div>
                        )}
                        {result.ultrasonic_energy !== undefined && (
                            <div className="flex justify-between">
                                <span className="text-gray-400">Ultrasonic Energy:</span>
                                <span className="text-white">{(result.ultrasonic_energy * 100).toFixed(3)}%</span>
                            </div>
                        )}
                    </div>
                </motion.div>
            )
        }

        if (result.success && !result.has_watermark) {
            return (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-gray-800 border border-gray-600 rounded-lg"
                >
                    <div className="flex items-center gap-2">
                        <XCircle className="w-5 h-5 text-gray-400" />
                        <span className="text-gray-300">Không tìm thấy watermark trong file này</span>
                    </div>
                </motion.div>
            )
        }

        if (result.success && result.output_path) {
            return (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-green-900/30 border border-green-500/30 rounded-lg"
                >
                    <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="w-5 h-5 text-green-400" />
                        <span className="font-medium text-green-400">Watermark đã được nhúng!</span>
                    </div>

                    <div className="text-sm text-gray-300">
                        <p>Producer: <span className="text-white font-mono">{result.producer_id}</span></p>
                        <p>Carrier: <span className="text-purple-400">19 kHz (Ultrasonic)</span></p>
                        <p className="mt-2 text-xs text-gray-500">
                            Output: {result.output_path}
                        </p>
                    </div>
                </motion.div>
            )
        }

        if (!result.success) {
            return (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-red-900/30 border border-red-500/30 rounded-lg"
                >
                    <div className="flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                        <span className="text-red-300">{result.message || 'Có lỗi xảy ra'}</span>
                    </div>
                </motion.div>
            )
        }

        return null
    }

    return (
        <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-xl p-6 border border-gray-700">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-purple-500/20 rounded-lg">
                    <Fingerprint className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                    <h3 className="text-lg font-semibold text-white">Ultrasonic Watermark</h3>
                    <p className="text-sm text-gray-400">Nhúng & phát hiện dấu âm thanh ẩn (18-20 kHz)</p>
                </div>
            </div>

            {/* Mode Tabs */}
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => { setMode('embed'); setResult(null) }}
                    className={`flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition ${mode === 'embed'
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                >
                    <Shield className="w-4 h-4" />
                    Nhúng Watermark
                </button>
                <button
                    onClick={() => { setMode('detect'); setResult(null) }}
                    className={`flex-1 py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition ${mode === 'detect'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                >
                    <Scan className="w-4 h-4" />
                    Phát hiện
                </button>
            </div>

            {/* File Selection */}
            <div className="mb-4">
                <label className="block text-sm text-gray-400 mb-2">File âm thanh</label>
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={audioPath}
                        onChange={(e) => setAudioPath(e.target.value)}
                        placeholder="Chọn file MP3, WAV, FLAC..."
                        className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                    />
                    <button
                        onClick={handleSelectFile}
                        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition"
                    >
                        <Upload className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Embed Mode Fields */}
            <AnimatePresence mode="wait">
                {mode === 'embed' && (
                    <motion.div
                        key="embed-form"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-4"
                    >
                        <div>
                            <label className="block text-sm text-gray-400 mb-2">
                                Producer ID <span className="text-red-400">*</span>
                            </label>
                            <input
                                type="text"
                                value={producerId}
                                onChange={(e) => setProducerId(e.target.value)}
                                placeholder="VD: VINAHOUSE_PRODUCER_001"
                                maxLength={32}
                                className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                            />
                            <p className="text-xs text-gray-500 mt-1">Tối đa 32 ký tự</p>
                        </div>

                        <div>
                            <label className="block text-sm text-gray-400 mb-2">Track ID</label>
                            <input
                                type="text"
                                value={trackId}
                                onChange={(e) => setTrackId(e.target.value)}
                                placeholder="VD: TRACK_2024_001"
                                maxLength={32}
                                className="w-full bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                            />
                        </div>

                        <button
                            onClick={handleEmbed}
                            disabled={!audioPath || !producerId || isLoading}
                            className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium text-white flex items-center justify-center gap-2 transition"
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Đang nhúng...
                                </>
                            ) : (
                                <>
                                    <Shield className="w-5 h-5" />
                                    Nhúng Watermark
                                </>
                            )}
                        </button>
                    </motion.div>
                )}

                {mode === 'detect' && (
                    <motion.div
                        key="detect-form"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                    >
                        <button
                            onClick={handleDetect}
                            disabled={!audioPath || isLoading}
                            className="w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium text-white flex items-center justify-center gap-2 transition"
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Đang quét...
                                </>
                            ) : (
                                <>
                                    <Scan className="w-5 h-5" />
                                    Quét Watermark
                                </>
                            )}
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Result */}
            {renderResult()}

            {/* Spectrum Visualization */}
            {renderSpectrum()}

            {/* Info */}
            <div className="mt-6 p-4 bg-gray-800/50 rounded-lg text-xs text-gray-500">
                <p className="flex items-center gap-2 mb-1">
                    <AudioWaveform className="w-4 h-4" />
                    <span className="font-medium text-gray-400">Cách hoạt động:</span>
                </p>
                <ul className="space-y-1 ml-6">
                    <li>• Watermark được nhúng ở tần số 19 kHz (siêu âm)</li>
                    <li>• Không thể nghe thấy bằng tai người</li>
                    <li>• Chứa Producer ID, Track ID và Timestamp</li>
                    <li>• Sử dụng Spread Spectrum để chống nén MP3</li>
                </ul>
            </div>
        </div>
    )
}