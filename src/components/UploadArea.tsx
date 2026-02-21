import { useCallback, useState } from 'react'
import { motion } from 'framer-motion'
import { Upload, Music, FileAudio, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { useRemixStore } from '../store/remixStore'

function UploadArea() {
    const { setOriginalFile, uploadFile, processAudio, isProcessing, processingStep, originalFile } = useRemixStore()
    const [error, setError] = useState<string | null>(null)

    const handleFileUpload = useCallback(async (file: File) => {
        setError(null)

        // Validate file
        if (!file.type.includes('audio') && !file.name.match(/\.(mp3|wav|flac)$/i)) {
            setError('Please upload an audio file (MP3, WAV, or FLAC)')
            return
        }

        // Set file for preview
        setOriginalFile(file)

        try {
            // Upload to backend
            await uploadFile(file)

            // Start processing (analyze + separate)
            await processAudio()
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed')
        }
    }, [setOriginalFile, uploadFile, processAudio])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        const file = e.dataTransfer.files[0]
        if (file) {
            handleFileUpload(file)
        }
    }, [handleFileUpload])

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            handleFileUpload(file)
        }
    }, [handleFileUpload])

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-8"
        >
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

            {/* Processing Status */}
            {isProcessing && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mb-4 p-3 bg-primary-500/20 border border-primary-500/30 rounded-lg flex items-center gap-3"
                >
                    <Loader2 className="w-5 h-5 animate-spin text-primary-400" />
                    <div>
                        <p className="text-primary-300 font-medium">{processingStep}</p>
                        <p className="text-primary-400/70 text-sm">Please wait...</p>
                    </div>
                </motion.div>
            )}

            {/* File Selected */}
            {originalFile && !isProcessing && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 p-3 bg-green-500/20 border border-green-500/30 rounded-lg flex items-center gap-3"
                >
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <div>
                        <p className="text-green-300 font-medium">{originalFile.name}</p>
                        <p className="text-green-400/70 text-sm">
                            {(originalFile.size / 1024 / 1024).toFixed(2)} MB - Ready to remix!
                        </p>
                    </div>
                </motion.div>
            )}

            <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer
                    ${isProcessing
                        ? 'border-slate-700 opacity-50'
                        : 'border-slate-600 hover:border-primary-500'
                    }`}
            >
                <input
                    type="file"
                    accept=".mp3,.wav,.flac,audio/*"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="audio-upload"
                    disabled={isProcessing}
                />
                <label htmlFor="audio-upload" className={`cursor-pointer ${isProcessing ? 'pointer-events-none' : ''}`}>
                    <motion.div
                        animate={isProcessing ? {} : { y: [0, -10, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="mx-auto w-20 h-20 rounded-full bg-gradient-to-br from-primary-500/20 to-accent-500/20 flex items-center justify-center mb-6"
                    >
                        {isProcessing ? (
                            <Loader2 className="w-10 h-10 text-primary-400 animate-spin" />
                        ) : (
                            <Upload className="w-10 h-10 text-primary-400" />
                        )}
                    </motion.div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                        {isProcessing ? 'Processing...' : 'Drop your music file here'}
                    </h3>
                    <p className="text-slate-400 mb-4">
                        {isProcessing ? 'AI is analyzing your track' : 'or click to browse'}
                    </p>
                    <div className="flex items-center justify-center gap-4 text-sm text-slate-500">
                        <span className="flex items-center gap-1">
                            <FileAudio className="w-4 h-4" /> MP3
                        </span>
                        <span className="flex items-center gap-1">
                            <Music className="w-4 h-4" /> WAV
                        </span>
                        <span className="flex items-center gap-1">
                            <Music className="w-4 h-4" /> FLAC
                        </span>
                    </div>
                </label>
            </div>
        </motion.div>
    )
}

export default UploadArea