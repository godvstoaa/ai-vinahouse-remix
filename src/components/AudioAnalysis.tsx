import { motion } from 'framer-motion'
import { BarChart2, Music, Activity, Clock, Zap } from 'lucide-react'
import { useRemixStore } from '../store/remixStore'

function AudioAnalysis() {
    const { audioAnalysis, originalFile, isProcessing } = useRemixStore()

    // Demo data when no analysis is available
    const demoAnalysis = {
        bpm: 128,
        key: 'A Minor',
        genre: 'EDM',
        energy: 0.75,
        danceability: 0.82,
        duration: 215,
        sections: [
            { start: 0, end: 15, label: 'Intro' },
            { start: 15, end: 45, label: 'Verse 1' },
            { start: 45, end: 75, label: 'Chorus' },
            { start: 75, end: 105, label: 'Verse 2' },
            { start: 105, end: 135, label: 'Chorus' },
            { start: 135, end: 175, label: 'Bridge' },
            { start: 175, end: 215, label: 'Outro' },
        ]
    }

    const analysis = audioAnalysis || (originalFile ? demoAnalysis : null)

    if (!analysis && !isProcessing) {
        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-2xl p-6"
            >
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-primary-400" />
                    Audio Analysis
                </h3>
                <div className="text-center py-8 text-slate-400">
                    <Music className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Upload a track to see analysis</p>
                </div>
            </motion.div>
        )
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-6"
        >
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-primary-400" />
                Audio Analysis
            </h3>

            {/* Key Metrics */}
            <div className="grid grid-cols-2 gap-3 mb-6">
                <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">BPM</div>
                    <div className="text-2xl font-bold text-white">{analysis?.bpm || '--'}</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Key</div>
                    <div className="text-2xl font-bold text-white">{analysis?.key || '--'}</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Genre</div>
                    <div className="text-lg font-semibold text-primary-400">{analysis?.genre || '--'}</div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Duration</div>
                    <div className="text-lg font-semibold text-white">
                        {Math.floor((analysis?.duration || 0) / 60)}:{((analysis?.duration || 0) % 60).toString().padStart(2, '0')}
                    </div>
                </div>
            </div>

            {/* Energy & Danceability */}
            <div className="space-y-4 mb-6">
                <div>
                    <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-400 flex items-center gap-1">
                            <Zap className="w-4 h-4" /> Energy
                        </span>
                        <span className="text-white">{Math.round((analysis?.energy || 0) * 100)}%</span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${(analysis?.energy || 0) * 100}%` }}
                            className="h-full bg-gradient-to-r from-orange-500 to-red-500"
                        />
                    </div>
                </div>
                <div>
                    <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-400 flex items-center gap-1">
                            <Activity className="w-4 h-4" /> Danceability
                        </span>
                        <span className="text-white">{Math.round((analysis?.danceability || 0) * 100)}%</span>
                    </div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${(analysis?.danceability || 0) * 100}%` }}
                            className="h-full bg-gradient-to-r from-green-500 to-emerald-500"
                        />
                    </div>
                </div>
            </div>

            {/* Song Structure */}
            <div>
                <div className="text-sm text-slate-400 mb-2 flex items-center gap-1">
                    <Clock className="w-4 h-4" /> Structure
                </div>
                <div className="flex h-8 rounded-lg overflow-hidden">
                    {analysis?.sections?.map((section, index) => (
                        <div
                            key={index}
                            className={`flex-1 flex items-center justify-center text-xs font-medium ${index % 2 === 0 ? 'bg-primary-500/30' : 'bg-accent-500/30'
                                }`}
                            title={`${section.label}: ${section.start}s - ${section.end}s`}
                        >
                            {section.label}
                        </div>
                    ))}
                </div>
            </div>
        </motion.div>
    )
}

export default AudioAnalysis