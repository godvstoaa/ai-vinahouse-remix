import { motion } from 'framer-motion'
import { Volume2, VolumeX, Ear } from 'lucide-react'
import { useRemixStore, StemTrack } from '../store/remixStore'

function StemPlayer() {
    const { stems, updateStem } = useRemixStore()

    const handleVolumeChange = (id: string, volume: number) => {
        updateStem(id, { volume })
    }

    const toggleMute = (id: string, currentMuted: boolean) => {
        updateStem(id, { muted: !currentMuted })
    }

    const toggleSolo = (id: string, currentSolo: boolean) => {
        // When soloing one track, mute others
        if (!currentSolo) {
            stems.forEach(stem => {
                if (stem.id === id) {
                    updateStem(stem.id, { solo: true, muted: false })
                } else {
                    updateStem(stem.id, { solo: false })
                }
            })
        } else {
            updateStem(id, { solo: false })
        }
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-6"
        >
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Ear className="w-5 h-5 text-primary-400" />
                Stem Mixer
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {stems.map((stem) => (
                    <StemControl
                        key={stem.id}
                        stem={stem}
                        onVolumeChange={(vol) => handleVolumeChange(stem.id, vol)}
                        onMuteToggle={() => toggleMute(stem.id, stem.muted)}
                        onSoloToggle={() => toggleSolo(stem.id, stem.solo)}
                    />
                ))}
            </div>
        </motion.div>
    )
}

interface StemControlProps {
    stem: StemTrack
    onVolumeChange: (volume: number) => void
    onMuteToggle: () => void
    onSoloToggle: () => void
}

function StemControl({ stem, onVolumeChange, onMuteToggle, onSoloToggle }: StemControlProps) {
    return (
        <div
            className={`p-4 rounded-xl border transition-all ${stem.solo
                    ? 'border-primary-500 bg-primary-500/10'
                    : stem.muted
                        ? 'border-slate-700 bg-slate-800/50 opacity-50'
                        : 'border-slate-700 bg-slate-800/30'
                }`}
        >
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: stem.color }}
                    />
                    <span className="font-medium text-white">{stem.name}</span>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={onSoloToggle}
                        className={`px-2 py-1 text-xs rounded font-medium transition-colors ${stem.solo
                                ? 'bg-primary-500 text-white'
                                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                            }`}
                    >
                        S
                    </button>
                    <button
                        onClick={onMuteToggle}
                        className={`p-1 rounded transition-colors ${stem.muted
                                ? 'bg-red-500/20 text-red-400'
                                : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                            }`}
                    >
                        {stem.muted ? (
                            <VolumeX className="w-4 h-4" />
                        ) : (
                            <Volume2 className="w-4 h-4" />
                        )}
                    </button>
                </div>
            </div>

            <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={stem.volume}
                onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
                className="w-full"
                disabled={stem.muted}
            />

            <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>0%</span>
                <span>{Math.round(stem.volume * 100)}%</span>
                <span>100%</span>
            </div>
        </div>
    )
}

export default StemPlayer