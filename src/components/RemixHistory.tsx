import { motion } from 'framer-motion'
import { History, Play, Download, Trash2 } from 'lucide-react'
import { useRemixStore } from '../store/remixStore'

function RemixHistory() {
    const { history } = useRemixStore()

    const formatDate = (date: Date) => {
        return new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }).format(date)
    }

    const formatDuration = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = seconds % 60
        return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-6"
        >
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <History className="w-5 h-5 text-primary-400" />
                Remix History
            </h3>

            {history.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                    <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No remixes yet</p>
                    <p className="text-sm mt-1">Your generated remixes will appear here</p>
                </div>
            ) : (
                <div className="space-y-3 max-h-80 overflow-y-auto">
                    {history.map((remix) => (
                        <div
                            key={remix.id}
                            className="flex items-center gap-3 p-3 bg-slate-800/50 rounded-lg hover:bg-slate-800 transition-colors"
                        >
                            <button className="p-2 rounded-lg bg-primary-500/20 hover:bg-primary-500/30 transition-colors">
                                <Play className="w-4 h-4 text-primary-400" />
                            </button>
                            <div className="flex-1 min-w-0">
                                <div className="font-medium text-white truncate">
                                    {remix.originalName}
                                </div>
                                <div className="flex items-center gap-2 text-xs text-slate-400">
                                    <span className="px-1.5 py-0.5 bg-slate-700 rounded">
                                        {remix.genre}
                                    </span>
                                    <span>{formatDuration(remix.duration)}</span>
                                    <span>{formatDate(remix.createdAt)}</span>
                                </div>
                            </div>
                            <div className="flex items-center gap-1">
                                <button className="p-2 rounded-lg hover:bg-slate-700 transition-colors">
                                    <Download className="w-4 h-4 text-slate-400" />
                                </button>
                                <button className="p-2 rounded-lg hover:bg-red-500/20 transition-colors">
                                    <Trash2 className="w-4 h-4 text-slate-400 hover:text-red-400" />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </motion.div>
    )
}

export default RemixHistory