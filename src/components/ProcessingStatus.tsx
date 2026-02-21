import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import { useRemixStore } from '../store/remixStore'

function ProcessingStatus() {
    const { processingProgress, processingStep } = useRemixStore()

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-6"
        >
            <div className="flex items-center gap-4 mb-4">
                <Loader2 className="w-8 h-8 text-primary-400 animate-spin" />
                <div>
                    <h3 className="text-lg font-semibold text-white">Processing Audio</h3>
                    <p className="text-sm text-slate-400">{processingStep || 'Analyzing...'}</p>
                </div>
            </div>

            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${processingProgress}%` }}
                    className="h-full bg-gradient-to-r from-primary-500 to-accent-500"
                />
            </div>

            <div className="text-right text-sm text-slate-400 mt-2">
                {processingProgress}%
            </div>
        </motion.div>
    )
}

export default ProcessingStatus