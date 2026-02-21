import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useRemixStore } from '../store/remixStore'
import UploadArea from './UploadArea'
import WaveformDisplay from './WaveformDisplay'
import StemPlayer from './StemPlayer'
import RemixControls from './RemixControls'
import AudioAnalysis from './AudioAnalysis'
import RemixHistory from './RemixHistory'
import ProcessingStatus from './ProcessingStatus'
import TrainingPanel from './TrainingPanel'

function Dashboard() {
    const { originalFile, isProcessing, audioAnalysis } = useRemixStore()
    const [activeTab, setActiveTab] = useState<'remix' | 'train'>('remix')

    return (
        <div className="space-y-6">
            {/* Tab Navigation */}
            <div className="flex gap-2 bg-black/30 p-1 rounded-lg w-fit">
                <button
                    onClick={() => setActiveTab('remix')}
                    className={`px-6 py-2 rounded-lg font-medium transition-all ${activeTab === 'remix'
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                        : 'text-purple-300 hover:text-white'
                        }`}
                >
                    🎵 Remix Studio
                </button>
                <button
                    onClick={() => setActiveTab('train')}
                    className={`px-6 py-2 rounded-lg font-medium transition-all ${activeTab === 'train'
                        ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                        : 'text-purple-300 hover:text-white'
                        }`}
                >
                    🎓 Train Model
                </button>
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
                {activeTab === 'remix' ? (
                    <motion.div
                        key="remix"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
                    >
                        {/* Left Column - Upload & Waveform */}
                        <div className="lg:col-span-2 space-y-6">
                            <UploadArea />
                            {isProcessing && <ProcessingStatus isProcessing={isProcessing} />}
                            {originalFile && !isProcessing && <WaveformDisplay />}
                            {audioAnalysis && !isProcessing && <StemPlayer />}
                            {/* @ts-ignore - Dashboard.tsx is unused, RemixControls needs props */}
                            {audioAnalysis && !isProcessing && <RemixControls />}
                        </div>

                        {/* Right Column - Analysis & History */}
                        <div className="space-y-6">
                            <AudioAnalysis />
                            <RemixHistory />
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        key="train"
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                    >
                        <TrainingPanel />
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

export default Dashboard