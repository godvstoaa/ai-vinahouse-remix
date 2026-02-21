import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react'
import { useRemixStore } from '../store/remixStore'
import WaveSurfer from 'wavesurfer.js'

function WaveformDisplay() {
    const waveformRef = useRef<HTMLDivElement>(null)
    const wavesurfer = useRef<WaveSurfer | null>(null)
    const [isPlaying, setIsPlaying] = useState(false)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)

    const { originalUrl, originalFile } = useRemixStore()

    useEffect(() => {
        if (!waveformRef.current || !originalUrl) return

        wavesurfer.current = WaveSurfer.create({
            container: waveformRef.current,
            waveColor: '#38bdf8',
            progressColor: '#0ea5e9',
            cursorColor: '#d946ef',
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            height: 80,
            normalize: true,
        })

        wavesurfer.current.load(originalUrl)

        wavesurfer.current.on('ready', () => {
            setDuration(wavesurfer.current?.getDuration() || 0)
        })

        wavesurfer.current.on('audioprocess', () => {
            setCurrentTime(wavesurfer.current?.getCurrentTime() || 0)
        })

        wavesurfer.current.on('play', () => setIsPlaying(true))
        wavesurfer.current.on('pause', () => setIsPlaying(false))

        return () => {
            wavesurfer.current?.destroy()
        }
    }, [originalUrl])

    const togglePlay = () => {
        wavesurfer.current?.playPause()
    }

    const skipBack = () => {
        if (wavesurfer.current) {
            const newTime = Math.max(0, wavesurfer.current.getCurrentTime() - 10)
            wavesurfer.current.setTime(newTime)
        }
    }

    const skipForward = () => {
        if (wavesurfer.current) {
            const newTime = Math.min(duration, wavesurfer.current.getCurrentTime() + 10)
            wavesurfer.current.setTime(newTime)
        }
    }

    const formatTime = (time: number) => {
        const minutes = Math.floor(time / 60)
        const seconds = Math.floor(time % 60)
        return `${minutes}:${seconds.toString().padStart(2, '0')}`
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-6"
        >
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary-500"></span>
                    {originalFile?.name || 'Original Track'}
                </h3>
                <span className="text-sm text-slate-400">
                    {formatTime(currentTime)} / {formatTime(duration)}
                </span>
            </div>

            <div ref={waveformRef} className="mb-4 rounded-lg overflow-hidden" />

            <div className="flex items-center justify-center gap-4">
                <button
                    onClick={skipBack}
                    className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
                >
                    <SkipBack className="w-5 h-5 text-white" />
                </button>
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={togglePlay}
                    className="p-4 rounded-full bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-400 hover:to-accent-400 transition-all"
                >
                    {isPlaying ? (
                        <Pause className="w-6 h-6 text-white" />
                    ) : (
                        <Play className="w-6 h-6 text-white ml-0.5" />
                    )}
                </motion.button>
                <button
                    onClick={skipForward}
                    className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
                >
                    <SkipForward className="w-5 h-5 text-white" />
                </button>
            </div>
        </motion.div>
    )
}

export default WaveformDisplay