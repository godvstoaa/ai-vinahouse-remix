import { create } from 'zustand'
import * as api from '../services/api'

export interface StemTrack {
    id: string
    name: string
    file: string | null
    volume: number
    muted: boolean
    solo: boolean
    color: string
}

export interface RemixSettings {
    genre: string
    bpm: number
    originalBpm: number
    energyLevel: number
    reverb: number
    delay: number
    filter: number
    style: string
}

export interface RemixHistory {
    id: string
    originalName: string
    genre: string
    createdAt: Date
    duration: number
    fileUrl: string
}

export interface AudioAnalysis {
    bpm: number
    key: string
    genre: string
    energy: number
    danceability: number
    duration: number
    sections: Array<{
        start: number
        end: number
        label: string
    }>
}

interface RemixState {
    // File state
    originalFile: File | null
    originalUrl: string | null
    jobId: string | null
    isProcessing: boolean
    processingProgress: number
    processingStep: string

    // Analysis
    audioAnalysis: AudioAnalysis | null

    // Stems
    stems: StemTrack[]

    // Remix settings
    remixSettings: RemixSettings

    // Remix output
    remixUrl: string | null
    isRemixing: boolean
    remixProgress: number

    // History
    history: RemixHistory[]

    // API Connected
    apiConnected: boolean

    // Actions
    setOriginalFile: (file: File | null) => void
    setJobId: (jobId: string | null) => void
    setProcessing: (isProcessing: boolean, progress?: number, step?: string) => void
    setAudioAnalysis: (analysis: AudioAnalysis | null) => void
    setStems: (stems: StemTrack[]) => void
    updateStem: (id: string, updates: Partial<StemTrack>) => void
    updateRemixSettings: (settings: Partial<RemixSettings>) => void
    setRemixUrl: (url: string | null) => void
    setRemixing: (isRemixing: boolean, progress?: number) => void
    addToHistory: (remix: RemixHistory) => void
    setApiConnected: (connected: boolean) => void

    // API Thunks
    uploadFile: (file: File) => Promise<api.UploadResponse | undefined>
    processAudio: () => Promise<void>
    createRemix: () => Promise<void>

    reset: () => void
}

const defaultStems: StemTrack[] = [
    { id: 'vocals', name: 'Vocals', file: null, volume: 1, muted: false, solo: false, color: '#ef4444' },
    { id: 'drums', name: 'Drums', file: null, volume: 1, muted: false, solo: false, color: '#22c55e' },
    { id: 'bass', name: 'Bass', file: null, volume: 1, muted: false, solo: false, color: '#3b82f6' },
    { id: 'other', name: 'Other', file: null, volume: 1, muted: false, solo: false, color: '#a855f7' },
]

const defaultRemixSettings: RemixSettings = {
    genre: 'edm',
    bpm: 128,
    originalBpm: 128,
    energyLevel: 0.7,
    reverb: 0.3,
    delay: 0.2,
    filter: 0,
    style: 'club',
}

export const useRemixStore = create<RemixState>((set, get) => ({
    originalFile: null,
    originalUrl: null,
    jobId: null,
    isProcessing: false,
    processingProgress: 0,
    processingStep: '',
    audioAnalysis: null,
    stems: defaultStems,
    remixSettings: defaultRemixSettings,
    remixUrl: null,
    isRemixing: false,
    remixProgress: 0,
    history: [],
    apiConnected: false,

    setOriginalFile: (file) => set(() => ({
        originalFile: file,
        originalUrl: file ? URL.createObjectURL(file) : null,
        remixUrl: null,
        stems: defaultStems,
        audioAnalysis: null,
        jobId: null,
    })),

    setJobId: (jobId) => set({ jobId }),

    setProcessing: (isProcessing, progress = 0, step = '') => set({
        isProcessing,
        processingProgress: progress,
        processingStep: step,
    }),

    setAudioAnalysis: (analysis) => set((state) => ({
        audioAnalysis: analysis,
        remixSettings: analysis ? {
            ...state.remixSettings,
            originalBpm: analysis.bpm,
            bpm: analysis.bpm,
        } : state.remixSettings,
    })),

    setStems: (stems) => set({ stems }),

    updateStem: (id, updates) => set((state) => ({
        stems: state.stems.map((stem) =>
            stem.id === id ? { ...stem, ...updates } : stem
        ),
    })),

    updateRemixSettings: (settings) => set((state) => ({
        remixSettings: { ...state.remixSettings, ...settings },
    })),

    setRemixUrl: (url) => set({ remixUrl: url }),

    setRemixing: (isRemixing, progress = 0) => set({
        isRemixing,
        remixProgress: progress,
    }),

    addToHistory: (remix) => set((state) => ({
        history: [remix, ...state.history].slice(0, 50),
    })),

    setApiConnected: (connected) => set({ apiConnected: connected }),

    // Upload file to backend
    uploadFile: async (file: File) => {
        try {
            set({ isProcessing: true, processingProgress: 10, processingStep: 'Uploading...' })

            const response = await api.uploadAudio(file)
            set({ jobId: response.job_id })

            return response
        } catch (error) {
            set({ isProcessing: false, processingStep: 'Upload failed' })
            throw error
        }
    },

    // Process audio (analyze + separate)
    processAudio: async () => {
        const { jobId } = get()
        if (!jobId) return

        try {
            // Start analysis
            set({ processingStep: 'Analyzing audio...', processingProgress: 20 })
            await api.analyzeAudio(jobId)

            // Poll for analysis completion
            set({ processingStep: 'Analyzing...', processingProgress: 30 })
            await new Promise<void>((resolve, reject) => {
                const poll = async () => {
                    try {
                        const status = await api.getJobStatus(jobId)
                        if (status.status === 'analyzed') {
                            // Update analysis in store
                            if (status.analysis) {
                                set({
                                    audioAnalysis: {
                                        bpm: status.analysis.bpm,
                                        key: status.analysis.key,
                                        genre: status.analysis.genre || 'unknown',
                                        energy: status.analysis.energy,
                                        danceability: status.analysis.danceability,
                                        duration: status.analysis.duration,
                                        sections: status.analysis.sections || []
                                    }
                                })
                            }
                            resolve()
                        } else if (status.status === 'error') {
                            reject(new Error(status.error))
                        } else {
                            setTimeout(poll, 1000)
                        }
                    } catch (e) {
                        reject(e)
                    }
                }
                poll()
            })

            // Start separation
            set({ processingStep: 'Separating stems...', processingProgress: 50 })
            await api.separateStems(jobId)

            // Poll for separation completion
            await new Promise<void>((resolve, reject) => {
                const poll = async () => {
                    try {
                        const status = await api.getJobStatus(jobId)
                        if (status.status === 'separated') {
                            // Update stems with URLs
                            if (status.stems) {
                                const stemUpdates: StemTrack[] = defaultStems.map(stem => ({
                                    ...stem,
                                    file: api.getStemUrl(jobId, stem.id)
                                }))
                                set({ stems: stemUpdates })
                            }
                            resolve()
                        } else if (status.status === 'error') {
                            reject(new Error(status.error))
                        } else {
                            setTimeout(poll, 1000)
                        }
                    } catch (e) {
                        reject(e)
                    }
                }
                poll()
            })

            set({ isProcessing: false, processingProgress: 100, processingStep: 'Ready' })
        } catch (error) {
            set({ isProcessing: false, processingStep: 'Processing failed' })
            throw error
        }
    },

    // Create remix
    createRemix: async () => {
        const { jobId, remixSettings, originalFile } = get()
        if (!jobId) return

        try {
            set({ isRemixing: true, remixProgress: 20 })

            const settings: api.RemixSettings = {
                genre: remixSettings.genre,
                style: remixSettings.style,
                bpm: remixSettings.bpm,
                original_bpm: remixSettings.originalBpm,
                energy_level: remixSettings.energyLevel,
                reverb: remixSettings.reverb,
                delay: remixSettings.delay,
                filter: remixSettings.filter,
            }

            set({ remixProgress: 40 })
            await api.generateRemix(jobId, settings)

            // Poll for completion
            set({ remixProgress: 60 })
            await new Promise<void>((resolve, reject) => {
                const poll = async () => {
                    try {
                        const status = await api.getJobStatus(jobId)
                        if (status.status === 'completed') {
                            // Set remix URL
                            const remixUrl = api.getRemixUrl(jobId)
                            set({ remixUrl })

                            // Add to history
                            if (originalFile) {
                                get().addToHistory({
                                    id: jobId,
                                    originalName: originalFile.name,
                                    genre: remixSettings.genre,
                                    createdAt: new Date(),
                                    duration: 0,
                                    fileUrl: remixUrl
                                })
                            }

                            resolve()
                        } else if (status.status === 'error') {
                            reject(new Error(status.error))
                        } else {
                            set({ remixProgress: (get().remixProgress + 5) % 90 })
                            setTimeout(poll, 1000)
                        }
                    } catch (e) {
                        reject(e)
                    }
                }
                poll()
            })

            set({ isRemixing: false, remixProgress: 100 })
        } catch (error) {
            set({ isRemixing: false, remixProgress: 0 })
            throw error
        }
    },

    reset: () => set({
        originalFile: null,
        originalUrl: null,
        jobId: null,
        isProcessing: false,
        processingProgress: 0,
        processingStep: '',
        audioAnalysis: null,
        stems: defaultStems,
        remixSettings: defaultRemixSettings,
        remixUrl: null,
        isRemixing: false,
        remixProgress: 0,
    }),
}))
