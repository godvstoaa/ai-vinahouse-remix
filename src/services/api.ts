/**
 * API Service for AI Remix Backend
 * Handles all communication with the Python FastAPI backend
 */

const API_BASE_URL = 'http://localhost:8000'

export interface UploadResponse {
    job_id: string
    filename: string
    status: string
    message: string
}

export interface AnalysisResult {
    duration: number
    bpm: number
    key: string
    energy: number
    danceability: number
    genre?: string
    sections: Array<{
        start: number
        end: number
        label: string
    }>
}

export interface StemsResult {
    vocals: string
    drums: string
    bass: string
    other: string
}

export interface RemixSettings {
    genre: string
    style: string
    bpm: number
    original_bpm: number
    energy_level: number
    reverb: number
    delay: number
    filter: number
}

export interface JobStatus {
    job_id: string
    status: 'uploaded' | 'analyzing' | 'analyzed' | 'separating' | 'separated' | 'remixing' | 'completed' | 'error'
    analysis?: AnalysisResult
    stems?: StemsResult
    remix?: {
        path: string
        url: string
    }
    error?: string
}

/**
 * Upload audio file to the server
 */
export async function uploadAudio(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData
    })

    if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
    }

    return response.json()
}

/**
 * Start audio analysis
 */
export async function analyzeAudio(jobId: string): Promise<{ job_id: string; status: string }> {
    const response = await fetch(`${API_BASE_URL}/api/analyze/${jobId}`, {
        method: 'POST'
    })

    if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
    }

    return response.json()
}

/**
 * Start source separation
 */
export async function separateStems(jobId: string): Promise<{ job_id: string; status: string }> {
    const response = await fetch(`${API_BASE_URL}/api/separate/${jobId}`, {
        method: 'POST'
    })

    if (!response.ok) {
        throw new Error(`Separation failed: ${response.statusText}`)
    }

    return response.json()
}

/**
 * Generate remix
 */
export async function generateRemix(jobId: string, settings: RemixSettings): Promise<{ job_id: string; status: string }> {
    const response = await fetch(`${API_BASE_URL}/api/remix/${jobId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    })

    if (!response.ok) {
        throw new Error(`Remix generation failed: ${response.statusText}`)
    }

    return response.json()
}

/**
 * Get job status
 */
export async function getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await fetch(`${API_BASE_URL}/api/status/${jobId}`)

    if (!response.ok) {
        throw new Error(`Status check failed: ${response.statusText}`)
    }

    return response.json()
}

/**
 * Get stem audio URL
 */
export function getStemUrl(jobId: string, stemName: string): string {
    return `${API_BASE_URL}/api/download/stem/${jobId}/${stemName}`
}

/**
 * Get remix audio URL
 */
export function getRemixUrl(jobId: string): string {
    return `${API_BASE_URL}/api/download/remix/${jobId}`
}

/**
 * Delete job
 */
export async function deleteJob(jobId: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/api/job/${jobId}`, {
        method: 'DELETE'
    })

    if (!response.ok) {
        throw new Error(`Delete failed: ${response.statusText}`)
    }

    return response.json()
}

/**
 * Check API health
 */
export async function checkHealth(): Promise<{ status: string; models_loaded: boolean }> {
    const response = await fetch(`${API_BASE_URL}/health`)

    if (!response.ok) {
        throw new Error(`Health check failed: ${response.statusText}`)
    }

    return response.json()
}

/**
 * Poll job status until completion
 */
export async function pollJobStatus(
    jobId: string,
    onProgress: (status: JobStatus) => void,
    intervalMs: number = 1000
): Promise<JobStatus> {
    return new Promise((resolve, reject) => {
        const poll = async () => {
            try {
                const status = await getJobStatus(jobId)
                onProgress(status)

                if (status.status === 'completed') {
                    resolve(status)
                } else if (status.status === 'error') {
                    reject(new Error(status.error || 'Processing failed'))
                } else {
                    setTimeout(poll, intervalMs)
                }
            } catch (error) {
                reject(error)
            }
        }

        poll()
    })
}