import React from 'react';

export interface ProcessingProgress {
    stage: string;
    current: number;
    total: number;
    percentage: number;
    message: string;
}

interface ProcessingStatusProps {
    isProcessing: boolean;
    progress?: ProcessingProgress | null;
    status?: string;
    error?: string | null;
    onCancel?: () => void;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
    isProcessing,
    progress,
    status,
    error,
    onCancel
}) => {
    if (!isProcessing && !error) return null;

    const getStageIcon = (stage: string) => {
        switch (stage?.toLowerCase()) {
            case 'uploading':
                return '📤';
            case 'analyzing':
                return '🔍';
            case 'separating':
            case 'separation':
                return '🎚️';
            case 'remixing':
            case 'generating':
                return '🎵';
            case 'mastering':
                return '🔊';
            case 'saving':
                return '💾';
            default:
                return '⏳';
        }
    };

    const getStageColor = (stage: string) => {
        switch (stage?.toLowerCase()) {
            case 'analyzing':
                return 'bg-blue-500';
            case 'separating':
            case 'separation':
                return 'bg-purple-500';
            case 'remixing':
            case 'generating':
                return 'bg-pink-500';
            case 'mastering':
                return 'bg-orange-500';
            default:
                return 'bg-cyan-500';
        }
    };

    if (error) {
        return (
            <div className="bg-red-900/50 border border-red-500 rounded-lg p-4">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">❌</span>
                    <div>
                        <h4 className="text-red-400 font-semibold">Processing Error</h4>
                        <p className="text-red-300 text-sm">{error}</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-gray-800/80 backdrop-blur border border-gray-700 rounded-lg p-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <div className="animate-spin text-xl">⚙️</div>
                    <span className="text-white font-medium">
                        {progress?.stage || status || 'Processing...'}
                    </span>
                </div>
                {onCancel && (
                    <button
                        onClick={onCancel}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm text-white transition"
                    >
                        Cancel
                    </button>
                )}
            </div>

            {/* Progress Bar */}
            {progress && (
                <div className="space-y-2">
                    <div className="flex justify-between text-sm text-gray-400">
                        <span>{progress.message}</span>
                        <span>{progress.percentage.toFixed(0)}%</span>
                    </div>

                    <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                        <div
                            className={`h-full ${getStageColor(progress.stage)} transition-all duration-300 ease-out`}
                            style={{ width: `${progress.percentage}%` }}
                        />
                    </div>

                    {progress.total > 0 && (
                        <div className="text-xs text-gray-500 text-right">
                            {progress.current} / {progress.total}
                        </div>
                    )}
                </div>
            )}

            {/* Processing Steps */}
            <div className="mt-4 flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center gap-1">
                    <span>📤</span>
                    <span>Upload</span>
                </div>
                <div className="h-px flex-1 mx-2 bg-gray-700" />
                <div className="flex items-center gap-1">
                    <span>🔍</span>
                    <span>Analyze</span>
                </div>
                <div className="h-px flex-1 mx-2 bg-gray-700" />
                <div className="flex items-center gap-1">
                    <span>🎚️</span>
                    <span>Separate</span>
                </div>
                <div className="h-px flex-1 mx-2 bg-gray-700" />
                <div className="flex items-center gap-1">
                    <span>🎵</span>
                    <span>Remix</span>
                </div>
                <div className="h-px flex-1 mx-2 bg-gray-700" />
                <div className="flex items-center gap-1">
                    <span>✅</span>
                    <span>Done</span>
                </div>
            </div>
        </div>
    );
};

export default ProcessingStatus;