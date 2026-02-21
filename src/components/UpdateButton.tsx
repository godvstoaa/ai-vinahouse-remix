import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, Check, AlertCircle, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface UpdateInfo {
    version: string;
    releaseDate?: string;
    releaseNotes?: string;
}

interface DownloadProgress {
    percent: number;
    transferred: number;
    total: number;
}

interface UpdateError {
    message: string;
}

// Declare electronAPI
declare global {
    interface Window {
        electronAPI?: {
            checkForUpdates: () => Promise<any>;
            downloadUpdate: () => Promise<any>;
            installUpdate: () => Promise<void>;
            getAppVersion: () => Promise<string>;
            onUpdateAvailable: (callback: (info: UpdateInfo) => void) => void;
            onUpdateNotAvailable: (callback: () => void) => void;
            onDownloadProgress: (callback: (progress: DownloadProgress) => void) => void;
            onUpdateDownloaded: (callback: (info: UpdateInfo) => void) => void;
            onUpdateError: (callback: (error: UpdateError) => void) => void;
            removeAllUpdateListeners: () => void;
            isMac: boolean;
        };
    }
}

type UpdateStatus = 'idle' | 'checking' | 'available' | 'downloading' | 'downloaded' | 'error';

const UpdateButton: React.FC = () => {
    const [status, setStatus] = useState<UpdateStatus>('idle');
    const [currentVersion, setCurrentVersion] = useState<string>('');
    const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
    const [downloadProgress, setDownloadProgress] = useState(0);
    const [errorMessage, setErrorMessage] = useState('');
    const [showNotification, setShowNotification] = useState(false);

    // Check if running in Electron
    const isElectron = window.electronAPI !== undefined;

    useEffect(() => {
        if (!isElectron) return;

        // Get current version
        window.electronAPI?.getAppVersion().then(setCurrentVersion);

        // Setup listeners
        window.electronAPI?.onUpdateAvailable((info) => {
            setStatus('available');
            setUpdateInfo(info);
            setShowNotification(true);
        });

        window.electronAPI?.onUpdateNotAvailable(() => {
            setStatus('idle');
        });

        window.electronAPI?.onDownloadProgress((progress) => {
            setDownloadProgress(progress.percent);
        });

        window.electronAPI?.onUpdateDownloaded(() => {
            setStatus('downloaded');
            setShowNotification(true);
        });

        window.electronAPI?.onUpdateError((error) => {
            setStatus('error');
            setErrorMessage(error.message);
            setShowNotification(true);
        });

        return () => {
            window.electronAPI?.removeAllUpdateListeners();
        };
    }, [isElectron]);

    const checkForUpdates = async () => {
        if (!isElectron) return;
        setStatus('checking');
        try {
            await window.electronAPI?.checkForUpdates();
        } catch (error) {
            setStatus('error');
            setErrorMessage('Failed to check for updates');
        }
    };

    const downloadUpdate = async () => {
        if (!isElectron) return;
        setStatus('downloading');
        setDownloadProgress(0);
        try {
            await window.electronAPI?.downloadUpdate();
        } catch (error) {
            setStatus('error');
            setErrorMessage('Failed to download update');
        }
    };

    const installUpdate = () => {
        if (!isElectron) return;
        window.electronAPI?.installUpdate();
    };

    if (!isElectron) {
        return null; // Don't show in web mode
    }

    return (
        <>
            {/* Update Button in Header */}
            <div className="flex items-center gap-2">
                {/* Current Version */}
                <span className="text-xs text-gray-500">v{currentVersion}</span>

                {/* Update Button */}
                <motion.button
                    onClick={status === 'available' ? downloadUpdate : status === 'downloaded' ? installUpdate : checkForUpdates}
                    disabled={status === 'checking' || status === 'downloading'}
                    className={`
                        flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all
                        ${status === 'idle' || status === 'error' ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' : ''}
                        ${status === 'checking' ? 'bg-gray-700 text-gray-400 cursor-wait' : ''}
                        ${status === 'available' ? 'bg-orange-600 hover:bg-orange-500 text-white' : ''}
                        ${status === 'downloading' ? 'bg-blue-600 text-white' : ''}
                        ${status === 'downloaded' ? 'bg-green-600 hover:bg-green-500 text-white' : ''}
                    `}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    {status === 'idle' && <RefreshCw className="w-4 h-4" />}
                    {status === 'checking' && <RefreshCw className="w-4 h-4 animate-spin" />}
                    {status === 'available' && <Download className="w-4 h-4" />}
                    {status === 'downloading' && (
                        <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            <span>{downloadProgress}%</span>
                        </>
                    )}
                    {status === 'downloaded' && <Check className="w-4 h-4" />}
                    {status === 'error' && <AlertCircle className="w-4 h-4" />}

                    <span className="hidden sm:inline">
                        {status === 'idle' && 'Check Update'}
                        {status === 'checking' && 'Checking...'}
                        {status === 'available' && `Update v${updateInfo?.version}`}
                        {status === 'downloading' && 'Downloading...'}
                        {status === 'downloaded' && 'Install Now'}
                        {status === 'error' && 'Retry'}
                    </span>
                </motion.button>
            </div>

            {/* Notification Popup */}
            <AnimatePresence>
                {showNotification && (
                    <motion.div
                        initial={{ opacity: 0, y: -20, x: '-50%' }}
                        animate={{ opacity: 1, y: 0, x: '-50%' }}
                        exit={{ opacity: 0, y: -20 }}
                        className="fixed top-4 left-1/2 z-50 bg-gray-800 border border-gray-700 rounded-xl shadow-2xl p-4 max-w-md"
                    >
                        <div className="flex items-start gap-3">
                            {status === 'available' && (
                                <div className="w-10 h-10 rounded-full bg-orange-600 flex items-center justify-center">
                                    <Download className="w-5 h-5 text-white" />
                                </div>
                            )}
                            {status === 'downloaded' && (
                                <div className="w-10 h-10 rounded-full bg-green-600 flex items-center justify-center">
                                    <Check className="w-5 h-5 text-white" />
                                </div>
                            )}
                            {status === 'error' && (
                                <div className="w-10 h-10 rounded-full bg-red-600 flex items-center justify-center">
                                    <AlertCircle className="w-5 h-5 text-white" />
                                </div>
                            )}

                            <div className="flex-1">
                                {status === 'available' && (
                                    <>
                                        <h4 className="font-semibold text-white">
                                            Update Available! 🎉
                                        </h4>
                                        <p className="text-sm text-gray-400 mt-1">
                                            Version {updateInfo?.version} is ready to download.
                                        </p>
                                        <div className="flex gap-2 mt-3">
                                            <button
                                                onClick={() => { downloadUpdate(); setShowNotification(false); }}
                                                className="px-3 py-1.5 bg-orange-600 hover:bg-orange-500 rounded-lg text-sm font-medium"
                                            >
                                                Download
                                            </button>
                                            <button
                                                onClick={() => setShowNotification(false)}
                                                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
                                            >
                                                Later
                                            </button>
                                        </div>
                                    </>
                                )}
                                {status === 'downloaded' && (
                                    <>
                                        <h4 className="font-semibold text-white">
                                            Ready to Install! ✅
                                        </h4>
                                        <p className="text-sm text-gray-400 mt-1">
                                            Restart the app to complete the update.
                                        </p>
                                        <button
                                            onClick={installUpdate}
                                            className="mt-3 px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded-lg text-sm font-medium"
                                        >
                                            Restart & Install
                                        </button>
                                    </>
                                )}
                                {status === 'error' && (
                                    <>
                                        <h4 className="font-semibold text-white">
                                            Update Failed
                                        </h4>
                                        <p className="text-sm text-red-400 mt-1">
                                            {errorMessage}
                                        </p>
                                        <button
                                            onClick={() => setShowNotification(false)}
                                            className="mt-3 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
                                        >
                                            Close
                                        </button>
                                    </>
                                )}
                            </div>

                            <button
                                onClick={() => setShowNotification(false)}
                                className="text-gray-500 hover:text-gray-300"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Download Progress Bar (fullscreen overlay) */}
            <AnimatePresence>
                {status === 'downloading' && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 flex items-center justify-center"
                    >
                        <motion.div
                            initial={{ scale: 0.9 }}
                            animate={{ scale: 1 }}
                            className="bg-gray-800 rounded-2xl p-6 shadow-2xl max-w-sm w-full mx-4"
                        >
                            <h3 className="text-lg font-semibold text-white mb-4">
                                Downloading Update...
                            </h3>
                            <div className="w-full bg-gray-700 rounded-full h-3 mb-2">
                                <motion.div
                                    className="bg-blue-500 h-3 rounded-full"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${downloadProgress}%` }}
                                />
                            </div>
                            <p className="text-sm text-gray-400 text-center">
                                {downloadProgress}% complete
                            </p>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default UpdateButton;