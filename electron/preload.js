const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('electronAPI', {
    // File selection
    selectFile: () => ipcRenderer.invoke('select-file'),
    selectFolder: () => ipcRenderer.invoke('select-folder'),
    getAppPath: () => ipcRenderer.invoke('get-app-path'),
    getResourcesPath: () => ipcRenderer.invoke('get-resources-path'),

    // Platform info
    platform: process.platform,
    isMac: process.platform === 'darwin',
    isWindows: process.platform === 'win32',
    isLinux: process.platform === 'linux',

    // =====================
    // AUTO UPDATE API
    // =====================

    // Kiểm tra update
    checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),

    // Download update
    downloadUpdate: () => ipcRenderer.invoke('download-update'),

    // Install update
    installUpdate: () => ipcRenderer.invoke('install-update'),

    // Get app version
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),

    // Update event listeners
    onUpdateAvailable: (callback) => {
        ipcRenderer.on('update-available', (event, info) => callback(info));
    },

    onUpdateNotAvailable: (callback) => {
        ipcRenderer.on('update-not-available', () => callback());
    },

    onDownloadProgress: (callback) => {
        ipcRenderer.on('download-progress', (event, progress) => callback(progress));
    },

    onUpdateDownloaded: (callback) => {
        ipcRenderer.on('update-downloaded', (event, info) => callback(info));
    },

    onUpdateError: (callback) => {
        ipcRenderer.on('update-error', (event, error) => callback(error));
    },

    // Remove listeners
    removeAllUpdateListeners: () => {
        ipcRenderer.removeAllListeners('update-available');
        ipcRenderer.removeAllListeners('update-not-available');
        ipcRenderer.removeAllListeners('download-progress');
        ipcRenderer.removeAllListeners('update-downloaded');
        ipcRenderer.removeAllListeners('update-error');
    }
});