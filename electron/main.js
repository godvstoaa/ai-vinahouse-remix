const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { autoUpdater } = require('electron-updater');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess = null;

// Kiểm tra môi trường development hay production
const isDev = !app.isPackaged;

// Đường dẫn đến Python backend
const getPythonPath = () => {
    if (isDev) {
        return 'python';
    }

    // Trong production, sử dụng Python đi kèm
    const platform = process.platform;
    if (platform === 'darwin') {
        return path.join(process.resourcesPath, 'backend', 'venv', 'bin', 'python');
    } else if (platform === 'win32') {
        return path.join(process.resourcesPath, 'backend', 'venv', 'Scripts', 'python.exe');
    }
    return 'python3';
};

// Đường dẫn đến backend
const getBackendPath = () => {
    if (isDev) {
        return path.join(__dirname, '..', 'backend', 'main.py');
    }
    return path.join(process.resourcesPath, 'backend', 'main.py');
};

// Khởi động Python backend
const startPythonBackend = () => {
    const pythonPath = getPythonPath();
    const backendPath = getBackendPath();

    console.log('Starting Python backend...');
    console.log('Python path:', pythonPath);
    console.log('Backend path:', backendPath);

    // Kill process cũ nếu có
    if (pythonProcess) {
        pythonProcess.kill();
    }

    pythonProcess = spawn(pythonPath, [backendPath], {
        cwd: path.dirname(backendPath),
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
        pythonProcess = null;
    });
};

// Tạo main window
const createWindow = () => {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 700,
        title: 'AI Vinahouse Remix Pro',
        icon: path.join(__dirname, '..', 'build', 'icon.png'),
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        show: false
    });

    // Load frontend
    if (isDev) {
        // Development: Load từ Vite dev server
        mainWindow.loadURL('http://localhost:3000');
        mainWindow.webContents.openDevTools();
    } else {
        // Production: Load từ file đã build
        mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
    }

    // Hiển thị window khi ready
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
};

// =====================
// AUTO UPDATE LOGIC
// =====================

// Cấu hình auto updater
autoUpdater.autoDownload = false;
autoUpdater.autoInstallOnAppQuit = true;

// Log update events
autoUpdater.logger = require('electron-log');
autoUpdater.logger.transports.file.level = 'info';

// Kiểm tra update khi app start
const checkForUpdates = () => {
    if (!isDev) {
        console.log('Checking for updates...');
        autoUpdater.checkForUpdates();
    }
};

// Khi có update mới
autoUpdater.on('update-available', (info) => {
    console.log('Update available:', info.version);

    // Gửi thông báo đến renderer
    if (mainWindow) {
        mainWindow.webContents.send('update-available', {
            version: info.version,
            releaseDate: info.releaseDate,
            releaseNotes: info.releaseNotes
        });
    }
});

// Khi không có update
autoUpdater.on('update-not-available', (info) => {
    console.log('No updates available');
    if (mainWindow) {
        mainWindow.webContents.send('update-not-available');
    }
});

// Tiến trình download
autoUpdater.on('download-progress', (progress) => {
    console.log(`Download progress: ${progress.percent}%`);
    if (mainWindow) {
        mainWindow.webContents.send('download-progress', {
            percent: Math.round(progress.percent),
            transferred: progress.transferred,
            total: progress.total
        });
    }
});

// Khi download xong
autoUpdater.on('update-downloaded', (info) => {
    console.log('Update downloaded:', info.version);
    if (mainWindow) {
        mainWindow.webContents.send('update-downloaded', {
            version: info.version
        });
    }
});

// Lỗi update
autoUpdater.on('error', (error) => {
    console.error('Update error:', error);
    if (mainWindow) {
        mainWindow.webContents.send('update-error', {
            message: error.message
        });
    }
});

// IPC handlers cho update
ipcMain.handle('check-for-updates', async () => {
    if (isDev) {
        return { available: false, message: 'Updates disabled in dev mode' };
    }
    return autoUpdater.checkForUpdates();
});

ipcMain.handle('download-update', async () => {
    if (!isDev) {
        return autoUpdater.downloadUpdate();
    }
    return false;
});

ipcMain.handle('install-update', async () => {
    if (!isDev) {
        // Quit và install
        autoUpdater.quitAndInstall();
    }
});

ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});

// =====================
// OTHER IPC HANDLERS
// =====================

ipcMain.handle('select-file', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: [
            { name: 'Audio Files', extensions: ['mp3', 'wav', 'flac', 'ogg', 'm4a'] }
        ]
    });
    return result;
});

ipcMain.handle('select-folder', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory']
    });
    return result;
});

ipcMain.handle('get-app-path', () => {
    return app.getAppPath();
});

ipcMain.handle('get-resources-path', () => {
    return process.resourcesPath;
});

// App ready
app.whenReady().then(() => {
    // Khởi động Python backend trước
    startPythonBackend();

    // Đợi backend start
    setTimeout(() => {
        createWindow();

        // Kiểm tra update sau 3 giây
        setTimeout(checkForUpdates, 3000);
    }, 2000);

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

// Quit app
app.on('window-all-closed', () => {
    // Kill Python process khi quit
    if (pythonProcess) {
        pythonProcess.kill();
    }

    if (process.platform !== 'darwin') {
        app.quit();
    }
});