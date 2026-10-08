const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const fs = require('fs');

let mainWindow;

// 判断是否为调试模式
const isDev = process.argv.includes('--dev') || process.env.NODE_ENV === 'development';

function createWindow() {
  // 去除默认菜单栏（调试模式下保留以便使用开发者工具快捷键）
  if (!isDev) {
    Menu.setApplicationMenu(null);
  }

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: 'Terravision - 遥感图像多标签分类系统',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    backgroundColor: '#0a0e17',
    frame: true,
    autoHideMenuBar: true
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // 调试模式下自动打开 DevTools
  if (isDev) {
    mainWindow.webContents.openDevTools();
    console.log('🔧 调试模式已启用，DevTools 已打开');
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ==================== IPC 通信 ====================

// 选择单个图像文件
ipcMain.handle('select-image', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择遥感图像',
    filters: [
      { name: '图像文件', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'] }
    ],
    properties: ['openFile']
  });
  if (result.canceled) return null;
  const filePath = result.filePaths[0];
  const buffer = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.bmp': 'image/bmp', '.tif': 'image/tiff', '.tiff': 'image/tiff' };
  const base64 = buffer.toString('base64');
  return {
    name: path.basename(filePath),
    path: filePath,
    base64: `data:${mimeTypes[ext] || 'image/jpeg'};base64,${base64}`,
    size: buffer.length
  };
});

// 选择文件夹批量导入图像
ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择包含遥感图像的文件夹',
    properties: ['openDirectory']
  });
  if (result.canceled) return null;
  const dirPath = result.filePaths[0];
  const imageExts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'];
  const files = fs.readdirSync(dirPath)
    .filter(f => imageExts.includes(path.extname(f).toLowerCase()))
    .map(f => path.join(dirPath, f));
  return { dirPath, files, count: files.length };
});

// 读取文件为 base64
ipcMain.handle('read-file-base64', async (event, filePath) => {
  const buffer = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.bmp': 'image/bmp', '.tif': 'image/tiff', '.tiff': 'image/tiff' };
  const base64 = buffer.toString('base64');
  return {
    name: path.basename(filePath),
    base64: `data:${mimeTypes[ext] || 'image/jpeg'};base64,${base64}`
  };
});

// 导出 CSV
ipcMain.handle('export-csv', async (event, csvContent, defaultName) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: '导出预测结果',
    defaultPath: defaultName || 'predictions.csv',
    filters: [{ name: 'CSV文件', extensions: ['csv'] }]
  });
  if (result.canceled) return false;
  fs.writeFileSync(result.filePath, '\uFEFF' + csvContent, 'utf-8'); // BOM for Excel
  return true;
});