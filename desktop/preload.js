const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  readFileBase64: (filePath) => ipcRenderer.invoke('read-file-base64', filePath),
  exportCsv: (csvContent, defaultName) => ipcRenderer.invoke('export-csv', csvContent, defaultName)
});
