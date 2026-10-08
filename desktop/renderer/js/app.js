// ==================== Terravision 桌面端主逻辑 ====================

// ---- 状态管理 ----
const state = {
  currentTab: 'home',
  currentModel: 'resnet50',
  models: [],
  selectedImage: null,
  batchImages: [],
  batchResults: [],
  apiOnline: false,
  statsData: null
};

// ---- 初始化 ----
document.addEventListener('DOMContentLoaded', () => {
  loadTheme();
  initNavigation();
  initModelSelect();
  initPredictTab();
  initBatchTab();
  initStatsTab();
  initModelsTab();
  initSettingsTab();
  initHomeTab();
  checkApiStatus();
  setInterval(checkApiStatus, 10000);
});

// ---- 导航 ----
function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      const tab = item.dataset.tab;
      state.currentTab = tab;
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      document.getElementById(`tab-${tab}`).classList.add('active');
      if (tab === 'stats') loadStats();
      if (tab === 'models') loadModels();
      if (tab === 'batch') renderBatchPreviews();
      if (tab === 'home') loadHomeData();
    });
  });
}

// ---- 切换标签页（供首页快速入口调用） ----
function switchTab(tabName) {
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  document.querySelector(`.nav-item[data-tab="${tabName}"]`).classList.add('active');
  state.currentTab = tabName;
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`tab-${tabName}`).classList.add('active');
  if (tabName === 'stats') loadStats();
  if (tabName === 'models') loadModels();
  if (tabName === 'batch') renderBatchPreviews();
}

// ---- 模型选择同步 ----
function initModelSelect() {
  const globalSelect = document.getElementById('global-model-select');
  const predictSelect = document.getElementById('predict-model-select');
  const batchSelect = document.getElementById('batch-model-select');
  
  globalSelect.addEventListener('change', () => {
    const val = globalSelect.value;
    state.currentModel = val;
    if (predictSelect) predictSelect.value = val;
    if (batchSelect) batchSelect.value = val;
  });
  if (predictSelect) {
    predictSelect.addEventListener('change', () => {
      globalSelect.value = predictSelect.value;
      state.currentModel = predictSelect.value;
    });
  }
  if (batchSelect) {
    batchSelect.addEventListener('change', () => {
      globalSelect.value = batchSelect.value;
      state.currentModel = batchSelect.value;
    });
  }
}

async function loadModelOptions() {
  try {
    const data = await API.getModels();
    state.models = data.models || [];
    const loadedModels = state.models.filter(m => m.loaded).map(m => m.id);
    console.log(data);
    const selects = [
      document.getElementById('global-model-select'),
      document.getElementById('predict-model-select'),
      document.getElementById('batch-model-select')
    ];
    selects.forEach(select => {
      if (!select) return;
      select.innerHTML = '';
      loadedModels.forEach(id => {
        const labelMap = { 'resnet50': 'ResNet-50', 'cnn': 'Custom CNN', 'svm': 'SVM', 'rf': 'Random Forest', 'xgb': 'XGBoost', 'early_fusion': 'Early Fusion', 'late_fusion': 'Late Fusion' };
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = labelMap[id] || id;
        select.appendChild(opt);
      });
    });
  } catch (err) {}
}

// ---- API 状态检查 ----
async function checkApiStatus() {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  const wasOnline = state.apiOnline;
  state.apiOnline = await API.healthCheck();
  if (state.apiOnline) {
    dot.className = 'status-dot online';
    text.textContent = '后端已连接';
    if (!wasOnline) {
      loadModelOptions();
      loadHomeData();
    }
  } else {
    dot.className = 'status-dot offline';
    text.textContent = '后端未连接';
  }
  updateHomeStatus();
}

// ---- Toast ----
function showToast(msg, type) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = `toast ${type || ''} show`;
  setTimeout(() => { toast.className = 'toast'; }, 2500);
}

// ---- 设置页 ----
function initSettingsTab() {
  const apiInput = document.getElementById('api-url-input');
  const saveBtn = document.getElementById('save-api-url');
  const saved = localStorage.getItem('terravision-api-url');
  if (saved) apiInput.value = saved;
  saveBtn.addEventListener('click', () => {
    const url = apiInput.value.trim().replace(/\/+$/, '');
    localStorage.setItem('terravision-api-url', url);
    API.setBaseUrl(url);
    showToast('API 地址已保存', 'success');
    checkApiStatus();
  });
}

// ---- 主题 ----
function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('terravision-theme', theme);
  document.querySelectorAll('.theme-option').forEach(el => {
    el.classList.toggle('active', el.dataset.themeValue === theme);
  });
  // 刷新echarts图表
  setTimeout(() => {
    if (state.currentTab === 'stats' && statsChart) loadStats();
    if (state.currentTab === 'home' && homeChart && state.statsData) renderHomeChart(state.statsData);
  }, 100);
}

function loadTheme() {
  const saved = localStorage.getItem('terravision-theme') || 'dark';
  setTheme(saved);
}

// ==================== 1. 单图预测 ====================
function initPredictTab() {
  const dropZone = document.getElementById('predict-drop-zone');
  const predictBtn = document.getElementById('predict-btn');
  const explainBtn = document.getElementById('explain-btn');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdBadge = document.getElementById('threshold-badge');
  const browseBtn = document.getElementById('predict-browse-btn');
  const changeBtn = document.getElementById('viewer-change-btn');

  async function browseImage() {
    const result = await window.electronAPI.selectImage();
    if (result) {
      state.selectedImage = result;
      showImageViewer(result);
    }
  }

  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('drag-over'); });
  dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && /\.(jpg|jpeg|png|bmp|tif|tiff)$/i.test(file.name)) {
      await handleImageFile(file);
    }
  });
  dropZone.addEventListener('click', browseImage);
  if (browseBtn) browseBtn.addEventListener('click', (e) => { e.stopPropagation(); browseImage(); });

  if (changeBtn) changeBtn.addEventListener('click', () => {
    showDropZone();
    state.selectedImage = null;
    predictBtn.disabled = true;
    explainBtn.disabled = true;
    document.getElementById('predict-result').innerHTML = '<div class="result-placeholder"><div class="placeholder-icon"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div><p>选择图像后点击"开始预测"查看结果</p><p class="sub">支持多标签分类，可识别 17 种地理特征</p></div>';
  });

  document.addEventListener('paste', (e) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          handleImageFile(file);
          showToast('图片已粘贴', 'success');
          if (state.currentTab !== 'predict') switchTab('predict');
        }
        break;
      }
    }
  });

  predictBtn.addEventListener('click', runPredict);
  explainBtn.addEventListener('click', runExplain);
  thresholdSlider.addEventListener('input', () => {
    if (thresholdBadge) thresholdBadge.textContent = parseFloat(thresholdSlider.value).toFixed(2);
  });

  setInterval(() => {
    const disabled = !state.apiOnline || !state.selectedImage;
    predictBtn.disabled = disabled;
    explainBtn.disabled = !state.apiOnline || !state.selectedImage;
  }, 1000);
}

function showDropZone() {
  document.getElementById('predict-drop-zone').style.display = '';
  document.getElementById('predict-viewer').style.display = 'none';
}

function showImageViewer(imgData) {
  document.getElementById('predict-drop-zone').style.display = 'none';
  document.getElementById('predict-viewer').style.display = '';
  const imgviewer = document.getElementById('viewer-img');
  imgviewer.src = imgData.base64;
  imgviewer.style.height = "150%";
  document.getElementById('viewer-filename').textContent = imgData.name;
  document.getElementById('viewer-size').textContent = (imgData.size / 1024).toFixed(1) + ' KB';
  const img = new Image();
  img.onload = () => {
    document.getElementById('viewer-dims').textContent = img.naturalWidth + '×' + img.naturalHeight;
  };
  img.src = imgData.base64;
  document.getElementById('predict-btn').disabled = !state.apiOnline;
}

async function handleImageFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    state.selectedImage = { name: file.name, base64: e.target.result, size: file.size };
    showImageViewer(state.selectedImage);
  };
  reader.readAsDataURL(file);
}

function displaySelectedImage(imgData) { showImageViewer(imgData); }

async function runPredict() {
  if (!state.selectedImage) return showToast('请先选择图像', 'warning');
  const model = document.getElementById('predict-model-select').value;
  const threshold = parseFloat(document.getElementById('threshold-slider').value);
  const resultDiv = document.getElementById('predict-result');
  resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><span>预测中...</span></div>';

  try {
    const data = await API.predict(state.selectedImage.base64, model, threshold);
    renderPredictResult(data);
  } catch (err) {
    resultDiv.innerHTML = `<div class="error-msg">预测失败: ${err.message}</div>`;
  }
}

function renderPredictResult(data) {
  const resultDiv = document.getElementById('predict-result');
  const labels = data.labels || [];
  const confidence = data.confidence || [];
  const allScores = data.all_scores || {};
  const modelName = data.model || 'unknown';
  const time = data.inference_time_ms || 0;

  const labelDescriptions = {
    'AnnualCrop': '一年生作物农田', 'Forest': '森林区域', 'HerbaceousVegetation': '草本植被',
    'Highway': '高速公路', 'Industrial': '工业区', 'Pasture': '牧场草地',
    'PermanentCrop': '多年生作物', 'Residential': '住宅区', 'River': '河流',
    'SeaLake': '海洋/湖泊', 'Vegetation': '植被覆盖', 'OpenLand': '开阔土地',
    'Artificial': '人工建筑', 'Urban': '城市区域', 'Transportation': '交通设施',
    'Water': '水体', 'NaturalArea': '自然区域'
  };
  const labelIcons = {
    'AnnualCrop': '🌾', 'Forest': '🌲', 'HerbaceousVegetation': '🌿', 'Highway': '🛣️',
    'Industrial': '🏭', 'Pasture': '🐄', 'PermanentCrop': '🌳', 'Residential': '🏠',
    'River': '🏞️', 'SeaLake': '🌊', 'Vegetation': '🌱', 'OpenLand': '🏜️',
    'Artificial': '🏗️', 'Urban': '🏙️', 'Transportation': '🚗', 'Water': '💧',
    'NaturalArea': '🏔️'
  };
  const modelNames = {
    'resnet50': 'ResNet-50 深度学习模型', 'cnn': '自定义 CNN 模型',
    'svm': 'SVM 支持向量机', 'rf': '随机森林', 'xgb': 'XGBoost',
    'early_fusion': '早融合模型', 'late_fusion': '晚融合模型'
  };

  const scoreEntries = Object.entries(allScores).sort((a, b) => b[1] - a[1]);

  let mainResultHtml = '';
  if (labels.length === 0) {
    mainResultHtml = '<div class="main-result-card empty"><div class="empty-icon">🔍</div><div class="empty-text">未检测到明显特征</div><div class="empty-hint">可尝试降低阈值后重新预测</div></div>';
  } else {
    const topLabel = labels[0], topConf = confidence[0] || 0, desc = labelDescriptions[topLabel] || topLabel, icon = labelIcons[topLabel] || '📍';
    mainResultHtml = '<div class="main-result-card"><div class="main-icon">' + icon + '</div><div class="main-info"><div class="main-label">' + topLabel + '</div><div class="main-desc">' + desc + '</div><div class="main-conf"><div class="conf-bar"><div class="conf-fill" style="width:' + (topConf * 100) + '%"></div></div><span class="conf-text">置信度 ' + (topConf * 100).toFixed(1) + '%</span></div></div></div>';
    if (labels.length > 1) {
      const otherHtml = labels.slice(1).map(l => '<span class="other-tag" title="' + (labelDescriptions[l] || l) + '">' + (labelIcons[l] || '📍') + ' ' + l + '</span>').join('');
      mainResultHtml += '<div class="other-labels"><span class="other-title">同时包含:</span>' + otherHtml + '</div>';
    }
  }

  const detailHtml = scoreEntries.map(([label, score]) => {
    const pct = (score * 100).toFixed(1);
    const icon = labelIcons[label] || '📍';
    const barColor = score >= 0.7 ? '#4ade80' : score >= 0.5 ? '#409cff' : score >= 0.3 ? '#f59e0b' : '#6b7280';
    return '<div class="detail-row' + (score >= 0.5 ? ' positive' : '') + '"><span class="detail-icon">' + icon + '</span><span class="detail-label" title="' + (labelDescriptions[label] || '') + '">' + label + '</span><div class="detail-bar-track"><div class="detail-bar-fill" style="width:' + pct + '%;background:' + barColor + '"></div></div><span class="detail-value" style="color:' + barColor + '">' + pct + '%</span></div>';
  }).join('');

  resultDiv.innerHTML = '<div class="result-summary"><div class="result-meta"><span class="meta-model">🤖 ' + (modelNames[modelName] || modelName) + '</span><span class="meta-time">⚡ ' + time + 'ms</span></div>' + mainResultHtml + '</div><div class="result-detail"><div class="detail-header"><span>📊 全部17个标签置信度详情</span><span class="detail-hint">（绿色=高置信，蓝色=中等，橙色=低置信）</span></div><div class="detail-list">' + detailHtml + '</div></div>';
}

async function runExplain() {
  if (!state.selectedImage) return showToast('请先选择图像', 'warning');
  const resultDiv = document.getElementById('predict-result');
  resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><span>生成热力图...</span></div>';

  try {
    const data = await API.explain(state.selectedImage.base64);
    resultDiv.innerHTML = `
      <div class="result-header">
        <span class="result-model">GradCAM 解释: ${data.label || ''}</span>
        <span class="result-time">置信度: ${((data.confidence || 0) * 100).toFixed(1)}%</span>
      </div>
      <div class="heatmap-container">
        <img src="${data.heatmap_base64}" alt="GradCAM Heatmap" />
      </div>
    `;
  } catch (err) {
    resultDiv.innerHTML = `<div class="error-msg">GradCAM 失败: ${err.message}</div>`;
  }
}

// ==================== 2. 批量预测 ====================
function initBatchTab() {
  const selectBtn = document.getElementById('batch-select-btn');
  const clearBtn = document.getElementById('batch-clear-btn');
  const runBtn = document.getElementById('batch-run-btn');
  const exportBtn = document.getElementById('batch-export-btn');

  selectBtn.addEventListener('click', selectBatchFolder);
  clearBtn.addEventListener('click', () => {
    state.batchImages = [];
    state.batchResults = [];
    renderBatchPreviews();
    document.getElementById('batch-results').innerHTML = '';
  });
  runBtn.addEventListener('click', runBatchPredict);
  exportBtn.addEventListener('click', exportBatchResults);
}

async function selectBatchFolder() {
  const result = await window.electronAPI.selectFolder();
  if (!result) return;
  showToast(`已选择 ${result.count} 张图像`, 'info');

  state.batchImages = [];
  state.batchResults = [];

  // 分批读取文件（最多100张）
  const files = result.files.slice(0, 100);
  for (const f of files) {
    try {
      const data = await window.electronAPI.readFileBase64(f);
      state.batchImages.push(data);
    } catch (e) { /* skip */ }
  }
  renderBatchPreviews();
}

function renderBatchPreviews() {
  const container = document.getElementById('batch-preview');
  if (state.batchImages.length === 0) {
    container.innerHTML = '<p class="placeholder-text">点击上方按钮选择图像文件夹</p>';
    document.getElementById('batch-count').textContent = '0';
    return;
  }
  document.getElementById('batch-count').textContent = state.batchImages.length;
  container.innerHTML = state.batchImages.slice(0, 50).map(img =>
    `<div class="batch-thumb" title="${img.name}">
      <img src="${img.base64}" alt="${img.name}" />
      <span>${img.name}</span>
    </div>`
  ).join('');
  if (state.batchImages.length > 50) {
    container.innerHTML += `<div class="batch-thumb more">+${state.batchImages.length - 50} 张</div>`;
  }
}

async function runBatchPredict() {
  if (state.batchImages.length === 0) return showToast('请先选择图像文件夹', 'warning');
  const model = document.getElementById('batch-model-select').value;
  const resultsDiv = document.getElementById('batch-results');
  const progressDiv = document.getElementById('batch-progress');

  resultsDiv.innerHTML = '';
  progressDiv.style.display = 'block';

  try {
    const data = await API.batchPredict(state.batchImages, model);
    state.batchResults = data.results || [];
    renderBatchResults(data);
    showToast(`批量预测完成，共 ${data.total || state.batchResults.length} 张`, 'success');
  } catch (err) {
    resultsDiv.innerHTML = `<div class="error-msg">批量预测失败: ${err.message}</div>`;
  } finally {
    progressDiv.style.display = 'none';
  }
}

function renderBatchResults(data) {
  const resultsDiv = document.getElementById('batch-results');
  const results = data.results || [];
  const modelName = data.model || 'unknown';

  // 标签分布统计
  const labelCount = {};
  results.forEach(r => {
    (r.labels || []).forEach(l => { labelCount[l] = (labelCount[l] || 0) + 1; });
  });

  const stats = Object.entries(labelCount).sort((a, b) => b[1] - a[1])
    .map(([l, c]) => `<span class="stat-tag">${l}: ${c}</span>`).join('');

  const table = results.map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${r.filename || 'unknown'}</td>
      <td>${(r.labels || []).join(', ') || '-'}</td>
      <td>${(r.confidence || []).map(c => (c * 100).toFixed(1) + '%').join(', ') || '-'}</td>
    </tr>
  `).join('');

  resultsDiv.innerHTML = `
    <div class="result-header">
      <span>模型: ${modelName}</span>
      <span>共计: ${results.length} 张</span>
    </div>
    <div class="batch-stats">${stats || '无预测标签'}</div>
    <div class="batch-table-wrap">
      <table class="batch-table">
        <thead><tr><th>#</th><th>文件名</th><th>预测标签</th><th>置信度</th></tr></thead>
        <tbody>${table}</tbody>
      </table>
    </div>
  `;
  document.getElementById('batch-export-btn').style.display = 'inline-block';
}

async function exportBatchResults() {
  if (state.batchResults.length === 0) return;
  let csv = '序号,文件名,预测标签,置信度\n';
  state.batchResults.forEach((r, i) => {
    csv += `${i + 1},"${r.filename || 'unknown'}","${(r.labels || []).join('; ')}","${(r.confidence || []).map(c => c.toFixed(4)).join('; ')}"\n`;
  });
  const ok = await window.electronAPI.exportCsv(csv, 'batch_predictions.csv');
  if (ok) showToast('CSV 已导出', 'success');
}

// ==================== 3. 模型对比统计 ====================
let statsChart = null;

function initStatsTab() {
  // 统计标签页初始化（无需额外操作）
}

async function loadStats() {
  const container = document.getElementById('stats-chart');
  try {
    const data = await API.getStats();
    renderStatsChart(data);
  } catch (err) {
    container.innerHTML = `<div class="placeholder-text">无法加载统计: ${err.message}</div>`;
  }
}

function renderStatsChart(data) {
  const models = data.models || [];
  const metrics = data.metrics || {};
  if (!models.length) return;

  const chartDom = document.getElementById('stats-chart');
  chartDom.innerHTML = '';

  if (statsChart) statsChart.dispose();
  statsChart = echarts.init(chartDom);

  const option = {
    title: { text: '模型性能对比', left: 'center', textStyle: { color: '#e0e0e0' } },
    tooltip: { trigger: 'axis' },
    legend: {
      data: ['Macro F1', 'Micro F1', 'mAP'],
      bottom: 0,
      textStyle: { color: '#aaa' }
    },
    grid: { left: '3%', right: '4%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: models,
      axisLabel: { color: '#aaa', rotate: 15 },
      axisLine: { lineStyle: { color: '#444' } }
    },
    yAxis: {
      type: 'value',
      min: 0, max: 1,
      axisLabel: { color: '#aaa', formatter: v => (v * 100).toFixed(0) + '%' },
      splitLine: { lineStyle: { color: '#333' } }
    },
    series: [
      {
        name: 'Macro F1', type: 'bar',
        data: metrics.macro_f1 || [],
        itemStyle: { color: '#4fc3f7' },
        barGap: '5%'
      },
      {
        name: 'Micro F1', type: 'bar',
        data: metrics.micro_f1 || [],
        itemStyle: { color: '#81c784' },
        barGap: '5%'
      },
      {
        name: 'mAP', type: 'bar',
        data: metrics.mAP || [],
        itemStyle: { color: '#ff8a65' },
        barGap: '5%'
      }
    ]
  };
  statsChart.setOption(option);
  window.addEventListener('resize', () => statsChart?.resize());
}

// ==================== 4. 模型管理 ====================
async function loadModels() {
  const container = document.getElementById('models-list');
  container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
  try {
    const data = await API.getModels();
    const models = data.models || [];
    container.innerHTML = models.map(m => `
      <div class="model-card">
        <div class="model-icon">${m.type === 'deep' ? '🧠' : m.type === 'traditional' ? '📊' : '🔗'}</div>
        <div class="model-info">
          <h4>${m.name}</h4>
          <span class="model-type">${m.type === 'deep' ? '深度学习' : m.type === 'traditional' ? '传统ML' : '融合模型'}</span>
          <span class="model-id">ID: ${m.id}</span>
        </div>
        <button class="btn btn-sm" onclick="state.currentModel='${m.id}'; document.querySelectorAll('.model-select').forEach(s=>s.value='${m.id}'); showToast('已切换为 ${m.name}', 'success')">选用</button>
      </div>
    `).join('');
  } catch (err) {
    container.innerHTML = `<div class="error-msg">加载失败: ${err.message}</div>`;
  }
}

function initModelsTab() {
  // 在 tab 切换时加载
}

// ==================== 5. 首页概览 ====================
let homeChart = null;

function initHomeTab() {
  // 首页初始化
}

function updateHomeStatus() {
  const backendStatus = document.getElementById('home-backend-status');
  const modelCount = document.getElementById('home-model-count');
  if (backendStatus) {
    if (state.apiOnline) {
      backendStatus.textContent = '已连接';
      backendStatus.className = 'status-value online';
    } else {
      backendStatus.textContent = '未连接';
      backendStatus.className = 'status-value offline';
    }
  }
  if (modelCount) {
    modelCount.textContent = state.models.length > 0 ? `${state.models.length} 个` : '-';
  }
}

async function loadHomeData() {
  if (!state.apiOnline) return;
  
  try {
    // 加载统计数据
    const data = await API.getStats();
    state.statsData = data;
    
    // 更新最佳模型显示
    updateBestModel(data);
    
    // 渲染首页迷你图表
    renderHomeChart(data);
    
    // 隐藏提示文字
    const note = document.getElementById('home-stats-note');
    if (note) note.style.display = 'none';
    
  } catch (err) {
    console.error('首页数据加载失败:', err);
  }
}

function updateBestModel(data) {
  const models = data.models || [];
  const metrics = data.metrics || {};
  const macroF1 = metrics.macro_f1 || [];
  
  if (models.length === 0 || macroF1.length === 0) return;
  
  // 找到最高 Macro F1 的模型
  let bestIdx = 0;
  let bestScore = 0;
  macroF1.forEach((score, idx) => {
    if (score > bestScore) {
      bestScore = score;
      bestIdx = idx;
    }
  });
  
  const bestModelDiv = document.getElementById('home-best-model');
  if (bestModelDiv) {
    const modelName = bestModelDiv.querySelector('.model-name');
    const scores = bestModelDiv.querySelectorAll('.score-value');
    if (modelName) modelName.textContent = models[bestIdx];
    if (scores[0]) scores[0].textContent = (macroF1[bestIdx] * 100).toFixed(1) + '%';
    if (scores[1]) scores[1].textContent = ((metrics.mAP || [])[bestIdx] * 100).toFixed(1) + '%';
  }
}

function renderHomeChart(data) {
  const chartDom = document.getElementById('home-stats-chart');
  if (!chartDom) return;
  
  chartDom.innerHTML = '';
  
  if (homeChart) homeChart.dispose();
  homeChart = echarts.init(chartDom);
  
  const models = data.models || [];
  const metrics = data.metrics || {};
  
  const computed = getComputedStyle(document.documentElement);
  const textColor = computed.getPropertyValue('--text-secondary').trim();
  const accentColor = computed.getPropertyValue('--accent').trim();
  
  const option = {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', top: '10%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: models,
      axisLabel: { color: textColor, fontSize: 10, rotate: 20 },
      axisLine: { lineStyle: { color: computed.getPropertyValue('--border-color').trim() } }
    },
    yAxis: {
      type: 'value',
      min: 0, max: 1,
      axisLabel: { color: textColor, fontSize: 10, formatter: v => (v * 100).toFixed(0) + '%' },
      splitLine: { lineStyle: { color: computed.getPropertyValue('--border-color').trim() } }
    },
    series: [
      {
        name: 'Macro F1', type: 'bar',
        data: metrics.macro_f1 || [],
        itemStyle: { color: accentColor },
        barWidth: '40%'
      }
    ]
  };
  
  homeChart.setOption(option);
}
