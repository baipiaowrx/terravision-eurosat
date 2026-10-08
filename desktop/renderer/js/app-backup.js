// ==================== 1. 单图预测 ====================
function initPredictTab() {
  const dropZone = document.getElementById('predict-drop-zone');
  const predictBtn = document.getElementById('predict-btn');
  const explainBtn = document.getElementById('explain-btn');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdBadge = document.getElementById('threshold-badge');
  const browseBtn = document.getElementById('predict-browse-btn');
  const changeBtn = document.getElementById('viewer-change-btn');

  // 浏览文件
  async function browseImage() {
    const result = await window.electronAPI.selectImage();
    if (result) {
      state.selectedImage = result;
      showImageViewer(result);
    }
  }

  // 拖拽上传
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

  // 更换图片
  if (changeBtn) changeBtn.addEventListener('click', () => {
    showDropZone();
    state.selectedImage = null;
    predictBtn.disabled = true;
    explainBtn.disabled = true;
    document.getElementById('predict-result').innerHTML = `
      <div class="result-placeholder">
        <div class="placeholder-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        </div>
        <p>选择图像后点击"开始预测"查看结果</p>
        <p class="sub">支持多标签分类，可识别 17 种地理特征</p>
      </div>`;
  });

  // 粘贴图片
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
  document.getElementById('viewer-img').src = imgData.base64;
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
    state.selectedImage = {
      name: file.name,
      base64: e.target.result,
      size: file.size
    };
    showImageViewer(state.selectedImage);
  };
  reader.readAsDataURL(file);
}

function displaySelectedImage(imgData) {
  showImageViewer(imgData);
}