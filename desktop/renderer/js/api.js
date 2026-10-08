// ==================== API 客户端 ====================
// 对接 Terravision FastAPI 后端

const API = {
  // 默认 baseURL，可通过设置面板修改
  baseURL: 'http://127.0.0.1:8000',
  timeout: 30000,

  setBaseURL(url) {
    this.baseURL = url.replace(/\/+$/, '');
    localStorage.setItem('api_base_url', this.baseURL);
  },

  getBaseURL() {
    const saved = localStorage.getItem('api_base_url');
    if (saved) this.baseURL = saved;
    return this.baseURL;
  },

  async request(method, endpoint, options = {}) {
    const url = `${this.getBaseURL()}${endpoint}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.timeout || this.timeout);

    try {
      const fetchOptions = {
        method,
        signal: controller.signal,
        ...options.fetchOptions
      };
      if (options.body && !(options.body instanceof FormData)) {
        fetchOptions.headers = { 'Content-Type': 'application/json', ...(fetchOptions.headers || {}) };
        fetchOptions.body = JSON.stringify(options.body);
      } else if (options.body instanceof FormData) {
        fetchOptions.body = options.body;
      }
      const res = await fetch(url, fetchOptions);
      clearTimeout(timer);
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText}`);
      }
      return await res.json();
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') throw new Error('请求超时，请检查后端服务是否启动');
      throw err;
    }
  },

  // ---- 预测推理 ----

  /** 单图预测 */
  async predict(imageBase64, model = 'resnet50', threshold = 0.5) {
    // base64 转 Blob
    const blob = this.base64ToBlob(imageBase64);
    const formData = new FormData();
    formData.append('file', blob, 'image.jpg');
    if (model) formData.append('model', model);
    if (threshold !== undefined) formData.append('threshold', String(threshold));
    return this.request('POST', '/predict', { body: formData, timeout: 60000 });
  },

  /** 批量预测 - 逐个文件上传 */
  async batchPredict(filesBase64, model = 'resnet50') {
    const formData = new FormData();
    filesBase64.forEach(({ base64, name }, i) => {
      const blob = this.base64ToBlob(base64);
      formData.append('files', blob, name || `img_${i}.jpg`);
    });
    if (model) formData.append('model', model);
    return this.request('POST', '/batch_predict', { body: formData, timeout: 120000 });
  },

  /** GradCAM 解释 */
  async explain(imageBase64, label = null) {
    const blob = this.base64ToBlob(imageBase64);
    const formData = new FormData();
    formData.append('file', blob, 'image.jpg');
    if (label) formData.append('label', label);
    return this.request('POST', '/explain', { body: formData, timeout: 60000 });
  },

  // ---- 可视化数据 ----

  async getStats() {
    return this.request('GET', '/stats');
  },

  async getTSNE(n = 500, feature = 'resnet') {
    return this.request('GET', `/tsne?n=${n}&feature=${feature}`);
  },

  async getConfusion(model = null) {
    const qs = model ? `?model=${model}` : '';
    return this.request('GET', `/confusion${qs}`);
  },

  async getLabelDistribution() {
    return this.request('GET', '/label_dist');
  },

  // ---- 数据集 ----

  async getModels() {
    return this.request('GET', '/models');
  },

  async getSamples(label = null, n = 9, split = 'test') {
    const params = new URLSearchParams();
    if (label) params.append('label', label);
    params.append('n', String(n));
    params.append('split', split);
    return this.request('GET', `/sample?${params.toString()}`);
  },

  // ---- 工具 ----

  base64ToBlob(base64) {
    const parts = base64.split(',');
    const mime = parts[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
    const bstr = atob(parts[1]);
    const n = bstr.length;
    const u8arr = new Uint8Array(n);
    for (let i = 0; i < n; i++) { u8arr[i] = bstr.charCodeAt(i); }
    return new Blob([u8arr], { type: mime });
  },

  /** 健康检查 */
  async healthCheck() {
    try {
      const res = await fetch(`${this.getBaseURL()}/models`, { signal: AbortSignal.timeout(3000) });
      return res.ok;
    } catch { return false; }
  }
};
