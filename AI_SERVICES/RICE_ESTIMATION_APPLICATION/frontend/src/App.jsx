import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const GATEWAY_URL = 'http://localhost:3000';

/* Component đếm số hạt mượt mà (Count-up animation) */
function CountUpNumber({ target }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (target === undefined || target === null) return;
    const end = typeof target === 'number' ? target : parseFloat(target) || 0;
    const duration = 1000;
    const startTime = performance.now();

    const step = (now) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setCount(Math.round(ease * end));
      if (progress < 1) {
        requestAnimationFrame(step);
      }
    };

    requestAnimationFrame(step);
  }, [target]);

  return <span>{count.toLocaleString()}</span>;
}

/* Component chỉ báo tiến trình phân tích đa giai đoạn kèm Timer */
function AnalysisProgressView() {
  const [elapsed, setElapsed] = useState('00:00.0s');
  const [progress, setProgress] = useState(6);
  const [stageIndex, setStageIndex] = useState(0);

  const stages = [
    { id: 1, name: "Đo cốc lúa", desc: "Đang hiệu chỉnh px/mm & phân tích kích thước cốc..." },
    { id: 2, name: "Bóc tách SAHI", desc: "AI đang cắt lát siêu phân giải & phát hiện hạt lúa (YOLOv8-seg)..." },
    { id: 3, name: "Phẩm cấp CNN", desc: "Đang làm sạch hình thái & phân loại hạt nguyên/lỗi (DenseNet121)..." },
    { id: 4, name: "Hồi quy ML", desc: "Đang trích xuất 31 đặc trưng & suy luận số lượng hạt bằng Machine Learning..." },
  ];

  useEffect(() => {
    const start = performance.now();

    const timerId = setInterval(() => {
      const sec = (performance.now() - start) / 1000;
      const mins = Math.floor(sec / 60);
      const s = (sec % 60).toFixed(1);
      setElapsed(`⏱️ ${String(mins).padStart(2, '0')}:${s.padStart(4, '0')}s`);

      if (sec < 1.5) {
        setStageIndex(0);
      } else if (sec < 6.0) {
        setStageIndex(1);
      } else if (sec < 8.5) {
        setStageIndex(2);
      } else {
        setStageIndex(3);
      }
    }, 100);

    const progressId = setInterval(() => {
      const sec = (performance.now() - start) / 1000;
      let target = 6;
      if (sec < 1.5) {
        target = 6 + (sec / 1.5) * 17;
      } else if (sec < 6.0) {
        target = 23 + ((sec - 1.5) / 4.5) * 45;
      } else if (sec < 8.5) {
        target = 68 + ((sec - 6.0) / 2.5) * 22;
      } else {
        target = Math.min(96, 90 + ((sec - 8.5) / 5.0) * 6);
      }
      setProgress(Math.round(target));
    }, 150);

    return () => {
      clearInterval(timerId);
      clearInterval(progressId);
    };
  }, []);

  return (
    <div className="analysis-progress-panel">
      <div className="progress-header">
        <div className="progress-pulse-badge">
          <span className="pulse-dot"></span>
          AI INFERENCE ACTIVE
        </div>
        <div className="progress-timer">{elapsed}</div>
      </div>

      <h3 className="progress-title">Hệ Thống Đang Xử Lý Mẫu Lúa</h3>
      <p className="progress-subtext">{stages[stageIndex].desc}</p>

      <div className="progress-bar-container">
        <div className="progress-bar-fill" style={{ width: `${progress}%` }}>
          <span className="shimmer-effect"></span>
        </div>
      </div>
      <div className="progress-percentage-label">{progress}% HOÀN THÀNH</div>

      <div className="pipeline-steps-checklist">
        {stages.map((st, idx) => (
          <div 
            key={st.id} 
            className={`step-chip ${idx < stageIndex ? 'completed' : idx === stageIndex ? 'active' : 'pending'}`}
          >
            <span className="chip-indicator">
              {idx < stageIndex ? '✓' : idx === stageIndex ? '⟳' : idx + 1}
            </span>
            <span className="chip-name">{st.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState('phone'); // 'phone' | 'computer'

  // Form State (Computer mode)
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [formData, setFormData] = useState({ 
    diam: '', 
    height: '', 
    empty: 0,
    wall_thickness: 1.0,
    weightTotal: '',  
    sampleCount: '',  
    sampleWeight: '',
    estimatorMode: 'auto',
    debug: true
  });
  
  // Results & UI State
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [artifacts, setArtifacts] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Phone Camera QR State
  const [qrData, setQrData] = useState(null);
  const [qrLoading, setQrLoading] = useState(false);
  const [qrError, setQrError] = useState(null);

  // Lắng nghe Server-Sent Events (SSE) để cập nhật realtime từ Điện thoại
  useEffect(() => {
    const eventSource = new EventSource(`${GATEWAY_URL}/api/capture/stream`);

    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'analyzing') {
          setLoading(true);
          setErrorMsg(null);
          setResult(null);
        } else if (payload.type === 'result' && payload.data?.status === 'success') {
          setResult(payload.data.estimation);
          setMetrics(payload.data.metrics_summary);
          setAnalysis(payload.data);
          setArtifacts(payload.data.artifacts || null);
          setLoading(false);
          setErrorMsg(null);
        } else if (payload.type === 'capture_ready' && payload.captureInfo) {
          setQrData(payload.captureInfo);
          setQrLoading(false);
        } else if (payload.type === 'error') {
          setErrorMsg(payload.error?.error || "Lỗi xử lý từ hệ thống AI");
          setLoading(false);
        }
      } catch (_) {}
    };

    eventSource.onerror = () => {};

    return () => {
      eventSource.close();
    };
  }, []);

  // Tải QR code khi chuyển sang chế độ Phone
  useEffect(() => {
    if (mode === 'phone' && !qrData) {
      loadQrCode();
    }
  }, [mode]);

  const loadQrCode = async () => {
    setQrLoading(true);
    setQrError(null);
    try {
      const res = await axios.get(`${GATEWAY_URL}/api/capture/qr`);
      if (res.data.success) {
        setQrData(res.data);
      } else {
        setQrError(res.data.error || 'Không thể tạo QR code.');
      }
    } catch (err) {
      setQrError('Không thể kết nối Gateway (Port 3000). Hãy kiểm tra server.js.');
    } finally {
      setQrLoading(false);
    }
  };

  const handleInputChange = (e) => {
    let raw = e.target.value;
    if (typeof raw === 'string') raw = raw.replace(/,/g, '.');
    const value = raw === '' ? '' : parseFloat(raw);
    setFormData({ ...formData, [e.target.name]: isNaN(value) ? '' : value });
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    if (file) setPreview(URL.createObjectURL(file));
  };

  // Tính toán hình học tức thì & Inline Validation
  const riceHeight = formData.height !== '' && formData.empty !== '' && Number(formData.height) >= Number(formData.empty)
    ? (Number(formData.height) - Number(formData.empty)).toFixed(1)
    : null;

  const isInvalidHeight = formData.height !== '' && formData.empty !== '' && Number(formData.empty) >= Number(formData.height);
  const isInvalidThickness = formData.diam !== '' && formData.wall_thickness !== '' && Number(formData.wall_thickness) * 2 >= Number(formData.diam);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    if (!image) {
      alert("Vui lòng chọn ảnh chụp mẫu lúa trước khi phân tích!");
      return;
    }
    if (isInvalidHeight) {
      alert("Lỗi: Mức hụt lúa không thể lớn hơn hoặc bằng chiều cao ly!");
      return;
    }
    if (isInvalidThickness) {
      alert("Lỗi: Độ dày thành ly vượt quá kích thước lòng trong cốc!");
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    setResult(null);

    const data = new FormData();
    data.append('file', image);
    data.append('diam', formData.diam);
    data.append('height', formData.height);
    data.append('empty', formData.empty);
    data.append('wall_thickness', formData.wall_thickness);

    if (formData.weightTotal) data.append('weight_total', formData.weightTotal);
    if (formData.sampleCount) data.append('sample_count', formData.sampleCount);
    if (formData.sampleWeight) data.append('sample_weight', formData.sampleWeight);
    data.append('estimator_mode', formData.estimatorMode);
    data.append('debug', formData.debug);

    try {
      const response = await axios.post(`${GATEWAY_URL}/api/predict`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000
      });

      const responseData = response.data;

      if (responseData.status === 'success') {
        setResult(responseData.estimation);
        setMetrics(responseData.metrics_summary);
        setAnalysis(responseData);
        setArtifacts(responseData.artifacts || null);
      } else {
        setErrorMsg(responseData.message || responseData.error || "Lỗi xử lý AI");
      }
    } catch (error) {
      setErrorMsg("Mất kết nối hoặc quá thời gian chờ (Timeout). Hãy kiểm tra Server AI.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header className="brand-header">
        <div className="brand-badge-row">
          <span className="brand-pill">✨ RESEARCH LAB EDITION · V2.5</span>
        </div>
        <h1 className="brand-title">RICE VISION AI</h1>
        <p className="brand-subtitle">Hệ Thống Ước Lượng Hạt Giống Thông Minh Đa Phương Thức</p>

        {/* Chuyển đổi chế độ phong cách Segmented Pill */}
        <div className="segmented-tabs">
          <button 
            type="button"
            className={`tab-item ${mode === 'phone' ? 'active' : ''}`}
            onClick={() => setMode('phone')}
          >
            <span className="tab-icon">📱</span>
            <span>Camera Điện thoại</span>
            <span className="tab-badge">Khuyên dùng</span>
          </button>
          <button 
            type="button"
            className={`tab-item ${mode === 'computer' ? 'active' : ''}`}
            onClick={() => setMode('computer')}
          >
            <span className="tab-icon">💻</span>
            <span>Tải ảnh từ máy tính</span>
          </button>
        </div>
      </header>

      <main className="dashboard-grid">
        {/* PANEL NHẬP LIỆU: TÙY THEO CHẾ ĐỘ */}
        {mode === 'phone' ? (
          <section className="input-panel qr-panel">
            <h2>Kết nối Điện thoại làm Camera Node</h2>
            <p className="panel-desc">
              Sử dụng camera điện thoại để chụp ảnh mẫu lúa với độ phân giải và chất lượng quang học tốt nhất.
            </p>

            {qrLoading && (
              <div className="qr-loading">
                <div className="spinner"></div>
                <p>Đang khởi tạo chứng chỉ HTTPS và mã QR...</p>
              </div>
            )}

            {qrError && (
              <div className="qr-error">
                <p>⚠️ {qrError}</p>
                <button onClick={loadQrCode} className="retry-btn">Thử lại</button>
              </div>
            )}

            {qrData && !qrLoading && (
              <div className="qr-content">
                <div className="qr-frame">
                  <img src={qrData.qr} alt="Mã QR kết nối điện thoại" className="qr-image" />
                </div>

                <div className="qr-instructions">
                  <div className="step-item">
                    <span className="step-number">1</span>
                    <span>Đảm bảo điện thoại và máy tính kết nối <strong>cùng mạng Wi-Fi</strong>.</span>
                  </div>
                  <div className="step-item">
                    <span className="step-number">2</span>
                    <span>Mở ứng dụng <strong>Camera</strong> trên điện thoại để quét mã QR.</span>
                  </div>
                  <div className="step-item">
                    <span className="step-number">3</span>
                    <span>Nhập thông số, căn chỉnh miệng cốc vào vòng tròn và bấm <strong>GỬI ẢNH</strong>.</span>
                  </div>
                </div>

                <div className="qr-direct-link">
                  <span>Hoặc truy cập trực tiếp:</span>
                  <a href={qrData.url} target="_blank" rel="noreferrer">{qrData.url}</a>
                </div>

                <div className="phone-waiting-status">
                  <span className="pulse-dot"></span>
                  <span>Đang chờ điện thoại gửi ảnh chụp mẫu lúa...</span>
                </div>
              </div>
            )}
          </section>
        ) : (
          /* CHẾ ĐỘ THỦ CÔNG TẢI FILE TỪ MÁY TÍNH */
          <section className="input-panel">
            <h2>Tham số cấu hình cốc lúa</h2>
            <form onSubmit={handleSubmit}>
              <div className="input-row">
                <div className="input-group">
                  <label>Đường kính ly (mm)</label>
                  <div className="input-with-unit">
                    <input type="number" step="0.01" name="diam" placeholder="VD: 17.8" value={formData.diam} onChange={handleInputChange} required />
                    <span className="unit-tag">mm</span>
                  </div>
                </div>
                <div className="input-group">
                  <label>Chiều cao ly (mm)</label>
                  <div className="input-with-unit">
                    <input type="number" step="0.01" name="height" placeholder="VD: 33.8" value={formData.height} onChange={handleInputChange} required />
                    <span className="unit-tag">mm</span>
                  </div>
                </div>
              </div>
              
              <div className="input-row">
                <div className="input-group">
                  <label>Mức hụt lúa (mm)</label>
                  <div className="input-with-unit">
                    <input type="number" step="0.01" name="empty" value={formData.empty} onChange={handleInputChange} required />
                    <span className="unit-tag">mm</span>
                  </div>
                </div>
                <div className="input-group">
                  <label>Độ dày thành ly (mm)</label>
                  <div className="input-with-unit">
                    <input type="number" step="0.01" name="wall_thickness" value={formData.wall_thickness} onChange={handleInputChange} required />
                    <span className="unit-tag">mm</span>
                  </div>
                </div>
              </div>

              {/* Thông báo chiều cao lớp gạo tức thì & Validation Error */}
              {riceHeight && !isInvalidHeight && (
                <div className="rice-height-badge">
                  <span>🌾 Chiều cao lớp gạo thực tế: <strong>{riceHeight} mm</strong></span>
                </div>
              )}
              {isInvalidHeight && (
                <div className="validation-error-badge">
                  <span>⚠️ Mức hụt lúa không thể lớn hơn hoặc bằng chiều cao ly!</span>
                </div>
              )}
              {isInvalidThickness && (
                <div className="validation-error-badge">
                  <span>⚠️ Độ dày thành vượt quá kích thước lòng trong cốc!</span>
                </div>
              )}

              <div className="optional-section">
                <h3>Tối ưu độ chính xác (Tùy chọn kết hợp Cân nặng)</h3>
                <div className="input-row" style={{ gridTemplateColumns: '1fr', marginBottom: '10px' }}>
                  <div className="input-group">
                    <label>Tổng trọng lượng khối lúa (g)</label>
                    <div className="input-with-unit">
                      <input type="number" step="0.01" name="weightTotal" placeholder="VD: 500" value={formData.weightTotal} onChange={handleInputChange} />
                      <span className="unit-tag">g</span>
                    </div>
                  </div>
                </div>
                <div className="input-row">
                  <div className="input-group">
                    <label>Số hạt đếm cân mẫu</label>
                    <div className="input-with-unit">
                      <input type="number" step="1" name="sampleCount" placeholder="VD: 10" value={formData.sampleCount} onChange={handleInputChange} />
                      <span className="unit-tag">hạt</span>
                    </div>
                  </div>
                  <div className="input-group">
                    <label>Trọng lượng hạt mẫu (g)</label>
                    <div className="input-with-unit">
                      <input type="number" step="0.01" name="sampleWeight" placeholder="VD: 0.15" value={formData.sampleWeight} onChange={handleInputChange} />
                      <span className="unit-tag">g</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="advanced-section">
                <h3>Chế độ suy luận và quan sát</h3>
                <div className="advanced-grid">
                  <div className="input-group">
                    <label>Phương pháp ước lượng</label>
                    <select 
                      name="estimatorMode" 
                      value={formData.estimatorMode} 
                      onChange={(e) => setFormData({ ...formData, estimatorMode: e.target.value })}
                    >
                      <option value="auto">Tự động (Ưu tiên Hồi quy ML + Hybrid Cân mẫu)</option>
                      <option value="regression">Chỉ mô hình Hồi quy ML (31 đặc trưng)</option>
                      <option value="geometry">Chỉ công thức hình học 3D</option>
                      <option value="weight">Chỉ suy luận theo Cân mẫu</option>
                    </select>
                  </div>
                  <div className="input-group">
                    <label>Tùy chọn hiển thị</label>
                    <label className="debug-toggle-card">
                      <input 
                        type="checkbox" 
                        checked={formData.debug} 
                        onChange={(e) => setFormData({ ...formData, debug: e.target.checked })} 
                      />
                      <span>Hiển thị phân tích chuyên sâu</span>
                    </label>
                  </div>
                </div>
              </div>

              <div className="upload-group">
                <label className="upload-label">
                  <span className="upload-btn-text">
                    <span className="upload-icon">📷</span>
                    <span>{image ? `Đã chọn: ${image.name}` : 'Nhấp hoặc Kéo thả ảnh mẫu lúa vào đây'}</span>
                  </span>
                  <input type="file" accept="image/*" onChange={handleFileChange} hidden />
                </label>
                {preview && (
                  <div className="preview-container">
                    <img src={preview} alt="Preview" className="img-preview" />
                  </div>
                )}
              </div>

              <button 
                type="submit" 
                className={`run-btn ${loading ? 'loading' : ''}`} 
                disabled={loading || isInvalidHeight || isInvalidThickness}
              >
                {loading ? 'ĐANG PHÂN TÍCH TIẾN TRÌNH...' : '⚡ BẮT ĐẦU ƯỚC LƯỢNG'}
              </button>
            </form>
          </section>
        )}

        {/* PANEL HIỂN THỊ KẾT QUẢ ĐỒNG BỘ */}
        <section className="result-panel">
          {!result && !loading && !errorMsg && (
            <div className="idle-state">
              <div className="pulse-circle"></div>
              <p className="idle-title">Hệ thống đang chờ dữ liệu đầu vào</p>
              <p className="idle-desc">Chụp ảnh từ điện thoại hoặc tải ảnh từ máy tính để bắt đầu phân tích.</p>
            </div>
          )}

          {/* Màn hình tiến trình thông minh Loading State */}
          {loading && <AnalysisProgressView />}

          {errorMsg && (
            <div className="error-state">
              <p>⚠️ Lỗi: {errorMsg}</p>
            </div>
          )}

          {result && !loading && (
            <div className="success-state">
              <div className="hero-result-card">
                <p className="result-subtitle">SỐ LƯỢNG ƯỚC TÍNH</p>
                <h2 className="result-number">
                  <CountUpNumber target={result.final} />
                  <span className="unit"> hạt</span>
                </h2>
                
                {/* Method indicator chuẩn xác */}
                <div className="method-badge">
                  {result.method_used?.includes('hybrid') ? (
                    <span className="badge badge-blue">
                      <span className="badge-dot"></span>
                      Ước lượng Hybrid (Hồi quy ML + Cân mẫu)
                    </span>
                  ) : result.method_used?.includes('regression') ? (
                    <span className="badge badge-green">
                      <span className="badge-dot"></span>
                      Mô hình Hồi quy ({result.model_bundle || metrics?.regression_model || 'Machine Learning'})
                    </span>
                  ) : result.method_used?.includes('weight') ? (
                    <span className="badge badge-amber">
                      <span className="badge-dot"></span>
                      Ước lượng Cân mẫu
                    </span>
                  ) : (
                    <span className="badge badge-gray">
                      <span className="badge-dot"></span>
                      Thể tích Hình học 3D (Tham khảo)
                    </span>
                  )}
                </div>
              </div>

              {/* Bento Grid Thống kê Hình thái học */}
              {metrics && (
                <div className="metrics-section">
                  <h3 className="section-title">Thông số hình thái học trung bình</h3>
                  <div className="bento-grid">
                    <div className="bento-card">
                      <div className="bento-icon">🔍</div>
                      <div className="bento-content">
                        <span className="bento-label">Hạt phát hiện</span>
                        <span className="bento-val">{metrics.total_grains_detected}</span>
                      </div>
                    </div>

                    <div className="bento-card">
                      <div className="bento-icon">🌾</div>
                      <div className="bento-content">
                        <span className="bento-label">Hạt nguyên</span>
                        <span className="bento-val">{metrics.whole_grains_surface}</span>
                      </div>
                    </div>

                    <div className="bento-card bento-wide">
                      <div className="bento-header-row">
                        <span className="bento-label">Độ đồng đều phẩm cấp</span>
                        <span className="bento-highlight">{metrics.uniformity_rate_pct?.toFixed(1)}%</span>
                      </div>
                      <div className="uniformity-bar-track">
                        <div 
                          className="uniformity-bar-fill" 
                          style={{ width: `${Math.min(100, Math.max(0, metrics.uniformity_rate_pct || 0))}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bento-card">
                      <div className="bento-icon">📏</div>
                      <div className="bento-content">
                        <span className="bento-label">Dài TB</span>
                        <span className="bento-val">{metrics.avg_length_mm} <small>mm</small></span>
                      </div>
                    </div>

                    <div className="bento-card">
                      <div className="bento-icon">📐</div>
                      <div className="bento-content">
                        <span className="bento-label">Rộng TB</span>
                        <span className="bento-val">{metrics.avg_width_mm} <small>mm</small></span>
                      </div>
                    </div>

                    <div className="bento-card">
                      <div className="bento-icon">📦</div>
                      <div className="bento-content">
                        <span className="bento-label">Dày TB</span>
                        <span className="bento-val">{metrics.avg_thickness_mm} <small>mm</small></span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Bảng đối sánh các phương pháp */}
              <div className="breakdown-card">
                <h3>Chi tiết các phương pháp suy luận</h3>
                <table className="breakdown-table">
                  <tbody>
                    <tr>
                      <td>
                        <strong>Hồi quy 31 đặc trưng</strong>
                        <small>{result.model_bundle || 'Machine Learning'}</small>
                      </td>
                      <td className="val highlight">{result.regression_est?.toLocaleString() || '-'} hạt</td>
                    </tr>
                    <tr>
                      <td>
                        <strong>Ước lượng theo Khối lượng mẫu</strong>
                        <small>Tỷ trọng hạt cân mẫu</small>
                      </td>
                      <td className="val">{result.weight_est?.toLocaleString() || '-'} hạt</td>
                    </tr>
                    <tr>
                      <td>
                        <strong>Hình học Thể tích 3D (Tham khảo)</strong>
                        <small>Độ xếp chặt Ellipsoid</small>
                      </td>
                      <td className="val text-muted">{result.ai_est?.toLocaleString() || result.geometry_est?.toLocaleString() || '-'} hạt</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {artifacts?.status === 'completed' && artifacts?.request_id && (
                <div className="artifact-actions">
                  <span>✅ Đã lưu kết quả: {artifacts.request_id}</span>
                  <a href={`${GATEWAY_URL}/api/results/${artifacts.request_id}/download`} target="_blank" rel="noreferrer">
                    Tải toàn bộ artifact
                  </a>
                </div>
              )}

              {analysis?.debug_info && (
                <details className="analysis-details">
                  <summary>Chi tiết pipeline và 31 đặc trưng kỹ thuật</summary>
                  <div className="analysis-grid">
                    <p>Scale: {analysis.debug_info.container?.pixels_per_mm?.toFixed?.(2) ?? '-'} px/mm</p>
                    <p>V_bulk: {analysis.debug_info.container?.bulk_volume_mm3?.toFixed?.(1) ?? '-'} mm³</p>
                    <p>Hệ số packing: {analysis.debug_info.physical_estimator?.packing_fraction ?? '-'}</p>
                    <p>Mean volume sau lọc: {analysis.debug_info.physical_estimator?.mean_clean_volume_mm3 ?? '-'} mm³</p>
                    <p>Trạng thái lọc: {analysis.debug_info.physical_estimator?.size_filter?.status ?? '-'}</p>
                    <p>Hạt bị loại nhánh physical: {analysis.debug_info.grain_details?.size_filter_rejected ?? 0}</p>
                  </div>
                  {Object.entries(analysis.debug_info.visuals || {}).map(([name, src]) => (
                    <figure className="debug-visual" key={name}>
                      <img src={src} alt={name} />
                      <figcaption>{name === 'container_detection' ? 'Phát hiện miệng ly' : 'Phân loại hạt: xanh = hạt nguyên, cam = chỉ dùng regression, đỏ = không đạt CNN'}</figcaption>
                    </figure>
                  ))}
                  <pre>{JSON.stringify({ timings_ms: analysis.timings_ms, features_31: analysis.debug_info.features_vector }, null, 2)}</pre>
                </details>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
