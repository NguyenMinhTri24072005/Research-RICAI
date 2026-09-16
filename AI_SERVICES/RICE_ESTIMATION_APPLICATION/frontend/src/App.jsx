import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const GATEWAY_URL = 'http://localhost:3000';

function App() {
  const [mode, setMode] = useState('phone'); // 'phone' | 'computer'

  // Form State (Computer mode)
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [formData, setFormData] = useState({ 
    diam: '', 
    height: '', 
    empty: 0,
    wall_thickness: 0.1,
    weightTotal: '',  
    sampleCount: '',  
    sampleWeight: ''  
  });
  
  // Results & UI State
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Phone Camera QR State
  const [qrData, setQrData] = useState(null);
  const [qrLoading, setQrLoading] = useState(false);
  const [qrError, setQrError] = useState(null);
  const [phoneConnected, setPhoneConnected] = useState(false);

  // Lắng nghe Server-Sent Events (SSE) để cập nhật kết quả realtime từ Điện thoại
  useEffect(() => {
    const eventSource = new EventSource(`${GATEWAY_URL}/api/capture/stream`);

    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'result' && payload.data?.status === 'success') {
          setResult(payload.data.estimation);
          setMetrics(payload.data.metrics_summary);
          setLoading(false);
          setErrorMsg(null);
        } else if (payload.type === 'capture_ready' && payload.captureInfo) {
          setQrData(payload.captureInfo);
          setQrLoading(false);
        }
      } catch (_) {}
    };

    eventSource.onerror = () => {
      // Tự động reconnect sau sự cố mạng
    };

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    if (!image) {
      alert("Vui lòng tải ảnh lên trước khi phân tích!");
      return;
    }

    setLoading(true);
    setResult(null);
    setMetrics(null);
    setErrorMsg(null);

    const data = new FormData();
    data.append('file', image);
    data.append('diam', formData.diam || 0);
    data.append('height', formData.height || 0);
    data.append('empty', formData.empty || 0);
    data.append('wall_thickness', formData.wall_thickness || 0);
    data.append('weight_total', formData.weightTotal || 0);
    data.append('sample_count', formData.sampleCount || 0);
    data.append('sample_weight', formData.sampleWeight || 0);

    try {
      const response = await axios.post(`${GATEWAY_URL}/api/predict`, data, {
        timeout: 120000,
      });

      const responseData = response.data;

      if (responseData.status === 'success') {
        setResult(responseData.estimation);
        setMetrics(responseData.metrics_summary);
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
        <h1>RICE VISION AI</h1>
        <p>Hệ Thống Ước Lượng Hạt Giống Thông Minh</p>

        {/* Chuyển đổi chế độ: Điện thoại làm Camera Node vs Tải file trực tiếp */}
        <div className="mode-toggle">
          <button 
            type="button"
            className={`mode-btn ${mode === 'phone' ? 'active' : ''}`}
            onClick={() => setMode('phone')}
          >
            📱 Camera Điện thoại (Khuyên dùng)
          </button>
          <button 
            type="button"
            className={`mode-btn ${mode === 'computer' ? 'active' : ''}`}
            onClick={() => setMode('computer')}
          >
            💻 Tải ảnh từ máy tính
          </button>
        </div>
      </header>

      <main className="dashboard-grid">
        {/* PANEL NHẬP LIỆU: TÙY THEO CHẾ ĐỘ */}
        {mode === 'phone' ? (
          <section className="input-panel qr-panel">
            <h2>Kết nối Điện thoại làm Camera</h2>
            <p className="panel-desc">
              Sử dụng camera điện thoại để chụp ảnh mẫu lúa với độ phân giải và chất lượng tốt nhất.
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
                    <span>Mở ứng dụng <strong>Camera</strong> hoặc trình quét QR trên điện thoại để quét mã.</span>
                  </div>
                  <div className="step-item">
                    <span className="step-number">3</span>
                    <span>Bấm vào link mở ra, bật camera, nhập thông số và bấm <strong>GỬI ẢNH VÀ ƯỚC LƯỢNG</strong>.</span>
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
            <h2>Tham số cấu hình</h2>
            <form onSubmit={handleSubmit}>
              <div className="input-row">
                <div className="input-group">
                  <label>Đường kính ly (cm)</label>
                  <input type="number" step="0.01" name="diam" placeholder="VD: 1.78" value={formData.diam} onChange={handleInputChange} required />
                </div>
                <div className="input-group">
                  <label>Chiều cao ly (cm)</label>
                  <input type="number" step="0.01" name="height" placeholder="VD: 3.38" value={formData.height} onChange={handleInputChange} required />
                </div>
              </div>
              
              <div className="input-row">
                <div className="input-group">
                  <label>Mức hụt lúa (cm)</label>
                  <input type="number" step="0.01" name="empty" value={formData.empty} onChange={handleInputChange} required />
                </div>
                <div className="input-group">
                  <label>Độ dày thành ly (cm)</label>
                  <input type="number" step="0.01" name="wall_thickness" value={formData.wall_thickness} onChange={handleInputChange} required />
                </div>
              </div>

              <div className="optional-section">
                <h3>Tối ưu độ chính xác (Tùy chọn kết hợp Cân nặng)</h3>
                <div className="input-row" style={{ gridTemplateColumns: '1fr' }}>
                  <div className="input-group" style={{ marginBottom: '10px' }}>
                    <label>Tổng trọng lượng khối lúa (g)</label>
                    <input type="number" step="0.01" name="weightTotal" placeholder="VD: 500" value={formData.weightTotal} onChange={handleInputChange} />
                  </div>
                </div>
                <div className="input-row">
                  <div className="input-group">
                    <label>Số hạt đếm cân mẫu</label>
                    <input type="number" step="1" name="sampleCount" placeholder="VD: 5, 10, 20..." value={formData.sampleCount} onChange={handleInputChange} />
                  </div>
                  <div className="input-group">
                    <label>Trọng lượng hạt mẫu (g)</label>
                    <input type="number" step="0.01" name="sampleWeight" placeholder="VD: 0.15" value={formData.sampleWeight} onChange={handleInputChange} />
                  </div>
                </div>
              </div>

              <div className="upload-group">
                <label className="upload-label">
                  <span className="upload-btn-text">Chọn ảnh mẫu lúa</span>
                  <input type="file" accept="image/*" onChange={handleFileChange} hidden />
                </label>
                {preview && <img src={preview} alt="Preview" className="img-preview" />}
              </div>

              <button type="submit" className={`run-btn ${loading ? 'loading' : ''}`} disabled={loading}>
                {loading ? 'Đang phân tích...' : 'BẮT ĐẦU ƯỚC LƯỢNG'}
              </button>
            </form>
          </section>
        )}

        {/* PANEL HIỂN THỊ KẾT QUẢ ĐỒNG BỘ */}
        <section className="result-panel">
          {!result && !loading && !errorMsg && (
            <div className="idle-state">
              <div className="pulse-circle"></div>
              <p>Hệ thống đang chờ dữ liệu đầu vào</p>
              <small style={{ color: '#9ca3af', marginTop: '6px' }}>
                {mode === 'phone' ? 'Hãy dùng điện thoại quét mã QR bên trái để chụp ảnh' : 'Hãy điền thông số và tải ảnh lên'}
              </small>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>AI đang xử lý trên GPU...</p>
            </div>
          )}

          {errorMsg && (
            <div className="error-state">
              <p>Lỗi: {errorMsg}</p>
            </div>
          )}

          {result && (
            <div className="success-state">
              <p className="result-subtitle">SỐ LƯỢNG ƯỚC TÍNH</p>
              <h2 className="result-number">{result.final?.toLocaleString() || 0} <span className="unit">hạt</span></h2>
              
              {/* Method indicator */}
              <div className="method-badge">
                {result.method_used?.includes('regression') ? (
                  <span className="badge badge-green">Mô hình Hồi quy ({result.method_used?.includes('ExtraTrees') ? 'Extra Trees' : 'OLS'})</span>
                ) : result.method_used === 'hybrid' ? (
                  <span className="badge badge-blue">Ước lượng Hybrid (AI + Cân nặng)</span>
                ) : (
                  <span className="badge badge-gray">{result.method_used || 'Thể tích thuần AI'}</span>
                )}
              </div>

              {/* Breakdown table */}
              <div className="breakdown-table">
                <h3>Chi tiết các phương pháp</h3>
                <table>
                  <tbody>
                    <tr>
                      <td>Hồi quy 31 biến (Extra Trees)</td>
                      <td className="val">{result.regression_est?.toLocaleString() || '-'} hạt</td>
                    </tr>
                    <tr>
                      <td>Thể tích thuần AI</td>
                      <td className="val">{result.ai_est?.toLocaleString() || '-'} hạt</td>
                    </tr>
                    <tr>
                      <td>Tỷ trọng Cân mẫu</td>
                      <td className="val">{result.weight_est?.toLocaleString() || '-'} hạt</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Metrics */}
              {metrics && (
                <div className="metrics-panel">
                  <h3>Thông số hình thái học trung bình</h3>
                  <div className="metrics-grid">
                    <div className="metric-item">
                      <span className="metric-label">Hạt phát hiện</span>
                      <span className="metric-value">{metrics.total_grains_detected}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Hạt nguyên</span>
                      <span className="metric-value">{metrics.whole_grains_surface}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Độ đồng đều</span>
                      <span className="metric-value">{metrics.uniformity_rate_pct?.toFixed(1)}%</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Dài TB</span>
                      <span className="metric-value">{metrics.avg_length_mm} mm</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Rộng TB</span>
                      <span className="metric-value">{metrics.avg_width_mm} mm</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Dày TB</span>
                      <span className="metric-value">{metrics.avg_thickness_mm} mm</span>
                    </div>
                  </div>
                  <p className="model-info">Mô hình AI: {metrics.regression_model || 'Extra Trees'}</p>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;