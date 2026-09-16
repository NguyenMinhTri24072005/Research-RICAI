const express  = require('express');
const multer   = require('multer');
const axios    = require('axios');
const FormData = require('form-data');
const fs       = require('fs');
const cors     = require('cors');

const app  = express();
const port = 3000;

app.use(cors());
app.use(express.json());


const path = require('path');

// ─────────────────────────────────────────────────────────────────────────────
// 🔑 TỰ ĐỘNG ĐỌC FILE QUẢN LÝ CẤU HÌNH & TOKEN (.env)
// ─────────────────────────────────────────────────────────────────────────────
function loadEnvConfig(filePath) {
    if (!fs.existsSync(filePath)) return {};
    const content = fs.readFileSync(filePath, 'utf-8');
    const config = {};
    for (const line of content.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) continue;
        const eqIdx = trimmed.indexOf('=');
        if (eqIdx !== -1) {
            const key = trimmed.slice(0, eqIdx).trim();
            let val = trimmed.slice(eqIdx + 1).trim();
            // Gọt bỏ dấu ngoặc kép hoặc ngoặc đơn nếu người dùng lỡ thêm vào
            val = val.replace(/^["']|["']$/g, '').trim();
            if (val) config[key] = val;
        }
    }
    return config;
}

const rootEnvPath = path.resolve(__dirname, '../../.env');
const localEnvPath = path.resolve(__dirname, '.env');
const envConfig = { ...loadEnvConfig(rootEnvPath), ...loadEnvConfig(localEnvPath) };

// URL máy chủ AI: Tự động ưu tiên NGROK_DOMAIN -> AI_SERVER_URL -> localhost:8000
let COLAB_BASE_URL = process.env.AI_SERVER_URL 
    || (envConfig.NGROK_DOMAIN ? `https://${envConfig.NGROK_DOMAIN.replace(/^https?:\/\//, '')}` : null)
    || envConfig.AI_SERVER_URL 
    || "http://localhost:8000";

if (!fs.existsSync('uploads')) fs.mkdirSync('uploads');

const upload = multer({ dest: 'uploads/' });

app.post('/api/set-url', (req, res) => {
    const { url } = req.body;
    if (!url || (!url.startsWith('https://') && !url.startsWith('http://'))) {
        return res.status(400).json({ error: "URL không hợp lệ. Phải bắt đầu bằng http:// hoặc https://" });
    }
    COLAB_BASE_URL = url.replace(/\/+$/, '');
    console.log(`✅ Đã cập nhật AI_SERVER_URL: ${COLAB_BASE_URL}`);

    // Lưu lại vào file .env để ghi nhớ cho lần sau
    try {
        if (fs.existsSync(rootEnvPath)) {
            let content = fs.readFileSync(rootEnvPath, 'utf-8');
            if (content.includes('AI_SERVER_URL=')) {
                content = content.replace(/AI_SERVER_URL=.*/g, `AI_SERVER_URL=${COLAB_BASE_URL}`);
            } else {
                content += `\nAI_SERVER_URL=${COLAB_BASE_URL}\n`;
            }
            fs.writeFileSync(rootEnvPath, content, 'utf-8');
            console.log(`💾 Đã đồng bộ URL mới vào file .env`);
        }
    } catch (e) {
        console.warn(`⚠️ Không thể ghi lại .env: ${e.message}`);
    }

    res.json({ success: true, url: COLAB_BASE_URL, predict_endpoint: `${COLAB_BASE_URL}/predict` });
});

// [THÊM MỚI] Endpoint kiểm tra URL hiện tại đang dùng
app.get('/api/status', (req, res) => {
    res.json({
        backend_running: true,
        colab_url: COLAB_BASE_URL,
        predict_endpoint: `${COLAB_BASE_URL}/predict`,
        capture_server_running: !!captureInfo,
        capture_url: captureInfo ? captureInfo.url : null,
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// 📱 QUẢN LÝ PHONE-CAMERA CAPTURE NODE (Port 8765 HTTPS)
// ─────────────────────────────────────────────────────────────────────────────
const { spawn } = require('child_process');
const pythonPath = path.resolve(__dirname, '../../.venv/Scripts/python.exe');
const startScript = path.resolve(__dirname, '../../capture_server/start.py');

let captureProcess = null;
let captureInfo = null; // { url, qr, port }
let sseClients = [];

// Stream sự kiện thời gian thực (SSE) gửi đến React Dashboard
app.get('/api/capture/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    sseClients.push(res);
    res.write(`data: ${JSON.stringify({ type: 'connected', captureInfo })}\n\n`);

    req.on('close', () => {
        sseClients = sseClients.filter(c => c !== res);
    });
});

function broadcastToDashboard(payload) {
    const data = `data: ${JSON.stringify(payload)}\n\n`;
    sseClients.forEach(client => {
        try { client.write(data); } catch (_) {}
    });
}

function ensureCaptureServer() {
    return new Promise((resolve, reject) => {
        if (captureInfo && captureProcess && !captureProcess.killed) {
            return resolve(captureInfo);
        }

        const pyExec = fs.existsSync(pythonPath) ? pythonPath : 'python';
        console.log(`[*] Khoi chay Phone Capture Node bang: ${pyExec}`);

        captureProcess = spawn(pyExec, [startScript, '8765'], {
            cwd: path.resolve(__dirname, '../..'),
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let outputBuffer = '';
        let resolved = false;

        captureProcess.stdout.on('data', (data) => {
            const text = data.toString();
            outputBuffer += text;
            for (const line of outputBuffer.split(/\r?\n/)) {
                if (line.startsWith('READY:')) {
                    try {
                        const jsonStr = line.slice(6).trim();
                        captureInfo = JSON.parse(jsonStr);
                        resolved = true;
                        console.log(`✅ Phone Capture Node da san sang tai: ${captureInfo.url}`);
                        broadcastToDashboard({ type: 'capture_ready', captureInfo });
                        resolve(captureInfo);
                        break;
                    } catch (e) {
                        console.error("Loi parse READY json:", e);
                    }
                }
            }
        });

        captureProcess.stderr.on('data', (data) => {
            console.error(`[CAPTURE ERR] ${data.toString().trim()}`);
        });

        captureProcess.on('exit', (code) => {
            console.log(`[CAPTURE] Tien trinh dung voi ma ${code}`);
            captureProcess = null;
            captureInfo = null;
            broadcastToDashboard({ type: 'capture_stopped' });
            if (!resolved) reject(new Error(`Capture Server dung voi ma ${code}`));
        });

        setTimeout(() => {
            if (!resolved) {
                if (captureInfo) resolve(captureInfo);
                else reject(new Error("Timeout khi khoi dong Capture Server (>15s)"));
            }
        }, 15000);
    });
}

// Endpoint lay QR Code va URL ket noi cho dien thoai
app.get('/api/capture/qr', async (req, res) => {
    try {
        const info = await ensureCaptureServer();
        res.json({ success: true, url: info.url, qr: info.qr, port: info.port });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

app.get('/api/capture/status', (req, res) => {
    res.json({
        running: !!captureInfo,
        url: captureInfo ? captureInfo.url : null,
    });
});

app.post('/api/capture/stop', (req, res) => {
    if (captureProcess) {
        captureProcess.kill();
        captureProcess = null;
    }
    captureInfo = null;
    broadcastToDashboard({ type: 'capture_stopped' });
    res.json({ success: true, message: "Capture Server da dung" });
});

// ─────────────────────────────────────────────────────────────────────────────
// Endpoint chính: nhận ảnh + thông số → chuyển tiếp tới Colab AI Server
// ─────────────────────────────────────────────────────────────────────────────
app.post('/api/predict', upload.any(), async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({ error: "Chưa có file ảnh được upload!" });
        }

        const file     = req.files[0];
        const formData = new FormData();

        formData.append('file',   fs.createReadStream(file.path), file.originalname);
        formData.append('diam',   req.body.diam);
        formData.append('height', req.body.height);
        formData.append('empty',  req.body.empty);
        formData.append('wall_thickness', req.body.wall_thickness || 0.1);
        formData.append('weight_total',   req.body.weight_total || 0);
        formData.append('sample_count',   req.body.sample_count || 0);
        formData.append('sample_weight',  req.body.sample_weight || 0);

        const targetUrl = `${COLAB_BASE_URL}/predict`;
        console.log(`🚀 Gửi request tới: ${targetUrl}`);
        console.log(`   diam=${req.body.diam} | height=${req.body.height} | empty=${req.body.empty}`);

        const response = await axios.post(targetUrl, formData, {
            timeout: 120000,
            headers: { ...formData.getHeaders(), 'ngrok-skip-browser-warning': 'true' },
        });

        if (fs.existsSync(file.path)) fs.unlinkSync(file.path);

        console.log(`✅ Nhận response từ Colab/Local: status=${response.data?.status}`);
        
        // Phát sóng kết quả qua SSE để React Dashboard cập nhật realtime
        broadcastToDashboard({ type: 'result', data: response.data });

        res.json(response.data);

    } catch (error) {
        // Dọn file tạm kể cả khi lỗi
        if (req.files?.[0]?.path && fs.existsSync(req.files[0].path)) {
            fs.unlinkSync(req.files[0].path);
        }

        console.error("--- LỖI SERVER.JS ---");

        if (error.code === 'ECONNABORTED') {
            console.error("Timeout: Colab không phản hồi trong 120s");
            const errObj = { error: "AI Server timeout (>120s). Kiểm tra Colab còn chạy không." };
            broadcastToDashboard({ type: 'error', error: errObj });
            return res.status(504).json(errObj);
        }

        if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
            console.error(`Không kết nối được tới: ${COLAB_BASE_URL}`);
            const errObj = {
                error: `Không kết nối được tới AI Server (${COLAB_BASE_URL}). Ngrok URL có thể đã hết hạn — cập nhật lại qua POST /api/set-url`
            };
            broadcastToDashboard({ type: 'error', error: errObj });
            return res.status(503).json(errObj);
        }

        if (error.response) {
            const status = error.response.status;
            const errData = error.response.data || { error: `Colab báo lỗi ${status}` };
            console.error("Colab phản hồi lỗi HTTP:", status, errData);
            broadcastToDashboard({ type: 'error', error: errData });
            return res.status(status).json(errData);
        }

        console.error("Lỗi hệ thống:", error.message);
        const genericErr = { error: error.message };
        broadcastToDashboard({ type: 'error', error: genericErr });
        res.status(500).json(genericErr);
    }
});

app.listen(port, () => {
    console.log(`🚀 Backend đang chạy tại http://localhost:${port}`);
    console.log(`   Colab URL hiện tại: ${COLAB_BASE_URL}`);
    console.log(`   Cập nhật ngrok URL: POST http://localhost:${port}/api/set-url`);
    console.log(`   Kiểm tra kết nối  : GET  http://localhost:${port}/api/status`);
});