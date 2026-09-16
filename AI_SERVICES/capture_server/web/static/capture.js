const token = new URLSearchParams(location.search).get("token") || "";
const fieldKeys = ["diam", "height", "empty", "wall_thickness", "weight_total", "sample_count", "sample_weight"];
const nodeId = localStorage.getItem("riceVision.nodeId") || (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
localStorage.setItem("riceVision.nodeId", nodeId);

const input = document.getElementById("cameraInput");
const video = document.getElementById("cameraVideo");
const canvas = document.getElementById("captureCanvas");
const preview = document.getElementById("preview");
const previewEmpty = document.getElementById("previewEmpty");
const cameraNotice = document.getElementById("cameraNotice");
const fallbackPanel = document.getElementById("fallbackPanel");
const startCameraButton = document.getElementById("startCamera");
const takePhotoButton = document.getElementById("takePhoto");
const switchCameraButton = document.getElementById("switchCamera");
const aspectRatioSelect = document.getElementById("aspectRatio");
const flashToggle = document.getElementById("flashToggle");
const clearButton = document.getElementById("clearImage");
const saveButton = document.getElementById("saveButton");
const status = document.getElementById("status");
const badge = document.getElementById("connectionBadge");
const resultCard = document.getElementById("resultCard");

let selectedFile = null;
let previewUrl = null;
let mediaStream = null;
let facingMode = "environment";
let flashEnabled = false;
let pageActive = true;

function setStatus(message, kind = "") {
  status.textContent = message;
  status.className = `status ${kind}`;
}

function selectedAspectRatio() {
  if (aspectRatioSelect.value === "4:3") return 4 / 3;
  if (aspectRatioSelect.value === "16:9") return 16 / 9;
  return null;
}

function videoConstraints() {
  const ratio = selectedAspectRatio();
  const constraints = { facingMode: { ideal: facingMode } };
  if (ratio) {
    constraints.aspectRatio = { ideal: ratio };
    constraints.width = { ideal: 1920 };
    constraints.height = { ideal: aspectRatioSelect.value === "4:3" ? 1440 : 1080 };
  }
  return constraints;
}

function activeVideoTrack() {
  return mediaStream?.getVideoTracks?.()[0] || null;
}

function resetFlashControl() {
  flashEnabled = false;
  flashToggle.textContent = "BẬT FLASH";
  flashToggle.classList.remove("flash-on");
  flashToggle.disabled = true;
  flashToggle.classList.add("hidden");
}

function updateFlashControl() {
  const capabilities = activeVideoTrack()?.getCapabilities?.() || {};
  const supported = Boolean(capabilities.torch);
  flashToggle.classList.toggle("hidden", !supported);
  flashToggle.disabled = !supported;
  if (!supported) {
    flashEnabled = false;
    flashToggle.textContent = "BẬT FLASH";
    flashToggle.classList.remove("flash-on");
  }
  return supported;
}

async function setFlash(enabled, silent = false) {
  const track = activeVideoTrack();
  const capabilities = track?.getCapabilities?.() || {};
  if (!track || !capabilities.torch) {
    resetFlashControl();
    return;
  }
  try {
    await track.applyConstraints({ advanced: [{ torch: enabled }] });
    flashEnabled = enabled;
    flashToggle.textContent = enabled ? "TẮT FLASH" : "BẬT FLASH";
    flashToggle.classList.toggle("flash-on", enabled);
  } catch (error) {
    if (!silent) setStatus("Không thể điều khiển đèn flash.", "error");
  }
}

function toStandardNumber(str) {
  if (typeof str !== "string") str = String(str || "");
  return str.replace(/,/g, ".").trim();
}

function updateComputed() {
  const container = parseFloat(toStandardNumber(document.getElementById("height").value));
  const empty = parseFloat(toStandardNumber(document.getElementById("empty").value));
  document.getElementById("riceHeight").textContent = Number.isFinite(container) && Number.isFinite(empty) && container >= empty
    ? String(Math.round((container - empty) * 100) / 100)
    : "—";
}

function persistFields() {
  const values = Object.fromEntries(fieldKeys.map(key => [key, document.getElementById(key).value]));
  localStorage.setItem("riceVision.manual", JSON.stringify(values));
}

function restoreFields() {
  try {
    const saved = JSON.parse(localStorage.getItem("riceVision.manual") || "{}");
    for (const key of fieldKeys) {
      if (saved[key] !== undefined && saved[key] !== "") {
        document.getElementById(key).value = saved[key];
      }
    }
  } catch (_) {}
  updateComputed();
}

function stopInlineCamera(showPlaceholder = true) {
  if (flashEnabled) void setFlash(false, true);
  if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
  mediaStream = null;
  video.srcObject = null;
  video.classList.add("hidden");
  startCameraButton.textContent = "BẬT CAMERA";
  takePhotoButton.disabled = true;
  switchCameraButton.disabled = true;
  resetFlashControl();
  if (showPlaceholder && !selectedFile) {
    previewEmpty.textContent = "Camera chưa bật";
    previewEmpty.classList.remove("hidden");
  }
}

function cameraErrorMessage(error) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    return "Camera trong khung cần HTTPS. Hãy đảm bảo bạn kết nối qua https://.";
  }
  if (error?.name === "NotAllowedError") return "Bạn chưa cấp quyền Camera cho trình duyệt.";
  if (error?.name === "NotFoundError") return "Không tìm thấy camera phù hợp trên điện thoại.";
  if (error?.name === "NotReadableError") return "Camera đang bị ứng dụng khác chiếm giữ.";
  return `Lỗi mở camera: ${error?.message || error}`;
}

async function startInlineCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    const message = cameraErrorMessage();
    cameraNotice.textContent = message;
    fallbackPanel.open = true;
    setStatus(message, "error");
    return;
  }
  startCameraButton.disabled = true;
  stopInlineCamera(false);
  clearSelectedImage();
  previewEmpty.textContent = "Đang mở camera…";
  previewEmpty.classList.remove("hidden");
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: videoConstraints(),
    });
    video.srcObject = mediaStream;
    await video.play();
    startCameraButton.textContent = "TẮT CAMERA";
    showLiveCamera();
    updateFlashControl();
    setStatus("Camera đã sẵn sàng. Căn chỉnh mẫu rồi chụp.");
  } catch (error) {
    stopInlineCamera(true);
    const message = cameraErrorMessage(error);
    cameraNotice.textContent = message;
    fallbackPanel.open = true;
    setStatus(message, "error");
  } finally {
    startCameraButton.disabled = false;
  }
}

startCameraButton.addEventListener("click", () => {
  if (mediaStream) {
    stopInlineCamera(true);
    cameraNotice.textContent = "Camera đã tắt. Bấm Bật camera khi cần chụp.";
  } else {
    startInlineCamera();
  }
});

switchCameraButton.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startInlineCamera();
});

aspectRatioSelect.addEventListener("change", async () => {
  if (!mediaStream) return;
  await startInlineCamera();
});

flashToggle.addEventListener("click", () => { void setFlash(!flashEnabled); });

function revokePreviewUrl() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

function clearSelectedImage() {
  selectedFile = null;
  input.value = "";
  preview.removeAttribute("src");
  preview.classList.add("hidden");
  clearButton.classList.add("hidden");
  saveButton.disabled = true;
  aspectRatioSelect.disabled = false;
  revokePreviewUrl();
}

function showLiveCamera() {
  clearSelectedImage();
  previewEmpty.classList.add("hidden");
  video.classList.remove("hidden");
  takePhotoButton.disabled = false;
  switchCameraButton.disabled = false;
  aspectRatioSelect.disabled = false;
  cameraNotice.textContent = "Camera trực tiếp. Căn giữa cốc lúa rồi bấm Chụp ảnh.";
}

function showCapturedImage(file, message) {
  clearSelectedImage();
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  video.classList.add("hidden");
  previewEmpty.classList.add("hidden");
  preview.classList.remove("hidden");
  clearButton.classList.remove("hidden");
  takePhotoButton.disabled = true;
  saveButton.disabled = false;
  aspectRatioSelect.disabled = true;
  cameraNotice.textContent = "Ảnh đã chụp. Kiểm tra ảnh và thông số rồi bấm Ước lượng.";
  setStatus(message);
}

takePhotoButton.addEventListener("click", () => {
  if (!mediaStream || !video.videoWidth || !video.videoHeight) {
    setStatus("Camera chưa sẵn sàng.", "error");
    return;
  }
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d", { alpha: false });
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    if (!blob) {
      setStatus("Không thể tạo ảnh từ camera.", "error");
      return;
    }
    const file = new File([blob], "capture.jpg", { type: "image/jpeg", lastModified: Date.now() });
    showCapturedImage(file, `Đã chụp ảnh ${canvas.width} × ${canvas.height}.`);
  }, "image/jpeg", 0.95);
});

clearButton.addEventListener("click", () => {
  if (mediaStream) {
    showLiveCamera();
  } else {
    clearSelectedImage();
    startInlineCamera();
  }
});

input.addEventListener("change", () => {
  const file = input.files && input.files[0] ? input.files[0] : null;
  if (!file) return;
  stopInlineCamera(false);
  showCapturedImage(file, `Đã chọn ảnh ${(file.size / 1024 / 1024).toFixed(1)} MB.`);
});

for (const key of fieldKeys) {
  const el = document.getElementById(key);
  if (el) {
    el.addEventListener("input", (e) => {
      // Tự động đổi dấu phẩy thành dấu chấm ngay khi gõ
      if (e.target.value.includes(",")) {
        const start = e.target.selectionStart;
        e.target.value = e.target.value.replace(/,/g, ".");
        e.target.setSelectionRange(start, start);
      }
      persistFields();
      updateComputed();
    });
  }
}

function renderResult(data) {
  if (!data || data.status !== "success") return;
  const est = data.estimation || {};
  const metrics = data.metrics_summary || {};
  
  document.getElementById("resFinal").textContent = est.final?.toLocaleString() || "0";
  document.getElementById("resMethod").textContent = est.method_used?.includes("regression")
    ? `Mô hình Hồi quy (${metrics.regression_model || "Extra Trees"})`
    : (est.method_used === "hybrid" ? "Ước lượng Hybrid" : "Thể tích thuần AI");
  
  document.getElementById("resReg").textContent = est.regression_est?.toLocaleString() || "—";
  document.getElementById("resAi").textContent = est.ai_est?.toLocaleString() || "—";
  document.getElementById("resDetected").textContent = metrics.total_grains_detected ?? "—";
  document.getElementById("resUniformity").textContent = metrics.uniformity_rate_pct != null ? metrics.uniformity_rate_pct.toFixed(1) : "—";
  
  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Gui anh va thong so len may tinh xu ly
saveButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  
  const diamVal = toStandardNumber(document.getElementById("diam").value);
  const heightVal = toStandardNumber(document.getElementById("height").value);
  if (!diamVal || !heightVal) {
    setStatus("Vui lòng nhập Đường kính ly và Chiều cao ly.", "error");
    return;
  }

  saveButton.disabled = true;
  saveButton.textContent = "ĐANG PHÂN TÍCH AI…";
  setStatus("Đang gửi ảnh sang máy tính xử lý… Vui lòng đợi trong giây lát.");

  try {
    const body = new FormData();
    body.append("file", selectedFile, "capture.jpg");
    for (const key of fieldKeys) {
      const val = toStandardNumber(document.getElementById(key).value);
      body.append(key, val || 0);
    }
    body.append("node_id", nodeId);

    const response = await fetch("/api/capture", {
      method: "POST",
      headers: { "X-Capture-Token": token },
      body,
    });

    const data = await response.json();
    if (!response.ok || data.status !== "success") {
      throw new Error(data.detail || data.error || data.message || "Lỗi xử lý AI.");
    }

    persistFields();
    renderResult(data);
    setStatus("Phân tích hoàn tất! Kết quả đã được hiển thị trên cả điện thoại và máy tính.", "success");

  } catch (error) {
    setStatus(error.message || String(error), "error");
  } finally {
    saveButton.disabled = false;
    saveButton.textContent = "GỬI ẢNH VÀ ƯỚC LƯỢNG";
  }
});

function connectSocket() {
  if (!pageActive) return;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const label = encodeURIComponent(navigator.userAgent.includes("iPhone") ? "iPhone" : "Android");
  const wsUrl = `${scheme}://${location.host}/ws/node?token=${encodeURIComponent(token)}&node_id=${encodeURIComponent(nodeId)}&label=${label}`;
  
  const socket = new WebSocket(wsUrl);
  socket.onopen = () => {
    badge.textContent = "Đã kết nối Host";
    badge.className = "badge online";
  };
  socket.onmessage = event => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "result") {
        renderResult(msg.data);
      }
    } catch (_) {}
  };
  const pingTimer = setInterval(() => {
    if (socket.readyState === WebSocket.OPEN) socket.send("ping");
  }, 5000);
  socket.onclose = () => {
    clearInterval(pingTimer);
    if (!pageActive) return;
    badge.textContent = "Mất kết nối";
    badge.className = "badge offline";
    setTimeout(connectSocket, 3000);
  };
}

window.addEventListener("beforeunload", () => {
  pageActive = false;
  stopInlineCamera(false);
  revokePreviewUrl();
});

restoreFields();
connectSocket();
