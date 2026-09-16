const token = new URLSearchParams(location.search).get("token") || "";
const fieldKeys = ["Weight_g", "Container_Height_mm", "Inner_Diameter_mm", "Empty_Height_mm", "Actual_Count"];
const nodeId = localStorage.getItem("riceCapture.nodeId") || (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
localStorage.setItem("riceCapture.nodeId", nodeId);

const input = document.getElementById("cameraInput");
const video = document.getElementById("cameraVideo");
const cameraStage = video.closest(".camera-stage");
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
let selectedFile = null;
let previewUrl = null;
let pendingRequestId = null;
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
  // Try to enable flash button if using environment camera
  const supported = facingMode === "environment";
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
  if (!track) {
    resetFlashControl();
    return false;
  }
  flashToggle.disabled = true;
  try {
    await track.applyConstraints({ advanced: [{ torch: enabled }] });
    flashEnabled = enabled;
    flashToggle.textContent = enabled ? "TẮT FLASH" : "BẬT FLASH";
    flashToggle.classList.toggle("flash-on", enabled);
    if (!silent) setStatus(enabled ? "Flash đang bật để trợ sáng khi chụp." : "Flash đã tắt.");
    return true;
  } catch (error) {
    flashEnabled = false;
    flashToggle.textContent = "BẬT FLASH";
    flashToggle.classList.remove("flash-on");
    if (!silent) setStatus("Trình duyệt hoặc camera này không hỗ trợ bật flash bằng web.", "error");
    return false;
  } finally {
    flashToggle.disabled = false;
  }
}

function updateComputed() {
  const container = Number(document.getElementById("Container_Height_mm").value);
  const empty = Number(document.getElementById("Empty_Height_mm").value);
  document.getElementById("riceHeight").textContent = Number.isFinite(container) && Number.isFinite(empty) && container >= empty
    ? String(Math.round((container - empty) * 1000000) / 1000000)
    : "—";
}

function persistFields() {
  const values = Object.fromEntries(fieldKeys.map(key => [key, document.getElementById(key).value]));
  localStorage.setItem("riceCapture.manual", JSON.stringify(values));
}

function applyFields(serverValues = {}) {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("riceCapture.manual") || "{}"); } catch (_) {}
  for (const key of fieldKeys) {
    const value = saved[key] ?? serverValues[key] ?? "";
    document.getElementById(key).value = value;
  }
  updateComputed();
}

function updateSessionDisplay(data) {
  document.getElementById("nextId").textContent = data.next_sample_id;
}

async function loadSession(initial = false) {
  const response = await fetch("/api/session", { headers: { "X-Capture-Token": token } });
  if (!response.ok) throw new Error("Không thể kết nối phiên chụp trên máy tính.");
  const data = await response.json();
  if (initial) applyFields(data.manual);
  updateSessionDisplay(data);
  badge.textContent = "Đã kết nối";
  badge.className = "badge online";
}

function connectSocket() {
  if (!pageActive) return;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const label = encodeURIComponent(navigator.userAgent.includes("iPhone") ? "iPhone" : "Điện thoại Android");
  const socket = new WebSocket(`${scheme}://${location.host}/ws/node?token=${encodeURIComponent(token)}&node_id=${encodeURIComponent(nodeId)}&label=${label}`);
  socket.onopen = () => { badge.textContent = "Đã kết nối"; badge.className = "badge online"; };
  socket.onmessage = event => {
    try {
      const data = JSON.parse(event.data);
      if (data.next_sample_id) updateSessionDisplay(data);
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

function revokePreviewUrl() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

function clearSelectedImage() {
  selectedFile = null;
  pendingRequestId = null;
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
  preview.classList.add("hidden");
  video.classList.remove("hidden");
  takePhotoButton.disabled = false;
  switchCameraButton.disabled = false;
  aspectRatioSelect.disabled = false;
  cameraNotice.textContent = "Camera đang hiển thị trực tiếp. Chạm vào mẫu để lấy nét rồi bấm Chụp ảnh.";
}

function showCapturedImage(file, message) {
  clearSelectedImage();
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  pendingRequestId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  preview.src = previewUrl;
  video.classList.add("hidden");
  previewEmpty.classList.add("hidden");
  preview.classList.remove("hidden");
  clearButton.classList.remove("hidden");
  takePhotoButton.disabled = true;
  saveButton.disabled = false;
  aspectRatioSelect.disabled = true;
  cameraNotice.textContent = "Ảnh đã chụp. Kiểm tra ảnh và thông số trước khi lưu.";
  setStatus(message);
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
    return "Camera trong khung cần HTTPS và chứng chỉ đã được tin cậy. Hãy cài chứng chỉ từ máy tính rồi quét lại QR.";
  }
  if (error?.name === "NotAllowedError") return "Bạn chưa cho phép trình duyệt dùng camera. Hãy cấp quyền Camera rồi thử lại.";
  if (error?.name === "NotFoundError") return "Không tìm thấy camera phù hợp trên điện thoại.";
  if (error?.name === "NotReadableError") return "Camera đang được ứng dụng khác sử dụng. Hãy đóng ứng dụng đó rồi thử lại.";
  return `Không thể bật camera trong khung: ${error?.message || error}`;
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
    setStatus("Camera đã sẵn sàng. Form vẫn được giữ ở phía trên.");
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
  if (!mediaStream) {
    cameraNotice.textContent = "Tỷ lệ đã chọn sẽ được dùng khi bạn bật camera.";
    return;
  }
  setStatus("Đang áp dụng tỷ lệ khung hình mới…");
  await startInlineCamera();
});

flashToggle.addEventListener("click", () => { void setFlash(!flashEnabled); });

function pointInsideVideo(clientX, clientY) {
  const rect = video.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const sourceWidth = video.videoWidth || rect.width;
  const sourceHeight = video.videoHeight || rect.height;
  const scale = Math.min(rect.width / sourceWidth, rect.height / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const left = rect.left + (rect.width - renderedWidth) / 2;
  const top = rect.top + (rect.height - renderedHeight) / 2;
  if (clientX < left || clientX > left + renderedWidth || clientY < top || clientY > top + renderedHeight) {
    return null;
  }
  return {
    x: Math.min(1, Math.max(0, (clientX - left) / renderedWidth)),
    y: Math.min(1, Math.max(0, (clientY - top) / renderedHeight)),
  };
}

function showFocusRing(clientX, clientY) {
  const stageRect = cameraStage.getBoundingClientRect();
  const focusRing = document.createElement("span");
  focusRing.className = "focus-ring";
  focusRing.style.left = `${clientX - stageRect.left}px`;
  focusRing.style.top = `${clientY - stageRect.top}px`;
  cameraStage.appendChild(focusRing);
  requestAnimationFrame(() => focusRing.classList.add("visible"));
  setTimeout(() => {
    focusRing.classList.remove("visible");
    setTimeout(() => focusRing.remove(), 220);
  }, 650);
}

async function focusCameraAt(event) {
  const track = activeVideoTrack();
  if (!track) return;
  const point = pointInsideVideo(event.clientX, event.clientY);
  if (!point) return;
  showFocusRing(event.clientX, event.clientY);

  const capabilities = track.getCapabilities?.() || {};
  const focusModes = Array.isArray(capabilities.focusMode) ? capabilities.focusMode : [];
  const supportsPoint = capabilities.pointsOfInterest === true
    || navigator.mediaDevices.getSupportedConstraints?.().pointsOfInterest === true;

  if (supportsPoint) {
    try {
      const pointConstraints = { pointsOfInterest: [point] };
      if (focusModes.includes("single-shot")) pointConstraints.focusMode = "single-shot";
      await track.applyConstraints({ advanced: [pointConstraints] });
      cameraNotice.textContent = "Đã lấy nét tại điểm chạm. Giữ máy ổn định rồi chụp ảnh.";
      return;
    } catch (error) {
      console.debug("Camera không nhận điểm lấy nét, chuyển sang autofocus:", error);
    }
  }
  const fallbackMode = focusModes.includes("single-shot")
    ? "single-shot"
    : (focusModes.includes("continuous") ? "continuous" : null);
  if (fallbackMode) {
    try {
      await track.applyConstraints({ advanced: [{ focusMode: fallbackMode }] });
      cameraNotice.textContent = "Camera đã chạy lấy nét tự động tại vùng trung tâm.";
      return;
    } catch (error) {
      console.debug("Camera không cho đổi chế độ lấy nét:", error);
    }
  }
  cameraNotice.textContent = "Điện thoại này tự điều khiển lấy nét; hãy giữ máy ổn định sau khi chạm.";
}

video.addEventListener("pointerup", event => {
  event.preventDefault();
  void focusCameraAt(event);
});

takePhotoButton.addEventListener("click", () => {
  if (!mediaStream || !video.videoWidth || !video.videoHeight) {
    setStatus("Camera chưa tạo được khung hình. Hãy chờ một chút rồi thử lại.", "error");
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
    showCapturedImage(file, `Đã chụp ảnh ${canvas.width} × ${canvas.height}. Kiểm tra rồi bấm lưu.`);
  }, "image/jpeg", 0.95);
});

clearButton.addEventListener("click", () => {
  if (mediaStream) {
    showLiveCamera();
    setStatus("Đã bỏ ảnh cũ. Hãy căn chỉnh và chụp lại.");
  } else {
    clearSelectedImage();
    startInlineCamera();
  }
});

input.addEventListener("change", () => {
  const file = input.files && input.files[0] ? input.files[0] : null;
  if (!file) return;
  stopInlineCamera(false);
  showCapturedImage(file, `Đã chọn ảnh ${(file.size / 1024 / 1024).toFixed(1)} MB. Kiểm tra rồi bấm lưu.`);
});

for (const key of fieldKeys) {
  const el = document.getElementById(key);
  el.addEventListener("input", () => {
    if (el.getAttribute("inputmode") === "decimal") {
      let val = el.value.replace(/,/g, ".");
      val = val.replace(/[^0-9.]/g, "");
      const parts = val.split(".");
      if (parts.length > 2) val = parts[0] + "." + parts.slice(1).join("");
      el.value = val;
    } else if (el.getAttribute("inputmode") === "numeric") {
      el.value = el.value.replace(/[^0-9]/g, "");
    }
    persistFields();
    updateComputed();
  });
}

async function convertToJpeg(file) {
  if (file.type === "image/jpeg") return file;
  const image = new Image();
  const url = URL.createObjectURL(file);
  try {
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = url; });
    const maxEdge = 4096;
    const scale = Math.min(1, maxEdge / Math.max(image.naturalWidth, image.naturalHeight));
    const conversionCanvas = document.createElement("canvas");
    conversionCanvas.width = Math.round(image.naturalWidth * scale);
    conversionCanvas.height = Math.round(image.naturalHeight * scale);
    conversionCanvas.getContext("2d").drawImage(image, 0, 0, conversionCanvas.width, conversionCanvas.height);
    return await new Promise((resolve, reject) => conversionCanvas.toBlob(
      blob => blob ? resolve(new File([blob], "capture.jpg", { type: "image/jpeg" })) : reject(new Error("Không thể đổi ảnh sang JPEG.")),
      "image/jpeg", 0.95
    ));
  } finally {
    URL.revokeObjectURL(url);
  }
}

saveButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  for (const key of fieldKeys) {
    if (document.getElementById(key).value === "") {
      setStatus("Hãy nhập đầy đủ tất cả thông số.", "error");
      document.getElementById(key).focus();
      return;
    }
  }
  saveButton.disabled = true;
  setStatus("Đang gửi ảnh về máy tính…");
  try {
    const jpeg = await convertToJpeg(selectedFile);
    const body = new FormData();
    body.append("image", jpeg, "capture.jpg");
    for (const key of fieldKeys) body.append(key, document.getElementById(key).value);
    body.append("node_id", nodeId);
    body.append("request_id", pendingRequestId || (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`));
    const response = await fetch("/api/samples", { method: "POST", headers: { "X-Capture-Token": token }, body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Máy tính không thể lưu mẫu.");
    persistFields();
    updateSessionDisplay({ next_sample_id: data.next_sample_id, manual: {} });
    clearSelectedImage();
    if (mediaStream) showLiveCamera();
    else previewEmpty.classList.remove("hidden");
    setStatus(`Đã lưu ${data.sample_id}. Các thông số được giữ nguyên; có thể chụp mẫu tiếp theo.`, "success");
  } catch (error) {
    saveButton.disabled = false;
    setStatus(error.message || String(error), "error");
  }
});

window.addEventListener("beforeunload", () => {
  pageActive = false;
  stopInlineCamera(false);
  revokePreviewUrl();
});

if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
  cameraNotice.textContent = "Camera trong khung chưa khả dụng: cần HTTPS và chứng chỉ Rice Capture đã được tin cậy.";
  fallbackPanel.open = true;
}

loadSession(true).then(connectSocket).catch(error => {
  badge.textContent = "Không kết nối";
  badge.className = "badge offline";
  setStatus(error.message, "error");
});
