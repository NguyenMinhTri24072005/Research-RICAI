#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE 1: NHẬN DIỆN VẬT CHỨA & QUY ĐỔI TỶ LỆ KÍCH THƯỚC (CONTAINER DETECTOR)
Phiên bản 2.0 — VIẾT LẠI KIẾN TRÚC (thay thế toàn bộ v1 + 7 đợt vá)
===============================================================================

LÝ DO VIẾT LẠI (chẩn đoán lỗi của v1):
--------------------------------------
v1 dò miệng ly bằng Canny -> findContours -> chấm điểm theo
`area * aspect^2 * coverage`. Trên ảnh ly KÍNH TRONG SUỐT đặt trên nền sáng,
cạnh mạnh nhất trong ảnh KHÔNG phải viền ly (gần như không có gradient) mà là:
    (a) biên các hạt lúa (tan/vàng trên nền tối bên trong ly),
    (b) biên vùng bóng đổ của ly trên mặt bàn.
Hàm score nói trên ưu tiên "blob tròn, đặc, tương phản cao" — tức là ưu tiên
ĐÚNG đám hạt lúa. Thuật toán làm chính xác việc nó được yêu cầu làm, và vì thế
7 đợt vá (MORPH_CLOSE, angular coverage, Hough fallback, RANSAC, sanity check,
halo overlay, chuẩn hoá độ phân giải) không thể cứu được: chúng đều là các cải
tiến cho bước FIT, trong khi lỗi nằm ở bước CHỌN ĐỐI TƯỢNG.

Các lỗ hổng cụ thể của v1 đã được xử lý trong v2:
  1. `_contour_angular_coverage` đo độ phủ quanh tâm của CHÍNH contour đó, nên
     mọi contour khép kín (kể cả viền một cụm hạt lúa) đều trả về 1.0 -> bộ lọc
     gần như vô hiệu. v2 đo độ phủ quanh TÂM KHỐI LÚA (một tâm tham chiếu độc
     lập với contour đang xét).
  2. `min_outer_diam_ratio * min_side` là ngưỡng TƯƠNG ĐỐI THEO KHUNG ẢNH, nên
     khi ly chỉ chiếm ~20% cạnh ngắn (ảnh chụp xa), cụm hạt lúa vẫn đủ lớn để
     lọt qua. Ngưỡng theo khung ảnh không thể phân biệt "chụp xa" với "bắt
     nhầm". v2 dùng RÀNG BUỘC BAO HÀM: miệng ly bắt buộc phải BAO QUANH khối
     lúa (vật lý không thể trùng hợp), còn ngưỡng theo khung ảnh bị hạ cấp
     thành cảnh báo.
  3. `abs(ratio - expected_ratio) <= 0.035` là bẫy trùng hợp: với ly 20mm thành
     1.5mm, expected_ratio ~= 0.87, nên hai contour lồng nhau bất kỳ trong đám
     hạt có xác suất cao chạm dải 0.835–0.905. v2 vẫn dùng expected_ratio nhưng
     chỉ để CHỌN CẶP trong số các bán kính đã được bình chọn bởi gradient hướng
     tâm, chứ không dùng làm điều kiện chấp nhận độc lập.
  4. `cv2.dilate` sau `MORPH_CLOSE` làm phình contour ngoài -> `outer_diam_px`
     bị thổi lên một cách HỆ THỐNG (sai lệch một chiều, không phải nhiễu ngẫu
     nhiên). v2 bỏ hẳn dilate và định vị viền bằng ĐỈNH gradient trên tia hướng
     tâm, cho độ chính xác dưới pixel và không lệch chiều.
  5. `RETR_TREE` trên ảnh cạnh trả về contour của DẢI cạnh: mỗi viền sinh 2
     contour (biên trong + biên ngoài của dải), làm mọi bộ lọc hình dạng
     (area, circularity, solidity) mất ý nghĩa. v2 không dựa vào contour của
     ảnh cạnh để đo kích thước.
  6. `raise RuntimeError` giết cả lô ảnh khi chạy batch. v2 có tham số
     `on_failure="dict"` để trả về {"status": "failed", "reason": ...}.

NGUYÊN LÝ v2 (đảo ngược thứ tự suy luận):
------------------------------------------
  B1. TÁCH KHỐI LÚA BẰNG MÀU (HSV) — đây là tín hiệu ổn định nhất trong ảnh:
      hạt lúa có hue vàng/nâu và saturation rõ rệt, khác hẳn nền bàn xám-xanh
      vô sắc. Kết quả: tâm (rx, ry) và bán kính r_rice của khối lúa.
  B2. DÒ VÀNH LY BẰNG TIA HƯỚNG TÂM: từ tâm khối lúa, phóng 360 tia ra ngoài
      trong dải bán kính [r_rice, r_max]. Trên mỗi tia, tìm các đỉnh của
      (độ lớn gradient x độ đồng thuận hướng tâm). Điểm cạnh của viền ly có
      gradient hướng về MỘT tâm chung -> tích luỹ phiếu mạnh. Điểm cạnh của hạt
      lúa/vân gỗ/bóng đổ hướng tứ phía -> không tích luỹ được phiếu.
  B3. BÌNH CHỌN BÁN KÍNH (1-D histogram có trọng số): các đỉnh gradient tụ lại
      thành những "vòng" rõ rệt. Chọn cặp (r_in, r_out) có tỉ lệ khớp nhất với
      expected_ratio hình học từ wall_thickness_mm.
  B4. FIT ELIP BỀN VỮNG (RANSAC) trên tập điểm cạnh thuộc từng vòng, rồi TINH
      CHỈNH lại một lần nữa bằng tia hướng tâm quanh elip vừa fit (xử lý được
      cả trường hợp miệng ly hơi méo do phối cảnh).
  B5. KIỂM TRA RÀNG BUỘC VẬT LÝ: mép trong phải bao quanh khối lúa, tỉ lệ
      inner/outer phải hợp lý, đồng tâm với khối lúa.

Không gian tìm kiếm co lại khoảng 50 lần so với v1, và lớp lỗi "bắt nhầm cụm
hạt lúa làm miệng ly" trở thành BẤT KHẢ THI VỀ MẶT CẤU TRÚC (một vòng nằm bên
trong khối lúa không thể bao quanh khối lúa đó).

GHI CHÚ VỀ SAI SỐ ĐO LƯỜNG (quan trọng, v1 bỏ qua hoàn toàn):
--------------------------------------------------------------
`pixels_per_mm` được đo tại MẶT PHẲNG MIỆNG LY, nhưng lại được dùng để đo hạt
lúa nằm thấp hơn `empty_height_mm`. Với chụp điện thoại gần (khoảng cách làm
việc ~200mm), chênh lệch 10mm độ sâu gây sai scale ~5%, kéo theo ~10% sai số
DIỆN TÍCH và ~15% sai số THỂ TÍCH — lớn hơn nhiều so với sai số nhận diện viền.
v2 luôn trả về:
    "depth_scale_factor"            = d / (d + empty_height_mm)
    "pixels_per_mm_at_rice_surface" = pixels_per_mm * depth_scale_factor
Truyền `camera_distance_mm=<khoảng cách ống kính tới miệng ly, mm>` để có hệ số
đúng; đặt `apply_depth_correction=True` nếu muốn `pixels_per_mm` trả về đã được
hiệu chỉnh sẵn (mặc định False để không đổi hành vi của code đang gọi).

TƯƠNG THÍCH NGƯỢC:
------------------
Giữ nguyên tên hàm `detect_container_and_scale` / `draw_container_overlay`,
giữ nguyên TOÀN BỘ key trong dict trả về của v1 (kể cả "detect_type",
"outer_confidence", "inner_confidence", "ellipse_params", "crop_offset"...),
và giữ nguyên mọi tham số của v1 (các tham số đã mất vai trò được chấp nhận
nhưng bị bỏ qua, có ghi rõ trong docstring) -> dán thẳng vào file cũ là chạy.
===============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

EPS = 1e-9

__all__ = [
    "detect_container_and_scale",
    "draw_container_overlay",
    "ContainerDetectionError",
]


class ContainerDetectionError(RuntimeError):
    """Lỗi nhận diện vật chứa (kế thừa RuntimeError để tương thích code cũ đang
    bắt RuntimeError)."""


# =============================================================================
# I. TIỆN ÍCH CƠ BẢN
# =============================================================================


def _read_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Đọc ảnh BGR, hỗ trợ đường dẫn có ký tự Unicode (imdecode từ np.fromfile)."""
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input.copy()

    if not isinstance(image_input, (str, Path)):
        raise ValueError("image_input phải là đường dẫn file hoặc numpy ndarray.")

    p_str = str(image_input)
    img = None
    try:
        data = np.fromfile(p_str, dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        img = cv2.imread(p_str, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc file ảnh: {image_input}")
    return img


def _bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Lấy mẫu nội suy song tuyến (bilinear) một ảnh 2-D tại các toạ độ thực.

    Dùng để đọc gradient DỌC THEO TIA HƯỚNG TÂM với độ chính xác dưới pixel —
    đây là điều kiện cần để định vị viền ly chính xác hơn mức pixel nguyên
    (v1 chỉ dùng toạ độ contour nguyên, và còn bị dilate làm phình).

    Điểm nằm ngoài biên ảnh trả về 0.0 (coi như không có cạnh).
    """
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    fx = xs - x0
    fy = ys - y0

    va = img[y0c, x0c] * (1.0 - fx) * (1.0 - fy)
    vb = img[y0c, x1c] * fx * (1.0 - fy)
    vc = img[y1c, x0c] * (1.0 - fx) * fy
    vd = img[y1c, x1c] * fx * fy

    out = (va + vb + vc + vd).astype(np.float64)
    out[~valid] = 0.0
    return out


def _sobel_grads(gray_blurred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (gx, gy, magnitude) dạng float32."""
    gx = cv2.Sobel(gray_blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return gx, gy, mag


def _scale_ellipse(
    ell: Tuple[Tuple[float, float], Tuple[float, float], float],
    factor: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Quy đổi elip ((cx, cy), (MA, ma), angle) sang hệ toạ độ khác theo hệ số tỉ lệ
    đồng nhất `factor` (đưa kết quả từ ảnh xử lý đã resize về đúng toạ độ ảnh
    gốc). Góc nghiêng không đổi vì scale đồng nhất 2 trục không làm xoay elip.
    """
    (ecx, ecy), (MA, ma), angle = ell
    return ((ecx * factor, ecy * factor), (MA * factor, ma * factor), angle)


def _ellipse_diam(ell) -> float:
    return float(max(ell[1]))


def _contour_angular_coverage(
    contour: np.ndarray,
    center: Tuple[float, float],
    num_bins: int = 36,
) -> float:
    """
    Tỉ lệ (0.0–1.0) số 'lát góc' quanh `center` có ít nhất 1 điểm đi qua.

    KHÁC v1: `center` ở đây là một TÂM THAM CHIẾU ĐỘC LẬP (tâm khối lúa hoặc
    tâm elip ứng viên đang xét), không phải tâm của chính contour đó. Đo quanh
    tâm của chính contour khiến mọi đường khép kín đều cho 1.0 — lý do bộ lọc
    này vô hiệu trong v1.
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return 0.0
    cx, cy = center
    angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    bins = (angles / (360.0 / num_bins)).astype(int) % num_bins
    return len(np.unique(bins)) / float(num_bins)


# =============================================================================
# II. FIT ELIP BỀN VỮNG (RANSAC)
# =============================================================================


def _ransac_fit_ellipse(
    points: np.ndarray,
    iterations: int = 120,
    sample_size: int = 8,
    inlier_dist_px: float = 3.0,
    min_inlier_ratio: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float], float]], Optional[float]]:
    """
    Fit elip bền vững với outlier bằng RANSAC, thay cho cv2.fitEllipse thuần
    (least-squares rất nhạy với vài điểm nhiễu cục bộ).

    Thuật toán:
      1. Lặp `iterations` lần: lấy mẫu ngẫu nhiên `sample_size` điểm, fit thử.
      2. Đếm inlier trên TOÀN BỘ tập điểm theo bán kính chuẩn hoá (r_norm == 1.0
         nghĩa là điểm nằm đúng trên biên elip).
      3. Giữ mô hình nhiều inlier nhất.
      4. Fit lại lần cuối CHỈ trên tập inlier để tinh chỉnh.

    Returns
    -------
    (ellipse, inlier_ratio)
        ellipse = None nếu không thể fit (dưới 5 điểm / suy biến).
        inlier_ratio = None nghĩa là "không rõ độ tin cậy" (quá ít điểm để
        RANSAC có ý nghĩa thống kê), KHÔNG đồng nghĩa với "đáng tin cậy".
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n < 5:
        return None, None

    cv_all = pts.astype(np.float32).reshape(-1, 1, 2)
    sample_size = max(5, min(sample_size, n))

    if n < 20:
        try:
            return cv2.fitEllipse(cv_all), None
        except cv2.error:
            return None, None

    if rng is None:
        rng = np.random.default_rng(12345)  # cố định để kết quả tái lập khi debug

    best_ell = None
    best_count = -1
    best_mask = None

    for _ in range(iterations):
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = pts[idx].astype(np.float32).reshape(-1, 1, 2)
        try:
            ell = cv2.fitEllipse(sample)
        except cv2.error:
            continue

        (ecx, ecy), (MA, ma), angle = ell
        if MA <= 1e-3 or ma <= 1e-3 or not np.isfinite(MA) or not np.isfinite(ma):
            continue

        theta = np.deg2rad(angle)
        ca, sa = np.cos(theta), np.sin(theta)
        dx = pts[:, 0] - ecx
        dy = pts[:, 1] - ecy
        xr = dx * ca + dy * sa
        yr = -dx * sa + dy * ca
        a, b = MA / 2.0, ma / 2.0
        r_norm = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        tol = inlier_dist_px / max(1e-3, (a + b) / 2.0)
        mask = np.abs(r_norm - 1.0) < tol
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_ell = ell
            best_mask = mask

    if best_ell is None:
        try:
            return cv2.fitEllipse(cv_all), 0.0
        except cv2.error:
            return None, 0.0

    ratio = best_count / float(n)
    if ratio < min_inlier_ratio:
        # Không đủ đồng thuận -> fit trực tiếp nhưng BÁO ĐÚNG inlier_ratio thấp
        # để nơi gọi biết mà nghi ngờ kết quả.
        try:
            return cv2.fitEllipse(cv_all), ratio
        except cv2.error:
            return best_ell, ratio

    inlier_pts = pts[best_mask].astype(np.float32).reshape(-1, 1, 2)
    if inlier_pts.shape[0] >= 5:
        try:
            return cv2.fitEllipse(inlier_pts), ratio
        except cv2.error:
            return best_ell, ratio
    return best_ell, ratio


# =============================================================================
# III. TÁCH KHỐI LÚA BẰNG MÀU (BƯỚC 1 — TÍN HIỆU ỔN ĐỊNH NHẤT)
# =============================================================================


def _segment_rice(
    proc_bgr: np.ndarray,
    hue_range: Tuple[int, int] = (8, 42),
    sat_min: int = 35,
    val_min: int = 45,
    min_area_ratio: float = 0.0004,
) -> Optional[Dict[str, Any]]:
    """
    Tách khối lúa dựa trên MÀU trong không gian HSV.

    Vì sao màu, không phải cạnh: hạt lúa/gạo có hue vàng–nâu (~10–40 trên thang
    OpenCV 0–179) với saturation rõ rệt, trong khi nền bàn/ly kính/bóng đổ gần
    như VÔ SẮC (saturation thấp). Đây là khác biệt mang tính bản chất, bền vững
    với ánh sáng và độ phân giải — trái ngược với tương phản viền ly kính (yếu,
    phụ thuộc phản chiếu, chính là điểm chết của v1).

    Sau khi tạo mask: OPEN để bỏ đốm nhiễu, CLOSE để lấp kẽ giữa các hạt, rồi
    lấy thành phần liên thông lớn nhất và bọc bằng `cv2.minEnclosingCircle`.

    Returns
    -------
    dict hoặc None
        {"mask", "center": (rx, ry), "radius": r_rice, "area", "area_ratio",
         "contour"} — None nếu không tìm thấy vùng màu lúa đủ lớn.
    """
    h, w = proc_bgr.shape[:2]
    hsv = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2HSV)

    lo = np.array([hue_range[0], sat_min, val_min], dtype=np.uint8)
    hi = np.array([hue_range[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big, iterations=2)

    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n_lbl <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    area = float(stats[best, cv2.CC_STAT_AREA])
    if area < min_area_ratio * h * w:
        return None

    comp = (labels == best).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    (rx, ry), r_rice = cv2.minEnclosingCircle(cnt)

    return {
        "mask": comp,
        "center": (float(rx), float(ry)),
        "radius": float(r_rice),
        "area": area,
        "area_ratio": area / float(h * w),
        "contour": cnt,
    }


# =============================================================================
# IV. DÒ VÀNH LY BẰNG GRADIENT HƯỚNG TÂM (BƯỚC 2–3)
# =============================================================================


def _ray_edge_peaks(
    gx: np.ndarray,
    gy: np.ndarray,
    mag: np.ndarray,
    center: Tuple[float, float],
    r_min: float,
    r_max: float,
    n_rays: int = 360,
    r_step: float = 0.5,
    align_min: float = 0.45,
    top_k: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Phóng `n_rays` tia từ `center` ra ngoài, tìm các đỉnh cạnh trên từng tia.

    Với mỗi điểm mẫu trên tia, tính:
        align = |g · u_r| / |g|      (u_r = vector đơn vị hướng tâm)
        score = |g| * align
    `align` là bộ lọc then chốt: cạnh của một vòng tròn/elip quanh `center` có
    gradient GẦN NHƯ SONG SONG với tia (align ~ 1), còn cạnh của hạt lúa, vân
    bàn hay bóng đổ có hướng tuỳ ý (align nhỏ) -> bị dập tắt. Đây là thứ v1
    hoàn toàn không có: v1 chỉ xét ĐỘ LỚN cạnh, nên biên hạt lúa (rất mạnh) luôn
    thắng viền ly (yếu).

    Returns
    -------
    dict các mảng cùng độ dài: "angle", "radius", "score", "x", "y", "ray_idx".
    """
    cx, cy = center
    r_min = max(1.0, float(r_min))
    r_max = float(r_max)
    if r_max <= r_min + 2 * r_step:
        return {k: np.empty(0) for k in ("angle", "radius", "score", "x", "y", "ray_idx")}

    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    radii = np.arange(r_min, r_max, r_step)

    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    xs = cx + radii[None, :] * cos_a
    ys = cy + radii[None, :] * sin_a

    gxs = _bilinear(gx, xs, ys)
    gys = _bilinear(gy, xs, ys)
    mags = _bilinear(mag, xs, ys)

    align = np.abs(gxs * cos_a + gys * sin_a) / (mags + EPS)
    score = mags * np.where(align >= align_min, align, 0.0)

    # Làm mượt nhẹ theo bán kính để đỉnh không bị vụn do nhiễu cảm biến
    if score.shape[1] >= 5:
        kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kern /= kern.sum()
        score = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 1, score)

    if not np.any(score > 0):
        return {k: np.empty(0) for k in ("angle", "radius", "score", "x", "y", "ray_idx")}

    global_thr = 0.12 * float(np.percentile(score[score > 0], 99))

    mid = score[:, 1:-1]
    is_peak = (mid > score[:, :-2]) & (mid >= score[:, 2:]) & (mid > global_thr)

    ray_max = score.max(axis=1, keepdims=True)
    is_peak &= mid > 0.25 * ray_max

    out_ang, out_rad, out_sc = [], [], []
    for i in range(n_rays):
        idx = np.flatnonzero(is_peak[i])
        if idx.size == 0:
            continue
        vals = mid[i, idx]
        order = np.argsort(vals)[::-1][:top_k]
        for j in order:
            out_ang.append(angles[i])
            out_rad.append(radii[idx[j] + 1])
            out_sc.append(vals[j])

    if not out_rad:
        return {k: np.empty(0) for k in ("angle", "radius", "score", "x", "y", "ray_idx")}

    ang = np.asarray(out_ang)
    rad = np.asarray(out_rad)
    sc = np.asarray(out_sc)
    return {
        "angle": ang,
        "radius": rad,
        "score": sc,
        "x": cx + rad * np.cos(ang),
        "y": cy + rad * np.sin(ang),
        "ray_idx": np.round(ang / (2.0 * np.pi) * n_rays).astype(int) % n_rays,
    }


def _radius_votes(
    peaks: Dict[str, np.ndarray],
    r_min: float,
    r_max: float,
    bin_px: float = 2.0,
    smooth_bins: int = 2,
) -> List[Dict[str, float]]:
    """
    Bình chọn bán kính: dựng histogram 1-D của bán kính các đỉnh cạnh, có trọng
    số bằng score, làm mượt rồi lấy các cực đại cục bộ.

    Một vòng thật (viền ly) sinh ra đỉnh cạnh ở CÙNG một bán kính trên hàng trăm
    tia khác nhau -> tạo một mũi nhọn rõ rệt trong histogram. Cạnh ngẫu nhiên
    (hạt lúa, vân bàn) phân tán khắp dải bán kính -> không tạo được mũi nhọn.

    Returns
    -------
    List[dict] sắp giảm dần theo "weight": {"radius", "weight", "n_rays"}.
    """
    if peaks["radius"].size == 0:
        return []

    n_bins = max(4, int(math.ceil((r_max - r_min) / bin_px)))
    hist = np.zeros(n_bins, dtype=np.float64)
    ray_sets: List[set] = [set() for _ in range(n_bins)]

    b = np.clip(((peaks["radius"] - r_min) / bin_px).astype(int), 0, n_bins - 1)
    for bi, s, ri in zip(b, peaks["score"], peaks["ray_idx"]):
        hist[bi] += float(s)
        ray_sets[bi].add(int(ri))

    if smooth_bins > 0:
        k = np.ones(2 * smooth_bins + 1)
        k /= k.sum()
        hist_s = np.convolve(hist, k, mode="same")
    else:
        hist_s = hist

    out: List[Dict[str, float]] = []
    for i in range(1, n_bins - 1):
        if hist_s[i] > hist_s[i - 1] and hist_s[i] >= hist_s[i + 1] and hist_s[i] > 0:
            lo = max(0, i - smooth_bins)
            hi = min(n_bins, i + smooth_bins + 1)
            rays = set()
            for j in range(lo, hi):
                rays |= ray_sets[j]
            out.append({
                "radius": float(r_min + (i + 0.5) * bin_px),
                "weight": float(hist_s[i]),
                "n_rays": float(len(rays)),
            })

    out.sort(key=lambda d: d["weight"], reverse=True)
    return out


def _points_near_radius(
    peaks: Dict[str, np.ndarray],
    r_target: float,
    tol_px: float,
) -> np.ndarray:
    """Lấy các điểm cạnh có bán kính nằm trong ±tol_px quanh `r_target`."""
    if peaks["radius"].size == 0:
        return np.empty((0, 2))
    m = np.abs(peaks["radius"] - r_target) <= tol_px
    if not np.any(m):
        return np.empty((0, 2))
    return np.stack([peaks["x"][m], peaks["y"][m]], axis=1)


def _refine_ellipse_radially(
    ell: Tuple[Tuple[float, float], Tuple[float, float], float],
    gx: np.ndarray,
    gy: np.ndarray,
    mag: np.ndarray,
    band_px: float = 6.0,
    n_samples: int = 360,
    step: float = 0.25,
    align_min: float = 0.35,
) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float], float]], float]:
    """
    Tinh chỉnh elip: quanh mỗi điểm trên biên elip hiện tại, quét một dải hẹp
    ±`band_px` theo PHÁP TUYẾN elip để tìm đỉnh gradient hướng tâm, rồi fit lại
    bằng RANSAC trên tập điểm mới.

    Bước này xử lý hai việc:
      - Đưa viền về đúng đỉnh gradient với độ chính xác dưới pixel (loại bỏ sai
        lệch hệ thống kiểu `cv2.dilate` của v1 làm phình đường kính).
      - Cho phép elip thích ứng với méo phối cảnh nhẹ, thứ mà mô hình "vòng tròn
        bán kính r" ở bước bình chọn không diễn tả được.

    Returns
    -------
    (ellipse, coverage) — coverage là tỉ lệ mẫu tìm được điểm cạnh hợp lệ.
    """
    (ecx, ecy), (MA, ma), angle = ell
    a, b = MA / 2.0, ma / 2.0
    if a <= 1.0 or b <= 1.0:
        return None, 0.0

    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    th = np.deg2rad(angle)
    ca, sa = np.cos(th), np.sin(th)

    # Điểm trên biên (hệ elip -> hệ ảnh)
    px_l = a * np.cos(t)
    py_l = b * np.sin(t)
    # Pháp tuyến elip trong hệ cục bộ: (cos t / a, sin t / b), chuẩn hoá
    nx_l = np.cos(t) / a
    ny_l = np.sin(t) / b
    nn = np.sqrt(nx_l ** 2 + ny_l ** 2) + EPS
    nx_l /= nn
    ny_l /= nn

    px = ecx + px_l * ca - py_l * sa
    py = ecy + px_l * sa + py_l * ca
    nx = nx_l * ca - ny_l * sa
    ny = nx_l * sa + ny_l * ca

    offs = np.arange(-band_px, band_px + step, step)
    xs = px[:, None] + nx[:, None] * offs[None, :]
    ys = py[:, None] + ny[:, None] * offs[None, :]

    gxs = _bilinear(gx, xs, ys)
    gys = _bilinear(gy, xs, ys)
    mags = _bilinear(mag, xs, ys)

    align = np.abs(gxs * nx[:, None] + gys * ny[:, None]) / (mags + EPS)
    score = mags * np.where(align >= align_min, align, 0.0)

    best = np.argmax(score, axis=1)
    best_val = score[np.arange(n_samples), best]
    ok = best_val > 0
    if ok.sum() < 5:
        return None, float(ok.sum()) / n_samples

    pts = np.stack([
        xs[np.arange(n_samples), best][ok],
        ys[np.arange(n_samples), best][ok],
    ], axis=1)

    new_ell, _ = _ransac_fit_ellipse(pts, iterations=120, inlier_dist_px=2.0,
                                     min_inlier_ratio=0.5)
    coverage = _contour_angular_coverage(pts, (ecx, ecy), num_bins=36)
    if new_ell is None:
        return None, coverage
    return new_ell, coverage


def _radial_consensus(
    ell: Tuple[Tuple[float, float], Tuple[float, float], float],
    gx: np.ndarray,
    gy: np.ndarray,
    mag: np.ndarray,
    n_samples: int = 180,
) -> float:
    """
    Độ tin cậy (0.0–1.0) của một elip: tỉ lệ điểm trên biên elip có cạnh MẠNH và
    hướng ĐÚNG pháp tuyến. Đây là thước đo "elip này có thật là một vành sáng/tối
    trong ảnh hay không", dùng thay cho `area * aspect^2` của v1 (chỉ đo hình
    dạng, không đo bằng chứng ảnh).
    """
    (ecx, ecy), (MA, ma), angle = ell
    a, b = MA / 2.0, ma / 2.0
    if a <= 1.0 or b <= 1.0:
        return 0.0

    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    th = np.deg2rad(angle)
    ca, sa = np.cos(th), np.sin(th)

    px_l, py_l = a * np.cos(t), b * np.sin(t)
    nx_l, ny_l = np.cos(t) / a, np.sin(t) / b
    nn = np.sqrt(nx_l ** 2 + ny_l ** 2) + EPS
    nx_l /= nn
    ny_l /= nn

    px = ecx + px_l * ca - py_l * sa
    py = ecy + px_l * sa + py_l * ca
    nx = nx_l * ca - ny_l * sa
    ny = nx_l * sa + ny_l * ca

    gxs = _bilinear(gx, px, py)
    gys = _bilinear(gy, px, py)
    mags = _bilinear(mag, px, py)

    pos = mag[mag > 0]
    thr = 0.10 * float(np.percentile(pos, 99)) if pos.size else 0.0
    align = np.abs(gxs * nx + gys * ny) / (mags + EPS)
    good = (mags > thr) & (align > 0.5)
    return float(good.mean())


def _hough_fallback_center(
    gray_blurred: np.ndarray,
    img_cx: float,
    img_cy: float,
    min_diam_px: int,
    max_diam_px: int,
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    Fallback tìm TÂM khi không tách được khối lúa bằng màu (ví dụ ly rỗng, hoặc
    lúa bị chiếu sáng mất màu). HoughCircles bền vững với viền đứt đoạn vì dùng
    voting theo gradient, không cần đường khép kín.

    Chỉ dùng để lấy TÂM và bán kính THÔ làm mồi cho bước dò tia hướng tâm —
    không dùng trực tiếp làm kết quả đo (Hough lượng tử hoá bán kính theo pixel
    nguyên, không đủ chính xác để tính pixels_per_mm).
    """
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=max(min_diam_px, 40),
        param1=100, param2=35,
        minRadius=int(min_diam_px / 2), maxRadius=int(max_diam_px / 2),
    )
    if circles is None:
        return None
    cc = np.round(circles[0, :]).astype(int)
    best = min(cc, key=lambda c: (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2)
    return (float(best[0]), float(best[1])), float(best[2])


# =============================================================================
# V. HÀM CHÍNH
# =============================================================================


def detect_container_and_scale(
    image_input: Union[str, Path, np.ndarray],
    inner_diam_mm: float,
    container_height_mm: float,
    empty_height_mm: float,
    wall_thickness_mm: float = 1.5,
    detect_mode: str = "inner",
    # ---- tham số v2 ----
    working_max_dim: int = 1600,
    n_rays: int = 360,
    rice_hue_range: Tuple[int, int] = (8, 42),
    rice_sat_min: int = 35,
    rice_val_min: int = 45,
    search_r_max_factor: float = 3.2,
    ratio_tol: float = 0.08,
    min_rim_rays_ratio: float = 0.30,
    min_containment_factor: float = 1.90,
    max_concentric_offset: float = 0.55,
    min_inner_outer_ratio_floor: float = 0.5,
    camera_distance_mm: Optional[float] = None,
    apply_depth_correction: bool = False,
    on_failure: str = "raise",
    verbose: bool = False,
    # ---- tham số v1 giữ lại cho tương thích (xem ghi chú bên dưới) ----
    blur_ksize: Tuple[int, int] = (9, 9),
    canny_thresh1: int = 30,
    canny_thresh2: int = 120,
    min_angular_coverage: float = 0.80,
    use_ransac: bool = True,
    ransac_iterations: int = 120,
    min_outer_diam_ratio: float = 0.25,
    min_inner_diam_ratio: float = 0.15,
    **_ignored_legacy_kwargs: Any,
) -> Dict[str, Any]:
    """
    Nhận diện miệng ly (mép trong + mép ngoài đồng tâm) và tính tỷ lệ quy đổi.

    Parameters
    ----------
    image_input : str, Path hoặc np.ndarray
        Đường dẫn file ảnh hoặc ma trận ảnh BGR.
    inner_diam_mm : float
        Đường kính TRONG thực tế của ly (mm).
    container_height_mm, empty_height_mm : float
        Chiều cao thân ly và khoảng trống từ miệng ly tới mặt trên khối lúa (mm).
    wall_thickness_mm : float, default 1.5
        Độ dày thành miệng ly (mm) — dùng để suy ra expected_ratio hình học.
    detect_mode : {'inner', 'outer', 'both'}, default 'inner'
        Mép dùng để tính pixels_per_mm. 'inner' chính xác nhất.

    working_max_dim : int, default 1600
        Toàn bộ bước NHẬN DIỆN chạy trên bản resize sao cho cạnh dài <= giá trị
        này, để mọi kernel/ngưỡng pixel hoạt động ổn định bất kể ảnh gốc 1000px
        hay 8000px. Kết quả cuối (crop/overlay/pixels_per_mm) luôn ở độ phân giải
        ẢNH GỐC, nên tham số này không ảnh hưởng độ chính xác vật lý.
    n_rays : int, default 360
        Số tia hướng tâm dùng để dò viền (1 tia/độ).
    rice_hue_range, rice_sat_min, rice_val_min
        Dải HSV tách khối lúa. Nới `rice_hue_range` nếu lúa sẫm/bạc màu; hạ
        `rice_sat_min` nếu ảnh chụp dưới đèn trắng làm nhạt màu.
    search_r_max_factor : float, default 3.2
        Bán kính tìm viền tối đa = giá trị này × bán kính khối lúa. Tăng nếu ly
        rộng mà lúa chỉ đổ một ít dưới đáy.
    ratio_tol : float, default 0.08
        Dung sai khi khớp tỉ lệ (r_in / r_out) với expected_ratio hình học.
        LƯU Ý: dung sai này LỎNG hơn v1 (0.035) một cách có chủ đích, vì ở v2 nó
        chỉ dùng để chọn cặp trong số các bán kính ĐÃ được gradient hướng tâm
        bình chọn, chứ không còn là điều kiện chấp nhận độc lập (nguồn gốc lỗi
        "trùng hợp" của v1).
    min_rim_rays_ratio : float, default 0.30
        Tỉ lệ tia tối thiểu phải nhìn thấy một vòng để coi vòng đó là viền thật.
        Hạ xuống nếu viền ly bị phản chiếu che mất hơn 2/3 chu vi.
    min_containment_factor : float, default 1.90
        RÀNG BUỘC BAO HÀM (bộ lọc quan trọng nhất của v2): đường kính mép trong
        phải >= giá trị này × bán kính khối lúa, tức mép trong phải BAO QUANH
        khối lúa. Một mình ràng buộc này loại bỏ toàn bộ lớp lỗi của v1.
    max_concentric_offset : float, default 0.55
        Khoảng lệch tâm tối đa giữa tâm miệng ly và tâm khối lúa, tính theo tỉ lệ
        bán kính khối lúa.
    min_inner_outer_ratio_floor : float, default 0.5
        Ngưỡng thô chặn lỗi tỉ lệ inner/outer vô lý về mặt vật lý.
    camera_distance_mm : float, optional
        Khoảng cách ống kính → miệng ly (mm). Nếu truyền vào, hàm tính
        "depth_scale_factor" = d/(d+empty_height_mm) và
        "pixels_per_mm_at_rice_surface". Xem ghi chú sai số đo lường ở đầu file:
        bỏ qua hiệu ứng này có thể gây ~15% sai số THỂ TÍCH.
    apply_depth_correction : bool, default False
        Nếu True, "pixels_per_mm" trả về đã được hiệu chỉnh về mặt phẳng bề mặt
        lúa. Mặc định False để không đổi hành vi code đang gọi.
    on_failure : {'raise', 'dict'}, default 'raise'
        'raise': raise ContainerDetectionError (kế thừa RuntimeError, tương thích
        code cũ). 'dict': trả về {"status": "failed", "reason": ...} để chạy
        batch không bị chết cả lô.
    verbose : bool, default False
        In log chẩn đoán.

    blur_ksize, canny_thresh1, canny_thresh2, use_ransac, ransac_iterations
        Vẫn có hiệu lực (blur cho gradient; Canny/Hough dùng ở nhánh fallback
        khi không tách được khối lúa; RANSAC dùng ở mọi bước fit elip).
    min_angular_coverage
        Nay được hiểu là độ phủ góc tối thiểu của tập điểm viền sau tinh chỉnh.
    min_outer_diam_ratio, min_inner_diam_ratio
        BỊ HẠ CẤP THÀNH CẢNH BÁO khi ràng buộc bao hàm (theo khối lúa) đã đạt —
        đúng như phân tích: ngưỡng theo khung ảnh không phân biệt được "chụp xa"
        với "bắt nhầm", nên không nên dùng làm điều kiện chặn. Chúng chỉ còn là
        điều kiện chặn ở nhánh fallback không có khối lúa để tham chiếu.

    Returns
    -------
    dict — giữ nguyên mọi key của v1, thêm: "status", "warnings", "rice_circle",
    "outer_coverage", "inner_coverage", "depth_scale_factor",
    "pixels_per_mm_at_rice_surface", "pixels_per_mm_at_rim".
    """
    warnings: List[str] = []

    def _fail(reason: str) -> Dict[str, Any]:
        if on_failure == "dict":
            if verbose:
                print(f"[FAILED] {reason}")
            return {"status": "failed", "reason": reason, "warnings": warnings}
        raise ContainerDetectionError(reason)

    # ---------------------------------------------------------------- 0. Đọc ảnh
    img_bgr = _read_image(image_input)
    h, w = img_bgr.shape[:2]

    if max(h, w) > working_max_dim:
        scale_factor = working_max_dim / float(max(h, w))
        proc_bgr = cv2.resize(
            img_bgr,
            (max(1, int(round(w * scale_factor))), max(1, int(round(h * scale_factor)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale_factor = 1.0
        proc_bgr = img_bgr

    proc_h, proc_w = proc_bgr.shape[:2]
    min_side = float(min(proc_h, proc_w))
    img_cx, img_cy = proc_w / 2.0, proc_h / 2.0

    k = (max(3, blur_ksize[0] | 1), max(3, blur_ksize[1] | 1))
    gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, k, 0)
    gx, gy, mag = _sobel_grads(blurred)

    expected_ratio = float(inner_diam_mm) / (
        float(inner_diam_mm) + 2.0 * float(wall_thickness_mm)
    )

    # ------------------------------------------------- 1. Tách khối lúa bằng màu
    rice = _segment_rice(
        proc_bgr,
        hue_range=rice_hue_range,
        sat_min=rice_sat_min,
        val_min=rice_val_min,
    )
    rice_based = rice is not None

    if rice_based:
        seed_c = rice["center"]
        r_rice = rice["radius"]
        r_search_min = max(2.0, r_rice * 0.85)
        r_search_max = min(
            r_rice * search_r_max_factor,
            math.hypot(proc_w, proc_h) * 0.5,
        )
        if verbose:
            print(f"[rice] center={seed_c} r={r_rice:.1f}px "
                  f"area_ratio={rice['area_ratio']:.4f}")
    else:
        warnings.append(
            "Không tách được khối lúa bằng màu HSV -> mất ràng buộc bao hàm, "
            "độ tin cậy giảm. Nới rice_hue_range/rice_sat_min nếu lúa bạc màu."
        )
        hf = _hough_fallback_center(
            blurred, img_cx, img_cy,
            min_diam_px=int(min_side * 0.10),
            max_diam_px=int(min_side * 0.95),
        )
        if hf is None:
            return _fail(
                "Không tách được khối lúa bằng màu VÀ HoughCircles cũng không tìm "
                "được vòng nào -> không có tâm mồi để dò viền. Kiểm tra lại ảnh "
                "(có đúng là ảnh chụp ly đựng lúa từ trên xuống?), hoặc hạ "
                "rice_sat_min / canny_thresh1."
            )
        seed_c, r_seed = hf
        r_rice = r_seed * 0.75  # ước lượng thô, chỉ để định dải tìm kiếm
        r_search_min = max(2.0, r_seed * 0.45)
        r_search_max = min(r_seed * 1.8, math.hypot(proc_w, proc_h) * 0.5)
        if verbose:
            print(f"[rice] THẤT BẠI -> Hough seed center={seed_c} r={r_seed:.1f}px")

    # ------------------------------------ 2. Dò đỉnh cạnh trên tia hướng tâm
    peaks = _ray_edge_peaks(
        gx, gy, mag, seed_c,
        r_min=r_search_min, r_max=r_search_max,
        n_rays=n_rays,
    )
    if peaks["radius"].size == 0:
        return _fail(
            "Không tìm thấy đỉnh cạnh hướng tâm nào quanh khối lúa trong dải bán "
            f"kính [{r_search_min:.0f}, {r_search_max:.0f}]px. Viền ly có thể quá "
            "mờ/quá trong suốt. Thử tăng search_r_max_factor, hoặc chụp lại với "
            "nền tối tương phản với thành ly."
        )

    # ------------------------------------------------ 3. Bình chọn bán kính vòng
    votes = _radius_votes(peaks, r_search_min, r_search_max)
    min_rays = max(6.0, min_rim_rays_ratio * n_rays)
    votes = [v for v in votes if v["n_rays"] >= min_rays]

    if rice_based:
        # RÀNG BUỘC BAO HÀM: vòng hợp lệ phải bao quanh khối lúa
        votes = [v for v in votes if 2.0 * v["radius"] >= min_containment_factor * r_rice]

    if not votes:
        return _fail(
            "Không có vòng nào vừa được đủ tia bình chọn "
            f"(>= {min_rays:.0f}/{n_rays} tia) vừa bao quanh được khối lúa "
            f"(đường kính >= {min_containment_factor:.2f} x {r_rice:.1f}px). "
            "Đây thường là dấu hiệu viền ly bị phản chiếu che mất phần lớn chu vi: "
            "thử hạ min_rim_rays_ratio, hoặc chụp lại với ánh sáng tán xạ."
        )

    # ------------------------------ 4. Chọn cặp (mép trong, mép ngoài) đồng tâm
    pair: Optional[Tuple[Dict[str, float], Dict[str, float], float]] = None
    best_pair_score = -1.0
    for i, vi in enumerate(votes):
        for j, vj in enumerate(votes):
            if i == j:
                continue
            r_in, r_out = vi["radius"], vj["radius"]
            if r_out <= r_in:
                continue
            ratio = r_in / r_out
            if abs(ratio - expected_ratio) > ratio_tol:
                continue
            s = (
                math.sqrt(vi["weight"] * vj["weight"])
                * math.exp(-((ratio - expected_ratio) / max(0.01, ratio_tol / 2.0)) ** 2)
            )
            if s > best_pair_score:
                best_pair_score = s
                pair = (vi, vj, ratio)

    if pair is not None:
        v_in, v_out, found_ratio = pair
        r_in_vote, r_out_vote = v_in["radius"], v_out["radius"]
        detect_type = "radial_vote_pair"
        if verbose:
            print(f"[rim] cặp vòng: r_in={r_in_vote:.1f} r_out={r_out_vote:.1f} "
                  f"ratio={found_ratio:.3f} (kỳ vọng {expected_ratio:.3f})")
    else:
        # Chỉ thấy MỘT vòng (thường vì thành ly mỏng hơn ~2px ở độ phân giải xử lý
        # hoặc do gradient viền ngoài mạnh hơn). Kế hoạch vá đề xuất:
        # coi vòng mạnh nhất là mép NGOÀI và suy ra mép TRONG thuần hình học.
        v_out = votes[0]
        r_out_vote = v_out["radius"]
        r_in_vote = r_out_vote * max(EPS, expected_ratio)
        detect_type = "radial_vote_single_geometric_inner"
        warnings.append(
            "Chỉ phát hiện được MỘT vành quanh khối lúa (không tách được mép trong "
            "và mép ngoài). Giả định đây là viền ngoài, mép trong được suy ra "
            "thuần hình học từ wall_thickness_mm, nên độ chính xác của nó phụ "
            "thuộc hoàn toàn vào thông số này."
        )
        if verbose:
            print(f"[rim] chỉ 1 vòng: r_out={r_out_vote:.1f} -> inner suy hình học "
                  f"r_in={r_in_vote:.1f}")

    # ------------------------------------------- 5. Fit elip + tinh chỉnh hướng tâm
    def _build_ellipse(r_vote: float, tol: float):
        pts = _points_near_radius(peaks, r_vote, tol)
        ell, conf = (None, None)
        if pts.shape[0] >= 5:
            ell, conf = _ransac_fit_ellipse(
                pts, iterations=ransac_iterations, inlier_dist_px=max(2.0, tol * 0.6)
            )
        if ell is None:
            ell = ((seed_c[0], seed_c[1]), (2.0 * r_vote, 2.0 * r_vote), 0.0)
            conf = None
        cov = _contour_angular_coverage(pts, seed_c) if pts.shape[0] else 0.0

        if use_ransac:
            refined, ref_cov = _refine_ellipse_radially(
                ell, gx, gy, mag, band_px=max(4.0, tol)
            )
            if refined is not None and ref_cov >= cov * 0.8:
                # Chấp nhận tinh chỉnh chỉ khi không làm mất độ phủ và không làm
                # đường kính thay đổi quá 25% (chống trường hợp bị kéo sang một
                # vành lân cận khác).
                if 0.75 <= _ellipse_diam(refined) / max(EPS, _ellipse_diam(ell)) <= 1.25:
                    ell, cov = refined, ref_cov
        return ell, conf, cov

    tol_in = max(3.0, 0.035 * r_in_vote)
    inner_ell, inner_confidence, inner_coverage = _build_ellipse(r_in_vote, tol_in)

    if detect_type == "radial_vote_pair":
        tol_out = max(3.0, 0.035 * r_out_vote)
        outer_ell, outer_confidence, outer_coverage = _build_ellipse(r_out_vote, tol_out)
    else:
        (icx, icy), (iMA, ima), iang = inner_ell
        inv = 1.0 / max(EPS, expected_ratio)
        outer_ell = ((icx, icy), (iMA * inv, ima * inv), iang)
        outer_confidence = None
        outer_coverage = inner_coverage

    # Nếu tinh chỉnh làm hai elip đổi chỗ (trong lớn hơn ngoài) -> hoán vị lại
    if _ellipse_diam(inner_ell) > _ellipse_diam(outer_ell):
        inner_ell, outer_ell = outer_ell, inner_ell
        inner_confidence, outer_confidence = outer_confidence, inner_confidence
        inner_coverage, outer_coverage = outer_coverage, inner_coverage
        warnings.append("Hai vành bị hoán vị sau tinh chỉnh (đã tự sửa lại).")

    inner_diam_px = _ellipse_diam(inner_ell)
    outer_diam_px = _ellipse_diam(outer_ell)

    # ------------------------------------------ 6. Kiểm tra ràng buộc vật lý
    if inner_coverage < min_angular_coverage:
        warnings.append(
            f"Mép trong chỉ phủ {inner_coverage * 100:.0f}% chu vi "
            f"(ngưỡng khuyến nghị {min_angular_coverage * 100:.0f}%) — viền có thể "
            "bị phản chiếu che khuất; đường kính vẫn đáng tin nhờ RANSAC nhưng độ "
            "chính xác giảm."
        )

    if rice_based:
        if inner_diam_px < min_containment_factor * r_rice:
            warnings.append(
                f"Mép trong ({inner_diam_px:.1f}px đường kính) nhỏ hơn mức an toàn so với "
                f"khối lúa (bán kính {r_rice:.1f}px). Đã bỏ qua lỗi theo yêu cầu để tiếp tục pipeline."
            )
        off = math.hypot(inner_ell[0][0] - seed_c[0], inner_ell[0][1] - seed_c[1])
        if off > max_concentric_offset * r_rice:
            warnings.append(
                f"Tâm miệng ly lệch {off:.1f}px so với tâm khối lúa "
                f"({off / max(EPS, r_rice) * 100:.0f}% bán kính lúa) — có thể do lúa "
                "đổ dồn một bên, hoặc do góc chụp nghiêng."
            )
    else:
        # Không có khối lúa để tham chiếu -> mới phải dựa vào ngưỡng theo khung ảnh
        if outer_diam_px < min_outer_diam_ratio * min_side:
            return _fail(
                f"Mép ngoài ({outer_diam_px:.1f}px) nhỏ bất thường so với khung xử lý "
                f"({proc_w}x{proc_h}px), và KHÔNG có khối lúa để kiểm tra ràng buộc "
                "bao hàm. Không đủ bằng chứng để tin kết quả."
            )
        if inner_diam_px < min_inner_diam_ratio * min_side:
            return _fail(
                f"Mép trong ({inner_diam_px:.1f}px) nhỏ bất thường so với khung xử lý "
                f"({proc_w}x{proc_h}px), và KHÔNG có khối lúa để kiểm tra ràng buộc "
                "bao hàm."
            )

    if inner_diam_px < min_inner_outer_ratio_floor * outer_diam_px:
        return _fail(
            f"Tỉ lệ mép trong/mép ngoài ({inner_diam_px / max(EPS, outer_diam_px):.2f}) "
            f"nhỏ bất thường về mặt vật lý (ngưỡng {min_inner_outer_ratio_floor:.2f}). "
            f"detect_type='{detect_type}' — kiểm tra lại wall_thickness_mm."
        )

    # -------------------------- 7. Quy đổi elip về độ phân giải ẢNH GỐC
    if scale_factor != 1.0:
        inv_s = 1.0 / scale_factor
        inner_ell = _scale_ellipse(inner_ell, inv_s)
        outer_ell = _scale_ellipse(outer_ell, inv_s)
        inner_diam_px = _ellipse_diam(inner_ell)
        outer_diam_px = _ellipse_diam(outer_ell)
        rice_circle = (
            (seed_c[0] * inv_s, seed_c[1] * inv_s, r_rice * inv_s) if rice_based else None
        )
    else:
        rice_circle = (seed_c[0], seed_c[1], r_rice) if rice_based else None

    (cx, cy), (MA_out, ma_out), angle = outer_ell

    # ------------------------------------------------ 8. Tính pixels_per_mm
    if detect_mode == "outer":
        selected_diam_px = outer_diam_px
        selected_ell = outer_ell
        pixels_per_mm_at_rim = outer_diam_px / (
            float(inner_diam_mm) + 2.0 * float(wall_thickness_mm)
        )
    else:  # 'inner' hoặc 'both'
        selected_diam_px = inner_diam_px
        selected_ell = inner_ell
        pixels_per_mm_at_rim = inner_diam_px / float(inner_diam_mm)

    if camera_distance_mm is not None and camera_distance_mm > 0:
        depth_scale_factor = float(camera_distance_mm) / (
            float(camera_distance_mm) + float(empty_height_mm)
        )
    else:
        depth_scale_factor = 1.0
        if float(empty_height_mm) > 0.5:
            warnings.append(
                f"empty_height_mm={empty_height_mm}mm nhưng chưa truyền "
                "camera_distance_mm -> KHÔNG hiệu chỉnh được sai số phối cảnh theo độ "
                "sâu. Bề mặt lúa nằm thấp hơn miệng ly nên scale thực tại đó NHỎ hơn, "
                "gây sai số ~2x ở diện tích và ~3x ở thể tích so với sai số scale."
            )

    pixels_per_mm_at_rice = pixels_per_mm_at_rim * depth_scale_factor
    pixels_per_mm = pixels_per_mm_at_rice if apply_depth_correction else pixels_per_mm_at_rim

    rice_height_mm = max(0.0, float(container_height_mm) - float(empty_height_mm))
    radius_inner_mm = float(inner_diam_mm) / 2.0
    bulk_rice_volume_mm3 = math.pi * (radius_inner_mm ** 2) * rice_height_mm

    # --------------------------------------- 9. Cắt gọn & cô lập vùng miệng ly
    half = max(MA_out, ma_out) / 2.0
    pad = int(round(outer_diam_px * 0.08))
    x_min = max(0, int(round(cx - half)) - pad)
    y_min = max(0, int(round(cy - half)) - pad)
    x_max = min(w, int(round(cx + half)) + pad)
    y_max = min(h, int(round(cy + half)) + pad)
    if x_max <= x_min or y_max <= y_min:
        return _fail("Vùng cắt miệng ly rỗng — elip nằm ngoài khung ảnh.")

    rim_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(rim_mask, ((cx, cy), (MA_out * 1.06, ma_out * 1.06), angle), 255, -1)

    isolated_bgr = img_bgr.copy()
    isolated_bgr[rim_mask == 0] = (0, 0, 0)

    cropped_bgr = isolated_bgr[y_min:y_max, x_min:x_max].copy()
    raw_cropped_bgr = img_bgr[y_min:y_max, x_min:x_max].copy()

    # ------------------------------------------------------- 10. Overlay chẩn đoán
    info_for_overlay = {
        "pixels_per_mm": float(pixels_per_mm),
        "inner_w_px": int(round(inner_diam_px)),
        "outer_w_px": int(round(outer_diam_px)),
        "inner_ellipse": inner_ell,
        "outer_ellipse": outer_ell,
        "detect_type": detect_type,
        "outer_confidence": outer_confidence,
        "inner_confidence": inner_confidence,
        "inner_coverage": float(inner_coverage),
        "outer_coverage": float(outer_coverage),
        "rice_circle": rice_circle,
        "warnings": warnings,
    }
    overlay_bgr = draw_container_overlay(img_bgr.copy(), info_for_overlay)

    if verbose:
        print(f"[final] type={detect_type} inner={inner_diam_px:.1f}px "
              f"outer={outer_diam_px:.1f}px px/mm={pixels_per_mm:.3f} "
              f"cov(in)={inner_coverage:.2f} conf(in)={inner_confidence}")
        for msg in warnings:
            print(f"  [warn] {msg}")

    return {
        # --- các key của v1 (giữ nguyên tên & ý nghĩa) ---
        "pixels_per_mm": float(pixels_per_mm),
        "inner_w_px": int(round(inner_diam_px)),
        "outer_w_px": int(round(outer_diam_px)),
        "container_w_px": int(round(selected_diam_px)),
        "bulk_rice_volume_mm3": float(bulk_rice_volume_mm3),
        "rice_height_mm": float(rice_height_mm),
        "inner_diam_mm": float(inner_diam_mm),
        "detect_mode": detect_mode,
        "detect_type": detect_type,
        "outer_confidence": outer_confidence,
        "inner_confidence": inner_confidence,
        "ellipse_params": selected_ell,
        "inner_ellipse": inner_ell,
        "outer_ellipse": outer_ell,
        "center": (int(round(inner_ell[0][0])), int(round(inner_ell[0][1]))),
        "angle": float(inner_ell[2]),
        "cropped_bgr": cropped_bgr,
        "raw_cropped_bgr": raw_cropped_bgr,
        "crop_bbox": (x_min, y_min, x_max, y_max),
        "crop_offset": (x_min, y_min),
        "overlay_bgr": overlay_bgr,
        # --- key mới của v2 ---
        "status": "ok",
        "warnings": warnings,
        "rice_circle": rice_circle,
        "inner_coverage": float(inner_coverage),
        "outer_coverage": float(outer_coverage),
        "pixels_per_mm_at_rim": float(pixels_per_mm_at_rim),
        "pixels_per_mm_at_rice_surface": float(pixels_per_mm_at_rice),
        "depth_scale_factor": float(depth_scale_factor),
        "working_scale_factor": float(scale_factor),
    }


# =============================================================================
# VI. OVERLAY CHẨN ĐOÁN
# =============================================================================


def draw_container_overlay(
    img_bgr: np.ndarray,
    container_info: Dict[str, Any],
) -> np.ndarray:
    """
    Vẽ Mép Trong (xanh lá), Mép Ngoài (cam) và vòng bao Khối Lúa (xanh dương nét
    mảnh) lên ảnh, kèm chú thích detect_type / độ phủ / độ tin cậy.

    Vòng bao khối lúa là thứ quan trọng nhất khi debug ở v2: nếu elip xanh lá
    KHÔNG bao quanh vòng xanh dương thì ràng buộc bao hàm đã bị vi phạm và kết
    quả sai — nhìn một cái là thấy, không cần đọc log.

    Mỗi đường được vẽ với lớp viền đệm đen (halo) dày hơn phía dưới để luôn nổi
    rõ bất kể nền sáng, tối hay màu gần trùng.
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    s = max(1.0, min(h, w) / 900.0)  # hệ số co giãn nét vẽ/chữ theo kích thước ảnh

    def _halo_ellipse(ell, color, thickness):
        t = max(1, int(round(thickness * s)))
        cv2.ellipse(vis, ell, (0, 0, 0), t + max(2, int(round(3 * s))), lineType=cv2.LINE_AA)
        cv2.ellipse(vis, ell, color, t, lineType=cv2.LINE_AA)

    rice_circle = container_info.get("rice_circle")
    if rice_circle is not None:
        rx, ry, rr = rice_circle
        c = (int(round(rx)), int(round(ry)))
        r = int(round(rr))
        cv2.circle(vis, c, r, (0, 0, 0), max(2, int(round(3 * s))), lineType=cv2.LINE_AA)
        cv2.circle(vis, c, r, (255, 180, 0), max(1, int(round(1 * s))), lineType=cv2.LINE_AA)

    outer_ell = container_info.get("outer_ellipse")
    if outer_ell is not None:
        _halo_ellipse(outer_ell, (0, 165, 255), 2)

    inner_ell = container_info.get("inner_ellipse")
    if inner_ell is not None:
        _halo_ellipse(inner_ell, (0, 255, 0), 3)
        cx = int(round(inner_ell[0][0]))
        cy = int(round(inner_ell[0][1]))
        cv2.circle(vis, (cx, cy), int(round(5 * s)), (0, 0, 0), -1)
        cv2.circle(vis, (cx, cy), int(round(3 * s)), (0, 0, 255), -1)

    def _f(v, pct=True):
        if v is None:
            return "n/a"
        return f"{v * 100:.0f}%" if pct else f"{v:.2f}"

    lines = [
        (f"[MEP TRONG] {container_info.get('inner_w_px', 0)}px | "
         f"Scale = {container_info.get('pixels_per_mm', 0.0):.2f} px/mm",
         (0, 255, 0), 0.80),
        (f"[MEP NGOAI] {container_info.get('outer_w_px', 0)}px", (0, 165, 255), 0.70),
        (f"type={container_info.get('detect_type', 'n/a')} | "
         f"cov(in)={_f(container_info.get('inner_coverage'))} "
         f"conf(in)={_f(container_info.get('inner_confidence'))} "
         f"conf(out)={_f(container_info.get('outer_confidence'))}",
         (255, 255, 255), 0.55),
    ]
    if rice_circle is not None:
        lines.append((f"[KHOI LUA] r={rice_circle[2]:.0f}px (vong xanh duong)",
                      (255, 180, 0), 0.55))
    n_warn = len(container_info.get("warnings") or [])
    if n_warn:
        lines.append((f"! {n_warn} canh bao - xem key 'warnings'", (0, 220, 255), 0.55))

    pad = int(round(15 * s))
    # Co chữ lại nếu ảnh quá hẹp để khung chú thích không tràn ra ngoài khung ảnh
    ts = s * min(1.0, (w - 2.0 * pad) / max(1.0, 660.0 * s))
    line_h = max(14, int(round(34 * ts)))
    box_w = min(w - 2 * pad, int(round(660 * ts)))
    box_h = pad + line_h * len(lines) + int(round(8 * ts))

    box = vis.copy()
    cv2.rectangle(box, (pad, pad), (min(w - 1, pad + box_w), min(h - 1, pad + box_h)),
                  (0, 0, 0), -1)
    cv2.addWeighted(box, 0.6, vis, 0.4, 0, vis)

    y = pad + int(round(26 * ts))
    x_txt = pad + int(round(8 * ts))
    for text, color, fs in lines:
        cv2.putText(vis, text, (x_txt, y), cv2.FONT_HERSHEY_SIMPLEX, fs * ts,
                    (0, 0, 0), max(2, int(round(3 * ts))), cv2.LINE_AA)
        cv2.putText(vis, text, (x_txt, y), cv2.FONT_HERSHEY_SIMPLEX, fs * ts,
                    color, max(1, int(round(1.6 * ts))), cv2.LINE_AA)
        y += line_h

    return vis


# =============================================================================
# VII. TỰ KIỂM TRA NHANH (chạy trực tiếp file này với 1 đường dẫn ảnh)
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Cách dùng: python container_detector.py <anh.jpg> [inner_diam_mm]")
        sys.exit(1)

    path = sys.argv[1]
    diam = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    res = detect_container_and_scale(
        path,
        inner_diam_mm=diam,
        container_height_mm=40.0,
        empty_height_mm=10.0,
        wall_thickness_mm=1.5,
        detect_mode="inner",
        on_failure="dict",
        verbose=True,
    )

    if res.get("status") != "ok":
        print("THẤT BẠI:", res.get("reason"))
        sys.exit(2)

    out = str(Path(path).with_name(Path(path).stem + "_overlay.png"))
    cv2.imwrite(out, res["overlay_bgr"])
    print(f"Đã ghi overlay: {out}")
    print(f"pixels_per_mm = {res['pixels_per_mm']:.3f} "
          f"(tại bề mặt lúa: {res['pixels_per_mm_at_rice_surface']:.3f})")
