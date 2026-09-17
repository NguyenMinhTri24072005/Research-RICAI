#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE 1: NHẬN DIỆN VẬT CHỨA & QUY ĐỔI TỶ LỆ KÍCH THƯỚC (CONTAINER DETECTOR)
===============================================================================
Mục đích:
  - Nhận diện chính xác cả MÉP TRONG (Inner Rim) và MÉP NGOÀI (Outer Rim) của ly đựng lúa.
  - Sử dụng thuật toán Khớp Cặp Elip Đồng Tâm Hình Học (Robust Geometric Concentric Rim).
  - Khắc phục hoàn toàn hiện tượng mép trong bị bẫy bởi các cạnh sắc của hạt lúa bên trong ly.
  - Tính tỷ lệ quy đổi chuẩn xác từ Mép Trong: pixels_per_mm = inner_diam_px / inner_diam_mm.
  - Tính thể tích khối lúa thực tế trong vật chứa (mm³).
  - Hỗ trợ cắt gọn và cô lập vùng miệng ly để loại bỏ nhiễu nền ngoài.

  FIX (đợt vá lỗi elip lệch tâm & co nhỏ trên một số ảnh):
  - Nguyên nhân gốc: viền ly trong suốt (kính/nhựa) bị phản chiếu ánh sáng làm Canny
    edge bị đứt đoạn ở một phần chu vi -> contour chỉ còn là 1 cung (partial arc) ->
    cv2.fitEllipse (least-squares) fit ra elip nhỏ hơn thật và tâm bị kéo lệch về
    phía cung còn lại.
  - Vá 1: cv2.morphologyEx(MORPH_CLOSE) trước khi tìm contour để nối các đoạn viền
    bị đứt do phản chiếu / tương phản yếu.
  - Vá 2: hàm _contour_angular_coverage() loại bỏ các contour chỉ phủ một cung hẹp
    quanh tâm (không đáng tin cho fitEllipse), giữ lại contour phủ gần trọn 360°.
  - Vá 3: fallback bằng cv2.HoughCircles (_hough_fallback_ellipse) khi không còn
    contour nào đạt độ phủ góc yêu cầu — Hough bền vững hơn với viền đứt đoạn vì
    dùng voting theo gradient, không cần đường khép kín.
===============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def _contour_angular_coverage(
    contour: np.ndarray,
    center: Tuple[float, float],
    num_bins: int = 36,
) -> float:
    """
    Tính tỉ lệ (0.0 - 1.0) số 'lát góc' quanh tâm có ít nhất 1 điểm contour đi qua.

    Dùng để phát hiện contour chỉ là một cung hở (partial arc) thay vì viền khép kín
    360 độ. Giá trị thấp (ví dụ < 0.8) nghĩa là contour bị đứt đoạn trên một phần chu
    vi (thường do phản chiếu ánh sáng trên thành ly trong suốt) -> không nên tin
    tưởng kết quả cv2.fitEllipse trên contour đó, vì least-squares sẽ cho ra elip
    nhỏ hơn thật và tâm bị kéo lệch về phía cung còn lại.

    Parameters
    ----------
    contour : np.ndarray
        Contour lấy từ cv2.findContours, shape (N, 1, 2).
    center : Tuple[float, float]
        Tâm ước lượng (cx, cy), thường lấy từ ellipse đã fit tạm thời trên contour đó.
    num_bins : int, default 36
        Số lát góc chia quanh tâm (36 lát = mỗi lát 10 độ).

    Returns
    -------
    float
        Tỉ lệ lát góc có điểm contour đi qua, từ 0.0 (không phủ) đến 1.0 (phủ trọn).
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    cx, cy = center
    angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    bin_size = 360.0 / num_bins
    bins = (angles / bin_size).astype(int) % num_bins
    covered = len(set(bins.tolist()))
    return covered / float(num_bins)


def _hough_fallback_ellipse(
    gray_blurred: np.ndarray,
    img_cx: float,
    img_cy: float,
    min_diam_px: int,
    max_diam_px: int,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Fallback dò viền tròn bằng cv2.HoughCircles khi contour bị đứt đoạn (viền ly
    trong suốt / phản chiếu ánh sáng khiến Canny không cho ra đường khép kín).

    HoughCircles bền vững hơn fitEllipse trong trường hợp này vì nó tích lũy phiếu
    bầu (voting) theo gradient ảnh, không đòi hỏi một đường contour khép kín liên tục.

    Parameters
    ----------
    gray_blurred : np.ndarray
        Ảnh xám đã làm mờ Gaussian (dùng lại `blurred` đã có sẵn trong pipeline).
    img_cx, img_cy : float
        Tâm ảnh, dùng để ưu tiên vòng tròn gần giữa khung hình nhất.
    min_diam_px, max_diam_px : int
        Khoảng đường kính hợp lệ (pixel) để giới hạn không gian tìm kiếm của Hough.

    Returns
    -------
    Optional[Tuple]
        Elip dạng ((cx, cy), (MA, ma), angle) tương thích với cv2.fitEllipse,
        hoặc None nếu không tìm thấy vòng tròn nào.
    """
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_diam_px, 50),
        param1=100,
        param2=40,
        minRadius=int(min_diam_px / 2),
        maxRadius=int(max_diam_px / 2),
    )
    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    # Ưu tiên vòng tròn gần tâm ảnh nhất (giả định vật chứa được chụp gần giữa khung)
    best = min(circles, key=lambda c: (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2)
    x, y, r = best
    return ((float(x), float(y)), (float(2 * r), float(2 * r)), 0.0)


def detect_container_and_scale(
    image_input: Union[str, Path, np.ndarray],
    inner_diam_mm: float,
    container_height_mm: float,
    empty_height_mm: float,
    wall_thickness_mm: float = 1.5,
    detect_mode: str = "inner",  # 'inner' (mép trong), 'outer' (mép ngoài), hoặc 'both'
    blur_ksize: Tuple[int, int] = (9, 9),
    canny_thresh1: int = 30,
    canny_thresh2: int = 120,
    min_angular_coverage: float = 0.80,
) -> Dict[str, Any]:
    """
    Nhận diện viền miệng ly chuẩn xác (Mép Ngoài và Mép Trong đồng tâm).

    Parameters
    ----------
    image_input : str, Path hoặc np.ndarray
        Đường dẫn file ảnh hoặc ma trận ảnh BGR.
    inner_diam_mm : float
        Đường kính TRONG thực tế của ly (mm) từ bảng thông số mẫu.
    container_height_mm : float
        Chiều cao toàn bộ thân ly (mm).
    empty_height_mm : float
        Khoảng trống từ miệng ly đến mặt trên khối lúa (mm).
    wall_thickness_mm : float, default 1.5
        Độ dày thành miệng ly (mm) để đối chiếu hình học đồng tâm.
    detect_mode : str, default 'inner'
        'inner': Dùng Mép Trong để tính pixels_per_mm (Độ chuẩn xác cao nhất).
        'outer': Dùng Mép Ngoài.
        'both': Trả về thông số cả 2 mép.
    min_angular_coverage : float, default 0.80
        Ngưỡng tối thiểu độ phủ góc quanh tâm (0.0 - 1.0) để chấp nhận một contour
        làm ứng viên mép ly. Contour phủ dưới ngưỡng này bị coi là cung hở (đứt đoạn
        do phản chiếu ánh sáng trên thành ly trong suốt) và bị loại, tránh trường hợp
        fitEllipse cho ra elip lệch tâm & nhỏ hơn thật. Hạ xuống (vd 0.65) nếu ảnh có
        độ đứt viền nặng hơn; tăng lên nếu vẫn còn bắt nhầm vật thể khác.
    """
    if isinstance(image_input, (str, Path)):
        p_str = str(image_input)
        try:
            data = np.fromfile(p_str, dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img_bgr = None
        if img_bgr is None:
            img_bgr = cv2.imread(p_str)
        if img_bgr is None:
            raise FileNotFoundError(f"Không thể đọc file ảnh: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        raise ValueError("image_input phải là đường dẫn file hoặc numpy ndarray.")

    h, w = img_bgr.shape[:2]
    img_cx, img_cy = w / 2.0, h / 2.0

    # 1. Chuyển sang ảnh xám và lọc nhiễu Gaussian
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)

    # 2. Bước A: Bắt viền Mép Ngoài của ly bằng Canny + Đóng khoảng hở (MORPH_CLOSE) + Dilate
    #    MORPH_CLOSE được thêm để nối các đoạn viền bị đứt do phản chiếu ánh sáng trên
    #    thành ly trong suốt / tương phản yếu — nguyên nhân chính khiến contour chỉ còn
    #    là 1 cung hở thay vì đường khép kín, làm fitEllipse ra elip lệch tâm & nhỏ hơn thật.
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    dilated = cv2.dilate(
        closed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    ext_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ext_cnts:
        raise RuntimeError("Không tìm thấy viền miệng cốc trong ảnh!")

    # Lọc các contour ứng viên cho Mép Ngoài dựa trên hình học
    outer_candidates: List[Dict[str, Any]] = []
    for c in ext_cnts:
        if c.shape[0] < 5:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 200:
            continue
        area = cv2.contourArea(c)
        if area < 500:
            continue

        ell = cv2.fitEllipse(c)
        (cx, cy), (MA, ma), angle = ell
        diam = max(MA, ma)
        minor = min(MA, ma)
        aspect = minor / max(1e-5, diam)

        # FIX: loại bỏ contour chỉ phủ một cung hẹp quanh tâm (viền bị đứt đoạn do
        # phản chiếu ánh sáng) — đây là nguyên nhân chính gây elip lệch tâm & co nhỏ.
        coverage = _contour_angular_coverage(c, (cx, cy))
        if coverage < min_angular_coverage:
            continue

        # Miệng ly tròn khi chụp từ trên xuống có độ tròn cao (aspect >= 0.70) và kích thước phù hợp
        if aspect >= 0.70 and 100 <= diam <= min(h, w) * 0.95:
            dist_center = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
            # Điểm ưu tiên: diện tích lớn + độ tròn cao + độ phủ góc cao + gần tâm ảnh
            score = area * (aspect ** 2) * coverage / (1.0 + 0.0008 * dist_center)
            outer_candidates.append({
                "ellipse": ell,
                "diam": diam,
                "minor": minor,
                "aspect": aspect,
                "area": area,
                "coverage": coverage,
                "score": score,
            })

    if outer_candidates:
        outer_candidates.sort(key=lambda x: x["score"], reverse=True)
        outer_ell = outer_candidates[0]["ellipse"]
    else:
        # FIX: fallback bằng HoughCircles trước khi rơi về "lấy đại contour lớn nhất"
        # (cách cũ dễ bắt nhầm cung hở làm elip lệch tâm & nhỏ hơn thật).
        hough_ell = _hough_fallback_ellipse(
            blurred,
            img_cx,
            img_cy,
            min_diam_px=int(min(h, w) * 0.15),
            max_diam_px=int(min(h, w) * 0.95),
        )
        if hough_ell is not None:
            outer_ell = hough_ell
        else:
            best_ext = max(ext_cnts, key=cv2.contourArea)
            if best_ext.shape[0] >= 5:
                outer_ell = cv2.fitEllipse(best_ext)
            else:
                x, y, w_box, h_box = cv2.boundingRect(best_ext)
                outer_ell = ((x + w_box / 2.0, y + h_box / 2.0), (max(w_box, h_box), min(w_box, h_box)), 0.0)

    (cx, cy), (MA_out, ma_out), angle = outer_ell
    outer_diam_px = float(max(MA_out, ma_out))

    # 3. Bước B: Xác định Mép Trong (Inner Rim) chuẩn xác
    # Tỷ lệ lý thuyết hình học giữa mép trong và mép ngoài
    expected_ratio = float(inner_diam_mm) / (float(inner_diam_mm) + 2.0 * float(wall_thickness_mm))

    # Quét contour bên trong kiểm tra với điều kiện hình học nghiêm ngặt
    # (dùng `closed` thay vì `edges` thô để cũng hưởng lợi từ việc nối viền đứt đoạn)
    cnts_all, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inner_candidates: List[Dict[str, Any]] = []

    for c in cnts_all:
        if c.shape[0] < 5:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 180:
            continue

        ell = cv2.fitEllipse(c)
        (ccx, ccy), (MA, ma), ang = ell
        dist = float(np.sqrt((ccx - cx) ** 2 + (ccy - cy) ** 2))
        diam = float(max(MA, ma))
        ratio = diam / outer_diam_px
        aspect = float(min(MA, ma) / max(1e-5, diam))
        outer_aspect = float(min(MA_out, ma_out) / max(1e-5, MA_out))

        # FIX: cùng lý do như mép ngoài — loại contour chỉ phủ một cung hẹp quanh tâm
        coverage = _contour_angular_coverage(c, (ccx, ccy))
        if coverage < min_angular_coverage:
            continue

        # Điều kiện nghiêm ngặt: gần như đồng tâm tuyệt đối (dist <= 4% đường kính), cùng độ tròn, cùng tỷ lệ
        if (
            dist <= 0.04 * outer_diam_px
            and abs(aspect - outer_aspect) <= 0.05
            and abs(ratio - expected_ratio) <= 0.035
        ):
            inner_candidates.append({
                "ellipse": ell,
                "diam": diam,
                "ratio": ratio,
                "dist": dist,
                "coverage": coverage,
            })

    if inner_candidates:
        inner_candidates.sort(key=lambda x: abs(x["ratio"] - expected_ratio))
        best_inner = inner_candidates[0]
        inner_ell = best_inner["ellipse"]
        inner_diam_px = float(best_inner["diam"])
        detect_type = "direct_inner_contour"
    else:
        # Suy luận Hình Học Đồng Tâm Chuẩn Xác (Geometric Concentric Rim)
        # Giữ nguyên tâm (cx, cy), góc nghiêng và tỷ lệ dài/rộng từ mép ngoài
        inner_diam_px = outer_diam_px * expected_ratio
        inner_ell = ((cx, cy), (MA_out * expected_ratio, ma_out * expected_ratio), angle)
        detect_type = "geometric_concentric_rim"

    # 4. Tính toán tỷ lệ quy đổi pixels_per_mm
    if detect_mode == "outer":
        selected_diam_px = outer_diam_px
        selected_ell = outer_ell
        pixels_per_mm = outer_diam_px / (float(inner_diam_mm) + 2.0 * float(wall_thickness_mm))
    else:  # 'inner' hoặc 'both'
        selected_diam_px = inner_diam_px
        selected_ell = inner_ell
        pixels_per_mm = inner_diam_px / float(inner_diam_mm)

    rice_height_mm = max(0.0, float(container_height_mm) - float(empty_height_mm))
    radius_inner_mm = float(inner_diam_mm) / 2.0
    bulk_rice_volume_mm3 = math.pi * (radius_inner_mm ** 2) * rice_height_mm

    # 5. Cô lập và cắt ảnh vùng miệng ly (loại bỏ hoàn toàn nền bên ngoài)
    pad = int(outer_diam_px * 0.08)  # 8% padding xung quanh mép ngoài
    x_min = max(0, int(cx) - int(max(MA_out, ma_out) / 2) - pad)
    y_min = max(0, int(cy) - int(max(MA_out, ma_out) / 2) - pad)
    x_max = min(w, int(cx) + int(max(MA_out, ma_out) / 2) + pad)
    y_max = min(h, int(cy) + int(max(MA_out, ma_out) / 2) + pad)

    # Tạo mask hình elip ôm sát mép ngoài miệng ly để triệt tiêu nhiễu nền bàn
    rim_mask = np.zeros((h, w), dtype=np.uint8)
    pad_ell = ((cx, cy), (MA_out * 1.06, ma_out * 1.06), angle)
    cv2.ellipse(rim_mask, pad_ell, 255, -1)

    isolated_bgr = img_bgr.copy()
    isolated_bgr[rim_mask == 0] = [0, 0, 0]

    cropped_bgr = isolated_bgr[y_min:y_max, x_min:x_max].copy()
    raw_cropped_bgr = img_bgr[y_min:y_max, x_min:x_max].copy()

    # 6. Tạo sẵn ảnh overlay gốc có vẽ khoanh viền
    temp_info = {
        "pixels_per_mm": float(pixels_per_mm),
        "inner_w_px": int(round(inner_diam_px)),
        "outer_w_px": int(round(outer_diam_px)),
        "inner_ellipse": inner_ell,
        "outer_ellipse": outer_ell,
    }
    overlay_bgr = draw_container_overlay(img_bgr.copy(), temp_info)

    return {
        "pixels_per_mm": float(pixels_per_mm),
        "inner_w_px": int(round(inner_diam_px)),
        "outer_w_px": int(round(outer_diam_px)),
        "container_w_px": int(round(selected_diam_px)),
        "bulk_rice_volume_mm3": float(bulk_rice_volume_mm3),
        "rice_height_mm": float(rice_height_mm),
        "inner_diam_mm": float(inner_diam_mm),
        "detect_mode": detect_mode,
        "detect_type": detect_type,
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
    }


def draw_container_overlay(
    img_bgr: np.ndarray,
    container_info: Dict[str, Any],
) -> np.ndarray:
    """
    Vẽ viền Mép Trong (Màu Xanh Lá Đậm) và Mép Ngoài (Màu Cam nét mảnh) lên ảnh để kiểm tra trực quan.
    """
    vis = img_bgr.copy()

    # 1. Vẽ Mép Ngoài (Màu Cam nét mảnh)
    outer_ell = container_info.get("outer_ellipse")
    if outer_ell is not None:
        cv2.ellipse(vis, outer_ell, (0, 165, 255), 2, lineType=cv2.LINE_AA)

    # 2. Vẽ Mép Trong (Màu Xanh Lá Đậm - Chuẩn)
    inner_ell = container_info.get("inner_ellipse")
    if inner_ell is not None:
        cv2.ellipse(vis, inner_ell, (0, 255, 0), 3, lineType=cv2.LINE_AA)
        cx = int(round(inner_ell[0][0]))
        cy = int(round(inner_ell[0][1]))
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

    # 3. Ghi chú thích với khung nền bán trong suốt để dễ đọc
    scale_val = container_info["pixels_per_mm"]
    in_px = container_info["inner_w_px"]
    out_px = container_info["outer_w_px"]

    txt1 = f"[MÉP TRONG]: {in_px}px | Scale = {scale_val:.2f} px/mm"
    txt2 = f"[MÉP NGOÀI]: {out_px}px"

    # Vẽ nền hộp đen mờ cho text
    overlay_box = vis.copy()
    cv2.rectangle(overlay_box, (15, 15), (550, 105), (0, 0, 0), -1)
    cv2.addWeighted(overlay_box, 0.6, vis, 0.4, 0, vis)

    cv2.putText(vis, txt1, (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, txt2, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

    return vis