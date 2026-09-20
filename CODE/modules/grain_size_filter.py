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

  FIX ĐỢT 2 (vá lỗi "mép trong" bị nhận nhầm thành một cung nhỏ lọt trong đám hạt lúa):
  - Nguyên nhân: contour của một cụm hạt lúa/vệt phản chiếu gần tâm ảnh có thể lọt qua
    hết các điều kiện lọc hình học (khoảng cách tâm, aspect, tỉ lệ đường kính) một
    cách "trùng hợp", đặc biệt khi mép ngoài (outer) ở bước trước cũng đã bị lệch nhẹ
    -> toàn bộ suy luận đồng tâm phía sau đúng về logic nhưng sai về vật lý, và code
    cũ KHÔNG có lớp kiểm tra cuối để tự phát hiện việc này, nên vẫn trả kết quả như
    thể đúng.
  - Vá 4: _ransac_fit_ellipse() — thay vì tin tưởng tuyệt đối cv2.fitEllipse (rất nhạy
    outlier vì là least-squares thuần), dùng RANSAC: lấy mẫu ngẫu nhiên nhiều lần,
    giữ lại mô hình có nhiều điểm đồng thuận (inlier) nhất, rồi fit lại lần cuối chỉ
    trên tập inlier đó. Bền vững hơn nhiều với nhiễu cục bộ.
  - Vá 5: Thêm lớp "sanity check" bắt buộc sau khi chọn outer/inner — nếu đường kính
    phát hiện được nhỏ bất thường so với kích thước ảnh, hoặc tỉ lệ inner/outer vô lý
    về mặt vật lý, hàm sẽ raise RuntimeError với thông báo rõ ràng thay vì âm thầm trả
    về một con số sai. Đây chính là lớp còn thiếu khiến lỗi trong ảnh gốc không bị
    chặn lại.
  - Vá 6: draw_container_overlay() vẽ viền có viền đệm đen (halo) phía dưới trước khi
    vẽ màu, giúp đường elip luôn nổi rõ bất kể màu nền, đồng thời in thêm dòng
    detect_type + độ tin cậy (RANSAC inlier ratio) để tự chẩn đoán khi xem ảnh debug.

  FIX ĐỢT 3 (vá lỗi trên ảnh độ phân giải rất cao — ví dụ ảnh chụp điện thoại 6000x8000):
  - Nguyên nhân: TOÀN BỘ tham số dạng pixel tuyệt đối trong pipeline (blur_ksize=(9,9),
    kernel morphology (9,9)/(5,5), canny_thresh1/2, ngưỡng area<500/peri<200/180,
    dung sai inlier của RANSAC...) được hiệu chỉnh ngầm định cho ảnh cỡ ~1000-1500px.
    Trên ảnh ~48 triệu pixel, các kernel này gần như không còn tác dụng làm mượt/nối
    viền tương ứng, khiến Canny sinh ra vô số cạnh vụn từ vân hạt gạo/nhiễu cảm biến,
    và contour vòng ngoài bị "gãy" thành nhiều mảnh nhỏ hơn hẳn miệng ly thật — dẫn
    đến việc chọn nhầm một contour nhỏ (thường nằm ngay trong đám hạt lúa) làm mép
    ngoài, dù vẫn đủ lớn để "qua mặt" sanity-check tuyệt đối ở Vá 5.
  - Vá 7: chuẩn hoá độ phân giải xử lý — toàn bộ bước phát hiện (Canny, contour,
    RANSAC, Hough fallback) chạy trên một bản resize xuống tối đa `working_max_dim`
    (mặc định 1600px cạnh dài), nơi mọi kernel/ngưỡng pixel phía trên hoạt động đúng
    như thiết kế ban đầu, bất kể ảnh gốc lớn cỡ nào. Elip kết quả (outer/inner) sau
    đó được quy đổi ngược lại đúng tỷ lệ ảnh gốc (_scale_ellipse) trước khi dùng để
    crop/mask/tính pixels_per_mm — nên độ chính xác vật lý cuối cùng không đổi, chỉ
    có bước NHẬN DIỆN là chạy ở độ phân giải ổn định.
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


def _ransac_fit_ellipse(
    contour: np.ndarray,
    iterations: int = 80,
    sample_size: int = 8,
    inlier_dist_px: float = 3.0,
    min_inlier_ratio: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float], float], Optional[float]]:
    """
    Fit elip bền vững với outlier bằng RANSAC, thay vì tin tưởng tuyệt đối
    cv2.fitEllipse (least-squares thuần, rất nhạy với vài điểm nhiễu cục bộ).

    Thuật toán:
      1. Lặp `iterations` lần: lấy mẫu ngẫu nhiên `sample_size` điểm trên contour,
         fit thử một elip từ mẫu đó.
      2. Với mỗi elip thử, đếm số điểm trong TOÀN BỘ contour nằm gần biên elip đó
         (sai số < `inlier_dist_px`, tính theo bán kính chuẩn hóa) -> đây là số
         inlier (điểm "đồng thuận" với mô hình).
      3. Giữ lại elip có số inlier cao nhất qua tất cả các lần lặp.
      4. Fit lại lần cuối (least-squares) chỉ trên tập inlier của elip tốt nhất để
         tinh chỉnh — cho kết quả mượt hơn so với chỉ giữ elip từ mẫu ngẫu nhiên.

    Nếu contour có quá ít điểm, hoặc không mô hình nào đạt `min_inlier_ratio`,
    hàm sẽ rơi về cv2.fitEllipse(contour) thông thường và trả confidence thấp/`None`
    để nơi gọi tự quyết định có tin tưởng kết quả hay không.

    Parameters
    ----------
    contour : np.ndarray
        Contour từ cv2.findContours, shape (N, 1, 2).
    iterations : int, default 80
        Số lần lặp lấy mẫu ngẫu nhiên.
    sample_size : int, default 8
        Số điểm lấy mẫu mỗi lần lặp (>= 5 vì cv2.fitEllipse cần tối thiểu 5 điểm).
    inlier_dist_px : float, default 3.0
        Sai số khoảng cách (pixel, xấp xỉ) để một điểm được coi là inlier.
    min_inlier_ratio : float, default 0.6
        Tỉ lệ inlier tối thiểu để tin tưởng kết quả RANSAC; dưới ngưỡng này coi như
        không đủ đồng thuận và rơi về fit trực tiếp trên toàn bộ contour.
    rng : np.random.Generator, optional
        Bộ sinh số ngẫu nhiên; truyền vào để kết quả tái lập được khi cần debug.

    Returns
    -------
    Tuple[ellipse, inlier_ratio]
        ellipse: ((cx, cy), (MA, ma), angle) — cùng định dạng cv2.fitEllipse.
        inlier_ratio: tỉ lệ đồng thuận (0.0 - 1.0) của mô hình tốt nhất, hoặc None
            nếu không chạy được RANSAC (contour quá ít điểm) — nghĩa là "không rõ
            độ tin cậy", KHÔNG đồng nghĩa với "đáng tin cậy".
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = pts.shape[0]
    sample_size = max(5, sample_size)

    if n < max(sample_size, 20):
        # Quá ít điểm để RANSAC có ý nghĩa thống kê -> fit trực tiếp, báo confidence None
        try:
            return cv2.fitEllipse(contour), None
        except cv2.error:
            raise RuntimeError("Không đủ điểm để fit elip (contour quá nhỏ/suy biến).")

    if rng is None:
        rng = np.random.default_rng()

    best_ell = None
    best_inlier_count = -1
    best_mask = None

    for _ in range(iterations):
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = pts[idx].astype(np.float32).reshape(-1, 1, 2)
        try:
            ell = cv2.fitEllipse(sample)
        except cv2.error:
            continue

        (ecx, ecy), (MA, ma), angle = ell
        if MA <= 1e-3 or ma <= 1e-3:
            continue

        theta = np.deg2rad(angle)
        cos_a, sin_a = np.cos(theta), np.sin(theta)
        dx = pts[:, 0] - ecx
        dy = pts[:, 1] - ecy
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a
        a, b = MA / 2.0, ma / 2.0
        # Bán kính chuẩn hóa: r_norm == 1.0 nghĩa là điểm nằm đúng trên biên elip
        r_norm = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        tol = inlier_dist_px / max(1e-3, (a + b) / 2.0)
        inlier_mask = np.abs(r_norm - 1.0) < tol
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_ell = ell
            best_mask = inlier_mask

    if best_ell is None:
        return cv2.fitEllipse(contour), 0.0

    inlier_ratio = best_inlier_count / float(n)
    if inlier_ratio < min_inlier_ratio:
        # Không mô hình nào đủ đồng thuận -> rơi về fit trực tiếp, nhưng vẫn báo
        # đúng inlier_ratio thấp để nơi gọi biết mà nghi ngờ kết quả này.
        return cv2.fitEllipse(contour), inlier_ratio

    inlier_pts = pts[best_mask].astype(np.float32).reshape(-1, 1, 2)
    if inlier_pts.shape[0] >= 5:
        refined_ell = cv2.fitEllipse(inlier_pts)
    else:
        refined_ell = best_ell

    return refined_ell, inlier_ratio


def _scale_ellipse(
    ell: Tuple[Tuple[float, float], Tuple[float, float], float],
    factor: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Quy đổi một elip ((cx, cy), (MA, ma), angle) sang một hệ tọa độ khác theo hệ số
    tỉ lệ đồng nhất `factor` (dùng để đưa kết quả từ ảnh xử lý đã resize về đúng
    tọa độ/kích thước của ảnh gốc). Góc nghiêng không đổi vì scale đồng nhất theo
    cả 2 trục không làm xoay elip.
    """
    (ecx, ecy), (MA, ma), angle = ell
    return ((ecx * factor, ecy * factor), (MA * factor, ma * factor), angle)


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
    use_ransac: bool = True,
    ransac_iterations: int = 80,
    min_outer_diam_ratio: float = 0.25,
    min_inner_diam_ratio: float = 0.15,
    min_inner_outer_ratio_floor: float = 0.5,
    working_max_dim: int = 1600,
    verbose: bool = False,
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
    use_ransac : bool, default True
        Bật tinh chỉnh elip bằng RANSAC (_ransac_fit_ellipse) sau khi chọn được
        contour ứng viên tốt nhất — bền vững hơn với outlier so với fitEllipse thuần.
    ransac_iterations : int, default 80
        Số lần lặp lấy mẫu ngẫu nhiên trong RANSAC.
    min_outer_diam_ratio : float, default 0.25
        SANITY CHECK: đường kính mép ngoài phát hiện được phải >= tỉ lệ này nhân với
        min(chiều rộng, chiều cao) ảnh, nếu không sẽ raise RuntimeError. Chặn trường
        hợp bắt nhầm một vật thể nhỏ (bóng, phản chiếu, vật thể khác) làm mép ngoài.
    min_inner_diam_ratio : float, default 0.15
        SANITY CHECK: đường kính mép trong phát hiện được phải >= tỉ lệ này nhân với
        min(chiều rộng, chiều cao) ảnh. Đây là lớp chặn trực tiếp cho lỗi "mép trong
        bị nhận nhầm thành một cung nhỏ lọt trong đám hạt lúa".
    min_inner_outer_ratio_floor : float, default 0.5
        SANITY CHECK: tỉ lệ inner/outer tối thiểu chấp nhận được (rất lỏng, chỉ để
        bắt lỗi thô — so với ngưỡng chặt ±0.035 dùng khi so khớp expected_ratio).
    working_max_dim : int, default 1600
        Toàn bộ bước NHẬN DIỆN (Canny/contour/RANSAC/Hough) chạy trên bản resize của
        ảnh sao cho cạnh dài nhất <= working_max_dim px, để mọi kernel/ngưỡng pixel
        (blur_ksize, canny_thresh1/2, min_angular_coverage...) hoạt động ổn định bất
        kể ảnh gốc có độ phân giải bao nhiêu — xem "FIX ĐỢT 3" ở đầu file. Ảnh nhỏ hơn
        working_max_dim thì giữ nguyên (không upscale). Kết quả cuối (crop, overlay,
        pixels_per_mm) luôn ở đúng độ phân giải ảnh GỐC — tham số này không ảnh hưởng
        độ chính xác vật lý, chỉ ảnh hưởng bước nhận diện.
    verbose : bool, default False
        In log chẩn đoán (số ứng viên, score, inlier_ratio, detect_type...) ra
        console để debug. Không bật mặc định để không đổi hành vi/])output hiện có.
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

    h, w = img_bgr.shape[:2]  # kích thước ảnh GỐC — dùng cho crop/mask/overlay cuối cùng

    # Vá 7: chuẩn hoá độ phân giải cho TOÀN BỘ bước nhận diện (không đụng đến ảnh gốc
    # dùng để crop/overlay ở cuối hàm). Xem "FIX ĐỢT 3" ở docstring đầu file.
    if max(h, w) > working_max_dim:
        scale_factor = working_max_dim / float(max(h, w))
        proc_w = max(1, int(round(w * scale_factor)))
        proc_h = max(1, int(round(h * scale_factor)))
        proc_bgr = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        scale_factor = 1.0
        proc_bgr = img_bgr

    proc_h, proc_w = proc_bgr.shape[:2]
    img_cx, img_cy = proc_w / 2.0, proc_h / 2.0
    min_side = min(proc_h, proc_w)

    # 1. Chuyển sang ảnh xám và lọc nhiễu Gaussian (trên ảnh đã chuẩn hoá độ phân giải)
    gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
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
        if aspect >= 0.70 and 100 <= diam <= min_side * 0.95:
            dist_center = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
            # Điểm ưu tiên: diện tích lớn + độ tròn cao + độ phủ góc cao + gần tâm ảnh
            score = area * (aspect ** 2) * coverage / (1.0 + 0.0008 * dist_center)
            outer_candidates.append({
                "ellipse": ell,
                "contour": c,
                "diam": diam,
                "minor": minor,
                "aspect": aspect,
                "area": area,
                "coverage": coverage,
                "score": score,
            })

    outer_contour_for_ransac: Optional[np.ndarray] = None
    outer_confidence: Optional[float] = None

    if outer_candidates:
        outer_candidates.sort(key=lambda x: x["score"], reverse=True)
        best_outer = outer_candidates[0]
        outer_ell = best_outer["ellipse"]
        outer_contour_for_ransac = best_outer["contour"]
        if verbose:
            print(
                f"[outer] {len(outer_candidates)} ứng viên | chọn diam={best_outer['diam']:.1f}px "
                f"score={best_outer['score']:.1f} coverage={best_outer['coverage']:.2f}"
            )
    else:
        # FIX: fallback bằng HoughCircles trước khi rơi về "lấy đại contour lớn nhất"
        # (cách cũ dễ bắt nhầm cung hở làm elip lệch tâm & nhỏ hơn thật).
        hough_ell = _hough_fallback_ellipse(
            blurred,
            img_cx,
            img_cy,
            min_diam_px=int(min_side * 0.15),
            max_diam_px=int(min_side * 0.95),
        )
        if hough_ell is not None:
            outer_ell = hough_ell
            if verbose:
                print("[outer] Không có ứng viên hợp lệ -> dùng HoughCircles fallback")
        else:
            best_ext = max(ext_cnts, key=cv2.contourArea)
            if best_ext.shape[0] >= 5:
                outer_ell = cv2.fitEllipse(best_ext)
                outer_contour_for_ransac = best_ext
            else:
                x, y, w_box, h_box = cv2.boundingRect(best_ext)
                outer_ell = ((x + w_box / 2.0, y + h_box / 2.0), (max(w_box, h_box), min(w_box, h_box)), 0.0)
            if verbose:
                print("[outer] Hough cũng thất bại -> dùng contour lớn nhất (kém tin cậy nhất)")

    # Vá 4: tinh chỉnh mép ngoài bằng RANSAC nếu có contour nguồn để fit lại
    if use_ransac and outer_contour_for_ransac is not None:
        outer_ell, outer_confidence = _ransac_fit_ellipse(
            outer_contour_for_ransac, iterations=ransac_iterations
        )

    (cx, cy), (MA_out, ma_out), angle = outer_ell
    outer_diam_px = float(max(MA_out, ma_out))

    # Vá 5a: SANITY CHECK mép ngoài — chặn ngay tại đây nếu outer đã sai, để không
    # lan truyền lỗi xuống toàn bộ suy luận đồng tâm phía sau.
    if outer_diam_px < min_outer_diam_ratio * min_side:
        raise RuntimeError(
            f"Mép NGOÀI phát hiện được ({outer_diam_px:.1f}px) nhỏ bất thường so với "
            f"khung xử lý ({proc_w}x{proc_h}px sau chuẩn hoá độ phân giải từ {w}x{h}px gốc, "
            f"ngưỡng tối thiểu {min_outer_diam_ratio * min_side:.1f}px). "
            f"Khả năng cao thuật toán đã bắt nhầm vật thể khác làm miệng ly — kiểm tra lại "
            f"ánh sáng/độ tương phản viền ly, hoặc hạ canny_thresh1/2, hoặc hạ "
            f"min_outer_diam_ratio nếu ảnh chụp cố ý zoom xa."
        )

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
                "contour": c,
                "diam": diam,
                "ratio": ratio,
                "dist": dist,
                "coverage": coverage,
            })

    inner_contour_for_ransac: Optional[np.ndarray] = None
    inner_confidence: Optional[float] = None

    if inner_candidates:
        inner_candidates.sort(key=lambda x: abs(x["ratio"] - expected_ratio))
        best_inner = inner_candidates[0]
        inner_ell = best_inner["ellipse"]
        inner_contour_for_ransac = best_inner["contour"]
        inner_diam_px = float(best_inner["diam"])
        detect_type = "direct_inner_contour"
        if verbose:
            print(
                f"[inner] {len(inner_candidates)} ứng viên | chọn diam={inner_diam_px:.1f}px "
                f"ratio={best_inner['ratio']:.3f} (kỳ vọng {expected_ratio:.3f})"
            )
    else:
        # Suy luận Hình Học Đồng Tâm Chuẩn Xác (Geometric Concentric Rim)
        # Giữ nguyên tâm (cx, cy), góc nghiêng và tỷ lệ dài/rộng từ mép ngoài
        inner_diam_px = outer_diam_px * expected_ratio
        inner_ell = ((cx, cy), (MA_out * expected_ratio, ma_out * expected_ratio), angle)
        detect_type = "geometric_concentric_rim"
        if verbose:
            print(f"[inner] Không có contour trực tiếp hợp lệ -> suy ra hình học từ mép ngoài")

    # Vá 4: tinh chỉnh mép trong bằng RANSAC nếu có contour nguồn (chỉ áp dụng cho
    # nhánh direct_inner_contour — nhánh geometric_concentric_rim không có contour
    # riêng vì nó được suy ra thuần hình học từ mép ngoài đã validate).
    if use_ransac and inner_contour_for_ransac is not None:
        inner_ell, inner_confidence = _ransac_fit_ellipse(
            inner_contour_for_ransac, iterations=ransac_iterations
        )
        inner_diam_px = float(max(inner_ell[1]))

    # Vá 5b: SANITY CHECK mép trong — đây là lớp chặn trực tiếp cho đúng lỗi đã thấy
    # trong ảnh debug (một cung nhỏ lọt trong đám hạt lúa bị nhận nhầm thành mép trong).
    if inner_diam_px < min_inner_diam_ratio * min_side:
        raise RuntimeError(
            f"Mép TRONG phát hiện được ({inner_diam_px:.1f}px) nhỏ bất thường so với "
            f"khung xử lý ({proc_w}x{proc_h}px sau chuẩn hoá độ phân giải, "
            f"ngưỡng tối thiểu {min_inner_diam_ratio * min_side:.1f}px). "
            f"Đây thường là dấu hiệu contour của một cụm hạt lúa/vệt phản chiếu đã bị nhận "
            f"nhầm thành viền ly (lọt qua điều kiện lọc do trùng hợp). Thử: giảm "
            f"min_angular_coverage nếu viền ly bị đứt nặng, hoặc kiểm tra lại wall_thickness_mm."
        )
    if inner_diam_px < min_inner_outer_ratio_floor * outer_diam_px:
        raise RuntimeError(
            f"Tỉ lệ mép trong/mép ngoài ({inner_diam_px / outer_diam_px:.2f}) nhỏ bất thường "
            f"về mặt vật lý (ngưỡng tối thiểu {min_inner_outer_ratio_floor:.2f}). "
            f"Kiểm tra lại detect_type='{detect_type}' và ảnh gốc — nhiều khả năng mép ngoài "
            f"hoặc mép trong đã bị bắt nhầm."
        )

    # Vá 7: elip đang ở tọa độ khung xử lý (đã resize) — quy đổi ngược về đúng tọa độ/
    # kích thước ảnh GỐC trước khi dùng cho crop/mask/overlay/pixels_per_mm bên dưới.
    if scale_factor != 1.0:
        inv_scale = 1.0 / scale_factor
        outer_ell = _scale_ellipse(outer_ell, inv_scale)
        inner_ell = _scale_ellipse(inner_ell, inv_scale)
        (cx, cy), (MA_out, ma_out), angle = outer_ell
        outer_diam_px = float(max(MA_out, ma_out))
        inner_diam_px = float(max(inner_ell[1]))

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

    # 6. Tạo sẵn ảnh overlay gốc có vẽ khoanh viền + chú thích chẩn đoán
    temp_info = {
        "pixels_per_mm": float(pixels_per_mm),
        "inner_w_px": int(round(inner_diam_px)),
        "outer_w_px": int(round(outer_diam_px)),
        "inner_ellipse": inner_ell,
        "outer_ellipse": outer_ell,
        "detect_type": detect_type,
        "outer_confidence": outer_confidence,
        "inner_confidence": inner_confidence,
    }
    overlay_bgr = draw_container_overlay(img_bgr.copy(), temp_info)

    if verbose:
        print(
            f"[final] detect_type={detect_type} pixels_per_mm={pixels_per_mm:.2f} "
            f"outer_confidence={outer_confidence} inner_confidence={inner_confidence}"
        )

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
    }


def draw_container_overlay(
    img_bgr: np.ndarray,
    container_info: Dict[str, Any],
) -> np.ndarray:
    """
    Vẽ viền Mép Trong (Màu Xanh Lá Đậm) và Mép Ngoài (Màu Cam) lên ảnh để kiểm tra
    trực quan, kèm chú thích detect_type + độ tin cậy RANSAC (nếu có) để tự chẩn
    đoán lỗi khi xem ảnh debug thay vì phải đọc log riêng.

    Vá 6: mỗi elip được vẽ với một lớp viền đệm đen (halo) dày hơn phía dưới trước
    khi vẽ màu lên trên — giúp đường elip luôn nổi rõ bất kể nó nằm trên nền sáng,
    tối hay có màu gần trùng (đây là lý do mép ngoài gần như "biến mất" trong ảnh
    debug trước đây).
    """
    vis = img_bgr.copy()

    def _draw_with_halo(ell, color, thickness):
        cv2.ellipse(vis, ell, (0, 0, 0), thickness + 3, lineType=cv2.LINE_AA)
        cv2.ellipse(vis, ell, color, thickness, lineType=cv2.LINE_AA)

    # 1. Vẽ Mép Ngoài (Màu Cam)
    outer_ell = container_info.get("outer_ellipse")
    if outer_ell is not None:
        _draw_with_halo(outer_ell, (0, 165, 255), 2)

    # 2. Vẽ Mép Trong (Màu Xanh Lá Đậm - Chuẩn)
    inner_ell = container_info.get("inner_ellipse")
    if inner_ell is not None:
        _draw_with_halo(inner_ell, (0, 255, 0), 3)
        cx = int(round(inner_ell[0][0]))
        cy = int(round(inner_ell[0][1]))
        cv2.circle(vis, (cx, cy), 5, (0, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

    # 3. Ghi chú thích với khung nền bán trong suốt để dễ đọc
    scale_val = container_info["pixels_per_mm"]
    in_px = container_info["inner_w_px"]
    out_px = container_info["outer_w_px"]
    detect_type = container_info.get("detect_type", "n/a")
    outer_conf = container_info.get("outer_confidence")
    inner_conf = container_info.get("inner_confidence")

    def _fmt_conf(v):
        return "n/a" if v is None else f"{v * 100:.0f}%"

    txt1 = f"[MEP TRONG]: {in_px}px | Scale = {scale_val:.2f} px/mm"
    txt2 = f"[MEP NGOAI]: {out_px}px"
    txt3 = f"type={detect_type} | conf(out)={_fmt_conf(outer_conf)} conf(in)={_fmt_conf(inner_conf)}"

    # Vẽ nền hộp đen mờ cho text (thêm 1 dòng so với bản gốc)
    overlay_box = vis.copy()
    cv2.rectangle(overlay_box, (15, 15), (620, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay_box, 0.6, vis, 0.4, 0, vis)

    cv2.putText(vis, txt1, (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, txt2, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, txt3, (25, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return vis