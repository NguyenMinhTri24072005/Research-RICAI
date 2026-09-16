#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
MODULE 3: PHÂN LOẠI PHẨM CẤP HẠT LÚA (DENSENET121 CNN CLASSIFIER)
===============================================================================
Mục đích:
  - Nạp mô hình CNN đã huấn luyện (hỗ trợ .keras và .h5).
  - Phân loại từng hạt lúa thành 'hat_nguyen' (hạt nguyên) hoặc 'hat_khuyet_tat' (hạt khuyết tật).
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class GrainClassifier:
    """Class quản lý mô hình phân loại hạt lúa bằng mạng học sâu DenseNet121."""

    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy trọng số model CNN tại: {self.model_path}")

        # Thứ tự chuẩn theo alphabet từ image_dataset_from_directory
        self.class_names = class_names or ["hat_khuyet_tat", "hat_nguyen"]
        self.target_size = target_size
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Nạp model với lazy import tensorflow."""
        import tensorflow as tf
        from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

        print(f"📦 Đang nạp mô hình CNN từ: {self.model_path}...")
        self.model = tf.keras.models.load_model(
            str(self.model_path),
            custom_objects={"preprocess_input": densenet_preprocess},
            compile=False,
            safe_mode=False,
        )
        print("✅ Nạp mô hình CNN thành công!")

    def preprocess_image(self, img_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Tiền xử lý ảnh hạt lúa về đúng kích thước chuẩn (224, 224) bằng Letterbox Padding.
        
        LƯU Ý QUAN TRỌNG:
        1. Giữ nguyên tỷ lệ khung hình (Aspect Ratio) của hạt lúa, chèn viền đen padding đối xứng.
        2. Mô hình DenseNet121 đã có sẵn layer Lambda(preprocess_input) bên trong cấu trúc mạng,
           do đó chỉ cần tensor RGB [0, 255] float32, KHÔNG gọi preprocess_input 2 lần!
        """
        if isinstance(img_input, (str, Path)):
            img = cv2.imread(str(img_input), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Không đọc được ảnh: {img_input}")
        else:
            img = img_input.copy()

        # Nếu là RGBA 4 kênh, gán vùng trong suốt thành màu đen
        if len(img.shape) == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3].copy()
            alpha = img[:, :, 3]
            bgr[alpha == 0] = [0, 0, 0]
        else:
            bgr = img.copy()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w = rgb.shape[:2]
        target_h, target_w = self.target_size

        if h == 0 or w == 0:
            return np.zeros((1, target_h, target_w, 3), dtype=np.float32)

        # Letterbox: Tính hệ số co giãn giữ tỷ lệ
        scale = min(target_h / h, target_w / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

        padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
        ph, pw = (target_h - nh) // 2, (target_w - nw) // 2
        padded[ph:ph + nh, pw:pw + nw] = resized

        tensor = np.expand_dims(padded, axis=0)
        return tensor

    def predict(
        self,
        img_input: Union[str, Path, np.ndarray],
        hat_nguyen_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Dự đoán nhãn phẩm cấp của 1 hạt lúa.

        Parameters
        ----------
        img_input: Ảnh đầu vào dạng đường dẫn file hoặc numpy ndarray (BGR/RGBA)
        hat_nguyen_threshold: Ngưỡng xác suất để quyết định là hạt nguyên (mặc định 0.50).

        Returns
        -------
        dict:
            - label: str ('hat_nguyen' hoặc 'hat_khuyet_tat')
            - confidence: float (0.0 đến 1.0)
            - probabilities: dict ({'hat_khuyet_tat': float, 'hat_nguyen': float})
        """
        if self.model is None:
            self._load_model()

        tensor = self.preprocess_image(img_input)
        preds = self.model.predict(tensor, verbose=0)[0]

        probs_dict = {
            self.class_names[i] if i < len(self.class_names) else f"class_{i}": float(preds[i])
            for i in range(len(preds))
        }

        # Nếu có nhãn hat_nguyen và áp dụng ngưỡng riêng
        if "hat_nguyen" in probs_dict and "hat_khuyet_tat" in probs_dict:
            p_nguyen = probs_dict["hat_nguyen"]
            if p_nguyen >= hat_nguyen_threshold:
                label = "hat_nguyen"
                conf = p_nguyen
            else:
                label = "hat_khuyet_tat"
                conf = probs_dict["hat_khuyet_tat"]
        else:
            pred_idx = int(np.argmax(preds))
            label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "unknown"
            conf = float(preds[pred_idx])

        return {
            "label": label,
            "confidence": conf,
            "probabilities": probs_dict,
        }

    def filter_grains(
        self,
        grains_list: List[Dict[str, Any]],
        target_label: str = "hat_nguyen",
        min_conf: float = 0.50,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Phân loại danh sách hạt lúa thành 2 nhóm: nhóm đạt chuẩn (target_label) và nhóm còn lại.
        Tự động chạy suy luận theo mẻ (Batch Inference) trên GPU để đạt tốc độ tối đa.

        Returns
        -------
        tuple: (list_target_grains, list_other_grains)
        """
        if not grains_list:
            return [], []

        if self.model is None:
            self._load_model()

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        try:
            # 1. Gom toàn bộ ảnh hạt lúa vào 1 mẻ (Batch Tensor)
            tensors = [
                self.preprocess_image(g["crop_rgba"] if "crop_rgba" in g else g.get("crop_bgr"))
                for g in grains_list
            ]
            batch_input = np.concatenate(tensors, axis=0)

            # 2. Suy luận 1 lần toàn bộ trên GPU (Batch forward pass)
            all_preds = self.model.predict(batch_input, batch_size=32, verbose=0)

            # 3. Phân phối kết quả phân loại
            for i, grain in enumerate(grains_list):
                preds = all_preds[i]
                probs_dict = {
                    self.class_names[c_idx] if c_idx < len(self.class_names) else f"class_{c_idx}": float(preds[c_idx])
                    for c_idx in range(len(preds))
                }

                if "hat_nguyen" in probs_dict and "hat_khuyet_tat" in probs_dict:
                    p_nguyen = probs_dict["hat_nguyen"]
                    if p_nguyen >= min_conf:
                        label = "hat_nguyen"
                        conf = p_nguyen
                    else:
                        label = "hat_khuyet_tat"
                        conf = probs_dict["hat_khuyet_tat"]
                else:
                    pred_idx = int(np.argmax(preds))
                    label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "unknown"
                    conf = float(preds[pred_idx])

                grain_info = dict(grain)
                grain_info["predicted_label"] = label
                grain_info["confidence"] = conf
                grain_info["probabilities"] = probs_dict

                if label == target_label:
                    accepted.append(grain_info)
                else:
                    rejected.append(grain_info)

        except Exception as e:
            # Fallback chạy từng hạt nếu có lỗi bộ nhớ
            for grain in grains_list:
                img = grain["crop_rgba"] if "crop_rgba" in grain else grain.get("crop_bgr")
                pred_res = self.predict(img, hat_nguyen_threshold=min_conf)

                grain_info = dict(grain)
                grain_info["predicted_label"] = pred_res["label"]
                grain_info["confidence"] = pred_res["confidence"]
                grain_info["probabilities"] = pred_res["probabilities"]

                if pred_res["label"] == target_label:
                    accepted.append(grain_info)
                else:
                    rejected.append(grain_info)

        return accepted, rejected
