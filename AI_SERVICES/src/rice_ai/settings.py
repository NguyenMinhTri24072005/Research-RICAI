"""Settings and path configuration for Rice Vision AI."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import dotenv_values


class Settings:
    """Quản lý cấu hình, biến môi trường và đường dẫn cho toàn bộ hệ thống Rice AI.
    
    Quy tắc ưu tiên (Precedence):
    1. Process Environment (os.environ)
    2. .env file tại AI_SERVICES/.env (hoặc custom env_file)
    3. Giá trị mặc định (Defaults)
    
    Mọi đường dẫn tương đối luôn được giải quyết từ thư mục gốc AI_SERVICES,
    không phụ thuộc vào working directory khi thực thi.
    """

    def __init__(self, env_file: Optional[Union[str, Path]] = None):
        # Xác định gốc AI_SERVICES: src/rice_ai/settings.py -> AI_SERVICES
        self.ai_services_root: Path = Path(__file__).resolve().parent.parent.parent
        self.project_root: Path = self.ai_services_root.parent

        # Đọc .env bằng dotenv_values để không làm ô nhiễm toàn cục os.environ
        target_env = Path(env_file) if env_file else self.ai_services_root / ".env"
        file_env: Dict[str, Optional[str]] = {}
        if target_env.exists():
            file_env = dotenv_values(dotenv_path=target_env)

        def get_val(key: str, default: Optional[str] = None) -> Optional[str]:
            if key in os.environ:
                return os.environ[key]
            if key in file_env and file_env[key] is not None:
                return file_env[key]
            return default

        # Đọc cấu hình mô hình từ môi trường
        self.yolo_model_path_str: Optional[str] = get_val("YOLO_MODEL_PATH")
        self.cnn_model_path_str: Optional[str] = get_val("CNN_MODEL_PATH")
        self.regression_model_dir_str: Optional[str] = get_val("REGRESSION_MODEL_DIR")

        # Cấu hình thiết bị GPU/CPU cho YOLO
        raw_yolo_dev = get_val("YOLO_DEVICE", "auto")
        self.yolo_device_str: str = raw_yolo_dev.strip().lower() if raw_yolo_dev else "auto"

        # Cổng mạng và tham số server
        self.port_ai: int = int(get_val("PORT_AI", "8000"))
        self.host_ai: str = get_val("HOST_AI", "0.0.0.0")
        
        # Concurrency check: chỉ chấp nhận 1 trong đợt này
        raw_concurrent = get_val("MAX_CONCURRENT_INFERENCES", "1")
        try:
            self.max_concurrent_inferences: int = int(raw_concurrent)
        except ValueError:
            raise ValueError(f"MAX_CONCURRENT_INFERENCES phải là số nguyên, nhận: '{raw_concurrent}'")
        if self.max_concurrent_inferences != 1:
            raise ValueError(
                f"MAX_CONCURRENT_INFERENCES hiện tại chỉ hỗ trợ giá trị 1 để bảo vệ an toàn "
                f"tài nguyên suy luận (nhận: {self.max_concurrent_inferences})."
            )

        # Hằng số tính toán vật lý & thị giác máy tính
        self.packing_fraction_geometry: float = self._float_in_range(get_val("PACKING_FRACTION_GEOMETRY", "0.55"), 0.0, 1.0, "PACKING_FRACTION_GEOMETRY")
        self.packing_fraction_feature_hybrid: float = self._float_in_range(
            get_val("PACKING_FRACTION_FEATURE_HYBRID", "0.62"), 0.0, 1.0,
            "PACKING_FRACTION_FEATURE_HYBRID",
        )
        self.sahi_slice_height: int = 640
        self.sahi_slice_width: int = 640
        self.sahi_overlap_height_ratio: float = self._float_in_range(
            get_val("SAHI_OVERLAP_RATIO", "0.25"), 0.0, 0.99, "SAHI_OVERLAP_RATIO"
        )
        self.sahi_overlap_width_ratio: float = self.sahi_overlap_height_ratio
        self.sahi_conf_threshold: float = self._float_in_range(
            get_val("SAHI_CONFIDENCE_THRESHOLD", "0.50"), 0.0, 1.0, "SAHI_CONFIDENCE_THRESHOLD"
        )
        self.cleaner_kernel_size: int = 3
        self.cleaner_neck_ratio: float = 0.15
        self.cleaner_step1_open_ksize: int = int(get_val("CLEANER_STEP1_OPEN_KSIZE", "5"))
        self.cleaner_step1_min_area: int = int(get_val("CLEANER_STEP1_MIN_AREA", "35"))
        self.cleaner_step1_centrality_weight: float = self._float_in_range(get_val("CLEANER_STEP1_CENTRALITY_WEIGHT", "2.5"), 0.0, 10.0, "CLEANER_STEP1_CENTRALITY_WEIGHT")
        self.cleaner_step2_open_ksize: int = int(get_val("CLEANER_STEP2_OPEN_KSIZE", "3"))
        self.cleaner_step2_min_area: int = int(get_val("CLEANER_STEP2_MIN_AREA", "25"))
        self.cleaner_step2_centrality_weight: float = self._float_in_range(get_val("CLEANER_STEP2_CENTRALITY_WEIGHT", "2.2"), 0.0, 10.0, "CLEANER_STEP2_CENTRALITY_WEIGHT")
        self.cnn_whole_confidence: float = self._float_in_range(get_val("CNN_WHOLE_CONFIDENCE", "0.90"), 0.0, 1.0, "CNN_WHOLE_CONFIDENCE")
        self.enable_size_filter: bool = (get_val("ENABLE_SIZE_FILTER", "true") or "true").strip().lower() in {"1", "true", "yes", "on"}
        self.size_filter_k: float = self._float_in_range(get_val("SIZE_FILTER_K", "0.10"), 0.0, 10.0, "SIZE_FILTER_K")
        self.size_filter_min_samples: int = int(get_val("SIZE_FILTER_MIN_SAMPLES", "8"))
        self.results_root_str: str = str(get_val("RESULTS_ROOT", "/content/pipeline_inference_results") or "").strip()
        self.save_inference_artifacts: bool = (
            str(get_val("SAVE_INFERENCE_ARTIFACTS", "true") or "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )
    @staticmethod
    def _float_in_range(raw: Optional[str], minimum: float, maximum: float, name: str) -> float:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} phải là số thực hợp lệ, nhận: {raw!r}") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} phải trong [{minimum}, {maximum}], nhận: {value}")
        return value

    def get_yolo_device(self) -> str:
        """Giải quyết thiết bị thực thi cho YOLO theo YOLO_DEVICE.
        
        Quy tắc:
        - 'auto' / '': ưu tiên 'cuda:0' nếu torch.cuda.is_available(), ngược lại 'cpu'.
        - 'cuda' / 'cuda:N': nếu torch.cuda.is_available() thì trả về device đó,
          nếu không có CUDA ném RuntimeError rõ ràng (không âm thầm fallback).
        - 'cpu': trả về 'cpu'.
        - khác: ném ValueError.
        """
        import torch
        dev = self.yolo_device_str
        if dev in ("auto", ""):
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        elif dev == "cpu":
            return "cpu"
        elif dev.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Cấu hình YOLO_DEVICE='{self.yolo_device_str}' yêu cầu GPU CUDA nhưng "
                    "CUDA không khả dụng trên hệ thống hiện tại."
                )
            if dev == "cuda":
                return "cuda:0"
            return dev
        else:
            raise ValueError(
                f"YOLO_DEVICE='{self.yolo_device_str}' không hợp lệ. "
                "Các giá trị hợp lệ: auto, cpu, cuda, cuda:N"
            )

    def resolve_path(self, path_input: Union[str, Path]) -> Path:
        """Giải quyết đường dẫn: đường dẫn tương đối luôn tính từ ai_services_root."""
        p = Path(path_input)
        if p.is_absolute():
            return p.resolve()
        return (self.ai_services_root / p).resolve()

    def get_yolo_path(self) -> Path:
        """Lấy đường dẫn tệp mô hình YOLO hợp lệ.
        
        Nếu YOLO_MODEL_PATH được cấu hình tường minh:
          - Bắt buộc tệp phải tồn tại và là file, nếu không ném FileNotFoundError (không fallback ngầm).
        Nếu không cấu hình:
          - Tìm kiếm theo danh sách candidate mặc định.
        """
        if self.yolo_model_path_str and self.yolo_model_path_str.strip():
            explicit_path = self.resolve_path(self.yolo_model_path_str.strip())
            if not explicit_path.is_file():
                raise FileNotFoundError(
                    f"Tệp YOLO được cấu hình tường minh không tồn tại hoặc không phải là file: {explicit_path} "
                    f"(từ YOLO_MODEL_PATH='{self.yolo_model_path_str}')"
                )
            return explicit_path

        # Candidates mặc định
        candidates: List[Path] = [
            self.ai_services_root / "artifacts" / "yolo" / "rice_segmentation" / "best.pt",
            self.project_root / "RESULTS" / "all-new-data-v1.yolov8_yolov8s-seg_trained" / "weights" / "best.pt",
            self.project_root / "RESULTS" / "35_special_images_segmentation.v1i.yolov8_v1_trained" / "weights" / "best.pt",
            self.project_root / "MODELS" / "yolo26s-seg.pt",
            Path("/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/RESULTS/35_special_images_segmentation.v1i.yolov8_v1_trained/weights/best.pt"),
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()

        raise FileNotFoundError(
            "Không tìm thấy mô hình YOLO trong các vị trí mặc định. "
            "Vui lòng thiết lập YOLO_MODEL_PATH trong file .env"
        )

    def get_cnn_path(self) -> Path:
        """Lấy đường dẫn tệp mô hình CNN hợp lệ.
        
        Nếu CNN_MODEL_PATH được cấu hình tường minh:
          - Bắt buộc tệp phải tồn tại và là file, nếu không ném FileNotFoundError (không fallback ngầm).
        Nếu không cấu hình:
          - Tìm kiếm theo danh sách candidate mặc định.
        """
        if self.cnn_model_path_str and self.cnn_model_path_str.strip():
            explicit_path = self.resolve_path(self.cnn_model_path_str.strip())
            if not explicit_path.is_file():
                raise FileNotFoundError(
                    f"Tệp CNN được cấu hình tường minh không tồn tại hoặc không phải là file: {explicit_path} "
                    f"(từ CNN_MODEL_PATH='{self.cnn_model_path_str}')"
                )
            return explicit_path

        # Candidates mặc định
        candidates: List[Path] = [
            self.ai_services_root / "artifacts" / "cnn" / "grain_classifier" / "best.keras",
            self.project_root / "RESULTS" / "CNN_DenseNet121_Trained" / "best_v3_step2.keras",
            self.project_root / "RESULTS" / "CNN_DenseNet121_Trained" / "best_rice_densenet121.keras",
            self.project_root / "RESULTS" / "35_special_images_segmentation.v1i.yolov8_v1_trained" / "rice_grain_classifier_cnn.h5",
            Path("/content/drive/MyDrive/NGHIÊN CỨU KHOA HỌC/GROUP_MEMBERS/NGUYEN MINH TRI/MAIN_SOURCES/DETECTED_OBJECTS/35_special_images_segmentation.v1i.yolov8_v1_trained/rice_grain_classifier_cnn.h5"),
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()

        raise FileNotFoundError(
            "Không tìm thấy mô hình CNN trong các vị trí mặc định. "
            "Vui lòng thiết lập CNN_MODEL_PATH trong file .env"
        )

    def get_regression_dir(self) -> Path:
        """Lấy đường dẫn thư mục bundle mô hình hồi quy.
        
        Nếu REGRESSION_MODEL_DIR được cấu hình tường minh:
          - Bắt buộc thư mục phải tồn tại, nếu không ném FileNotFoundError/NotADirectoryError.
        Nếu không cấu hình:
          - Trỏ về thư mục mặc định chứa ExtraTrees đang hoạt động: LINEAR_REGRESSION_MODEL/models.
        """
        if self.regression_model_dir_str and self.regression_model_dir_str.strip():
            explicit_dir = self.resolve_path(self.regression_model_dir_str.strip())
            if not explicit_dir.exists():
                raise FileNotFoundError(
                    f"Thư mục hồi quy được cấu hình tường minh không tồn tại: {explicit_dir} "
                    f"(từ REGRESSION_MODEL_DIR='{self.regression_model_dir_str}')"
                )
            if not explicit_dir.is_dir():
                raise NotADirectoryError(
                    f"Đường dẫn REGRESSION_MODEL_DIR không phải thư mục: {explicit_dir}"
                )
            return explicit_dir

        # Mặc định tương thích: thư mục models của LINEAR_REGRESSION_MODEL
        default_dir = (self.project_root / "LINEAR_REGRESSION_MODEL" / "models").resolve()
        if not default_dir.exists():
            raise FileNotFoundError(
                f"Thư mục regression mặc định không tồn tại: {default_dir}. "
                "Vui lòng thiết lập REGRESSION_MODEL_DIR trong .env"
            )
        return default_dir

    def get_results_root(self) -> Path:
        """Resolve the per-request inference result directory."""
        if not self.results_root_str:
            raise ValueError("RESULTS_ROOT không được để trống khi lưu artifact.")
        return self.resolve_path(self.results_root_str)

    @property
    def port(self) -> int:
        """Cổng mạng của AI Inference Server (alias của port_ai)."""
        return self.port_ai

    @property
    def host(self) -> str:
        """Địa chỉ host của AI Inference Server (alias của host_ai)."""
        return self.host_ai