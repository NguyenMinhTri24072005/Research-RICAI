from __future__ import annotations

from datetime import datetime
from typing import Any


MANUAL_HEADERS = (
    "Sample_ID",
    "Weight_g",
    "Container_Height_mm",
    "Inner_Diameter_mm",
    "Empty_Height_mm",
    "Rice_Height_mm",
    "Actual_Count",
    "Capture_Timestamp",
)


def parse_number(value: Any, field_label: str, *, integer: bool = False) -> float | int:
    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError(f"{field_label} không được để trống.")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{field_label} phải là một số hợp lệ.") from exc
    if number < 0:
        raise ValueError(f"{field_label} không được âm.")
    if integer:
        if not number.is_integer():
            raise ValueError(f"{field_label} phải là số nguyên.")
        return int(number)
    return number


def parse_capture_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Thời điểm lưu không được để trống.")
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Thời điểm lưu không đúng định dạng ISO 8601.") from exc
    if timestamp.tzinfo is None:
        raise ValueError("Thời điểm lưu phải có múi giờ.")
    return timestamp.isoformat(timespec="seconds")


def make_manual_record(sample_id: str, values: dict[str, Any]) -> dict[str, Any]:
    container_height = parse_number(values.get("Container_Height_mm"), "Chiều cao ly")
    empty_height = parse_number(values.get("Empty_Height_mm"), "Chiều cao khoảng trống")
    if empty_height > container_height:
        raise ValueError("Chiều cao khoảng trống không thể lớn hơn chiều cao ly.")

    return {
        "Sample_ID": sample_id.strip().upper(),
        "Weight_g": parse_number(values.get("Weight_g"), "Khối lượng"),
        "Container_Height_mm": container_height,
        "Inner_Diameter_mm": parse_number(values.get("Inner_Diameter_mm"), "Đường kính trong"),
        "Empty_Height_mm": empty_height,
        "Rice_Height_mm": round(float(container_height) - float(empty_height), 6),
        "Actual_Count": parse_number(values.get("Actual_Count"), "Số hạt thực tế", integer=True),
        "Capture_Timestamp": parse_capture_timestamp(values.get("Capture_Timestamp")),
    }
