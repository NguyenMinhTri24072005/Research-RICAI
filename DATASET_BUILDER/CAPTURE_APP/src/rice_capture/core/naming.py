from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NamingConfig:
    prefix: str = "M"
    sample_number: int = 1
    sample_digits: int = 4

    def validate(self) -> None:
        self.prefix = self.prefix.strip().upper()
        if self.prefix != "M":
            raise ValueError("Tiền tố ảnh được cố định là M.")
        if self.sample_number < 1:
            raise ValueError("Số thứ tự ảnh phải từ 1 trở lên.")
        if self.sample_digits != 4:
            raise ValueError("Số chữ số sau M được cố định là 4 (M0001, M0002, ...).")
        if self.sample_number >= 10**self.sample_digits:
            raise ValueError(
                f"Số {self.sample_number} không vừa {self.sample_digits} chữ số. "
                "Kho mã hợp lệ hiện tại là M0001-M9999."
            )

    @property
    def sample_code(self) -> str:
        self.validate()
        return f"{self.prefix}{self.sample_number:0{self.sample_digits}d}"

    @property
    def sample_id(self) -> str:
        return self.sample_code

    def next_id(self) -> None:
        self.sample_number += 1

