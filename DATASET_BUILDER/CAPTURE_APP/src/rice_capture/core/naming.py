from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NamingConfig:
    prefix: str = "M"
    sample_number: int = 1
    sample_digits: int = 3

    def validate(self) -> None:
        self.prefix = self.prefix.strip().upper()
        if self.prefix != "M":
            raise ValueError("Tiền tố ảnh được cố định là M.")
        if self.sample_number < 1:
            raise ValueError("Số thứ tự ảnh phải từ 1 trở lên.")
        if not 1 <= self.sample_digits <= 8:
            raise ValueError("Số chữ số sau M phải nằm trong khoảng 1-8.")
        if self.sample_number >= 10**self.sample_digits:
            raise ValueError(
                f"Số {self.sample_number} không vừa {self.sample_digits} chữ số. "
                "Hãy tăng số chữ số sau M."
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

