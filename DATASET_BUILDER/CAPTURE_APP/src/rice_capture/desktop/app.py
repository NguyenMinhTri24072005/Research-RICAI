#!/usr/bin/env python3
"""Desktop camera capture and manual-data entry app for DATASET_BUILDER."""

from __future__ import annotations

import os
import queue
import shutil
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from rice_capture.core import MANUAL_HEADERS, NamingConfig, make_manual_record
from rice_capture.server import MobileServerController, discover_lan_addresses
from rice_capture.services import CaptureCoordinator
from rice_capture.storage import SettingsStore, WorkbookStore


PACKAGE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parents[3]
DATASET_BUILDER_DIR = APP_DIR.parent
DEFAULT_IMAGE_DIR = DATASET_BUILDER_DIR / "1_Raw_Images"
DEFAULT_WORKBOOK = DATASET_BUILDER_DIR / "2_Manual_Records" / "manual_data.xlsx"
RUNTIME_DIR = APP_DIR / "runtime"
SETTINGS_FILE = RUNTIME_DIR / "settings.json"
LEGACY_SETTINGS_FILE = APP_DIR / ".dataset_capture_settings.json"
WEB_ROOT = PACKAGE_DIR / "web"


class DatasetCaptureApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rice Dataset Capture")
        self.root.geometry("1400x880")
        self.root.minsize(1120, 720)

        self.settings_store = SettingsStore(SETTINGS_FILE)
        saved = self.settings_store.load()
        if not saved and LEGACY_SETTINGS_FILE.exists():
            saved = SettingsStore(LEGACY_SETTINGS_FILE).load()

        self.cv2 = None
        self.pil_image = None
        self.image_tk = None
        self.camera = None
        self.camera_job = None
        self.current_frame = None
        self.captured_frame = None
        self.preview_photo = None
        self.workbook_store: WorkbookStore | None = None
        self.mobile_events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.coordinator = CaptureCoordinator(self.mobile_events)
        self.mobile_server = MobileServerController(
            self.coordinator,
            WEB_ROOT,
            port=int(saved.get("mobile_port", 8765)),
            cert_dir=RUNTIME_DIR / "certificates",
        )
        self.mobile_nodes: dict[str, str] = {}
        self.mobile_qr_photo = None
        self.coordinator_sync_job = None

        self.image_dir_var = tk.StringVar(value=saved.get("image_dir", str(DEFAULT_IMAGE_DIR)))
        self.workbook_var = tk.StringVar(value=saved.get("workbook", str(DEFAULT_WORKBOOK)))
        self.camera_source_var = tk.StringVar(value=str(saved.get("camera_source", "0")))
        self.camera_backend_var = tk.StringVar(value=saved.get("camera_backend", "Auto"))

        naming = saved.get("naming", {})
        self.prefix_var = tk.StringVar(value="M")
        self.sample_number_var = tk.IntVar(value=int(naming.get("sample_number", 1)))
        self.sample_digits_var = tk.IntVar(value=int(naming.get("sample_digits", 3)))

        manual = saved.get("manual", {})
        self.manual_vars = {
            "Weight_g": tk.StringVar(value=str(manual.get("Weight_g", ""))),
            "Container_Height_mm": tk.StringVar(value=str(manual.get("Container_Height_mm", ""))),
            "Inner_Diameter_mm": tk.StringVar(value=str(manual.get("Inner_Diameter_mm", ""))),
            "Empty_Height_mm": tk.StringVar(value=str(manual.get("Empty_Height_mm", ""))),
            "Actual_Count": tk.StringVar(value=str(manual.get("Actual_Count", ""))),
        }
        self.rice_height_var = tk.StringVar(value="—")
        self.auto_save_var = tk.BooleanVar(value=bool(saved.get("auto_save", True)))
        self.naming_preview_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Sẵn sàng")
        addresses = discover_lan_addresses()
        saved_address = saved.get("mobile_address", addresses[0])
        self.mobile_address_var = tk.StringVar(value=saved_address if saved_address in addresses else addresses[0])
        self.mobile_port_var = tk.IntVar(value=int(saved.get("mobile_port", 8765)))
        self.mobile_url_var = tk.StringVar(value="Chưa khởi động")
        self.mobile_status_var = tk.StringVar(value="Máy chủ điện thoại đang tắt")
        self.mobile_node_var = tk.StringVar(value="Chưa có điện thoại kết nối")

        self._configure_style()
        self._build_ui()
        self._bind_updates()
        self.load_workbook(silent=True)
        self._update_naming_preview()
        self._update_rice_height()
        self._schedule_coordinator_sync()
        self.root.after(200, self._poll_mobile_events)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Sub.TLabel", foreground="#4f5b57")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        style.configure("Treeview", rowheight=28)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        title_row = ttk.Frame(outer)
        title_row.pack(fill="x", pady=(0, 10))
        ttk.Label(title_row, text="Rice Dataset Capture", style="Title.TLabel").pack(side="left")
        ttk.Label(
            title_row,
            text="Chụp ảnh • giữ nguyên thông số ly • ghi trực tiếp vào Excel",
            style="Sub.TLabel",
        ).pack(side="left", padx=18)

        self._build_paths(outer)
        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True, pady=(10, 8))
        capture_tab = ttk.Frame(notebook, padding=8)
        excel_tab = ttk.Frame(notebook, padding=8)
        mobile_tab = ttk.Frame(notebook, padding=8)
        notebook.add(capture_tab, text="  Chụp & nhập dữ liệu  ")
        notebook.add(excel_tab, text="  Bảng Excel  ")
        notebook.add(mobile_tab, text="  Thiết bị di động  ")
        self._build_capture_tab(capture_tab)
        self._build_excel_tab(excel_tab)
        self._build_mobile_tab(mobile_tab)

        status = ttk.Frame(outer)
        status.pack(fill="x")
        ttk.Separator(status, orient="horizontal").pack(fill="x", pady=(0, 7))
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        ttk.Label(status, text="Ảnh chưa lưu" if self.captured_frame is None else "Đã chụp").pack(side="right")

    def _build_paths(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Nơi lưu dữ liệu", padding=8)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Thư mục ảnh:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(box, textvariable=self.image_dir_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(box, text="Chọn…", command=self.choose_image_dir).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(box, text="File Excel:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(7, 0))
        ttk.Entry(box, textvariable=self.workbook_var).grid(row=1, column=1, sticky="ew", pady=(7, 0))
        ttk.Button(box, text="Chọn/Mở…", command=self.choose_workbook).grid(row=1, column=2, padx=(8, 0), pady=(7, 0))
        ttk.Button(box, text="Nạp lại", command=self.load_workbook).grid(row=1, column=3, padx=(6, 0), pady=(7, 0))

    def _build_capture_tab(self, parent: ttk.Frame) -> None:
        pane = ttk.Panedwindow(parent, orient="horizontal")
        pane.pack(fill="both", expand=True)
        camera_side = ttk.Frame(pane, padding=(0, 0, 8, 0))
        form_side = ttk.Frame(pane, padding=(8, 0, 0, 0))
        pane.add(camera_side, weight=3)
        pane.add(form_side, weight=2)

        controls = ttk.LabelFrame(camera_side, text="Camera", padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="Nguồn:").pack(side="left")
        self.camera_combo = ttk.Combobox(
            controls, textvariable=self.camera_source_var, values=("0", "1", "2", "3"), width=18
        )
        self.camera_combo.pack(side="left", padx=6)
        ttk.Label(controls, text="Backend:").pack(side="left", padx=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.camera_backend_var,
            values=("Auto", "DirectShow", "MSMF"),
            state="readonly",
            width=12,
        ).pack(side="left", padx=6)
        ttk.Button(controls, text="Quét", command=self.scan_cameras).pack(side="left", padx=3)
        ttk.Button(controls, text="Bật", command=self.start_camera).pack(side="left", padx=3)
        ttk.Button(controls, text="Tắt", command=self.stop_camera).pack(side="left", padx=3)

        preview_box = ttk.Frame(camera_side, relief="sunken", borderwidth=1)
        preview_box.pack(fill="both", expand=True, pady=8)
        self.preview_label = tk.Label(
            preview_box,
            text="Chọn nguồn camera rồi bấm Bật",
            anchor="center",
            background="#18201d",
            foreground="white",
        )
        self.preview_label.pack(fill="both", expand=True)

        capture_bar = ttk.Frame(camera_side)
        capture_bar.pack(fill="x")
        ttk.Button(capture_bar, text="CHỤP KHUNG HÌNH", style="Primary.TButton", command=self.capture_frame).pack(
            side="left"
        )
        ttk.Button(capture_bar, text="Chụp lại", command=self.capture_frame).pack(side="left", padx=8)
        self.capture_state_label = ttk.Label(capture_bar, text="Chưa chụp")
        self.capture_state_label.pack(side="left", padx=10)

        naming_box = ttk.LabelFrame(form_side, text="Mã ảnh tuần tự", padding=10)
        naming_box.pack(fill="x")
        for column in range(4):
            naming_box.columnconfigure(column, weight=1 if column in {1, 3} else 0)

        ttk.Label(naming_box, text="Tiền tố cố định").grid(row=0, column=0, sticky="w")
        ttk.Label(naming_box, text="M", font=("Consolas", 11, "bold")).grid(
            row=0, column=1, sticky="w", padx=(6, 12)
        )
        ttk.Label(naming_box, text="Số ảnh kế tiếp").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(naming_box, from_=1, to=99999999, textvariable=self.sample_number_var, width=10).grid(
            row=0, column=3, sticky="ew", padx=(6, 0)
        )
        ttk.Label(naming_box, text="Số chữ số sau M").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(naming_box, from_=1, to=8, textvariable=self.sample_digits_var, width=8).grid(
            row=1, column=1, sticky="ew", padx=(6, 12), pady=(8, 0)
        )
        ttk.Label(naming_box, text="Ví dụ: 3 → M001; 4 → M0001", style="Sub.TLabel").grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(6, 0), pady=(8, 0)
        )
        ttk.Label(naming_box, text="ID kế tiếp:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Label(naming_box, textvariable=self.naming_preview_var, font=("Consolas", 12, "bold")).grid(
            row=2, column=1, columnspan=3, sticky="w", padx=(6, 0), pady=(10, 0)
        )
        naming_actions = ttk.Frame(naming_box)
        naming_actions.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(naming_actions, text="Tăng ID", command=self.next_image_id).pack(side="left")
        ttk.Button(naming_actions, text="Tìm ID trống", command=self.find_next_available).pack(side="left", padx=7)

        manual_box = ttk.LabelFrame(form_side, text="Thông số nhập tay (được giữ nguyên sau khi lưu)", padding=10)
        manual_box.pack(fill="x", pady=10)
        manual_box.columnconfigure(1, weight=1)
        fields = (
            ("Weight_g", "Khối lượng", "g"),
            ("Container_Height_mm", "Chiều cao ly", "mm"),
            ("Inner_Diameter_mm", "Đường kính trong", "mm"),
            ("Empty_Height_mm", "Chiều cao khoảng trống", "mm"),
            ("Actual_Count", "Số hạt thực tế", "hạt"),
        )
        for row, (key, label, unit) in enumerate(fields):
            ttk.Label(manual_box, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(manual_box, textvariable=self.manual_vars[key]).grid(
                row=row, column=1, sticky="ew", padx=8, pady=4
            )
            ttk.Label(manual_box, text=unit).grid(row=row, column=2, sticky="w", pady=4)
        ttk.Separator(manual_box).grid(row=5, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(manual_box, text="Chiều cao lớp gạo").grid(row=6, column=0, sticky="w")
        ttk.Label(manual_box, textvariable=self.rice_height_var, font=("Segoe UI", 10, "bold")).grid(
            row=6, column=1, sticky="w", padx=8
        )
        ttk.Label(manual_box, text="mm (tự tính)").grid(row=6, column=2, sticky="w")

        save_box = ttk.LabelFrame(form_side, text="Lưu", padding=10)
        save_box.pack(fill="x")
        ttk.Checkbutton(save_box, text="Tự lưu file Excel sau mỗi ảnh", variable=self.auto_save_var).pack(anchor="w")
        ttk.Button(save_box, text="LƯU ẢNH + DÒNG DỮ LIỆU", style="Primary.TButton", command=self.save_sample).pack(
            fill="x", pady=(8, 4)
        )
        ttk.Label(
            save_box,
            text="Sau khi lưu, mã ảnh tăng 1. Các thông số của ly vẫn được giữ nguyên.",
            style="Sub.TLabel",
            wraplength=400,
        ).pack(anchor="w", pady=(4, 0))

    def _build_excel_tab(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="Lưu Excel", command=self.save_workbook).pack(side="left")
        ttk.Button(toolbar, text="Lưu thành…", command=self.save_workbook_as).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Nạp lại", command=self.load_workbook).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Mở bằng Excel", command=self.open_in_excel).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Nạp dòng vào form", command=self.load_selected_into_form).pack(side="left", padx=(18, 5))
        ttk.Button(toolbar, text="Xóa dòng đã chọn", command=self.delete_selected_rows).pack(side="left", padx=5)
        ttk.Label(toolbar, text="Nhấp đúp một ô để sửa", style="Sub.TLabel").pack(side="right")

        table_frame = ttk.Frame(parent)
        table_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table_frame, show="headings", selectmode="extended")
        vbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree.bind("<Double-1>", self.edit_tree_cell)

    def _build_mobile_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill="x")

        connection = ttk.LabelFrame(top, text="Kết nối điện thoại bằng QR", padding=12)
        connection.pack(side="left", fill="both", expand=True, padx=(0, 8))
        connection.columnconfigure(1, weight=1)

        ttk.Label(connection, text="Địa chỉ mạng:").grid(row=0, column=0, sticky="w")
        self.mobile_address_combo = ttk.Combobox(
            connection,
            textvariable=self.mobile_address_var,
            values=tuple(discover_lan_addresses()),
            width=24,
        )
        self.mobile_address_combo.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(connection, text="Làm mới", command=self.refresh_mobile_addresses).grid(row=0, column=2)

        ttk.Label(connection, text="Cổng:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(connection, from_=1024, to=65535, textvariable=self.mobile_port_var, width=10).grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )

        actions = ttk.Frame(connection)
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        ttk.Button(
            actions,
            text="KHỞI ĐỘNG KẾT NỐI",
            style="Primary.TButton",
            command=self.start_mobile_server,
        ).pack(side="left")
        ttk.Button(actions, text="Dừng", command=self.stop_mobile_server).pack(side="left", padx=8)
        ttk.Button(actions, text="Sao chép địa chỉ", command=self.copy_mobile_url).pack(side="left")
        ttk.Button(actions, text="Xuất chứng chỉ…", command=self.export_mobile_certificate).pack(side="left", padx=8)
        ttk.Button(actions, text="Cách cài chứng chỉ", command=self.show_certificate_help).pack(side="left")

        ttk.Label(connection, textvariable=self.mobile_status_var, font=("Segoe UI", 10, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(4, 0)
        )
        ttk.Label(connection, textvariable=self.mobile_node_var, style="Sub.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(5, 0)
        )
        ttk.Entry(connection, textvariable=self.mobile_url_var, state="readonly").grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0)
        )
        ttk.Label(
            connection,
            text=(
                "Camera trong khung dùng HTTPS. Lần đầu, hãy xuất và cài chứng chỉ RiceCapture-CA.cer "
                "trên điện thoại; sau đó điện thoại và máy tính phải ở cùng Wi‑Fi hoặc USB tethering."
            ),
            style="Sub.TLabel",
            wraplength=650,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(9, 0))

        qr_box = ttk.LabelFrame(top, text="Quét bằng điện thoại", padding=12)
        qr_box.pack(side="right", fill="y")
        self.mobile_qr_label = ttk.Label(
            qr_box,
            text="Khởi động kết nối\nđể tạo mã QR",
            anchor="center",
            justify="center",
            width=28,
        )
        self.mobile_qr_label.pack(fill="both", expand=True)

        info = ttk.LabelFrame(parent, text="Luồng làm việc", padding=14)
        info.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(
            info,
            text=(
                "1. Lần đầu: xuất, chuyển sang điện thoại và cài chứng chỉ CA.\n"
                "2. Khởi động kết nối HTTPS và quét QR.\n"
                "3. Nhập thông số; bật camera trong khung, căn ảnh và bấm Chụp ảnh.\n"
                "4. Máy tính cấp mã Mxxx, lưu ảnh, cập nhật SQLite và đồng bộ Excel.\n"
                "5. Ảnh cùng thông số vừa nhận sẽ xuất hiện ở tab Chụp & nhập dữ liệu.\n\n"
                "Form trên điện thoại được giữ nguyên sau mỗi lần lưu. Máy tính là nơi duy nhất "
                "cấp mã chính thức để tránh trùng dữ liệu."
            ),
            justify="left",
            wraplength=1000,
        ).pack(anchor="nw")

    def _bind_updates(self) -> None:
        for variable in (
            self.sample_number_var,
            self.sample_digits_var,
            self.image_dir_var,
        ):
            variable.trace_add("write", lambda *_: self._update_naming_preview())
        for key in ("Container_Height_mm", "Empty_Height_mm"):
            self.manual_vars[key].trace_add("write", lambda *_: self._update_rice_height())
        for variable in (
            self.sample_number_var,
            self.sample_digits_var,
            self.image_dir_var,
            self.workbook_var,
            *self.manual_vars.values(),
        ):
            variable.trace_add("write", lambda *_: self._schedule_coordinator_sync())

    def _database_path(self) -> Path:
        workbook = Path(self.workbook_var.get() or DEFAULT_WORKBOOK)
        return workbook.parent / "capture_data.sqlite3"

    def _existing_excel_ids(self) -> set[str]:
        if not self.workbook_store:
            return set()
        return {
            str(record.get("Sample_ID") or "").strip().upper()
            for _, record in self.workbook_store.rows()
            if record.get("Sample_ID")
        }

    def _configure_coordinator(self) -> None:
        self.coordinator.configure(
            image_dir=self.image_dir_var.get(),
            database_path=self._database_path(),
            sample_number=self.sample_number_var.get(),
            sample_digits=self.sample_digits_var.get(),
            existing_excel_ids=self._existing_excel_ids(),
            manual_defaults={key: variable.get() for key, variable in self.manual_vars.items()},
        )

    def _schedule_coordinator_sync(self) -> None:
        if self.coordinator_sync_job is not None:
            try:
                self.root.after_cancel(self.coordinator_sync_job)
            except tk.TclError:
                pass
        self.coordinator_sync_job = self.root.after(250, self._run_coordinator_sync)

    def _run_coordinator_sync(self) -> None:
        self.coordinator_sync_job = None
        try:
            self._configure_coordinator()
        except (ValueError, tk.TclError, OSError):
            pass

    def refresh_mobile_addresses(self) -> None:
        addresses = discover_lan_addresses()
        self.mobile_address_combo.configure(values=tuple(addresses))
        if self.mobile_address_var.get() not in addresses:
            self.mobile_address_var.set(addresses[0])
        if self.mobile_server.running:
            self._refresh_mobile_qr()

    def start_mobile_server(self) -> None:
        try:
            self._configure_coordinator()
            requested_port = int(self.mobile_port_var.get())
            address = self.mobile_address_var.get().strip()
            if self.mobile_server.running:
                self.mobile_server.stop()
            self.mobile_server.port = requested_port
            self.mobile_server.start(address)
            self._refresh_mobile_qr()
            self.mobile_status_var.set("HTTPS đang hoạt động – hãy quét mã QR")
            self._set_status("Đã mở kết nối HTTPS cho điện thoại.")
        except Exception as exc:
            messagebox.showerror("Không thể kết nối điện thoại", str(exc))

    def _refresh_mobile_qr(self) -> None:
        if not self.mobile_server.running:
            return
        address = self.mobile_address_var.get().strip()
        url = self.mobile_server.connection_url(address)
        self.mobile_url_var.set(url)
        image = self.mobile_server.make_qr_image(address).convert("RGB")
        image.thumbnail((260, 260))
        from PIL import ImageTk

        self.mobile_qr_photo = ImageTk.PhotoImage(image)
        self.mobile_qr_label.configure(image=self.mobile_qr_photo, text="")

    def stop_mobile_server(self) -> None:
        self.mobile_server.stop()
        self.mobile_nodes.clear()
        self.mobile_url_var.set("Chưa khởi động")
        self.mobile_status_var.set("Máy chủ điện thoại đang tắt")
        self.mobile_node_var.set("Chưa có điện thoại kết nối")
        self.mobile_qr_label.configure(image="", text="Khởi động kết nối\nđể tạo mã QR")
        self.mobile_qr_photo = None
        self._set_status("Đã dừng kết nối điện thoại.")

    def copy_mobile_url(self) -> None:
        value = self.mobile_url_var.get()
        if not value.startswith("http"):
            messagebox.showinfo("Chưa có địa chỉ", "Hãy khởi động kết nối trước.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self._set_status("Đã sao chép địa chỉ kết nối.")

    def export_mobile_certificate(self) -> None:
        try:
            material = self.mobile_server.prepare_certificates(self.mobile_address_var.get().strip())
            selected = filedialog.asksaveasfilename(
                title="Xuất chứng chỉ để cài trên điện thoại",
                initialdir=str(APP_DIR),
                initialfile="RiceCapture-CA.cer",
                defaultextension=".cer",
                filetypes=(("Chứng chỉ X.509", "*.cer"), ("All files", "*.*")),
            )
            if not selected:
                return
            shutil.copyfile(material.ca_install_path, selected)
            self._set_status(f"Đã xuất chứng chỉ: {selected}")
            messagebox.showinfo(
                "Đã xuất chứng chỉ",
                "Hãy chuyển RiceCapture-CA.cer sang điện thoại và cài theo nút 'Cách cài chứng chỉ'. "
                "Chỉ cần thực hiện một lần trên mỗi điện thoại.",
            )
        except Exception as exc:
            messagebox.showerror("Không thể xuất chứng chỉ", str(exc))

    def show_certificate_help(self) -> None:
        messagebox.showinfo(
            "Cài chứng chỉ cho camera trong khung",
            "ANDROID\n"
            "1. Chuyển RiceCapture-CA.cer vào điện thoại.\n"
            "2. Mở Cài đặt > Bảo mật và quyền riêng tư > Cài đặt bảo mật khác.\n"
            "3. Chọn Cài đặt chứng chỉ > Chứng chỉ CA và chọn file.\n\n"
            "IPHONE / IPAD\n"
            "1. Mở file RiceCapture-CA.cer và cho phép tải hồ sơ.\n"
            "2. Cài đặt > Đã tải về hồ sơ > Cài đặt.\n"
            "3. Cài đặt > Cài đặt chung > Giới thiệu > Cài đặt tin cậy chứng chỉ.\n"
            "4. Bật tin cậy hoàn toàn cho Rice Dataset Capture Local CA.\n\n"
            "Sau đó đóng tab cũ, quét lại QR HTTPS và cấp quyền Camera khi trình duyệt hỏi.",
        )

    def _poll_mobile_events(self) -> None:
        try:
            while True:
                event = self.mobile_events.get_nowait()
                event_type = event.get("type")
                if event_type == "sample_saved":
                    self._handle_sample_saved_event(event)
                elif event_type == "node_connected":
                    self.mobile_nodes[event["node_id"]] = event.get("label", "Điện thoại")
                    self.mobile_node_var.set(
                        f"Đang kết nối: {', '.join(self.mobile_nodes.values())}"
                    )
                    self.mobile_status_var.set("Điện thoại đã kết nối")
                elif event_type == "node_disconnected":
                    self.mobile_nodes.pop(event.get("node_id", ""), None)
                    self.mobile_node_var.set(
                        f"Đang kết nối: {', '.join(self.mobile_nodes.values())}"
                        if self.mobile_nodes
                        else "Chưa có điện thoại kết nối"
                    )
        except queue.Empty:
            pass
        try:
            self.root.after(200, self._poll_mobile_events)
        except tk.TclError:
            pass

    def _handle_sample_saved_event(self, event: dict[str, Any]) -> None:
        record = dict(event["record"])
        for key, variable in self.manual_vars.items():
            if record.get(key) is not None:
                variable.set(str(record[key]))
        next_id = str(event["next_sample_id"])
        if next_id.startswith("M") and next_id[1:].isdigit():
            self.sample_digits_var.set(len(next_id) - 1)
            self.sample_number_var.set(int(next_id[1:]))
        self.captured_frame = None
        self.capture_state_label.configure(text=f"Đã nhận {event['sample_id']} từ điện thoại")
        self._show_received_image(Path(event["image_path"]))
        self._sync_pending_excel()
        self._set_status(f"Đã nhận và lưu {event['sample_id']} từ điện thoại.")

    def _show_received_image(self, image_path: Path) -> None:
        try:
            from PIL import Image, ImageOps, ImageTk

            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            width = max(self.preview_label.winfo_width(), 640)
            height = max(self.preview_label.winfo_height(), 420)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Đã nhận ảnh nhưng không thể xem trước:\n{exc}")

    @staticmethod
    def _database_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "Sample_ID": row["sample_id"],
            "Weight_g": row["weight_g"],
            "Container_Height_mm": row["container_height_mm"],
            "Inner_Diameter_mm": row["inner_diameter_mm"],
            "Empty_Height_mm": row["empty_height_mm"],
            "Rice_Height_mm": row["rice_height_mm"],
            "Actual_Count": row["actual_count"],
        }

    def _sync_pending_excel(self) -> None:
        try:
            if not self.workbook_store:
                store = WorkbookStore(self.workbook_var.get())
                store.load_or_create()
                self.workbook_store = store
            pending = self.coordinator.database.pending_excel_records()
            if not pending:
                self.refresh_tree()
                return
            for row in pending:
                record = self._database_row_to_record(row)
                if not self.workbook_store.contains_sample_id(record["Sample_ID"]):
                    self.workbook_store.append(record)
            self.workbook_store.save()
            for row in pending:
                self.coordinator.database.mark_excel_status(row["sample_id"], "synced")
            self.refresh_tree()
        except Exception as exc:
            self.refresh_tree()
            self.mobile_status_var.set(f"Đã lưu mẫu; Excel đang chờ đồng bộ: {exc}")
            self._set_status("Mẫu an toàn trong SQLite; hãy đóng Excel rồi bấm Lưu Excel.")

    def _naming_config(self) -> NamingConfig:
        return NamingConfig(
            prefix="M",
            sample_number=int(self.sample_number_var.get()),
            sample_digits=int(self.sample_digits_var.get()),
        )

    def _update_naming_preview(self) -> None:
        try:
            config = self._naming_config()
            path = Path(self.image_dir_var.get()) / f"{config.sample_id}.jpg"
            self.naming_preview_var.set(f"{config.sample_id}  →  {path.name}")
        except (ValueError, tk.TclError):
            self.naming_preview_var.set("Mã chưa hợp lệ")

    def _update_rice_height(self) -> None:
        try:
            container = float(self.manual_vars["Container_Height_mm"].get().replace(",", "."))
            empty = float(self.manual_vars["Empty_Height_mm"].get().replace(",", "."))
            value = container - empty
            self.rice_height_var.set(f"{value:g}" if value >= 0 else "Không hợp lệ")
        except ValueError:
            self.rice_height_var.set("—")

    def choose_image_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.image_dir_var.get() or str(APP_DIR))
        if selected:
            self.image_dir_var.set(selected)

    def choose_workbook(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(Path(self.workbook_var.get()).parent),
            filetypes=(("Excel workbook", "*.xlsx *.xlsm"), ("All files", "*.*")),
        )
        if not selected:
            selected = filedialog.asksaveasfilename(
                initialdir=str(Path(self.workbook_var.get()).parent),
                defaultextension=".xlsx",
                filetypes=(("Excel workbook", "*.xlsx"),),
            )
        if selected:
            self.workbook_var.set(selected)
            self.load_workbook()

    def load_workbook(self, silent: bool = False) -> None:
        try:
            if self.workbook_store and self.workbook_store.dirty and not silent:
                if not messagebox.askyesno("Có thay đổi chưa lưu", "Nạp lại sẽ bỏ các thay đổi chưa lưu. Tiếp tục?"):
                    return
            store = WorkbookStore(self.workbook_var.get())
            store.load_or_create()
            self.workbook_store = store
            self.refresh_tree()
            self._configure_coordinator()
            self._sync_pending_excel()
            self._set_status(f"Đã nạp workbook: {store.path.name}")
        except Exception as exc:
            if silent:
                self._set_status(f"Chưa thể nạp workbook: {exc}")
            else:
                messagebox.showerror("Không thể mở Excel", str(exc))

    def refresh_tree(self) -> None:
        if not self.workbook_store:
            return
        columns = ("_row",) + tuple(self.workbook_store.headers)
        self.tree.configure(columns=columns)
        self.tree.heading("_row", text="#")
        self.tree.column("_row", width=55, minwidth=45, anchor="center", stretch=False)
        for header in self.workbook_store.headers:
            self.tree.heading(header, text=header)
            width = 185 if header == "Sample_ID" else 155
            self.tree.column(header, width=width, minwidth=110, anchor="center")
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row_number, record in self.workbook_store.rows():
            values = (row_number,) + tuple(
                "" if record.get(header) is None else record.get(header, "")
                for header in self.workbook_store.headers
            )
            self.tree.insert("", "end", iid=str(row_number), values=values)
        children = self.tree.get_children()
        if children:
            self.tree.see(children[-1])

    def _load_camera_dependencies(self) -> None:
        if self.cv2 is not None:
            return
        try:
            import cv2
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu OpenCV hoặc Pillow. Hãy chạy: python -m pip install -r requirements.txt"
            ) from exc
        self.cv2 = cv2
        self.pil_image = Image
        self.image_tk = ImageTk

    def _camera_backend(self) -> int | None:
        if self.camera_backend_var.get() == "DirectShow":
            return self.cv2.CAP_DSHOW
        if self.camera_backend_var.get() == "MSMF":
            return self.cv2.CAP_MSMF
        return None

    def _camera_source(self) -> int | str:
        source = self.camera_source_var.get().strip()
        return int(source) if source.isdigit() else source

    def _open_capture(self, source: int | str):
        backend = self._camera_backend()
        return self.cv2.VideoCapture(source, backend) if backend is not None else self.cv2.VideoCapture(source)

    def scan_cameras(self) -> None:
        try:
            self._load_camera_dependencies()
            available = []
            unavailable_in_a_row = 0
            previous_log_level = self.cv2.getLogLevel() if hasattr(self.cv2, "getLogLevel") else None
            if hasattr(self.cv2, "setLogLevel"):
                self.cv2.setLogLevel(0)
            try:
                for index in range(8):
                    capture = self._open_capture(index)
                    ok, _ = capture.read() if capture.isOpened() else (False, None)
                    capture.release()
                    if ok:
                        available.append(str(index))
                        unavailable_in_a_row = 0
                    else:
                        unavailable_in_a_row += 1
                        if available and unavailable_in_a_row >= 3:
                            break
            finally:
                if previous_log_level is not None and hasattr(self.cv2, "setLogLevel"):
                    self.cv2.setLogLevel(previous_log_level)
            self.camera_combo.configure(values=available or ("0",))
            if available and self.camera_source_var.get() not in available:
                self.camera_source_var.set(available[0])
            self._set_status(f"Tìm thấy {len(available)} camera.")
        except Exception as exc:
            messagebox.showerror("Không thể quét camera", str(exc))

    def start_camera(self) -> None:
        try:
            self._load_camera_dependencies()
            self.stop_camera(clear_preview=False)
            self.camera = self._open_capture(self._camera_source())
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = None
                raise RuntimeError("Không mở được nguồn camera đã chọn.")
            self.camera.set(self.cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self._set_status(f"Camera {self.camera_source_var.get()} đang hoạt động.")
            self._update_camera_frame()
        except Exception as exc:
            messagebox.showerror("Lỗi camera", str(exc))

    def _update_camera_frame(self) -> None:
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if ok:
            self.current_frame = frame
            self._show_frame(frame)
        self.camera_job = self.root.after(30, self._update_camera_frame)

    def _show_frame(self, frame) -> None:
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        image = self.pil_image.fromarray(rgb)
        width = max(self.preview_label.winfo_width(), 640)
        height = max(self.preview_label.winfo_height(), 420)
        image.thumbnail((width, height), self.pil_image.Resampling.LANCZOS)
        self.preview_photo = self.image_tk.PhotoImage(image=image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def stop_camera(self, clear_preview: bool = True) -> None:
        if self.camera_job is not None:
            self.root.after_cancel(self.camera_job)
            self.camera_job = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if clear_preview and self.current_frame is None:
            self.preview_label.configure(image="", text="Camera đã tắt")

    def capture_frame(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("Chưa có hình", "Hãy bật camera và chờ hình xem trước xuất hiện.")
            return
        self.captured_frame = self.current_frame.copy()
        self.capture_state_label.configure(text="Đã chụp – sẵn sàng lưu")
        self._set_status("Đã giữ khung hình. Kiểm tra thông số rồi bấm lưu.")

    def _write_jpeg_atomic(self, path: Path, frame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = self.cv2.imencode(".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("OpenCV không mã hóa được ảnh JPEG.")
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".jpg", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            temp_path.write_bytes(encoded.tobytes())
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def save_sample(self) -> None:
        if self.captured_frame is None:
            messagebox.showwarning("Chưa chụp ảnh", "Hãy bấm CHỤP KHUNG HÌNH trước khi lưu ảnh.")
            return
        try:
            self._load_camera_dependencies()
            self._configure_coordinator()
            ok, encoded = self.cv2.imencode(
                ".jpg",
                self.captured_frame,
                [int(self.cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            if not ok:
                raise RuntimeError("OpenCV không mã hóa được ảnh JPEG.")
            result = self.coordinator.save_jpeg(
                encoded.tobytes(),
                {key: variable.get() for key, variable in self.manual_vars.items()},
                source_type="desktop",
            )
            self.captured_frame = None
            self.capture_state_label.configure(text=f"Đã lưu {result.sample_id} – chụp ảnh kế tiếp")
            self._set_status(f"Đã lưu {result.sample_id}; đang đồng bộ Excel.")
        except Exception as exc:
            messagebox.showerror("Không thể lưu mẫu", str(exc))

    def next_image_id(self) -> None:
        try:
            config = self._naming_config()
            self.sample_number_var.set(config.sample_number + 1)
            self.captured_frame = None
            self.capture_state_label.configure(text="ID mới – chưa chụp")
            self._set_status("Đã tăng ID ảnh; các thông số form vẫn được giữ nguyên.")
        except Exception as exc:
            messagebox.showerror("Mã không hợp lệ", str(exc))

    def find_next_available(self) -> None:
        try:
            config = self._naming_config()
            for _ in range(100000):
                image_path = Path(self.image_dir_var.get()) / f"{config.sample_id}.jpg"
                in_excel = self.workbook_store.contains_sample_id(config.sample_id) if self.workbook_store else False
                if not image_path.exists() and not in_excel:
                    self.sample_number_var.set(config.sample_number)
                    self._set_status(f"ID trống kế tiếp: {config.sample_id}")
                    return
                config.next_id()
            raise RuntimeError("Không tìm được ID trống trong giới hạn tìm kiếm.")
        except Exception as exc:
            messagebox.showerror("Không thể tìm ID", str(exc))

    def save_workbook(self) -> None:
        if not self.workbook_store:
            return
        try:
            self._configure_coordinator()
            self._sync_pending_excel()
            path = self.workbook_store.path
            self.workbook_var.set(str(path))
            self._set_status(f"Đã lưu workbook: {path}")
        except Exception as exc:
            messagebox.showerror("Không thể lưu Excel", str(exc))

    def save_workbook_as(self) -> None:
        if not self.workbook_store:
            return
        selected = filedialog.asksaveasfilename(
            initialdir=str(Path(self.workbook_var.get()).parent),
            initialfile=Path(self.workbook_var.get()).name,
            defaultextension=".xlsx",
            filetypes=(("Excel workbook", "*.xlsx"), ("Excel macro workbook", "*.xlsm")),
        )
        if selected:
            try:
                path = self.workbook_store.save(selected)
                self.workbook_var.set(str(path))
                self._set_status(f"Đã lưu workbook thành: {path}")
            except Exception as exc:
                messagebox.showerror("Không thể lưu Excel", str(exc))

    def open_in_excel(self) -> None:
        try:
            if self.workbook_store and self.workbook_store.dirty:
                self.workbook_store.save()
            path = Path(self.workbook_var.get())
            if not path.exists():
                raise FileNotFoundError(path)
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Không thể mở Excel", str(exc))

    def edit_tree_cell(self, event: tk.Event) -> None:
        if not self.workbook_store:
            return
        item = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        if not item or not column_id:
            return
        column_index = int(column_id[1:]) - 1
        if column_index <= 0:
            return
        header = self.workbook_store.headers[column_index - 1]
        values = self.tree.item(item, "values")
        current = values[column_index] if column_index < len(values) else ""
        new_value = simpledialog.askstring("Sửa ô", f"{header}:", initialvalue=str(current), parent=self.root)
        if new_value is None:
            return
        try:
            self.workbook_store.update_manual_value(int(item), header, new_value)
            self.refresh_tree()
            self._set_status(f"Đã sửa {header} tại dòng {item}; hãy lưu workbook.")
        except Exception as exc:
            messagebox.showerror("Giá trị không hợp lệ", str(exc))

    def delete_selected_rows(self) -> None:
        selected = self.tree.selection()
        if not selected or not self.workbook_store:
            return
        if not messagebox.askyesno("Xóa dòng", f"Xóa {len(selected)} dòng khỏi bảng đang mở?"):
            return
        self.workbook_store.delete_rows(int(item) for item in selected)
        self.refresh_tree()
        self._set_status("Đã xóa dòng trong bộ nhớ; hãy lưu workbook để ghi thay đổi.")

    def load_selected_into_form(self) -> None:
        selected = self.tree.selection()
        if not selected or not self.workbook_store:
            messagebox.showinfo("Chưa chọn", "Hãy chọn một dòng trong bảng.")
            return
        row_number = int(selected[0])
        record = dict(self.workbook_store.rows()).get(row_number, {})
        for key, variable in self.manual_vars.items():
            if record.get(key) not in (None, ""):
                variable.set(str(record[key]))
        self._set_status(f"Đã nạp thông số từ dòng {row_number} vào form.")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _settings_payload(self) -> dict[str, Any]:
        try:
            naming = {
                "prefix": "M",
                "sample_number": self.sample_number_var.get(),
                "sample_digits": self.sample_digits_var.get(),
            }
        except tk.TclError:
            naming = {}
        return {
            "image_dir": self.image_dir_var.get(),
            "workbook": self.workbook_var.get(),
            "camera_source": self.camera_source_var.get(),
            "camera_backend": self.camera_backend_var.get(),
            "naming": naming,
            "manual": {key: variable.get() for key, variable in self.manual_vars.items()},
            "auto_save": self.auto_save_var.get(),
            "mobile_address": self.mobile_address_var.get(),
            "mobile_port": self.mobile_port_var.get(),
        }

    def on_close(self) -> None:
        if self.workbook_store and self.workbook_store.dirty:
            answer = messagebox.askyesnocancel("Thay đổi chưa lưu", "Lưu thay đổi Excel trước khi thoát?")
            if answer is None:
                return
            if answer:
                try:
                    self.workbook_store.save()
                except Exception as exc:
                    messagebox.showerror("Không thể lưu Excel", str(exc))
                    return
        self.stop_camera(clear_preview=False)
        self.mobile_server.stop()
        try:
            self.settings_store.save(self._settings_payload())
        except OSError:
            pass
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    DatasetCaptureApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
