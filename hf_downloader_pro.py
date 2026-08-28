import sys
import os
import time
import shutil
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional

import requests
from PySide6.QtCore import Qt, Signal, QObject, QSettings
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QComboBox,
    QHeaderView,
    QGroupBox,
    QTabWidget,
    QSpinBox,
)

from huggingface_hub import HfApi, ModelInfo

# ----------------------------------------------------------------------
# Yardımcı Fonksiyonlar: Boyut ve Zaman Formatlama
# ----------------------------------------------------------------------
def human_readable_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"

def format_time(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ----------------------------------------------------------------------
# Hugging Face Metadata Boyut Çözücü (Kesin ve Değişmez Kaynak)
# ----------------------------------------------------------------------
def fetch_exact_repo_sizes(api: HfApi, repo_id: str, token: Optional[str] = None) -> Dict[str, int]:
    """
    Hugging Face API üzerinden LFS ve standart dosyaların net bayt boyutlarını döner.
    """
    try:
        info = api.model_info(repo_id=repo_id, files_metadata=True, token=token)
        sizes: Dict[str, int] = {}
        if info.siblings:
            for s in info.siblings:
                real_sz = 0
                lfs = getattr(s, "lfs", None)
                if lfs:
                    if isinstance(lfs, dict):
                        real_sz = int(lfs.get("size", 0) or 0)
                    else:
                        real_sz = int(getattr(lfs, "size", 0) or 0)
                if real_sz == 0 and getattr(s, "size", None) is not None:
                    real_sz = int(s.size or 0)
                sizes[s.rfilename] = real_sz
        return sizes
    except Exception:
        return {}

# ----------------------------------------------------------------------
# Görev Modeli (Task Data Object)
# ----------------------------------------------------------------------
class DownloadTask:
    def __init__(self, repo_id: str, filename: str, url: str, dest_path: Path, total_bytes: int = 0):
        self.id = str(uuid.uuid4())
        self.repo_id = repo_id
        self.filename = filename
        self.url = url
        self.dest_path = dest_path
        self.part_path = dest_path.parent / (dest_path.name + ".part")
        self.total_bytes = total_bytes
        
        # Disk kontrolü
        if self.dest_path.exists() and total_bytes > 0 and self.dest_path.stat().st_size == total_bytes:
            self.downloaded_bytes = total_bytes
            self.status = "Tamamlandı"
        elif self.part_path.exists():
            self.downloaded_bytes = self.part_path.stat().st_size
            self.status = "Duraklatıldı"
        else:
            self.downloaded_bytes = 0
            self.status = "Kuyrukta"

        self.speed = 0.0
        self.eta = 0.0
        self.cancel_event = threading.Event()

# ----------------------------------------------------------------------
# Sinyal Yöneticisi
# ----------------------------------------------------------------------
class ManagerSignals(QObject):
    task_progress = Signal(str, object, object, float, float)
    task_status = Signal(str, str)
    task_size_updated = Signal(str, object)
    error = Signal(str)
    search_results = Signal(list)
    files_listed = Signal(list)

# ----------------------------------------------------------------------
# Ana Pencere ve İndirme Yöneticisi
# ----------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚡ Hugging Face Ultra Downloader & Queue Manager")
        self.setMinimumSize(1120, 740)

        self.settings = QSettings("HFDownloaderApp", "QueueConfig")
        self.api = HfApi()
        self.signals = ManagerSignals()

        self.tasks: Dict[str, DownloadTask] = {}
        self.active_workers: Dict[str, threading.Thread] = {}
        self.metadata_cache: Dict[str, Dict[str, int]] = {}
        self.worker_lock = threading.Lock()

        self._setup_ui()
        self._load_settings()
        self._bind_signals()

        # Açılışta diskteki yarım (.part) dosyaları tara
        self.scan_and_restore_part_files()

        # Kuyruk Dağıtıcı Thread
        self.queue_active = True
        self.queue_thread = threading.Thread(target=self._queue_dispatcher_loop, daemon=True)
        self.queue_thread.start()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        # Tab 1: Model Arama & Seçim
        self.tab_search = QWidget()
        self._build_search_tab()
        self.tabs.addTab(self.tab_search, "🔍 Model Arama & Dosya Listesi")

        # Tab 2: İndirme Yöneticisi
        self.tab_queue = QWidget()
        self._build_queue_tab()
        self.tabs.addTab(self.tab_queue, "📥 İndirme Yöneticisi (Kuyruk)")

        # Tab 3: Disk Listesi
        self.tab_completed = QWidget()
        self._build_completed_tab()
        self.tabs.addTab(self.tab_completed, "📁 Yerel Depolama (Disk)")

        # Tab 4: Ayarlar
        self.tab_settings = QWidget()
        self._build_settings_tab()
        self.tabs.addTab(self.tab_settings, "⚙️ Ayarlar & HF Token")

        main_layout.addWidget(self.tabs)

    def _build_search_tab(self):
        layout = QVBoxLayout(self.tab_search)

        search_group = QGroupBox("Model Deposu Arama")
        s_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Örn: unsloth/DeepSeek-R1-GGUF veya Qwen/Qwen2.5-7B-Instruct")
        self.search_input.returnPressed.connect(self.search_models)
        self.search_btn = QPushButton("Depoyu Getir / Ara")
        self.search_btn.clicked.connect(self.search_models)
        s_layout.addWidget(self.search_input)
        s_layout.addWidget(self.search_btn)
        search_group.setLayout(s_layout)
        layout.addWidget(search_group)

        self.match_combo = QComboBox()
        self.match_combo.setVisible(False)
        self.match_combo.currentIndexChanged.connect(self._on_combo_repo_selected)
        layout.addWidget(self.match_combo)

        bar = QHBoxLayout()
        self.repo_label = QLabel("Aktif Depo: —")
        self.repo_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Tabloda filtrele (örn: .gguf, Q4_K_M)...")
        self.filter_input.textChanged.connect(self._filter_search_table)
        bar.addWidget(self.repo_label)
        bar.addStretch()
        bar.addWidget(QLabel("Hızlı Filtre:"))
        bar.addWidget(self.filter_input)
        layout.addLayout(bar)

        self.file_table = QTableWidget(0, 3)
        self.file_table.setHorizontalHeaderLabels(["Dosya Adı", "Gerçek Boyut", "Depolama Tipi"])
        h = self.file_table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.ExtendedSelection)
        layout.addWidget(self.file_table)

        btn_layout = QHBoxLayout()
        self.btn_add_selected = QPushButton("➕ Seçilenleri İndirme Kuyruğuna Ekle")
        self.btn_add_selected.setEnabled(False)
        self.btn_add_selected.clicked.connect(self.add_selected_to_queue)

        self.btn_add_all = QPushButton("➕ Tüm Depoyu Kuyruğa Ekle")
        self.btn_add_all.setEnabled(False)
        self.btn_add_all.clicked.connect(self.add_all_to_queue)

        btn_layout.addWidget(self.btn_add_selected)
        btn_layout.addWidget(self.btn_add_all)
        layout.addLayout(btn_layout)

    def _build_queue_tab(self):
        layout = QVBoxLayout(self.tab_queue)

        action_bar = QHBoxLayout()
        self.btn_resume_all = QPushButton("▶ Tümünü Başlat / Devam Ettir")
        self.btn_resume_all.clicked.connect(self.resume_all_tasks)
        self.btn_pause_all = QPushButton("⏸ Tümünü Duraklat")
        self.btn_pause_all.clicked.connect(self.pause_all_tasks)
        self.btn_scan_part_queue = QPushButton("🔄 Yarım Kalan (.part) Dosyaları Tara")
        self.btn_scan_part_queue.clicked.connect(self.scan_and_restore_part_files)
        self.btn_clear_completed = QPushButton("🧹 Tamamlananları Temizle")
        self.btn_clear_completed.clicked.connect(self.clear_completed_tasks)

        action_bar.addWidget(self.btn_resume_all)
        action_bar.addWidget(self.btn_pause_all)
        action_bar.addWidget(self.btn_scan_part_queue)
        action_bar.addWidget(self.btn_clear_completed)
        action_bar.addStretch()
        layout.addLayout(action_bar)

        self.queue_table = QTableWidget(0, 8)
        self.queue_table.setHorizontalHeaderLabels([
            "Model / Depo", "Dosya Adı", "İndirilen / Toplam Boyut", "İlerleme", "Hız", "Kalan Süre", "Durum", "İşlem"
        ])
        qh = self.queue_table.horizontalHeader()
        qh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        qh.setSectionResizeMode(1, QHeaderView.Stretch)
        qh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        qh.setSectionResizeMode(3, QHeaderView.Fixed)
        self.queue_table.setColumnWidth(3, 160)
        qh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        qh.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        qh.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        qh.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.queue_table)

    def _build_completed_tab(self):
        layout = QVBoxLayout(self.tab_completed)
        top_bar = QHBoxLayout()
        self.btn_open_folder = QPushButton("📁 Klasörü Aç")
        self.btn_open_folder.clicked.connect(self.open_download_dir)
        self.btn_refresh_downloads = QPushButton("🔄 Listeyi Yenile")
        self.btn_refresh_downloads.clicked.connect(self._refresh_completed_list)
        self.btn_scan_all_part = QPushButton("📥 Yarım Kalanları Kuyruğa Yükle")
        self.btn_scan_all_part.clicked.connect(self.scan_and_restore_part_files)

        top_bar.addWidget(self.btn_open_folder)
        top_bar.addWidget(self.btn_refresh_downloads)
        top_bar.addWidget(self.btn_scan_all_part)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.completed_table = QTableWidget(0, 5)
        self.completed_table.setHorizontalHeaderLabels([
            "Dosya / Model Yolu", "Diskteki Boyut", "Durum", "Değiştirilme Tarihi", "İşlemler"
        ])
        dh = self.completed_table.horizontalHeader()
        dh.setSectionResizeMode(0, QHeaderView.Stretch)
        dh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        dh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        dh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        dh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.completed_table.setAlternatingRowColors(True)
        layout.addWidget(self.completed_table)

    def _build_settings_tab(self):
        layout = QVBoxLayout(self.tab_settings)

        dir_group = QGroupBox("İndirme Dizini")
        d_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setReadOnly(True)
        self.btn_browse = QPushButton("Gözat...")
        self.btn_browse.clicked.connect(self.select_download_dir)
        d_layout.addWidget(self.dir_input)
        d_layout.addWidget(self.btn_browse)
        dir_group.setLayout(d_layout)
        layout.addWidget(dir_group)

        auth_group = QGroupBox("Hugging Face API Token")
        a_layout = QVBoxLayout()
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxx (Gated/Özel modeller için)")
        self.token_input.textChanged.connect(lambda: self.settings.setValue("hf_token", self.token_input.text().strip()))
        a_layout.addWidget(self.token_input)
        auth_group.setLayout(a_layout)
        layout.addWidget(auth_group)

        perf_group = QGroupBox("Paralel İndirme & Ağ Performansı")
        p_layout = QVBoxLayout()

        conc_layout = QHBoxLayout()
        conc_layout.addWidget(QLabel("Aynı Anda İndirilecek Dosya Sayısı (Paralel):"))
        self.spin_concurrency = QSpinBox()
        self.spin_concurrency.setRange(1, 4)
        self.spin_concurrency.setValue(2)
        self.spin_concurrency.valueChanged.connect(lambda val: self.settings.setValue("concurrency", val))
        conc_layout.addWidget(self.spin_concurrency)
        conc_layout.addStretch()
        p_layout.addLayout(conc_layout)

        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Aktarım Arabellek (Buffer) Boyutu:"))
        self.chunk_combo = QComboBox()
        self.chunk_combo.addItems(["1 MiB (Düşük Bellek)", "4 MiB (Önerilen)", "8 MiB (Gigabit Fiber / Ultra Hızlı)"])
        self.chunk_combo.setCurrentIndex(1)
        self.chunk_combo.currentIndexChanged.connect(lambda idx: self.settings.setValue("chunk_idx", idx))
        chunk_layout.addWidget(self.chunk_combo)
        chunk_layout.addStretch()
        p_layout.addLayout(chunk_layout)

        perf_group.setLayout(p_layout)
        layout.addWidget(perf_group)
        layout.addStretch()

    def _load_settings(self):
        default_dir = self.settings.value("download_dir", str(Path.home() / "HF_Models"))
        self.download_dir = Path(default_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.dir_input.setText(str(self.download_dir))

        token = self.settings.value("hf_token", "")
        self.token_input.setText(token)

        chunk_idx = self.settings.value("chunk_idx", 1, type=int)
        self.chunk_combo.setCurrentIndex(chunk_idx)

        concurrency = self.settings.value("concurrency", 2, type=int)
        self.spin_concurrency.setValue(concurrency)

        self._refresh_completed_list()

    def select_download_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Klasör Seçin", str(self.download_dir))
        if folder:
            self.download_dir = Path(folder)
            self.dir_input.setText(folder)
            self.settings.setValue("download_dir", folder)
            self._refresh_completed_list()
            self.scan_and_restore_part_files()

    def get_chunk_size(self) -> int:
        idx = self.chunk_combo.currentIndex()
        if idx == 0:
            return 1 * 1024 * 1024
        elif idx == 2:
            return 8 * 1024 * 1024
        return 4 * 1024 * 1024

    def _bind_signals(self):
        self.signals.error.connect(lambda msg: QMessageBox.critical(self, "Hata", msg))
        self.signals.search_results.connect(self._on_search_results_ready)
        self.signals.files_listed.connect(self._populate_search_table)
        self.signals.task_progress.connect(self._update_task_row_progress)
        self.signals.task_status.connect(self._update_task_row_status)
        self.signals.task_size_updated.connect(self._on_task_size_updated)

    # ------------------------------------------------------------------
    # Model Arama ve Dosya Listeleme
    # ------------------------------------------------------------------
    def search_models(self):
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir model adı yazın.")
            return

        self.repo_label.setText("Aranıyor...")
        self.file_table.setRowCount(0)
        self.match_combo.clear()
        self.match_combo.setVisible(False)
        self.btn_add_selected.setEnabled(False)
        self.btn_add_all.setEnabled(False)

        token = self.token_input.text().strip() or None

        if "/" in query:
            self._finalize_repo(query)
            return

        def _worker():
            try:
                models = list(self.api.list_models(search=query, limit=30, token=token))
                if not models:
                    self.signals.error.emit(f"'{query}' ile eşleşen model bulunamadı.")
                    return
                self.signals.search_results.emit(models)
            except Exception as e:
                self.signals.error.emit(f"Arama hatası: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_search_results_ready(self, models: List[ModelInfo]):
        self.match_combo.addItem("— Listeden Model Seçin —")
        for m in models:
            self.match_combo.addItem(m.modelId)
        self.match_combo.setVisible(True)
        self.match_combo.setCurrentIndex(0)
        self.repo_label.setText("Model seçimi bekleniyor...")

    def _on_combo_repo_selected(self, idx: int):
        if idx <= 0:
            return
        repo_id = self.match_combo.itemText(idx)
        self._finalize_repo(repo_id)

    def _finalize_repo(self, repo_id: str):
        self.current_repo_id = repo_id
        self.repo_label.setText(f"Aktif Depo: {repo_id}")
        token = self.token_input.text().strip() or None

        def _fetch_meta():
            try:
                sizes = fetch_exact_repo_sizes(self.api, repo_id, token)
                self.metadata_cache[repo_id] = sizes

                file_data = []
                for fn, sz in sizes.items():
                    file_data.append({"name": fn, "size": sz})

                self.signals.files_listed.emit(file_data)
            except Exception as e:
                self.signals.error.emit(f"Metadata alınamadı: {e}")

        threading.Thread(target=_fetch_meta, daemon=True).start()

    def _populate_search_table(self, file_data: List[Dict]):
        self.current_search_files = file_data
        self.file_table.setRowCount(0)

        for row, item in enumerate(file_data):
            self.file_table.insertRow(row)
            name_item = QTableWidgetItem(item["name"])
            size_item = QTableWidgetItem(human_readable_size(item["size"]))
            size_item.setTextAlignment(Qt.AlignCenter)
            
            is_lfs = item["size"] > 5 * 1024 * 1024
            lfs_item = QTableWidgetItem("Git LFS (Büyük Model)" if is_lfs else "Standart Dosya")
            lfs_item.setTextAlignment(Qt.AlignCenter)

            self.file_table.setItem(row, 0, name_item)
            self.file_table.setItem(row, 1, size_item)
            self.file_table.setItem(row, 2, lfs_item)

        self.btn_add_selected.setEnabled(True)
        self.btn_add_all.setEnabled(True)

    def _filter_search_table(self, text: str):
        query = text.lower()
        for row in range(self.file_table.rowCount()):
            item = self.file_table.item(row, 0)
            if item:
                self.file_table.setRowHidden(row, query not in item.text().lower())

    # ------------------------------------------------------------------
    # Kuyruğa Ekleme
    # ------------------------------------------------------------------
    def add_selected_to_queue(self):
        selected_rows = set(item.row() for item in self.file_table.selectedItems())
        if not selected_rows:
            QMessageBox.warning(self, "Uyarı", "Lütfen en az bir dosya seçin.")
            return

        targets = [self.file_table.item(r, 0).text() for r in selected_rows]
        files_to_add = [f for f in self.current_search_files if f["name"] in targets]
        self._add_files_to_queue(files_to_add)

    def add_all_to_queue(self):
        if hasattr(self, "current_search_files") and self.current_search_files:
            self._add_files_to_queue(self.current_search_files)

    def _add_files_to_queue(self, files_list: List[Dict]):
        repo_id = self.current_repo_id
        added = 0

        for f in files_list:
            fn = f["name"]
            dest_file = self.download_dir / repo_id / fn
            url = f"https://huggingface.co/{repo_id}/resolve/main/{fn}"

            if any(t.repo_id == repo_id and t.filename == fn for t in self.tasks.values()):
                continue

            task = DownloadTask(
                repo_id=repo_id,
                filename=fn,
                url=url,
                dest_path=dest_file,
                total_bytes=f["size"]
            )

            self.tasks[task.id] = task
            self._create_queue_table_row(task)
            added += 1

        if added > 0:
            self.tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Yarım Kalan (.part) Dosyaları Tarama
    # ------------------------------------------------------------------
    def scan_and_restore_part_files(self):
        if not self.download_dir.exists():
            return

        token = self.token_input.text().strip() or None

        for root, _, files in os.walk(self.download_dir):
            for f in files:
                if not f.endswith(".part"):
                    continue

                full_part_path = Path(root) / f
                actual_filename = f[:-5]
                actual_dest_path = Path(root) / actual_filename

                try:
                    rel_parts = full_part_path.relative_to(self.download_dir).parts
                    if len(rel_parts) >= 3:
                        repo_id = f"{rel_parts[0]}/{rel_parts[1]}"
                    elif len(rel_parts) == 2:
                        repo_id = rel_parts[0]
                    else:
                        repo_id = "Bilinmeyen-Depo"
                except Exception:
                    repo_id = "Bilinmeyen-Depo"

                exists = any(t.dest_path == actual_dest_path for t in self.tasks.values())
                if exists:
                    continue

                known_size = self.metadata_cache.get(repo_id, {}).get(actual_filename, 0)

                url = f"https://huggingface.co/{repo_id}/resolve/main/{actual_filename}"
                task = DownloadTask(
                    repo_id=repo_id,
                    filename=actual_filename,
                    url=url,
                    dest_path=actual_dest_path,
                    total_bytes=known_size
                )
                task.status = "Duraklatıldı"

                self.tasks[task.id] = task
                self._create_queue_table_row(task)

                if known_size == 0 and "/" in repo_id:
                    def _fetch_single_meta(t_id=task.id, r_id=repo_id, fn_name=actual_filename):
                        sizes = fetch_exact_repo_sizes(self.api, r_id, token)
                        self.metadata_cache[r_id] = sizes
                        real_sz = sizes.get(fn_name, 0)
                        if real_sz > 0:
                            self.signals.task_size_updated.emit(t_id, real_sz)

                    threading.Thread(target=_fetch_single_meta, daemon=True).start()

        self._refresh_completed_list()

    def _on_task_size_updated(self, task_id: str, new_size: int):
        task = self.tasks.get(task_id)
        if task:
            task.total_bytes = new_size
            row = self._get_row_by_task_id(task_id)
            if row >= 0:
                size_item = self.queue_table.item(row, 2)
                if size_item:
                    size_item.setText(f"{human_readable_size(task.downloaded_bytes)} / {human_readable_size(new_size)}")
                pbar = self.queue_table.cellWidget(row, 3)
                if isinstance(pbar, QProgressBar) and new_size > 0:
                    pbar.setValue(int(task.downloaded_bytes * 100 / new_size))

    # ------------------------------------------------------------------
    # Kuyruk Tablosu ve Hücre Yönetimi
    # ------------------------------------------------------------------
    def _create_queue_table_row(self, task: DownloadTask):
        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)

        repo_item = QTableWidgetItem(task.repo_id)
        repo_item.setData(Qt.UserRole, task.id)

        name_item = QTableWidgetItem(task.filename)
        
        total_str = human_readable_size(task.total_bytes) if task.total_bytes > 0 else "Alınıyor..."
        size_item = QTableWidgetItem(f"{human_readable_size(task.downloaded_bytes)} / {total_str}")
        size_item.setTextAlignment(Qt.AlignCenter)

        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pct = int(task.downloaded_bytes * 100 / task.total_bytes) if task.total_bytes > 0 else 0
        pbar.setValue(pct)
        pbar.setAlignment(Qt.AlignCenter)

        speed_item = QTableWidgetItem("0 B/s")
        speed_item.setTextAlignment(Qt.AlignCenter)

        eta_item = QTableWidgetItem("--:--:--")
        eta_item.setTextAlignment(Qt.AlignCenter)

        status_item = QTableWidgetItem(task.status)
        status_item.setTextAlignment(Qt.AlignCenter)

        action_widget = QWidget()
        act_layout = QHBoxLayout(action_widget)
        act_layout.setContentsMargins(2, 2, 2, 2)
        act_layout.setSpacing(4)

        btn_toggle = QPushButton("⏸ Durdur" if task.status == "İndiriliyor" else "▶ Devam")
        btn_toggle.clicked.connect(lambda _, tid=task.id: self._toggle_task_state(tid))

        btn_delete = QPushButton("🗑 Sil")
        btn_delete.clicked.connect(lambda _, tid=task.id: self._remove_task(tid))

        act_layout.addWidget(btn_toggle)
        act_layout.addWidget(btn_delete)

        self.queue_table.setItem(row, 0, repo_item)
        self.queue_table.setItem(row, 1, name_item)
        self.queue_table.setItem(row, 2, size_item)
        self.queue_table.setCellWidget(row, 3, pbar)
        self.queue_table.setItem(row, 4, speed_item)
        self.queue_table.setItem(row, 5, eta_item)
        self.queue_table.setItem(row, 6, status_item)
        self.queue_table.setCellWidget(row, 7, action_widget)

    def _get_row_by_task_id(self, task_id: str) -> int:
        for r in range(self.queue_table.rowCount()):
            item = self.queue_table.item(r, 0)
            if item and item.data(Qt.UserRole) == task_id:
                return r
        return -1

    def _update_task_row_progress(self, task_id: str, downloaded: int, total: int, speed: float, eta: float):
        row = self._get_row_by_task_id(task_id)
        if row < 0:
            return

        size_item = self.queue_table.item(row, 2)
        if size_item:
            size_item.setText(f"{human_readable_size(downloaded)} / {human_readable_size(total)}")

        pbar = self.queue_table.cellWidget(row, 3)
        if isinstance(pbar, QProgressBar) and total > 0:
            pbar.setValue(min(100, max(0, int(downloaded * 100 / total))))

        speed_item = self.queue_table.item(row, 4)
        if speed_item:
            speed_item.setText(f"{human_readable_size(int(speed))}/s")

        eta_item = self.queue_table.item(row, 5)
        if eta_item:
            eta_item.setText(format_time(eta))

    def _update_task_row_status(self, task_id: str, status: str):
        row = self._get_row_by_task_id(task_id)
        if row < 0:
            return

        status_item = self.queue_table.item(row, 6)
        if status_item:
            status_item.setText(status)

        action_widget = self.queue_table.cellWidget(row, 7)
        if action_widget:
            btn_toggle = action_widget.findChild(QPushButton)
            if btn_toggle:
                if status == "İndiriliyor":
                    btn_toggle.setText("⏸ Durdur")
                    btn_toggle.setEnabled(True)
                elif status in ["Duraklatıldı", "Kuyrukta", "Hata"]:
                    btn_toggle.setText("▶ Devam")
                    btn_toggle.setEnabled(True)
                elif status == "Tamamlandı":
                    btn_toggle.setText("✓ Bitti")
                    btn_toggle.setEnabled(False)

        if status == "Tamamlandı":
            self._refresh_completed_list()

    def _toggle_task_state(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task:
            return

        if task.status == "İndiriliyor":
            task.status = "Duraklatıldı"
            task.cancel_event.set()
            self._update_task_row_status(task_id, "Duraklatıldı")
        elif task.status in ["Duraklatıldı", "Hata"]:
            task.cancel_event.clear()
            task.status = "Kuyrukta"
            self._update_task_row_status(task_id, "Kuyrukta")

    def _remove_task(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task:
            return

        if task.status == "İndiriliyor":
            task.cancel_event.set()

        row = self._get_row_by_task_id(task_id)
        if row >= 0:
            self.queue_table.removeRow(row)
            del self.tasks[task_id]

    def pause_all_tasks(self):
        for task in self.tasks.values():
            if task.status in ["İndiriliyor", "Kuyrukta"]:
                task.status = "Duraklatıldı"
                task.cancel_event.set()
                self._update_task_row_status(task.id, "Duraklatıldı")

    def resume_all_tasks(self):
        for task in self.tasks.values():
            if task.status in ["Duraklatıldı", "Hata"]:
                task.cancel_event.clear()
                task.status = "Kuyrukta"
                self._update_task_row_status(task.id, "Kuyrukta")

    def clear_completed_tasks(self):
        to_remove = [tid for tid, t in self.tasks.items() if t.status == "Tamamlandı"]
        for tid in to_remove:
            self._remove_task(tid)

    # ------------------------------------------------------------------
    # Çoklu / Paralel Kuyruk Dağıtıcı (Thread-Safe Dispatcher)
    # ------------------------------------------------------------------
    def _queue_dispatcher_loop(self):
        while self.queue_active:
            time.sleep(0.3)

            with self.worker_lock:
                dead_workers = [tid for tid, th in self.active_workers.items() if not th.is_alive()]
                for tid in dead_workers:
                    del self.active_workers[tid]

                max_concurrent = self.spin_concurrency.value()
                available_slots = max_concurrent - len(self.active_workers)

                if available_slots > 0:
                    for tid, task in list(self.tasks.items()):
                        if available_slots <= 0:
                            break
                        if task.status == "Kuyrukta" and tid not in self.active_workers:
                            worker = threading.Thread(target=self._execute_download_task, args=(task,), daemon=True)
                            self.active_workers[tid] = worker
                            worker.start()
                            available_slots -= 1

    # ------------------------------------------------------------------
    # Kesintisiz İndirme Motoru (Auto-Reconnect & Full Size Guard)
    # ------------------------------------------------------------------
    def _execute_download_task(self, task: DownloadTask):
        task.status = "İndiriliyor"
        task.cancel_event.clear()
        self.signals.task_status.emit(task.id, "İndiriliyor")

        token = self.token_input.text().strip() or None
        chunk_size = self.get_chunk_size()
        dest_path = task.dest_path
        part_path = task.part_path

        # 1. Kesin dosya boyutu yoksa API'den çek
        if task.total_bytes <= 0:
            if task.repo_id in self.metadata_cache and task.filename in self.metadata_cache[task.repo_id]:
                task.total_bytes = self.metadata_cache[task.repo_id][task.filename]
            else:
                sizes = fetch_exact_repo_sizes(self.api, task.repo_id, token)
                self.metadata_cache[task.repo_id] = sizes
                task.total_bytes = sizes.get(task.filename, 0)

            if task.total_bytes > 0:
                self.signals.task_size_updated.emit(task.id, task.total_bytes)

        expected_size = task.total_bytes
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 2. Dosya diskte zaten tam mı kontrolü
        if dest_path.exists() and expected_size > 0 and dest_path.stat().st_size == expected_size:
            task.status = "Tamamlandı"
            self.signals.task_status.emit(task.id, "Tamamlandı")
            self.signals.task_progress.emit(task.id, expected_size, expected_size, 0, 0)
            return

        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        if expected_size > 0 and existing_bytes >= expected_size:
            if dest_path.exists():
                dest_path.unlink()
            part_path.rename(dest_path)
            task.status = "Tamamlandı"
            self.signals.task_status.emit(task.id, "Tamamlandı")
            self.signals.task_progress.emit(task.id, expected_size, expected_size, 0, 0)
            return

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=5, pool_maxsize=5, max_retries=3)
        session.mount("https://", adapter)

        speed_window = deque(maxlen=8)
        last_time = time.time()
        last_bytes = existing_bytes
        downloaded = existing_bytes

        # --- KESİNTİSİZ AKIŞ YENİLEYİCİ DÖNGÜ (AUTO-RECONNECT LOOP) ---
        # CDN 900MB veya 1GB'ta stream'i kapatsa dahi dosya boyutu tam dolana kadar döngü devam eder.
        while not task.cancel_event.is_set():
            if expected_size > 0 and downloaded >= expected_size:
                break

            headers = {"Accept-Encoding": "identity"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"

            try:
                with session.get(task.url, stream=True, timeout=(15, 30), headers=headers, allow_redirects=True) as r:
                    if r.status_code == 416:
                        # Range hatası -> dosya bitmiş olabilir veya sıfırlanmalı
                        if expected_size > 0 and downloaded >= expected_size:
                            break
                        downloaded = 0
                        part_path.unlink(missing_ok=True)
                        continue

                    r.raise_for_status()

                    # 200 dönerse sıfırdan başla, 206 dönerse kaldığı yere ekle
                    if r.status_code == 200:
                        mode = "wb"
                        downloaded = 0
                    else:
                        mode = "ab"

                    # API metadata boyutu yoksa HTTP başlıklarından GERÇEK TOPLAM boyutu bul.
                    # 206 cevaplarında Content-Range: bytes START-END/TOTAL
                    # 200 cevaplarında ise Content-Length doğrudan dosyanın toplam boyutudur.
                    if expected_size <= 0:
                        detected_total = 0
                        cr = r.headers.get("Content-Range", "").strip()

                        if cr and "/" in cr:
                            total_str = cr.rsplit("/", 1)[-1].strip()
                            if total_str.isdigit():
                                detected_total = int(total_str)

                        if detected_total <= 0:
                            content_length = r.headers.get("Content-Length", "").strip()
                            if content_length.isdigit():
                                remaining = int(content_length)
                                # Range isteği kabul edilmişse Content-Length sadece kalan kısmı gösterir.
                                detected_total = downloaded + remaining if r.status_code == 206 else remaining

                        if detected_total > 0:
                            expected_size = detected_total
                            task.total_bytes = expected_size
                            self.signals.task_size_updated.emit(task.id, expected_size)

                    with open(part_path, mode, buffering=chunk_size) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if task.cancel_event.is_set():
                                return

                            if not chunk:
                                continue

                            f.write(chunk)
                            downloaded += len(chunk)
                            task.downloaded_bytes = downloaded

                            now = time.time()
                            dt = now - last_time
                            if dt >= 0.3:
                                instant_speed = (downloaded - last_bytes) / dt
                                speed_window.append(instant_speed)
                                avg_speed = sum(speed_window) / len(speed_window) if speed_window else 0
                                eta = (expected_size - downloaded) / avg_speed if (avg_speed > 0 and expected_size > downloaded) else 0

                                task.speed = avg_speed
                                task.eta = eta
                                self.signals.task_progress.emit(task.id, downloaded, expected_size, avg_speed, eta)

                                last_time = now
                                last_bytes = downloaded

                # Eğer sunucu stream'i erken kapattıysa ama dosya henüz bitmediyse
                if expected_size > 0 and downloaded < expected_size and not task.cancel_event.is_set():
                    # Bağlantı koptu, 1 saniye bekle ve Range ile kaldığı yerden devam et
                    time.sleep(1)
                    continue
                else:
                    break

            except (requests.exceptions.RequestException, IOError) as net_err:
                if task.cancel_event.is_set():
                    return
                # Ağ hatasında (timeout/reset) 2 saniye bekleyip kaldığı bayttan tekrar bağlan
                time.sleep(2)
                continue

        # --- BİTİŞ KONTROLÜ VE DOĞRULAMA ---
        if not task.cancel_event.is_set():
            # Dosya diskte gerçekten expected_size kadar mı kontrol et
            actual_on_disk = part_path.stat().st_size if part_path.exists() else 0
            if expected_size <= 0:
                # Toplam boyut hâlâ bilinmiyorsa .part dosyasını tamamlandı diye
                # yeniden adlandırma. Böylece eksik stream yanlışlıkla tam dosya olmaz.
                task.status = "Duraklatıldı"
                self.signals.task_status.emit(task.id, "Boyut Bilinmiyor (Devam Et)")
            elif actual_on_disk < expected_size:
                task.status = "Duraklatıldı"
                self.signals.task_status.emit(task.id, "Koptu (Devam Et)")
            elif actual_on_disk > expected_size:
                task.status = "Hata"
                self.signals.task_status.emit(task.id, "Boyut Hatası")
            else:
                if part_path.exists():
                    if dest_path.exists():
                        dest_path.unlink()
                    part_path.rename(dest_path)
                task.status = "Tamamlandı"
                self.signals.task_progress.emit(task.id, expected_size, expected_size, 0, 0)
                self.signals.task_status.emit(task.id, "Tamamlandı")

    # ------------------------------------------------------------------
    # Yerel Dosyalar Sekmesi
    # ------------------------------------------------------------------
    def _refresh_completed_list(self):
        self.completed_table.setRowCount(0)
        if not self.download_dir.exists():
            return

        row = 0
        for root, _, files in os.walk(self.download_dir):
            for f in files:
                full_p = Path(root) / f
                rel_p = full_p.relative_to(self.download_dir)
                size_str = human_readable_size(full_p.stat().st_size)
                mtime_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(full_p.stat().st_mtime))

                self.completed_table.insertRow(row)
                self.completed_table.setItem(row, 0, QTableWidgetItem(str(rel_p)))
                self.completed_table.setItem(row, 1, QTableWidgetItem(size_str))

                is_part = f.endswith(".part")
                status_item = QTableWidgetItem("⏳ Yarım Kaldı (.part)" if is_part else "✓ Tamamlandı")
                status_item.setTextAlignment(Qt.AlignCenter)
                if is_part:
                    status_item.setForeground(QColor("#e67e22"))
                else:
                    status_item.setForeground(QColor("#27ae60"))
                self.completed_table.setItem(row, 2, status_item)
                self.completed_table.setItem(row, 3, QTableWidgetItem(mtime_str))

                act_widget = QWidget()
                act_layout = QHBoxLayout(act_widget)
                act_layout.setContentsMargins(2, 2, 2, 2)
                act_layout.setSpacing(4)

                if is_part:
                    btn_resume = QPushButton("▶ Devam Et")
                    btn_resume.clicked.connect(lambda _, fp=full_p: self._resume_part_from_disk(fp))
                    act_layout.addWidget(btn_resume)

                btn_del = QPushButton("🗑 Sil")
                btn_del.clicked.connect(lambda _, fp=full_p: self._delete_disk_file(fp))
                act_layout.addWidget(btn_del)

                self.completed_table.setCellWidget(row, 4, act_widget)
                row += 1

    def _resume_part_from_disk(self, part_path: Path):
        actual_filename = part_path.name[:-5]
        dest_path = part_path.with_name(actual_filename)

        try:
            rel_parts = part_path.relative_to(self.download_dir).parts
            if len(rel_parts) >= 3:
                repo_id = f"{rel_parts[0]}/{rel_parts[1]}"
            elif len(rel_parts) == 2:
                repo_id = rel_parts[0]
            else:
                repo_id = "Bilinmeyen-Depo"
        except Exception:
            repo_id = "Bilinmeyen-Depo"

        task = None
        for t in self.tasks.values():
            if t.dest_path == dest_path:
                task = t
                break

        if not task:
            known_size = self.metadata_cache.get(repo_id, {}).get(actual_filename, 0)
            url = f"https://huggingface.co/{repo_id}/resolve/main/{actual_filename}"
            task = DownloadTask(repo_id=repo_id, filename=actual_filename, url=url, dest_path=dest_path, total_bytes=known_size)
            self.tasks[task.id] = task
            self._create_queue_table_row(task)

        task.status = "Kuyrukta"
        task.cancel_event.clear()
        self._update_task_row_status(task.id, "Kuyrukta")
        self.tabs.setCurrentIndex(1)

    def _delete_disk_file(self, path: Path):
        if QMessageBox.question(self, "Onay", f"{path.name} dosyası silinsin mi?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            try:
                path.unlink()
                for tid, t in list(self.tasks.items()):
                    if t.dest_path == path or t.part_path == path:
                        self._remove_task(tid)
                self._refresh_completed_list()
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya silinemedi: {e}")

    def open_download_dir(self):
        if sys.platform == "win32":
            os.startfile(self.download_dir)
        elif sys.platform == "darwin":
            os.system(f'open "{self.download_dir}"')
        else:
            os.system(f'xdg-open "{self.download_dir}"')

    def closeEvent(self, event):
        self.queue_active = False
        for task in self.tasks.values():
            task.cancel_event.set()
        event.accept()

# ----------------------------------------------------------------------
# Uygulama Başlatıcı
# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()