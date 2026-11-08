"""GUI для Industrial Vision на CustomTkinter."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

from src.capture.video_capture import VideoSource
from src.capture.fps import FPSMeter
from src.inference.model_manager import ModelManager
from src.config import load_config
from src.events.event_engine import EventEngine
from src.events.evidence import EvidenceManager
from src.logging.setup import setup_logging


class VisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Industrial Vision — Промышленный контроль")
        self.geometry("1400x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Состояние
        self.is_running = False
        self.video_thread: threading.Thread | None = None
        self.current_frame = None
        self.fps_value = 0.0
        self.event_count = 0

        # Конфиг и модели
        self.config = load_config()
        log_cfg = self.config["logging"]
        setup_logging(log_cfg["dir"], log_cfg["level"], log_cfg["max_bytes"], log_cfg["backup_count"])

        self.model_manager = ModelManager(self.config)
        self.event_engine = EventEngine(self.config)
        self.evidence_mgr = EvidenceManager(
            self.config["events"]["evidence_dir"],
            self.config["events"]["snapshot_format"],
        )

        self._build_ui()
        self._update_stats_loop()

    # ================================================================
    # UI
    # ================================================================
    def _build_ui(self):
        # Левая панель
        left = ctk.CTkFrame(self, width=320, corner_radius=10)
        left.pack(side="left", fill="y", padx=10, pady=10)
        left.pack_propagate(False)

        ctk.CTkLabel(
            left, text="Industrial Vision",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(20, 10))

        # Источник
        ctk.CTkLabel(left, text="Источник видео", anchor="w").pack(pady=(10, 0), padx=10, fill="x")
        self.source_type = ctk.CTkOptionMenu(left, values=["Файл", "Камера", "RTSP"])
        self.source_type.pack(pady=5, padx=10, fill="x")

        self.source_entry = ctk.CTkEntry(left, placeholder_text="Путь к файлу / индекс / URL")
        self.source_entry.pack(pady=5, padx=10, fill="x")

        ctk.CTkButton(
            left, text="📁 Выбрать файл",
            command=self._pick_file,
            fg_color="gray30", hover_color="gray40",
        ).pack(pady=5, padx=10, fill="x")

        # Кнопки управления
        ctk.CTkButton(
            left, text="▶  Старт",
            command=self._start,
            fg_color="#2e7d32", hover_color="#1b5e20",
            height=45, font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(pady=(20, 5), padx=10, fill="x")

        ctk.CTkButton(
            left, text="⏹  Стоп",
            command=self._stop,
            fg_color="#c62828", hover_color="#8e0000",
            height=45, font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(pady=5, padx=10, fill="x")

        # Статистика
        stats_frame = ctk.CTkFrame(left, corner_radius=10)
        stats_frame.pack(pady=20, padx=10, fill="x")

        ctk.CTkLabel(stats_frame, text="Статистика", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        self.fps_label = ctk.CTkLabel(stats_frame, text="FPS: —", font=ctk.CTkFont(size=16))
        self.fps_label.pack(pady=2)
        self.events_label = ctk.CTkLabel(stats_frame, text="Событий: 0", font=ctk.CTkFont(size=14))
        self.events_label.pack(pady=(2, 10))

        # Список событий
        ctk.CTkLabel(left, text="Последние события", anchor="w").pack(pady=(5, 0), padx=10, fill="x")
        self.events_box = ctk.CTkTextbox(left, height=200)
        self.events_box.pack(pady=5, padx=10, fill="both", expand=True)

        # Правая панель (видео)
        right = ctk.CTkFrame(self, corner_radius=10)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(right, text="Нажмите «Старт» для запуска", font=ctk.CTkFont(size=16))
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

    # ================================================================
    # Обработчики
    # ================================================================
    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[("Видео", "*.mp4 *.avi *.mkv *.mov"), ("Все файлы", "*.*")],
        )
        if path:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, path)

    def _start(self):
        if self.is_running:
            return

        raw_source = self.source_entry.get().strip()
        if not raw_source:
            messagebox.showwarning("Ошибка", "Укажите источник видео")
            return

        # Камера — конвертируем в int
        if self.source_type.get() == "Камера":
            try:
                source = int(raw_source)
            except ValueError:
                messagebox.showerror("Ошибка", "Для камеры нужен индекс (0, 1, ...)")
                return
        else:
            source = raw_source

        self.is_running = True
        self.events_box.delete("1.0", "end")
        self.event_count = 0

        self.video_thread = threading.Thread(target=self._video_loop, args=(source,), daemon=True)
        self.video_thread.start()

    def _stop(self):
        self.is_running = False
        self.video_label.configure(image="", text="Остановлено")

    # ================================================================
    # Видео-поток (в отдельном потоке!)
    # ================================================================
    def _video_loop(self, source):
        try:
            src = VideoSource(source, reconnect_sec=1.0)
            src.open()
        except Exception as exc:
            self._log_event(f"ОШИБКА: {exc}")
            self.is_running = False
            return

        # Модели (ленивая загрузка при первом обращении)
        person_tracker = self.model_manager.get_tracker("person_tracker")
        ppe_detector = self.model_manager.get_detector("ppe_detector")
        sf_detector = self.model_manager.get_detector("smoke_fire_detector")
        spill_detector = self.model_manager.get_detector("spill_detector")

        meter = FPSMeter(window=30)
        frame_idx = 0
        last_ppe_boxes = []
        last_spill_dets = []
        last_sf_dets = []

        while self.is_running:
            try:
                frame = src.read()
            except RuntimeError:
                break

            frame_idx += 1

            # Трекинг каждый кадр
            persons = person_tracker.update(frame)

            # Огонь каждый 3-й
            if frame_idx % 3 == 0:
                sf_dets = sf_detector.predict(frame)
                last_sf_dets = sf_dets
            else:
                sf_dets = last_sf_dets

            # Разлив каждый 5-й
            if frame_idx % 5 == 0:
                spill_dets = spill_detector.predict(frame)
                last_spill_dets = spill_dets
            else:
                spill_dets = last_spill_dets

            # EventEngine каждый 5-й
            if frame_idx % 5 == 0:
                events = self.event_engine.process_frame(
                    persons, ppe_detector, sf_dets, spill_dets, frame
                )
                last_ppe_boxes = self.event_engine.last_ppe_detections
                for ev in events:
                    self.evidence_mgr.save(ev, frame)
                    self._log_event(f"{ev.event_type}  (ID: {ev.track_id})")
            else:
                events = []

            # Отрисовка
            for d in persons:
                if d.class_name == "person":
                    cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{d.track_id}", (d.x1, d.y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for d in last_ppe_boxes:
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 255), 1)
                cv2.putText(frame, d.class_name, (d.x1, d.y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            for d in sf_dets:
                color = (0, 0, 255) if d.class_name == "Fire" else (128, 128, 128)
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
                cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (d.x1, d.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for d in spill_dets:
                cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Spill {d.confidence:.2f}", (d.x1, d.y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            self.fps_value = meter.tick()
            cv2.putText(frame, f"FPS: {self.fps_value:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Обновление картинки в главном потоке
            self.after(0, self._update_frame, frame)

        src.close()
        self.is_running = False

    # ================================================================
    # Обновление UI (в главном потоке)
    # ================================================================
    def _update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Подгонка под размер окна (с сохранением пропорций)
        label_w = max(self.video_label.winfo_width(), 400)
        label_h = max(self.video_label.winfo_height(), 300)
        img.thumbnail((label_w, label_h), Image.LANCZOS)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img  # критично — иначе Python соберёт мусор

    def _log_event(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.event_count += 1
        self.after(0, lambda: (
            self.events_box.insert("end", f"[{ts}] {text}\n"),
            self.events_box.see("end"),
        ))

    def _update_stats_loop(self):
        self.fps_label.configure(text=f"FPS: {self.fps_value:.1f}")
        self.events_label.configure(text=f"Событий: {self.event_count}")
        self.after(200, self._update_stats_loop)


if __name__ == "__main__":
    app = VisionApp()
    app.mainloop()