"""
Owner: Thành viên 4 - GUI / Integration.

CustomTkinter desktop UI with sidebar layout and Matplotlib visualization.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Literal

import customtkinter as ctk
import numpy as np
from CTkToolTip import CTkToolTip
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

from app.controller import (
    CsvPreviewData,
    PreprocessResult,
    TestEvaluationResult,
    TrainingResult,
    load_csv_for_preview,
    prepare_csv_preview,
    preprocess_csv_selection,
    train_from_csv_selection,
    train_from_text,
)
from core.preprocessing import PreprocessOptions
from data.sample_data import SAMPLE_DATASET_NAMES, make_sample_dataset
from visualization.animation import next_frame_index
from visualization.plotter import (
    PlotArtists,
    PreviewArtists,
    TrainingFrameView,
    draw_csv_preview,
    draw_empty_plot,
    draw_training_frame,
    init_training_artists,
)
from visualization.theme import ThemeMode

VizMode = Literal["empty", "data_preview", "training"]


def _make_icon(text: str, size: int = 24) -> ctk.CTkImage:
    """Render a simple text icon for control buttons."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((4, 2), text, fill=(255, 255, 255, 255))
    return ctk.CTkImage(light_image=image, dark_image=image, size=(size, size))


class PolynomialRegressionApp(ctk.CTk):
    """Main GUI application."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Polynomial Regression Visualizer")
        self.geometry("1280x820")
        self.minsize(1100, 700)

        self.theme_mode: ThemeMode = "dark"
        self.viz_mode: VizMode = "empty"

        self.model = None
        self.x_data: np.ndarray | None = None
        self.y_data: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.visual_feature_index = 0
        self.target_column_name = "y"
        self.training_result: TrainingResult | None = None
        self._training_tree_row_ids: list[str] = []

        self.csv_preview: CsvPreviewData | None = None
        self.preprocess_result: PreprocessResult | None = None
        self._train_csv_config: dict[str, object] | None = None
        self.csv_path_var = tk.StringVar(value="No file selected")
        self.drop_missing_var = tk.BooleanVar(value=True)
        self.remove_outliers_var = tk.BooleanVar(value=True)
        self.iqr_multiplier_var = tk.DoubleVar(value=1.5)
        self.normalize_target_var = tk.BooleanVar(value=True)
        self.manual_text_default = "x,y\n-2.0,5.1\n-1.0,3.2\n0.0,1.0\n1.0,2.8\n2.0,6.5\n3.0,10.2"
        self.sample_text: str | None = None

        self.frame_index = 0
        self.playing = False
        self.animation_speed = 50
        self._animation_job: str | None = None
        self._train_thread: threading.Thread | None = None

        self.plot_artists = PlotArtists()
        self.preview_artists = PreviewArtists()
        self.test_eval_window: ctk.CTkToplevel | None = None
        self._last_test_evaluation: TestEvaluationResult | None = None
        self._test_eval_visible = True

        self._build_layout()
        self._show_empty_plots()
        self._set_animation_enabled(False)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="Polynomial AI",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar, corner_radius=0)
        self.sidebar_scroll.grid(row=1, column=0, sticky="nsew")

        self.input_tabs = ctk.CTkTabview(self.sidebar_scroll, width=280)
        self.input_tabs.pack(fill="x", padx=12, pady=8)
        self._build_manual_tab()
        self._build_csv_tab()
        self._build_sample_tab()

        ctk.CTkLabel(self.sidebar_scroll, text="Hyperparameters", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=16, pady=(12, 4)
        )

        self.degree_var = tk.IntVar(value=3)
        self.lr_var = tk.DoubleVar(value=0.03)
        self.epochs_var = tk.IntVar(value=500)
        self.save_every_var = tk.IntVar(value=1)

        degree_label = self._add_labeled_slider(self.sidebar_scroll, "Degree", self.degree_var, 1, 10, is_int=True)
        CTkToolTip(degree_label, message="Bậc đa thức (1=linear, 2=quadratic, ...)")

        lr_label = self._add_labeled_slider(
            self.sidebar_scroll, "Learning Rate", self.lr_var, 0.001, 0.2, is_int=False
        )
        CTkToolTip(lr_label, message="Tốc độ cập nhật weights mỗi epoch")

        epochs_label = self._add_labeled_slider(
            self.sidebar_scroll, "Epochs", self.epochs_var, 100, 2000, is_int=True
        )
        CTkToolTip(epochs_label, message="Số vòng lặp Gradient Descent")

        save_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        save_frame.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(save_frame, text="Save every").pack(side="left")
        ctk.CTkEntry(save_frame, textvariable=self.save_every_var, width=60).pack(side="right")

        self.sidebar_actions = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.sidebar_actions.grid(row=2, column=0, sticky="sew", padx=0, pady=0)
        self.sidebar_actions.grid_columnconfigure(0, weight=1)

        self.train_button = ctk.CTkButton(
            self.sidebar_actions,
            text="Train Model",
            height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.on_train_clicked,
        )
        self.train_button.grid(row=0, column=0, padx=16, pady=(8, 4), sticky="ew")

        self.reset_button = ctk.CTkButton(
            self.sidebar_actions,
            text="Reset",
            fg_color="transparent",
            border_width=1,
            command=self.on_reset_clicked,
        )
        self.reset_button.grid(row=1, column=0, padx=16, pady=4, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.sidebar_actions, mode="indeterminate")
        self.progress_bar.grid(row=2, column=0, padx=16, pady=4, sticky="ew")
        self.progress_bar.grid_remove()

        self.status_label = ctk.CTkLabel(self.sidebar_actions, text="Ready", text_color="gray70")
        self.status_label.grid(row=3, column=0, padx=16, pady=(4, 12), sticky="w")

        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.main.grid_rowconfigure(4, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        self._build_csv_preview_section()
        self._build_training_data_section()
        self._build_equation_bar()
        self._init_test_evaluation_state()
        self._build_chart_area()
        self._build_animation_bar()

    def _add_labeled_slider(
        self,
        parent: ctk.CTkFrame,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        is_int: bool,
    ) -> ctk.CTkLabel:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=16, pady=4)
        value_label = ctk.CTkLabel(frame, text="{}: {}".format(label, variable.get()))
        value_label.pack(anchor="w")

        def on_change(value: float) -> None:
            if is_int:
                variable.set(int(round(float(value))))
            else:
                variable.set(round(float(value), 4))
            value_label.configure(text="{}: {}".format(label, variable.get()))

        ctk.CTkSlider(
            frame,
            from_=from_,
            to=to,
            number_of_steps=int(to - from_) if is_int else 100,
            command=on_change,
        ).pack(fill="x", pady=(4, 0))
        return value_label

    def _build_manual_tab(self) -> None:
        tab = self.input_tabs.add("Manual")
        tab.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(tab, text="Enter x,y data:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.manual_text = ctk.CTkTextbox(tab, height=180)
        self.manual_text.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.manual_text.insert("1.0", self.manual_text_default)
        ctk.CTkButton(tab, text="Load Default", command=self._load_manual_default).grid(
            row=2, column=0, sticky="ew", padx=4, pady=4
        )

    def _build_csv_tab(self) -> None:
        tab = self.input_tabs.add("CSV")
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(tab, text="Browse CSV File", command=self.on_browse_csv).grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ctk.CTkLabel(tab, textvariable=self.csv_path_var, wraplength=250, justify="left").grid(
            row=1, column=0, sticky="w", padx=4, pady=2
        )

        ctk.CTkLabel(tab, text="Feature columns (Ctrl+click):").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 2))
        feature_frame = ctk.CTkFrame(tab)
        feature_frame.grid(row=3, column=0, sticky="nsew", padx=4, pady=2)
        self.feature_listbox = tk.Listbox(
            feature_frame,
            selectmode=tk.EXTENDED,
            height=5,
            bg="#2b2b2b",
            fg="#E0E0E0",
            selectbackground="#1f538d",
            borderwidth=0,
            highlightthickness=0,
        )
        self.feature_listbox.pack(fill="both", expand=True)
        self.feature_listbox.bind("<<ListboxSelect>>", lambda _e: self.on_csv_columns_changed())

        ctk.CTkLabel(tab, text="Target column:").grid(row=4, column=0, sticky="w", padx=4, pady=(8, 2))
        self.target_combo = ctk.CTkComboBox(tab, values=["—"], command=lambda _v: self.on_csv_columns_changed())
        self.target_combo.grid(row=5, column=0, sticky="ew", padx=4, pady=2)

        ctk.CTkLabel(tab, text="Visual X column:").grid(row=6, column=0, sticky="w", padx=4, pady=(8, 2))
        self.visual_combo = ctk.CTkComboBox(tab, values=["—"], command=lambda _v: self.on_csv_columns_changed())
        self.visual_combo.grid(row=7, column=0, sticky="ew", padx=4, pady=2)

        ctk.CTkLabel(tab, text="Pre-processing", font=ctk.CTkFont(weight="bold")).grid(
            row=8, column=0, sticky="w", padx=4, pady=(10, 2)
        )
        ctk.CTkCheckBox(
            tab,
            text="Drop null / N-A rows",
            variable=self.drop_missing_var,
        ).grid(row=9, column=0, sticky="w", padx=4, pady=2)
        ctk.CTkCheckBox(
            tab,
            text="Remove outliers (residual)",
            variable=self.remove_outliers_var,
        ).grid(row=10, column=0, sticky="w", padx=4, pady=2)

        iqr_frame = ctk.CTkFrame(tab, fg_color="transparent")
        iqr_frame.grid(row=11, column=0, sticky="ew", padx=4, pady=2)
        ctk.CTkLabel(iqr_frame, text="IQR multiplier").pack(side="left")
        ctk.CTkEntry(iqr_frame, textvariable=self.iqr_multiplier_var, width=60).pack(side="right")

        ctk.CTkCheckBox(
            tab,
            text="Normalize target (÷1e9)",
            variable=self.normalize_target_var,
        ).grid(row=12, column=0, sticky="w", padx=4, pady=2)

        ctk.CTkButton(tab, text="Pre-process Data", command=self.on_preprocess_clicked).grid(
            row=13, column=0, sticky="ew", padx=4, pady=4
        )
        self.preprocess_report_label = ctk.CTkLabel(tab, text="", justify="left", wraplength=250)
        self.preprocess_report_label.grid(row=14, column=0, sticky="w", padx=4, pady=2)

        self.csv_stats_label = ctk.CTkLabel(tab, text="", justify="left")
        self.csv_stats_label.grid(row=15, column=0, sticky="w", padx=4, pady=4)

    def _build_sample_tab(self) -> None:
        tab = self.input_tabs.add("Sample")
        tab.grid_rowconfigure(2, weight=1)
        ctk.CTkLabel(tab, text="Choose a synthetic dataset:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.sample_combo = ctk.CTkComboBox(tab, values=list(SAMPLE_DATASET_NAMES))
        self.sample_combo.set("Cubic")
        self.sample_combo.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.sample_preview = ctk.CTkTextbox(tab, height=180)
        self.sample_preview.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        self.sample_preview.configure(state="disabled")
        ctk.CTkButton(tab, text="Generate", command=self.on_generate_sample).grid(
            row=3, column=0, sticky="ew", padx=4, pady=4
        )

    def _build_csv_preview_section(self) -> None:
        self.preview_header = ctk.CTkFrame(self.main, fg_color="transparent")
        self.preview_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.preview_header.grid_columnconfigure(0, weight=1)

        self.preview_toggle_btn = ctk.CTkButton(
            self.preview_header,
            text="▼ CSV Table Preview",
            width=180,
            fg_color="transparent",
            border_width=1,
            command=self._toggle_csv_table,
        )
        self.preview_toggle_btn.grid(row=0, column=0, sticky="w")
        self.preview_header.grid_remove()

        self.table_frame = ctk.CTkFrame(self.main)
        self.table_frame.grid(row=1, column=0, sticky="ew", pady=4)
        self.table_frame.grid_columnconfigure(0, weight=1)
        self.table_frame.grid_remove()
        self._table_visible = True

        self.csv_tree = ttk.Treeview(self.table_frame, show="headings", height=5)
        self.csv_tree.grid(row=0, column=0, sticky="ew")
        tree_scroll = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.csv_tree.xview)
        tree_scroll.grid(row=1, column=0, sticky="ew")
        self.csv_tree.configure(xscrollcommand=tree_scroll.set)

        self.cards_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.cards_frame.grid(row=2, column=0, sticky="ew", pady=4)
        self.cards_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self.cards_frame.grid_remove()

        self.stat_cards: dict[str, ctk.CTkLabel] = {}
        for index, key in enumerate(["count", "mean_x", "std_y", "missing", "outliers"]):
            card = ctk.CTkFrame(self.cards_frame, corner_radius=10)
            card.grid(row=0, column=index, padx=4, sticky="ew")
            title = ctk.CTkLabel(card, text=key.replace("_", " ").title(), text_color="gray70")
            title.pack(padx=12, pady=(8, 0))
            value = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=16, weight="bold"))
            value.pack(padx=12, pady=(0, 8))
            self.stat_cards[key] = value

    def _build_training_data_section(self) -> None:
        """Table showing all x/y rows with per-frame y_pred and residual."""
        self.training_header = ctk.CTkFrame(self.main, fg_color="transparent")
        self.training_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.training_header.grid_columnconfigure(1, weight=1)
        self.training_header.grid_remove()

        self.training_toggle_btn = ctk.CTkButton(
            self.training_header,
            text="▼ Training Data (x, y, y_pred, residual)",
            width=320,
            fg_color="transparent",
            border_width=1,
            command=self._toggle_training_table,
        )
        self.training_toggle_btn.grid(row=0, column=0, sticky="w")

        self.training_summary_label = ctk.CTkLabel(
            self.training_header,
            text="",
            text_color="gray70",
            anchor="e",
        )
        self.training_summary_label.grid(row=0, column=1, sticky="e", padx=8)

        self.training_table_frame = ctk.CTkFrame(self.main)
        self.training_table_frame.grid(row=1, column=0, sticky="ew", pady=4)
        self.training_table_frame.grid_columnconfigure(0, weight=1)
        self.training_table_frame.grid_remove()
        self._training_table_visible = True

        columns = ("idx", "x", "y", "y_pred", "residual")
        self.training_tree = ttk.Treeview(
            self.training_table_frame,
            columns=columns,
            show="headings",
            height=6,
        )
        self.training_tree.grid(row=0, column=0, sticky="ew")
        y_scroll = ttk.Scrollbar(self.training_table_frame, orient="vertical", command=self.training_tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(self.training_table_frame, orient="horizontal", command=self.training_tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.training_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        for col, heading, width in (
            ("idx", "#", 50),
            ("x", "x", 120),
            ("y", "y", 120),
            ("y_pred", "y_pred", 120),
            ("residual", "residual", 120),
        ):
            self.training_tree.heading(col, text=heading)
            self.training_tree.column(col, width=width, anchor="center")

    def _toggle_training_table(self) -> None:
        self._training_table_visible = not self._training_table_visible
        if self._training_table_visible:
            self.training_table_frame.grid()
            self.training_toggle_btn.configure(text="▼ Training Data (x, y, y_pred, residual)")
        else:
            self.training_table_frame.grid_remove()
            self.training_toggle_btn.configure(text="▶ Training Data (x, y, y_pred, residual)")

    def _populate_training_data_table(self, x_data: np.ndarray, y_data: np.ndarray, feature_names: list[str]) -> None:
        """Fill training table with static feature/x and y columns."""
        self.training_tree.delete(*self.training_tree.get_children())
        self._training_tree_row_ids = []

        x_array = np.asarray(x_data, dtype=float)
        y_array = np.asarray(y_data, dtype=float).ravel()
        if x_array.ndim == 1:
            x_array = x_array.reshape(-1, 1)

        columns = ["idx"] + list(feature_names) + [self.target_column_name, "y_pred", "residual"]
        self.training_tree["columns"] = columns
        for col in columns:
            if col == "idx":
                self.training_tree.heading(col, text="#")
                self.training_tree.column(col, width=50, anchor="center")
            elif col == "y_pred":
                self.training_tree.heading(col, text="y_pred")
                self.training_tree.column(col, width=110, anchor="center")
            elif col == "residual":
                self.training_tree.heading(col, text="residual")
                self.training_tree.column(col, width=110, anchor="center")
            else:
                self.training_tree.heading(col, text=col)
                self.training_tree.column(col, width=110, anchor="center")

        for index in range(len(y_array)):
            feature_values = [f"{x_array[index, col]:.4f}" for col in range(x_array.shape[1])]
            row_id = self.training_tree.insert(
                "",
                "end",
                values=[str(index + 1), *feature_values, f"{y_array[index]:.4f}", "—", "—"],
            )
            self._training_tree_row_ids.append(row_id)

    def _update_training_data_rows(self, frame_view: TrainingFrameView) -> None:
        """Update y_pred and residual columns for the current animation frame."""
        if not self._training_tree_row_ids:
            return
        for index, row_id in enumerate(self._training_tree_row_ids):
            if index >= len(frame_view.y_pred):
                break
            current_values = list(self.training_tree.item(row_id, "values"))
            current_values[-2] = f"{frame_view.y_pred[index]:.4f}"
            current_values[-1] = f"{frame_view.residuals[index]:.4f}"
            self.training_tree.item(row_id, values=current_values)

        self.training_summary_label.configure(
            text="Epoch {}/{} | Loss: {:.6f} | n={}".format(
                frame_view.epoch,
                self.model.frames[-1].epoch if self.model and self.model.frames else frame_view.epoch,
                frame_view.loss,
                len(frame_view.y_actual),
            )
        )

    def _build_equation_bar(self) -> None:
        self.equation_frame = ctk.CTkFrame(self.main, corner_radius=8)
        self.equation_frame.grid(row=2, column=0, sticky="ew", pady=4)
        self.equation_frame.grid_remove()
        ctk.CTkLabel(self.equation_frame, text="Equation", text_color="gray70").pack(anchor="w", padx=12, pady=(8, 0))
        self.equation_label = ctk.CTkLabel(
            self.equation_frame,
            text="y = ...",
            font=ctk.CTkFont(family="Consolas", size=15),
            anchor="w",
        )
        self.equation_label.pack(fill="x", padx=12, pady=(0, 4))
        self.frame_detail_label = ctk.CTkLabel(
            self.equation_frame,
            text="Weights: —",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color="gray70",
            anchor="w",
        )
        self.frame_detail_label.pack(fill="x", padx=12, pady=(0, 8))

    def _init_test_evaluation_state(self) -> None:
        self.test_metric_labels: dict[str, ctk.CTkLabel] = {}
        self.test_metric_cards: list[ctk.CTkFrame] = []

    def _ensure_test_eval_window(self) -> ctk.CTkToplevel:
        if self.test_eval_window is not None and self.test_eval_window.winfo_exists():
            return self.test_eval_window

        window = ctk.CTkToplevel(self)
        window.title("Test Accuracy (20% hold-out)")
        window.geometry("760x480")
        window.minsize(640, 360)
        window.transient(self)

        container = ctk.CTkFrame(window, corner_radius=0)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        container.grid_columnconfigure((0, 1, 2), weight=1)
        container.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(container, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(header, text="Test Accuracy — 80/20 split", font=ctk.CTkFont(size=18, weight="bold")).pack(
            side="left"
        )
        self.test_eval_toggle_btn = ctk.CTkButton(
            header,
            text="▼ Hide details",
            width=110,
            fg_color="transparent",
            border_width=1,
            command=self._toggle_test_eval_section,
        )
        self.test_eval_toggle_btn.pack(side="right")

        self.test_metric_labels = {}
        self.test_metric_cards = []
        for index, key in enumerate(["rmse", "mae", "r2"]):
            card = ctk.CTkFrame(container, corner_radius=8)
            card.grid(row=1, column=index, padx=6, pady=4, sticky="ew")
            self.test_metric_cards.append(card)
            ctk.CTkLabel(card, text=key.upper(), text_color="gray70").pack(padx=10, pady=(6, 0))
            label = ctk.CTkLabel(card, text="—", font=ctk.CTkFont(size=15, weight="bold"))
            label.pack(padx=10, pady=(0, 8))
            self.test_metric_labels[key] = label

        self.test_table_frame = ctk.CTkFrame(container)
        self.test_table_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        self.test_table_frame.grid_columnconfigure(0, weight=1)
        self.test_table_frame.grid_rowconfigure(0, weight=1)

        test_columns = ("idx", "x", "y_actual", "y_pred", "accuracy")
        self.test_tree = ttk.Treeview(
            self.test_table_frame,
            columns=test_columns,
            show="headings",
            height=12,
        )
        self.test_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(self.test_table_frame, orient="vertical", command=self.test_tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(self.test_table_frame, orient="horizontal", command=self.test_tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.test_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        for col, heading, width in (
            ("idx", "#", 40),
            ("x", "x", 100),
            ("y_actual", "y_actual (VND)", 150),
            ("y_pred", "y_pred (VND)", 150),
            ("accuracy", "accuracy (%)", 120),
        ):
            self.test_tree.heading(col, text=heading)
            self.test_tree.column(col, width=width, anchor="center")

        window.protocol("WM_DELETE_WINDOW", self._close_test_eval_window)
        self.test_eval_window = window
        return window

    def _close_test_eval_window(self) -> None:
        if self.test_eval_window is not None and self.test_eval_window.winfo_exists():
            self.test_eval_window.withdraw()

    def _toggle_test_eval_section(self) -> None:
        self._test_eval_visible = not self._test_eval_visible
        if self._test_eval_visible:
            self.test_table_frame.grid()
            for card in self.test_metric_cards:
                card.grid()
            self.test_eval_toggle_btn.configure(text="▼ Hide details")
        else:
            self.test_table_frame.grid_remove()
            for card in self.test_metric_cards:
                card.grid_remove()
            self.test_eval_toggle_btn.configure(text="▶ Show details")

    def _build_chart_area(self) -> None:
        chart_frame = ctk.CTkFrame(self.main)
        chart_frame.grid(row=3, column=0, sticky="nsew", pady=4)
        chart_frame.grid_rowconfigure(1, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(11, 6.0), dpi=100)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.14, wspace=0.22)
        self.ax_fit = self.fig.add_subplot(1, 2, 1)
        self.ax_loss = self.fig.add_subplot(1, 2, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        toolbar_frame = tk.Frame(chart_frame, bg="#242424")
        toolbar_frame.grid(row=0, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _build_animation_bar(self) -> None:
        self.anim_bar = ctk.CTkFrame(self.main)
        self.anim_bar.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self.anim_bar.grid_columnconfigure(5, weight=1)

        self.check_accuracy_btn = ctk.CTkButton(
            self.anim_bar,
            text="View Accuracy",
            command=self.on_view_accuracy_clicked,
            height=32,
            width=130,
        )
        self.check_accuracy_btn.grid(row=0, column=0, padx=(8, 4), pady=8)
        self.check_accuracy_btn.grid_remove()

        self.view_results_btn = ctk.CTkButton(
            self.anim_bar,
            text="View Results",
            command=self.on_view_test_results_clicked,
            height=32,
            width=110,
            fg_color="transparent",
            border_width=1,
        )
        self.view_results_btn.grid(row=0, column=1, padx=4, pady=8)
        self.view_results_btn.grid_remove()

        icon_first = _make_icon("|◀")
        icon_play = _make_icon("▶")
        icon_last = _make_icon("▶|")

        self.btn_first = ctk.CTkButton(
            self.anim_bar, text="", image=icon_first, width=36, command=self.on_first_frame
        )
        self.btn_first.grid(row=0, column=2, padx=4, pady=8)

        self.btn_play = ctk.CTkButton(
            self.anim_bar, text="", image=icon_play, width=36, command=self.on_play_pause
        )
        self.btn_play.grid(row=0, column=3, padx=4, pady=8)

        self.btn_last = ctk.CTkButton(
            self.anim_bar, text="", image=icon_last, width=36, command=self.on_last_frame
        )
        self.btn_last.grid(row=0, column=4, padx=4, pady=8)

        self.frame_slider = ctk.CTkSlider(self.anim_bar, from_=0, to=1, number_of_steps=1, command=self.on_seek)
        self.frame_slider.grid(row=0, column=5, sticky="ew", padx=8, pady=8)
        self.frame_slider.set(0)

        self.anim_stats_label = ctk.CTkLabel(self.anim_bar, text="Epoch: 0/0 | Loss: — | Speed: 1.0x")
        self.anim_stats_label.grid(row=0, column=6, padx=8, pady=8)

        speed_frame = ctk.CTkFrame(self.anim_bar, fg_color="transparent")
        speed_frame.grid(row=0, column=7, padx=4, pady=8)
        ctk.CTkLabel(speed_frame, text="Speed").pack()
        self.speed_combo = ctk.CTkComboBox(
            speed_frame,
            values=["0.5x", "1.0x", "1.5x", "2.0x", "4.0x"],
            width=80,
            command=self._on_speed_changed,
        )
        self.speed_combo.set("1.0x")
        self.speed_combo.pack()

    def _load_manual_default(self) -> None:
        self.manual_text.delete("1.0", "end")
        self.manual_text.insert("1.0", self.manual_text_default)

    def on_generate_sample(self) -> None:
        name = self.sample_combo.get()
        data = make_sample_dataset(name)
        self.sample_text = data
        self.sample_preview.configure(state="normal")
        self.sample_preview.delete("1.0", "end")
        self.sample_preview.insert("1.0", data)
        self.sample_preview.configure(state="disabled")
        self.input_tabs.set("Sample")
        self.status_label.configure(text="Generated sample: {}".format(name))

    def _format_large_number(self, value: float) -> str:
        return f"{value:,.0f}"

    def _display_scale(self) -> float:
        """Return multiplier to convert normalized target values back to VND."""
        if self.preprocess_result and self.preprocess_result.target_scale:
            return float(self.preprocess_result.target_scale)
        if self.training_result and self.training_result.preprocess_report:
            report = self.training_result.preprocess_report
            if report.target_normalized and report.target_scale:
                return float(report.target_scale)
        return 1.0

    def _format_target_value(self, value: float, scale: float | None = None) -> str:
        """Format target values for display, denormalizing when needed."""
        multiplier = scale if scale is not None else self._display_scale()
        return self._format_large_number(float(value) * multiplier)

    def _format_metric_value(self, name: str, value: float, scale: float | None = None) -> str:
        if name == "r2":
            return f"{value:.4f}"
        multiplier = scale if scale is not None else self._display_scale()
        return self._format_large_number(float(value) * multiplier)

    def _prediction_accuracy_percent(self, y_actual: float, y_pred: float) -> float:
        """Match ratio: how close prediction is to actual price, as 0–100%."""
        if not np.isfinite(y_actual) or abs(y_actual) < 1e-12:
            return 0.0
        relative_error = abs(y_pred - y_actual) / abs(y_actual)
        return max(0.0, min(100.0, 100.0 * (1.0 - relative_error)))

    def _get_preprocess_options(self) -> PreprocessOptions:
        try:
            multiplier = float(self.iqr_multiplier_var.get())
        except (tk.TclError, ValueError):
            multiplier = 1.5
        if multiplier <= 0:
            multiplier = 1.5
        return PreprocessOptions(
            drop_missing=bool(self.drop_missing_var.get()),
            remove_outliers=bool(self.remove_outliers_var.get()),
            iqr_multiplier=multiplier,
            normalize_target=bool(self.normalize_target_var.get()),
        )

    def _apply_preset_column_selection(self) -> None:
        if self.csv_preview is None:
            return
        names = self.csv_preview.numeric_names
        feature_name = "Dien_Tich_m2" if "Dien_Tich_m2" in names else names[0]
        target_name = "Gia_Nha_VND" if "Gia_Nha_VND" in names else names[-1]

        self.feature_listbox.selection_clear(0, "end")
        for index, name in enumerate(names):
            if name == feature_name:
                self.feature_listbox.selection_set(index)
        self.target_combo.set(target_name)
        self.visual_combo.set(feature_name)

    def _load_train_csv(self, path: str) -> None:
        preview = load_csv_for_preview(path)
        self.csv_preview = preview
        self.preprocess_result = None
        self.csv_path_var.set(path)
        self._populate_csv_controls(preview)
        self._populate_csv_table(preview)
        self.preview_header.grid()
        self.table_frame.grid()
        self.cards_frame.grid()
        self.preprocess_report_label.configure(text="")
        self.on_csv_columns_changed()
        self.input_tabs.set("CSV")
        self.status_label.configure(text="Loaded {} rows".format(len(preview.rows)))

    def on_preprocess_clicked(self) -> None:
        if self.csv_preview is None:
            messagebox.showwarning("Pre-process", "Please load a training CSV file first.")
            return
        feature_names = self._selected_feature_names()
        target_name = self.target_combo.get()
        visual_name = self.visual_combo.get()
        if not feature_names or not target_name or not visual_name:
            messagebox.showwarning("Pre-process", "Please select feature and target columns.")
            return
        if visual_name not in feature_names:
            feature_names = list(dict.fromkeys(feature_names + [visual_name]))
        try:
            self.preprocess_result = preprocess_csv_selection(
                self.csv_preview.numeric_data,
                feature_names,
                target_name,
                visual_name,
                self._get_preprocess_options(),
            )
        except Exception as exc:
            messagebox.showerror("Pre-process Error", str(exc))
            return

        report = self.preprocess_result.report
        self.preprocess_report_label.configure(
            text=(
                "Original: {orig} | Missing dropped: {miss} | "
                "Outliers dropped: {out} | Ready: {final}"
            ).format(
                orig=report.n_original,
                miss=report.n_dropped_missing,
                out=report.n_dropped_outliers,
                final=report.n_final,
            )
        )
        self._update_preprocess_preview()
        self.status_label.configure(text="Pre-processing complete — {} rows ready".format(report.n_final))

    def on_view_accuracy_clicked(self) -> None:
        """Re-open the 20% hold-out accuracy window from the last training run."""
        if self._last_test_evaluation is None:
            messagebox.showinfo("View Accuracy", "No hold-out results yet. Train a model first.")
            return
        self._show_test_evaluation_with_eval(self._last_test_evaluation)

    def _show_test_evaluation_with_eval(self, evaluation: TestEvaluationResult) -> None:
        """Display test evaluation metrics and table in a separate window."""
        self._last_test_evaluation = evaluation
        window = self._ensure_test_eval_window()
        metrics = evaluation.metrics
        scale = evaluation.target_scale or self._display_scale()

        self.test_metric_labels["rmse"].configure(text=self._format_metric_value("rmse", metrics["rmse"], scale))
        self.test_metric_labels["mae"].configure(text=self._format_metric_value("mae", metrics["mae"], scale))
        self.test_metric_labels["r2"].configure(text=self._format_metric_value("r2", metrics["r2"]))

        self.test_tree.delete(*self.test_tree.get_children())
        visual_index = self.visual_feature_index
        x_test = evaluation.x_test
        x_visual = x_test[:, visual_index] if x_test.ndim == 2 else x_test
        for index, (x_value, y_actual, y_pred) in enumerate(
            zip(x_visual, evaluation.y_test, evaluation.y_pred)
        ):
            accuracy = self._prediction_accuracy_percent(y_actual, y_pred)
            self.test_tree.insert(
                "",
                "end",
                values=[
                    str(index + 1),
                    f"{x_value:.2f}",
                    self._format_target_value(y_actual, scale),
                    self._format_target_value(y_pred, scale),
                    f"{accuracy:.2f}%",
                ],
            )

        if not self._test_eval_visible:
            self._toggle_test_eval_section()

        window.deiconify()
        window.lift()
        window.focus_force()
        self.view_results_btn.grid()

    def on_view_test_results_clicked(self) -> None:
        self.on_view_accuracy_clicked()

    def _update_preprocess_preview(self) -> None:
        if self.preprocess_result is None:
            return
        result = self.preprocess_result
        visual_index = result.feature_names.index(result.visual_name)
        x_visual = result.x_data[:, visual_index] if result.x_data.ndim == 2 else result.x_data

        self._update_stat_cards_from_preprocess(result)
        self.csv_stats_label.configure(
            text="{} rows ready for training".format(result.report.n_final)
        )
        self.viz_mode = "data_preview"
        self.preview_artists = PreviewArtists()
        draw_csv_preview(
            x_visual,
            result.y_data,
            result.visual_name,
            result.target_name,
            self.ax_fit,
            self.ax_loss,
            self.fig,
            self.theme_mode,
            self.preview_artists,
            preprocess_report=result.report,
            x_visual_raw=result.x_visual_raw,
            y_target_raw=result.y_target_raw,
            target_scale=result.target_scale,
        )
        self._set_animation_enabled(False)
        self.equation_frame.grid_remove()
        self.canvas.draw_idle()

    def _update_stat_cards_from_preprocess(self, result: PreprocessResult) -> None:
        visual_index = result.feature_names.index(result.visual_name)
        x_visual = result.x_data[:, visual_index] if result.x_data.ndim == 2 else result.x_data
        self.stat_cards["count"].configure(text=str(result.report.n_final))
        self.stat_cards["mean_x"].configure(text="{:.2f}".format(float(np.mean(x_visual))))
        self.stat_cards["std_y"].configure(text="{:.2f}".format(float(np.std(result.y_data))))
        self.stat_cards["missing"].configure(text=str(result.report.n_dropped_missing))
        self.stat_cards["outliers"].configure(text=str(result.report.n_dropped_outliers))

    def on_browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._load_train_csv(path)
        except Exception as exc:
            messagebox.showerror("CSV Error", str(exc))

    def _populate_csv_controls(self, preview: CsvPreviewData) -> None:
        self.feature_listbox.delete(0, "end")
        for name in preview.numeric_names:
            self.feature_listbox.insert("end", name)
        if preview.numeric_names:
            self.feature_listbox.selection_set(0)

        self.target_combo.configure(values=preview.numeric_names)
        self.visual_combo.configure(values=preview.numeric_names)
        if "Dien_Tich_m2" in preview.numeric_names:
            default_visual = "Dien_Tich_m2"
        elif "total_sqft" in preview.numeric_names:
            default_visual = "total_sqft"
        else:
            default_visual = preview.numeric_names[0]

        if "Gia_Nha_VND" in preview.numeric_names:
            default_target = "Gia_Nha_VND"
        elif "price" in preview.numeric_names:
            default_target = "price"
        else:
            default_target = preview.numeric_names[-1]

        self.target_combo.set(default_target)
        self.visual_combo.set(default_visual)

    def _populate_csv_table(self, preview: CsvPreviewData) -> None:
        self.csv_tree.delete(*self.csv_tree.get_children())
        columns = list(preview.rows[0].keys()) if preview.rows else []
        self.csv_tree["columns"] = columns
        for column in columns:
            self.csv_tree.heading(column, text=column)
            self.csv_tree.column(column, width=120, anchor="center")
        for row in preview.rows[:10]:
            self.csv_tree.insert("", "end", values=[row.get(col, "") for col in columns])

    def _toggle_csv_table(self) -> None:
        self._table_visible = not self._table_visible
        if self._table_visible:
            self.table_frame.grid()
            self.preview_toggle_btn.configure(text="▼ CSV Table Preview")
        else:
            self.table_frame.grid_remove()
            self.preview_toggle_btn.configure(text="▶ CSV Table Preview")

    def _selected_feature_names(self) -> list[str]:
        indices = self.feature_listbox.curselection()
        return [self.feature_listbox.get(index) for index in indices]

    def on_csv_columns_changed(self) -> None:
        if self.csv_preview is None:
            return
        self.preprocess_result = None
        self.preprocess_report_label.configure(text="")
        feature_names = self._selected_feature_names()
        target_name = self.target_combo.get()
        visual_name = self.visual_combo.get()
        if not feature_names or not target_name or not visual_name:
            return
        try:
            result = prepare_csv_preview(
                self.csv_preview.numeric_data,
                self.csv_preview.rows,
                feature_names,
                target_name,
                visual_name,
            )
        except Exception as exc:
            self.status_label.configure(text=str(exc))
            return

        self._update_stat_cards(result, visual_name, target_name)
        self.stat_cards["outliers"].configure(text="—")
        self.csv_stats_label.configure(
            text="{} valid / {} dropped (before preprocess)".format(result.n_valid, result.n_dropped)
        )
        self.viz_mode = "data_preview"
        self.preview_artists = PreviewArtists()
        draw_csv_preview(
            result.x_visual,
            result.y_target,
            visual_name,
            target_name,
            self.ax_fit,
            self.ax_loss,
            self.fig,
            self.theme_mode,
            self.preview_artists,
        )
        self._set_animation_enabled(False)
        self.equation_frame.grid_remove()
        self.canvas.draw_idle()

    def _update_stat_cards(self, result: object, visual_name: str, target_name: str) -> None:
        stats = result.column_stats
        count = int(stats.get(target_name, {}).get("count", result.n_valid))
        mean_x = stats.get(visual_name, {}).get("mean", float("nan"))
        std_y = stats.get(target_name, {}).get("std", float("nan"))
        self.stat_cards["count"].configure(text=str(count))
        self.stat_cards["mean_x"].configure(text="{:.2f}".format(mean_x))
        self.stat_cards["std_y"].configure(text="{:.2f}".format(std_y))
        self.stat_cards["missing"].configure(text=str(result.n_dropped))

    def on_train_clicked(self) -> None:
        if self._train_thread and self._train_thread.is_alive():
            return
        self._train_csv_config = self._snapshot_csv_train_config()
        self._set_training_ui_busy(True)
        self._train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._train_thread.start()

    def _snapshot_csv_train_config(self) -> dict[str, object] | None:
        """Capture CSV column selection on the main thread before training."""
        if self.input_tabs.get() != "CSV":
            return None
        if self.csv_preview is None:
            return None

        feature_names = self._selected_feature_names()
        target_name = self.target_combo.get()
        visual_name = self.visual_combo.get()
        if not feature_names or not target_name or not visual_name:
            return None
        if visual_name not in feature_names:
            feature_names = list(dict.fromkeys(feature_names + [visual_name]))
        return {
            "feature_names": feature_names,
            "target_name": target_name,
            "visual_name": visual_name,
        }

    def _train_worker(self) -> None:
        try:
            result = self._run_training()
        except Exception as exc:
            self.after(0, self._on_train_failed, str(exc))
            return
        self.after(0, self._on_train_complete, result)

    def _run_training(self) -> TrainingResult:
        degree = int(self.degree_var.get())
        lr = float(self.lr_var.get())
        epochs = int(self.epochs_var.get())
        save_every = max(1, int(self.save_every_var.get()))
        active_tab = self.input_tabs.get()

        if active_tab == "CSV":
            if self.csv_preview is None:
                raise ValueError("Please load a training CSV file first.")
            cfg = self._train_csv_config or self._snapshot_csv_train_config()
            if cfg is None:
                raise ValueError("Please select feature and target columns.")
            feature_names = list(cfg["feature_names"])
            target_name = str(cfg["target_name"])
            visual_name = str(cfg["visual_name"])
            self.target_column_name = target_name
            return train_from_csv_selection(
                self.csv_preview.numeric_data,
                feature_names,
                target_name,
                visual_name,
                degree,
                lr,
                epochs,
                save_every,
                preprocessed=self.preprocess_result,
                iqr_multiplier=self._get_preprocess_options().iqr_multiplier,
            )

        if active_tab == "Sample":
            if not self.sample_text:
                raise ValueError("Please generate a sample dataset first.")
            self.target_column_name = "y"
            return train_from_text(self.sample_text.strip(), degree, lr, epochs, save_every)

        self.target_column_name = "y"
        raw_text = self.manual_text.get("1.0", "end").strip()
        return train_from_text(raw_text, degree, lr, epochs, save_every)

    def _set_training_ui_busy(self, busy: bool) -> None:
        if busy:
            self.train_button.configure(state="disabled", text="Training...")
            self.reset_button.configure(state="disabled")
            self.check_accuracy_btn.configure(state="disabled")
            self.progress_bar.grid()
            self.progress_bar.start()
            self.status_label.configure(text="Training in progress...")
        else:
            self.train_button.configure(state="normal", text="Train Model")
            self.reset_button.configure(state="normal")
            if self.model is not None:
                self.check_accuracy_btn.configure(state="normal")
            self.progress_bar.stop()
            self.progress_bar.grid_remove()

    def _on_train_failed(self, message: str) -> None:
        self._set_training_ui_busy(False)
        messagebox.showerror("Training Error", message)
        self.status_label.configure(text="Training failed")

    def _on_train_complete(self, result: TrainingResult) -> None:
        self.training_result = result
        self.model = result.model
        self.x_data = result.x_data
        self.y_data = result.y_data
        self.feature_names = result.feature_names
        self.target_column_name = result.target_name
        self.visual_feature_index = result.visual_feature_index
        self.viz_mode = "training"
        self.frame_index = 0
        self.playing = False

        self.preview_header.grid_remove()
        self.table_frame.grid_remove()
        self.cards_frame.grid_remove()
        self.equation_frame.grid()
        self.training_header.grid()
        self._training_table_visible = False
        self.training_table_frame.grid_remove()
        self.training_toggle_btn.configure(text="▶ Training Data (x, y, y_pred, residual)")
        self.check_accuracy_btn.grid()
        self.check_accuracy_btn.configure(state="normal")

        self._set_training_ui_busy(False)

        self._populate_training_data_table(result.x_data, result.y_data, result.feature_names)

        x_test_visual = None
        y_test = None
        y_test_pred = None
        if result.x_test is not None and result.y_test is not None:
            x_test = result.x_test
            y_test = result.y_test
            if x_test.ndim == 2:
                x_test_visual = x_test[:, result.visual_feature_index]
            else:
                x_test_visual = x_test
            if result.test_evaluation is not None:
                y_test_pred = result.test_evaluation.y_pred

        self.plot_artists = init_training_artists(
            self.fig,
            self.ax_fit,
            self.ax_loss,
            result.model,
            result.x_data,
            result.y_data,
            result.visual_feature_index,
            result.feature_names,
            self.theme_mode,
            x_test_visual=x_test_visual,
            y_test=y_test,
            y_test_pred=y_test_pred,
        )
        total_frames = len(result.model.frames)
        self.frame_slider.configure(to=max(total_frames - 1, 0), number_of_steps=max(total_frames - 1, 1))
        self.frame_slider.set(0)
        self._set_animation_enabled(True)
        self._update_training_view()
        self.playing = True
        self._animation_loop()

        if result.test_evaluation is not None:
            self._show_test_evaluation_with_eval(result.test_evaluation)
            n_test = len(result.test_evaluation.y_test)
            self.status_label.configure(
                text="Training complete — Test R² = {:.4f} on {} hold-out samples (20%)".format(
                    result.test_evaluation.metrics["r2"], n_test
                )
            )
        else:
            self.status_label.configure(text="Training complete — {} frames".format(total_frames))

    def _set_animation_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.btn_first, self.btn_play, self.btn_last, self.frame_slider, self.speed_combo):
            widget.configure(state=state)

    def _on_speed_changed(self, value: str) -> None:
        multiplier = float(value.replace("x", ""))
        self.animation_speed = max(10, int(50 / multiplier))

    def on_first_frame(self) -> None:
        self.playing = False
        self.frame_index = 0
        self.frame_slider.set(0)
        self._update_training_view()

    def on_last_frame(self) -> None:
        if self.model is None:
            return
        self.playing = False
        self.frame_index = len(self.model.frames) - 1
        self.frame_slider.set(self.frame_index)
        self._update_training_view()

    def on_play_pause(self) -> None:
        if self.model is None:
            return
        self.playing = not self.playing
        if self.playing:
            self._animation_loop()

    def on_seek(self, value: float) -> None:
        if self.model is None:
            return
        self.playing = False
        self.frame_index = int(round(value))
        self._update_training_view()

    def _update_training_view(self) -> None:
        if self.model is None or self.x_data is None or self.y_data is None:
            return
        frame_view = draw_training_frame(
            self.model,
            self.x_data,
            self.y_data,
            self.frame_index,
            self.feature_names,
            self.visual_feature_index,
            self.ax_fit,
            self.ax_loss,
            self.fig,
            self.plot_artists,
            self.theme_mode,
        )
        self.equation_label.configure(text=frame_view.equation)
        weights_text = ", ".join(f"{value:.4f}" for value in frame_view.weights[:6])
        if len(frame_view.weights) > 6:
            weights_text += ", ..."
        self.frame_detail_label.configure(
            text="Epoch {} | Loss {:.6f} | Weights: [{}]".format(
                frame_view.epoch, frame_view.loss, weights_text
            )
        )
        self._update_training_data_rows(frame_view)

        frames = self.model.frames
        if not frames:
            return
        speed = 50 / max(self.animation_speed, 1)
        self.anim_stats_label.configure(
            text="Epoch: {}/{} | Loss: {:.6f} | Speed: {:.1f}x | Points: {}".format(
                frame_view.epoch,
                frames[-1].epoch,
                frame_view.loss,
                speed,
                len(frame_view.y_actual),
            )
        )

    def _animation_loop(self) -> None:
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None
        if not self.playing or self.model is None:
            return

        total_frames = len(self.model.frames)
        next_index, should_continue = next_frame_index(self.frame_index, total_frames)
        self.frame_index = next_index
        self.frame_slider.set(self.frame_index)
        self._update_training_view()

        if should_continue and self.playing:
            self._animation_job = self.after(self.animation_speed, self._animation_loop)
        else:
            self.playing = False

    def _show_empty_plots(self) -> None:
        draw_empty_plot(self.fig, self.ax_fit, self.ax_loss, self.theme_mode)
        self.canvas.draw_idle()

    def on_reset_clicked(self) -> None:
        self.playing = False
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None

        self.model = None
        self.x_data = None
        self.y_data = None
        self.training_result = None
        self.csv_preview = None
        self.preprocess_result = None
        self._train_csv_config = None
        self.sample_text = None
        self.feature_names = []
        self.frame_index = 0
        self.viz_mode = "empty"

        self.plot_artists = PlotArtists()
        self.preview_artists = PreviewArtists()
        self.preview_header.grid_remove()
        self.table_frame.grid_remove()
        self.cards_frame.grid_remove()
        self.equation_frame.grid_remove()
        self._close_test_eval_window()
        self._last_test_evaluation = None
        self.check_accuracy_btn.grid_remove()
        self.view_results_btn.grid_remove()
        self.training_header.grid_remove()
        self.training_table_frame.grid_remove()
        self.training_tree.delete(*self.training_tree.get_children())
        self._training_tree_row_ids = []
        self.training_summary_label.configure(text="")
        self.frame_detail_label.configure(text="Weights: —")
        self.csv_path_var.set("No file selected")
        self.preprocess_report_label.configure(text="")
        self.csv_stats_label.configure(text="")
        for card in self.stat_cards.values():
            card.configure(text="—")
        if hasattr(self, "sample_preview"):
            self.sample_preview.configure(state="normal")
            self.sample_preview.delete("1.0", "end")
            self.sample_preview.configure(state="disabled")

        self._show_empty_plots()
        self._set_animation_enabled(False)
        self.anim_stats_label.configure(text="Epoch: 0/0 | Loss: — | Speed: 1.0x")
        self.status_label.configure(text="Reset complete")
