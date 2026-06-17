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
    TrainingResult,
    load_csv_for_preview,
    prepare_csv_preview,
    train_from_csv_selection,
    train_from_text,
)
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
        self.csv_path_var = tk.StringVar(value="No file selected")
        self.manual_text_default = "x,y\n-2.0,5.1\n-1.0,3.2\n0.0,1.0\n1.0,2.8\n2.0,6.5\n3.0,10.2"

        self.frame_index = 0
        self.playing = False
        self.animation_speed = 50
        self._animation_job: str | None = None
        self._train_thread: threading.Thread | None = None

        self.plot_artists = PlotArtists()
        self.preview_artists = PreviewArtists()

        self._build_layout()
        self._show_empty_plots()
        self._set_animation_enabled(False)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)

        ctk.CTkLabel(
            self.sidebar,
            text="Polynomial AI",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.input_tabs = ctk.CTkTabview(self.sidebar, width=280)
        self.input_tabs.grid(row=1, column=0, padx=12, pady=8, sticky="ew")
        self._build_manual_tab()
        self._build_csv_tab()
        self._build_sample_tab()

        ctk.CTkLabel(self.sidebar, text="Hyperparameters", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=16, pady=(12, 4), sticky="w"
        )

        self.degree_var = tk.IntVar(value=3)
        self.lr_var = tk.DoubleVar(value=0.03)
        self.epochs_var = tk.IntVar(value=500)
        self.save_every_var = tk.IntVar(value=1)

        self._add_labeled_slider(self.sidebar, 3, "Degree", self.degree_var, 1, 10, is_int=True)
        degree_label = self.sidebar.grid_slaves(row=3, column=0)[0]
        CTkToolTip(degree_label, message="Bậc đa thức (1=linear, 2=quadratic, ...)")

        self._add_labeled_slider(self.sidebar, 4, "Learning Rate", self.lr_var, 0.001, 0.2, is_int=False)
        lr_label = self.sidebar.grid_slaves(row=4, column=0)[0]
        CTkToolTip(lr_label, message="Tốc độ cập nhật weights mỗi epoch")

        self._add_labeled_slider(self.sidebar, 5, "Epochs", self.epochs_var, 100, 2000, is_int=True)
        epochs_label = self.sidebar.grid_slaves(row=5, column=0)[0]
        CTkToolTip(epochs_label, message="Số vòng lặp Gradient Descent")

        save_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        save_frame.grid(row=6, column=0, padx=16, pady=4, sticky="ew")
        ctk.CTkLabel(save_frame, text="Save every").pack(side="left")
        ctk.CTkEntry(save_frame, textvariable=self.save_every_var, width=60).pack(side="right")

        self.train_button = ctk.CTkButton(
            self.sidebar,
            text="Train Model",
            height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.on_train_clicked,
        )
        self.train_button.grid(row=7, column=0, padx=16, pady=(16, 4), sticky="ew")

        self.reset_button = ctk.CTkButton(
            self.sidebar,
            text="Reset",
            fg_color="transparent",
            border_width=1,
            command=self.on_reset_clicked,
        )
        self.reset_button.grid(row=8, column=0, padx=16, pady=4, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.sidebar, mode="indeterminate")
        self.progress_bar.grid(row=9, column=0, padx=16, pady=4, sticky="ew")
        self.progress_bar.grid_remove()

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray70")
        self.status_label.grid(row=10, column=0, padx=16, pady=(4, 16), sticky="w")

        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.main.grid_rowconfigure(4, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        self._build_csv_preview_section()
        self._build_training_data_section()
        self._build_equation_bar()
        self._build_chart_area()
        self._build_animation_bar()

    def _add_labeled_slider(
        self,
        parent: ctk.CTkFrame,
        row: int,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        is_int: bool,
    ) -> None:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, padx=16, pady=4, sticky="ew")
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

        self.csv_stats_label = ctk.CTkLabel(tab, text="", justify="left")
        self.csv_stats_label.grid(row=8, column=0, sticky="w", padx=4, pady=4)

    def _build_sample_tab(self) -> None:
        tab = self.input_tabs.add("Sample")
        ctk.CTkLabel(tab, text="Choose a synthetic dataset:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.sample_combo = ctk.CTkComboBox(tab, values=list(SAMPLE_DATASET_NAMES))
        self.sample_combo.set("Cubic")
        self.sample_combo.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ctk.CTkButton(tab, text="Generate", command=self.on_generate_sample).grid(
            row=2, column=0, sticky="ew", padx=4, pady=4
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
        self.cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.cards_frame.grid_remove()

        self.stat_cards: dict[str, ctk.CTkLabel] = {}
        for index, key in enumerate(["count", "mean_x", "std_y", "missing"]):
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
        self.equation_frame.grid(row=3, column=0, sticky="ew", pady=4)
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

    def _build_chart_area(self) -> None:
        chart_frame = ctk.CTkFrame(self.main)
        chart_frame.grid(row=4, column=0, sticky="nsew", pady=4)
        chart_frame.grid_rowconfigure(1, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(11, 5.2), dpi=100)
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
        self.anim_bar.grid_columnconfigure(3, weight=1)

        icon_first = _make_icon("|◀")
        icon_play = _make_icon("▶")
        icon_last = _make_icon("▶|")

        self.btn_first = ctk.CTkButton(
            self.anim_bar, text="", image=icon_first, width=36, command=self.on_first_frame
        )
        self.btn_first.grid(row=0, column=0, padx=4, pady=8)

        self.btn_play = ctk.CTkButton(
            self.anim_bar, text="", image=icon_play, width=36, command=self.on_play_pause
        )
        self.btn_play.grid(row=0, column=1, padx=4, pady=8)

        self.btn_last = ctk.CTkButton(
            self.anim_bar, text="", image=icon_last, width=36, command=self.on_last_frame
        )
        self.btn_last.grid(row=0, column=2, padx=4, pady=8)

        self.frame_slider = ctk.CTkSlider(self.anim_bar, from_=0, to=1, number_of_steps=1, command=self.on_seek)
        self.frame_slider.grid(row=0, column=3, sticky="ew", padx=8, pady=8)
        self.frame_slider.set(0)

        self.anim_stats_label = ctk.CTkLabel(self.anim_bar, text="Epoch: 0/0 | Loss: — | Speed: 1.0x")
        self.anim_stats_label.grid(row=0, column=4, padx=8, pady=8)

        speed_frame = ctk.CTkFrame(self.anim_bar, fg_color="transparent")
        speed_frame.grid(row=0, column=5, padx=4, pady=8)
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
        self.manual_text.delete("1.0", "end")
        self.manual_text.insert("1.0", data)
        self.input_tabs.set("Manual")
        self.status_label.configure(text="Generated sample: {}".format(name))

    def on_browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            preview = load_csv_for_preview(path)
        except Exception as exc:
            messagebox.showerror("CSV Error", str(exc))
            return

        self.csv_preview = preview
        self.csv_path_var.set(path)
        self._populate_csv_controls(preview)
        self._populate_csv_table(preview)
        self.preview_header.grid()
        self.table_frame.grid()
        self.cards_frame.grid()
        self.on_csv_columns_changed()
        self.status_label.configure(text="Loaded {} rows".format(len(preview.rows)))

    def _populate_csv_controls(self, preview: CsvPreviewData) -> None:
        self.feature_listbox.delete(0, "end")
        for name in preview.numeric_names:
            self.feature_listbox.insert("end", name)
        if preview.numeric_names:
            self.feature_listbox.selection_set(0)

        self.target_combo.configure(values=preview.numeric_names)
        self.visual_combo.configure(values=preview.numeric_names)
        default_target = "price" if "price" in preview.numeric_names else preview.numeric_names[0]
        default_visual = (
            "total_sqft"
            if "total_sqft" in preview.numeric_names
            else preview.numeric_names[0]
        )
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
        self.csv_stats_label.configure(
            text="{} valid / {} dropped".format(result.n_valid, result.n_dropped)
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
        self._set_training_ui_busy(True)
        self._train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._train_thread.start()

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
                raise ValueError("Please load a CSV file first.")
            feature_names = self._selected_feature_names()
            target_name = self.target_combo.get()
            visual_name = self.visual_combo.get()
            self.target_column_name = target_name
            if visual_name not in feature_names:
                feature_names = list(dict.fromkeys(feature_names + [visual_name]))
            return train_from_csv_selection(
                self.csv_preview.numeric_data,
                feature_names,
                target_name,
                visual_name,
                degree,
                lr,
                epochs,
                save_every,
            )

        self.target_column_name = "y"
        raw_text = self.manual_text.get("1.0", "end").strip()
        return train_from_text(raw_text, degree, lr, epochs, save_every)

    def _set_training_ui_busy(self, busy: bool) -> None:
        if busy:
            self.train_button.configure(state="disabled", text="Training...")
            self.reset_button.configure(state="disabled")
            self.progress_bar.grid()
            self.progress_bar.start()
            self.status_label.configure(text="Training in progress...")
        else:
            self.train_button.configure(state="normal", text="Train Model")
            self.reset_button.configure(state="normal")
            self.progress_bar.stop()
            self.progress_bar.grid_remove()

    def _on_train_failed(self, message: str) -> None:
        self._set_training_ui_busy(False)
        messagebox.showerror("Training Error", message)
        self.status_label.configure(text="Training failed")

    def _on_train_complete(self, result: TrainingResult) -> None:
        self._set_training_ui_busy(False)
        self.training_result = result
        self.model = result.model
        self.x_data = result.x_data
        self.y_data = result.y_data
        self.feature_names = result.feature_names
        self.visual_feature_index = result.visual_feature_index
        self.viz_mode = "training"
        self.frame_index = 0
        self.playing = False

        self.preview_header.grid_remove()
        self.table_frame.grid_remove()
        self.cards_frame.grid_remove()
        self.equation_frame.grid()
        self.training_header.grid()
        self.training_table_frame.grid()

        self._populate_training_data_table(result.x_data, result.y_data, result.feature_names)

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
        )
        total_frames = len(result.model.frames)
        self.frame_slider.configure(to=max(total_frames - 1, 0), number_of_steps=max(total_frames - 1, 1))
        self.frame_slider.set(0)
        self._set_animation_enabled(True)
        self._update_training_view()
        self.playing = True
        self._animation_loop()
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
        self.feature_names = []
        self.frame_index = 0
        self.viz_mode = "empty"

        self.plot_artists = PlotArtists()
        self.preview_artists = PreviewArtists()
        self.preview_header.grid_remove()
        self.table_frame.grid_remove()
        self.cards_frame.grid_remove()
        self.equation_frame.grid_remove()
        self.training_header.grid_remove()
        self.training_table_frame.grid_remove()
        self.training_tree.delete(*self.training_tree.get_children())
        self._training_tree_row_ids = []
        self.training_summary_label.configure(text="")
        self.frame_detail_label.configure(text="Weights: —")
        self.csv_path_var.set("No file selected")
        self.csv_stats_label.configure(text="")
        for card in self.stat_cards.values():
            card.configure(text="—")

        self._show_empty_plots()
        self._set_animation_enabled(False)
        self.anim_stats_label.configure(text="Epoch: 0/0 | Loss: — | Speed: 1.0x")
        self.status_label.configure(text="Reset complete")
