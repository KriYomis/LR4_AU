import tkinter as tk
from PIL import Image, ImageTk


CELL_SIZE = 30
CELL_GAP = 3
PADDING = 12
MAX_CANVAS_WIDTH = 1400
MAX_CANVAS_HEIGHT = 620


def clamp(value: float, left: float, right: float) -> float:
    return max(left, min(right, value))


def mix(color_a: tuple, color_b: tuple, factor: float) -> str:
    factor = clamp(factor, 0.0, 1.0)
    red = round(color_a[0] + (color_b[0] - color_a[0]) * factor)
    green = round(color_a[1] + (color_b[1] - color_a[1]) * factor)
    blue = round(color_a[2] + (color_b[2] - color_a[2]) * factor)
    return f"#{red:02x}{green:02x}{blue:02x}"


def weight_to_color(value: float, limit: float) -> str:
    if limit <= 0:
        return "#f4f1ea"
    normalized = clamp(value / limit, -1.0, 1.0)
    neutral = (244, 241, 234)
    positive = (32, 166, 110)
    negative = (217, 85, 67)
    if normalized >= 0:
        return mix(neutral, positive, normalized)
    return mix(neutral, negative, abs(normalized))


def delta_to_outline(delta: float, limit: float) -> str:
    if limit <= 0:
        return "#d9d2c3"
    normalized = clamp(abs(delta) / limit, 0.0, 1.0)
    return mix((217, 210, 195), (62, 76, 89), normalized)


def text_color_for_fill(fill: str) -> str:
    red, green, blue = hex_to_rgb(fill)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    if luminance < 150:
        return "#fffaf2"
    return "#23313a"


def hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


class WeightHistoryViewer:
    def __init__(self, history: list, class_names: list, input_side: int = 9):
        self.history = history
        self.class_names = class_names
        self.input_side = input_side
        self.epoch_index = len(history) - 1
        self.selected_output = None

        self.hidden_limit = self._matrix_abs_max("hidden")
        self.output_limit = self._matrix_abs_max("output")
        self.hidden_delta_limit = self._matrix_delta_abs_max("hidden")
        self.output_delta_limit = self._matrix_delta_abs_max("output")

        self.root = tk.Tk()
        self.root.title("Визуализация весов по эпохам")
        self.root.configure(bg="#f7f2e8")

        self.epoch_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.epoch_scale_var = tk.IntVar(value=self.epoch_index)

        self._build_layout()
        self._render()

    def _matrix_abs_max(self, key: str) -> float:
        maximum = 0.0
        for snapshot in self.history:
            for row in snapshot[key]:
                for value in row:
                    maximum = max(maximum, abs(value))
        return maximum

    def _matrix_delta_abs_max(self, key: str) -> float:
        maximum = 0.0
        for index in range(1, len(self.history)):
            prev = self.history[index - 1][key]
            curr = self.history[index][key]
            for row_prev, row_curr in zip(prev, curr):
                for value_prev, value_curr in zip(row_prev, row_curr):
                    maximum = max(maximum, abs(value_curr - value_prev))
        return maximum

    def _build_layout(self) -> None:
        controls = tk.Frame(self.root, bg="#f7f2e8")
        controls.pack(fill="x", padx=12, pady=(12, 6))

        tk.Label(
            controls, text="Эпоха:", bg="#f7f2e8",
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=(0, 10))
        self.epoch_scale = tk.Scale(
            controls,
            from_=0,
            to=len(self.history) - 1,
            orient="horizontal",
            variable=self.epoch_scale_var,
            command=self._on_epoch_scale,
            resolution=1,
            showvalue=False,
            length=420,
            bg="#f7f2e8",
            troughcolor="#d8e2dc",
            activebackground="#81b29a",
            highlightthickness=0
        )
        self.epoch_scale.pack(side="left")
        tk.Label(
            controls, textvariable=self.epoch_var, bg="#f7f2e8",
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=12)

        class_frame = tk.Frame(self.root, bg="#f7f2e8")
        class_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(
            class_frame, text="Подсветка выхода:", bg="#f7f2e8",
            font=("Segoe UI", 10, "bold")
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            class_frame, text="Все", command=lambda: self._set_output(None),
            bg="#f2cc8f", activebackground="#e7b86a", relief="flat", padx=10
        ).pack(side="left", padx=(0, 6))

        for index, class_name in enumerate(self.class_names):
            tk.Button(
                class_frame, text=class_name,
                command=lambda idx=index: self._set_output(idx),
                bg="#81b29a", activebackground="#6a9f84", relief="flat", padx=10
            ).pack(side="left", padx=(0, 6))

        tk.Label(
            self.root, textvariable=self.summary_var, bg="#f7f2e8",
            justify="left", anchor="w", font=("Consolas", 10)
        ).pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(
            self.root,
            text=(
                "Цвет ячейки: зелёный = положительный вес, красный = отрицательный, "
                "светлый = вес близок к нулю. Контур темнее = вес сильнее изменился "
                "по сравнению с предыдущей эпохой."
            ),
            bg="#f7f2e8", fg="#5c6770", justify="left", anchor="w",
            wraplength=1040, font=("Segoe UI", 9)
        ).pack(fill="x", padx=12, pady=(0, 8))

        hidden_frame = tk.Frame(self.root, bg="#f7f2e8")
        hidden_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.hidden_canvas = tk.Canvas(
            hidden_frame, bg="#fffaf2", highlightthickness=0,
            width=MAX_CANVAS_WIDTH, height=min(MAX_CANVAS_HEIGHT, 900)
        )
        self.hidden_canvas.grid(row=0, column=0, sticky="nsew")
        hidden_x = tk.Scrollbar(hidden_frame, orient="horizontal", command=self.hidden_canvas.xview)
        hidden_y = tk.Scrollbar(hidden_frame, orient="vertical", command=self.hidden_canvas.yview)
        self.hidden_canvas.configure(xscrollcommand=hidden_x.set, yscrollcommand=hidden_y.set)
        hidden_x.grid(row=1, column=0, sticky="ew")
        hidden_y.grid(row=0, column=1, sticky="ns")
        hidden_frame.grid_columnconfigure(0, weight=1)
        hidden_frame.grid_rowconfigure(0, weight=1)

        output_frame = tk.Frame(self.root, bg="#f7f2e8")
        output_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.output_canvas = tk.Canvas(
            output_frame, bg="#fffaf2", highlightthickness=0,
            width=min(MAX_CANVAS_WIDTH, 900), height=280
        )
        self.output_canvas.grid(row=0, column=0, sticky="nsew")
        output_x = tk.Scrollbar(output_frame, orient="horizontal", command=self.output_canvas.xview)
        output_y = tk.Scrollbar(output_frame, orient="vertical", command=self.output_canvas.yview)
        self.output_canvas.configure(xscrollcommand=output_x.set, yscrollcommand=output_y.set)
        output_x.grid(row=1, column=0, sticky="ew")
        output_y.grid(row=0, column=1, sticky="ns")
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

    def _set_output(self, output_index: int) -> None:
        self.selected_output = output_index
        self._render()

    def _on_epoch_scale(self, value: str) -> None:
        self.epoch_index = int(float(value))
        self._render()

    def _render(self) -> None:
        snapshot = self.history[self.epoch_index]
        prev_snapshot = None
        if self.epoch_index > 0:
            prev_snapshot = self.history[self.epoch_index - 1]

        self.epoch_scale_var.set(self.epoch_index)
        self.epoch_var.set(self._epoch_caption(snapshot))
        self.summary_var.set(self._summary_text(snapshot, prev_snapshot))

        self._draw_hidden_matrix(snapshot, prev_snapshot)
        self._draw_output_matrix(snapshot, prev_snapshot)

    def _epoch_caption(self, snapshot: dict) -> str:
        return f"Эпоха: {snapshot['epoch']} из {self.history[-1]['epoch']}"

    def _summary_text(self, snapshot: dict, prev_snapshot: dict) -> str:
        lines = []
        if snapshot["avg_error"] is None:
            lines.append("Начальное состояние сети до обучения.")
        else:
            lines.append(
                f"MSE: {snapshot['avg_error']:.6f}   "
                f"Точность: {snapshot['accuracy']:.1%}   "
                f"Ошибка валидации: {snapshot['val_error']:.1%}"
            )

        if self.selected_output is None:
            lines.append("Подсветка: все веса.")
        else:
            cls = self.class_names[self.selected_output]
            lines.append(
                f"Подсветка: выход '{cls}'. "
                f"Скрытый слой подсвечен по силе влияния на этот выход."
            )

        if prev_snapshot is not None:
            hidden_delta = self._mean_delta(snapshot["hidden"], prev_snapshot["hidden"])
            output_delta = self._mean_delta(snapshot["output"], prev_snapshot["output"])
            lines.append(
                f"Среднее изменение весов от прошлой эпохи: "
                f"hidden {hidden_delta:.5f}, output {output_delta:.5f}"
            )
        return "\n".join(lines)

    def _mean_delta(self, matrix: list, prev_matrix: list) -> float:
        total = 0.0
        count = 0
        for row, prev_row in zip(matrix, prev_matrix):
            for value, prev_value in zip(row, prev_row):
                total += abs(value - prev_value)
                count += 1
        return total / max(count, 1)

    def _draw_hidden_matrix(self, snapshot: dict, prev_snapshot: dict) -> None:
        self.hidden_canvas.delete("all")

        hidden = snapshot["hidden"]
        rows = len(hidden)
        cols = len(hidden[0]) if rows else 0
        cell = self._cell_size_for(cols)
        grid_side = self.input_side
        neuron_gap_x = 26
        neuron_gap_y = 34
        neurons_per_row = 4
        mini_width = grid_side * cell + max(grid_side - 1, 0) * CELL_GAP
        mini_height = grid_side * cell + max(grid_side - 1, 0) * CELL_GAP
        label_pad = 24
        total_rows = (rows + neurons_per_row - 1) // neurons_per_row
        total_width = neurons_per_row * mini_width + max(neurons_per_row - 1, 0) * neuron_gap_x
        total_height = total_rows * (mini_height + label_pad) + max(total_rows - 1, 0) * neuron_gap_y

        canvas_height = total_height + PADDING * 2 + 36
        canvas_width = total_width + PADDING * 2 + 30
        self.hidden_canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))
        self.hidden_canvas.create_text(
            PADDING, 8, anchor="nw",
            text=f"Скрытый слой: {rows} нейронов, каждый как карта весов {grid_side} x {grid_side}",
            font=("Segoe UI", 11, "bold"), fill="#2f3e46"
        )

        influence = None
        if self.selected_output is not None:
            influence = snapshot["output"][self.selected_output]
            influence_limit = max(abs(value) for value in influence) or 1.0
        else:
            influence_limit = 1.0

        for row_index, row in enumerate(hidden):
            if influence is None:
                row_boost = 1.0
            else:
                row_boost = 0.35 + 0.65 * abs(influence[row_index]) / influence_limit

            block_col = row_index % neurons_per_row
            block_row = row_index // neurons_per_row
            base_x = PADDING + block_col * (mini_width + neuron_gap_x)
            base_y = PADDING + 28 + block_row * (mini_height + label_pad + neuron_gap_y)

            neuron_title = f"h{row_index + 1}"
            if influence is not None:
                neuron_title += f"  ->  {influence[row_index]:+.2f}"

            self.hidden_canvas.create_text(
                base_x,
                base_y - 14,
                anchor="nw",
                text=neuron_title,
                font=("Segoe UI", 10, "bold"),
                fill="#2f3e46"
            )

            self.hidden_canvas.create_rectangle(
                base_x - 4,
                base_y - 4,
                base_x + mini_width + 4,
                base_y + mini_height + 4,
                outline="#d9d2c3",
                width=1
            )

            for col_index, value in enumerate(row):
                pixel_row = col_index // grid_side
                pixel_col = col_index % grid_side
                x0 = base_x + pixel_col * (cell + CELL_GAP)
                y0 = base_y + pixel_row * (cell + CELL_GAP)
                x1 = x0 + cell
                y1 = y0 + cell

                fill = weight_to_color(value, self.hidden_limit)
                if row_boost < 1.0:
                    fill = mix((246, 241, 232), hex_to_rgb(fill), row_boost)

                delta = 0.0
                if prev_snapshot is not None:
                    delta = value - prev_snapshot["hidden"][row_index][col_index]

                self.hidden_canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=fill,
                    outline=delta_to_outline(delta, self.hidden_delta_limit),
                    width=1
                )
                self.hidden_canvas.create_text(
                    (x0 + x1) / 2, (y0 + y1) / 2,
                    text=f"{value:.2f}",
                    fill=text_color_for_fill(fill),
                    font=("Consolas", max(4, min(7, cell - 18)))
                )

    def _draw_output_matrix(self, snapshot: dict, prev_snapshot: dict) -> None:
        self.output_canvas.delete("all")

        output = snapshot["output"]
        rows = len(output)
        cols = len(output[0]) if rows else 0
        cell = max(20, self._cell_size_for(cols))
        total_width = cols * cell + max(cols - 1, 0) * CELL_GAP
        total_height = rows * cell + max(rows - 1, 0) * CELL_GAP

        canvas_height = total_height + PADDING * 2 + 48
        canvas_width = total_width + PADDING * 2 + 54
        self.output_canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))
        self.output_canvas.create_text(
            PADDING, 8, anchor="nw",
            text=f"Матрица весов выходного слоя ({rows} x {cols})",
            font=("Segoe UI", 11, "bold"), fill="#2f3e46"
        )

        for row_index, row in enumerate(output):
            label_y = PADDING + 24 + row_index * (cell + CELL_GAP) + cell / 2
            self.output_canvas.create_text(
                PADDING - 2, label_y, anchor="e",
                text=self.class_names[row_index],
                font=("Segoe UI", 10, "bold"), fill="#2f3e46"
            )

            is_selected = self.selected_output == row_index or self.selected_output is None

            for col_index, value in enumerate(row):
                x0 = PADDING + 20 + col_index * (cell + CELL_GAP)
                y0 = PADDING + 24 + row_index * (cell + CELL_GAP)
                x1 = x0 + cell
                y1 = y0 + cell

                fill = weight_to_color(value, self.output_limit)
                if not is_selected:
                    fill = mix((246, 241, 232), hex_to_rgb(fill), 0.35)

                delta = 0.0
                if prev_snapshot is not None:
                    delta = value - prev_snapshot["output"][row_index][col_index]

                border_width = 2 if self.selected_output == row_index else 1
                self.output_canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=fill,
                    outline=delta_to_outline(delta, self.output_delta_limit),
                    width=border_width
                )
                self.output_canvas.create_text(
                    (x0 + x1) / 2, (y0 + y1) / 2,
                    text=f"{value:.2f}",
                    fill=text_color_for_fill(fill),
                    font=("Consolas", max(5, min(8, cell - 16)))
                )

    def _cell_size_for(self, cols: int) -> int:
        return CELL_SIZE

    def run(self) -> None:
        self.root.mainloop()


def show_weight_history(history: list, class_names: list, input_side: int = 9) -> None:
    if not history:
        return
    viewer = WeightHistoryViewer(history, class_names, input_side=input_side)
    viewer.run()


def _activation_to_color(value: float) -> str:
    return mix((244, 241, 234), (52, 152, 219), clamp(value, 0.0, 1.0))


def show_inference_path(
    image_path: str,
    pixels: list,
    hidden: list,
    outputs: list,
    class_names: list,
    predicted_index: int,
) -> None:
    root = tk.Tk()
    root.title("Путь изображения по нейронной сети")
    root.configure(bg="#f7f2e8")

    tk.Label(
        root,
        text=f"Изображение: {image_path}",
        bg="#f7f2e8",
        fg="#2f3e46",
        anchor="w",
        justify="left",
        font=("Segoe UI", 10, "bold"),
        wraplength=1180
    ).pack(fill="x", padx=16, pady=(14, 6))

    predicted_name = class_names[predicted_index]
    tk.Label(
        root,
        text=f"Предсказание: {predicted_name}    Максимальный выход: {outputs[predicted_index]:.4f}",
        bg="#f7f2e8",
        fg="#5c6770",
        anchor="w",
        justify="left",
        font=("Consolas", 10)
    ).pack(fill="x", padx=16, pady=(0, 10))

    canvas = tk.Canvas(root, width=1220, height=720, bg="#fffaf2", highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    input_x = 40
    input_y = 100
    pixel_size = 28
    pixel_gap = 4

    try:
        image = Image.open(image_path).convert("L").resize((180, 180))
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo
        canvas.create_text(130, 32, text="Исходное изображение", font=("Segoe UI", 11, "bold"), fill="#2f3e46")
        canvas.create_image(130, 120, image=photo)
    except Exception:
        canvas.create_text(130, 120, text="Не удалось\nзагрузить\nизображение", font=("Segoe UI", 11, "bold"), fill="#aa4439")

    grid_top = 250
    canvas.create_text(170, grid_top - 26, text="Вход 9 x 9", font=("Segoe UI", 11, "bold"), fill="#2f3e46")
    for row in range(9):
        for col in range(9):
            value = pixels[row * 9 + col]
            x0 = input_x + col * (pixel_size + pixel_gap)
            y0 = grid_top + row * (pixel_size + pixel_gap)
            x1 = x0 + pixel_size
            y1 = y0 + pixel_size
            fill = mix((248, 246, 240), (47, 62, 70), value)
            canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#d9d2c3", width=1)
            canvas.create_text(
                (x0 + x1) / 2, (y0 + y1) / 2,
                text=str(int(value)),
                fill=text_color_for_fill(fill),
                font=("Consolas", 10, "bold")
            )

    hidden_x = 430
    hidden_y = 90
    hidden_w = 160
    hidden_h = 26
    hidden_gap = 8
    canvas.create_text(hidden_x + 95, 32, text="Скрытый слой", font=("Segoe UI", 11, "bold"), fill="#2f3e46")

    for index, value in enumerate(hidden):
        y0 = hidden_y + index * (hidden_h + hidden_gap)
        y1 = y0 + hidden_h
        fill = _activation_to_color(value)
        canvas.create_rectangle(hidden_x, y0, hidden_x + hidden_w, y1, fill=fill, outline="#a7bbc7", width=1)
        canvas.create_text(hidden_x - 10, (y0 + y1) / 2, text=f"h{index + 1}", anchor="e", font=("Consolas", 10), fill="#2f3e46")
        canvas.create_text(hidden_x + hidden_w / 2, (y0 + y1) / 2, text=f"{value:.4f}", font=("Consolas", 10, "bold"), fill=text_color_for_fill(fill))

    output_x = 820
    output_y = 180
    output_w = 230
    output_h = 44
    output_gap = 20
    canvas.create_text(output_x + 115, 32, text="Выходной слой", font=("Segoe UI", 11, "bold"), fill="#2f3e46")

    max_output = max(outputs) if outputs else 1.0
    for index, value in enumerate(outputs):
        y0 = output_y + index * (output_h + output_gap)
        y1 = y0 + output_h
        bar_w = max(14, int(output_w * (value / max_output))) if max_output > 0 else 14
        fill = "#2a9d8f" if index == predicted_index else "#f4a261"
        canvas.create_rectangle(output_x, y0, output_x + output_w, y1, fill="#efe7d7", outline="#d9d2c3", width=1)
        canvas.create_rectangle(output_x, y0, output_x + bar_w, y1, fill=fill, outline=fill, width=1)
        canvas.create_text(output_x - 12, (y0 + y1) / 2, text=class_names[index], anchor="e", font=("Segoe UI", 11, "bold"), fill="#2f3e46")
        canvas.create_text(output_x + output_w / 2, (y0 + y1) / 2, text=f"{value:.4f}", font=("Consolas", 11, "bold"), fill="#23313a")

    canvas.create_line(310, 395, hidden_x - 28, 355, fill="#8aa1b1", width=3, smooth=True, arrow="last")
    canvas.create_line(hidden_x + hidden_w + 24, 355, output_x - 24, 250, fill="#8aa1b1", width=3, smooth=True, arrow="last")
    canvas.create_text(360, 340, text="81 входов", font=("Segoe UI", 10), fill="#5c6770")
    canvas.create_text(715, 315, text="20 активаций", font=("Segoe UI", 10), fill="#5c6770")

    tk.Label(
        root,
        text="Слева показан бинарный вход 9x9, в центре активации скрытых нейронов, справа итоговые выходы.",
        bg="#f7f2e8",
        fg="#5c6770",
        anchor="w",
        justify="left",
        font=("Segoe UI", 9)
    ).pack(fill="x", padx=16, pady=(0, 14))

    root.mainloop()
