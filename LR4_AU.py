"""
Распознавание изображений на базе НС обратного распространения
Классы: «+», «V», «O», «sq» — изображения PNG 9x9
Датасет: train/+/*.png, train/V/*.png, train/O/*.png, train/sq/*.png
 
pip install pillow
"""
 
import os
import math
import random
from dataclasses import dataclass
from PIL import Image
 
 
# ─────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────
 
@dataclass
class NetworkConfig:
    input_size:      int   = 81     # 9x9 пикселей
    hidden_size:     int   = 20     # нейронов в скрытом слое
    output_size:     int   = 4      # количество классов
    learning_rate:   float = 0.1    # скорость обучения
    epochs:          int   = 500    # максимум эпох
    error_threshold: float = 0.01   # порог ошибки для ранней остановки
    random_seed:     int   = 42
 
 
# ─────────────────────────────────────────────────────────
#  СТРУКТУРЫ ДАННЫХ
# ─────────────────────────────────────────────────────────
 
@dataclass
class EpochResult:
    epoch:       int
    total_error: float
    accuracy:    float
 
 
@dataclass
class EpochTrace:
    epoch:          int
    total_error:    float
    accuracy:       float
    weights_hidden: list   # снимок весов скрытого слоя
    weights_output: list   # снимок весов выходного слоя
    details:        list   # текстовый протокол шага
 
 
# ─────────────────────────────────────────────────────────
#  МАТЕМАТИКА  (без сторонних библиотек)
# ─────────────────────────────────────────────────────────
 
def sigmoid(x: float) -> float:
    """Сжимает любое число в диапазон (0, 1)."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
 
 
def sigmoid_derivative(output: float) -> float:
    """Производная сигмоиды через уже вычисленный выход."""
    return output * (1.0 - output)
 
 
def dot(a: list, b: list) -> float:
    """Скалярное произведение двух векторов."""
    return sum(x * y for x, y in zip(a, b))
 
 
# ─────────────────────────────────────────────────────────
#  НЕЙРОННАЯ СЕТЬ
#
#  Архитектура: вход(81) -> скрытый(20) -> выход(4)
#
#  Прямой проход:
#    hidden[j] = sigmoid( dot(w_hidden[j], input) + bias_h[j] )
#    output[k] = sigmoid( dot(w_output[k], hidden) + bias_o[k] )
#
#  Обратное распространение:
#    delta_out[k]    = (target[k] - output[k]) * sigmoid'(output[k])
#    delta_hidden[j] = sum_k(w_output[k][j] * delta_out[k]) * sigmoid'(hidden[j])
#    w_output[k][j] += lr * delta_out[k]    * hidden[j]
#    w_hidden[j][i] += lr * delta_hidden[j] * input[i]
# ─────────────────────────────────────────────────────────
 
class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        rng = random.Random(config.random_seed)
 
        def rand_matrix(rows, cols):
            return [[rng.uniform(-0.5, 0.5) for _ in range(cols)]
                    for _ in range(rows)]
 
        self.weights_hidden = rand_matrix(config.hidden_size, config.input_size)
        self.bias_hidden    = [rng.uniform(-0.5, 0.5) for _ in range(config.hidden_size)]
        self.weights_output = rand_matrix(config.output_size, config.hidden_size)
        self.bias_output    = [rng.uniform(-0.5, 0.5) for _ in range(config.output_size)]
 
    def forward(self, inputs: list) -> tuple:
        """Прямой проход. Возвращает (hidden_outputs, final_outputs)."""
        hidden = [
            sigmoid(dot(self.weights_hidden[j], inputs) + self.bias_hidden[j])
            for j in range(self.config.hidden_size)
        ]
        output = [
            sigmoid(dot(self.weights_output[k], hidden) + self.bias_output[k])
            for k in range(self.config.output_size)
        ]
        return hidden, output
 
    def backward(self, inputs: list, hidden: list, outputs: list, targets: list) -> float:
        """Обратное распространение ошибки. Обновляет веса, возвращает MSE."""
        lr = self.config.learning_rate
 
        delta_output = [
            (targets[k] - outputs[k]) * sigmoid_derivative(outputs[k])
            for k in range(self.config.output_size)
        ]
        delta_hidden = [
            sum(self.weights_output[k][j] * delta_output[k]
                for k in range(self.config.output_size))
            * sigmoid_derivative(hidden[j])
            for j in range(self.config.hidden_size)
        ]
 
        for k in range(self.config.output_size):
            for j in range(self.config.hidden_size):
                self.weights_output[k][j] += lr * delta_output[k] * hidden[j]
            self.bias_output[k] += lr * delta_output[k]
 
        for j in range(self.config.hidden_size):
            for i in range(self.config.input_size):
                self.weights_hidden[j][i] += lr * delta_hidden[j] * inputs[i]
            self.bias_hidden[j] += lr * delta_hidden[j]
 
        return sum((targets[k] - outputs[k]) ** 2
                   for k in range(self.config.output_size)) / self.config.output_size
 
    def predict(self, inputs: list) -> int:
        """Возвращает индекс класса с наибольшим выходом."""
        _, outputs = self.forward(inputs)
        return outputs.index(max(outputs))
 
    def predict_proba(self, inputs: list) -> list:
        """Возвращает значения всех выходных нейронов."""
        _, outputs = self.forward(inputs)
        return outputs
 
    def snapshot_weights(self) -> tuple:
        """Копии весов обоих слоёв для протокола."""
        return ([row[:] for row in self.weights_hidden],
                [row[:] for row in self.weights_output])
 
    def train(self, dataset: list, class_names: list, progress_callback=None) -> tuple:
        """
        Обучение методом обратного распространения.
        Возвращает (history: list[EpochResult], scores: dict).
        """
        history = []
 
        for epoch in range(1, self.config.epochs + 1):
            random.shuffle(dataset)
            total_error, correct = 0.0, 0
 
            for sample in dataset:
                targets = [0.0] * self.config.output_size
                targets[sample["label"]] = 1.0
                hidden, outputs = self.forward(sample["pixels"])
                total_error += self.backward(sample["pixels"], hidden, outputs, targets)
                if outputs.index(max(outputs)) == sample["label"]:
                    correct += 1
 
            avg_error = total_error / len(dataset)
            accuracy  = correct / len(dataset)
            history.append(EpochResult(epoch=epoch, total_error=avg_error, accuracy=accuracy))
 
            if progress_callback is not None:
                wh, wo = self.snapshot_weights()
                trace = EpochTrace(
                    epoch=epoch,
                    total_error=avg_error,
                    accuracy=accuracy,
                    weights_hidden=wh,
                    weights_output=wo,
                    details=[
                        f"Эпоха {epoch}/{self.config.epochs}",
                        f"Средняя ошибка MSE : {avg_error:.6f}",
                        f"Точность           : {accuracy:.1%}",
                    ],
                )
                if progress_callback(trace) is False:
                    break
 
            if avg_error < self.config.error_threshold:
                print(f"\n  Достигнут порог ошибки на эпохе {epoch}!")
                break
 
        scores = self._evaluate(dataset)
        return history, scores
 
    def _evaluate(self, dataset: list) -> dict:
        correct = sum(1 for s in dataset if self.predict(s["pixels"]) == s["label"])
        return {"correct": correct, "total": len(dataset),
                "accuracy": correct / len(dataset)}
 
 
# ─────────────────────────────────────────────────────────
#  ЧТЕНИЕ PNG
# ─────────────────────────────────────────────────────────
 
def read_png(filepath: str) -> list:
    """PNG любого размера -> список из 81 числа (0.0 или 1.0)."""
    img = Image.open(filepath).convert("L").resize((9, 9))
    return [1.0 if img.getpixel((c, r)) < 128 else 0.0
            for r in range(9) for c in range(9)]
 
 
def load_dataset(folder: str, class_names: list) -> list:
    """Читает папки folder/<class>/*.png. Индекс в class_names = метка."""
    data = []
    for label, cls in enumerate(class_names):
        path = os.path.join(folder, cls)
        if not os.path.isdir(path):
            print(f"  Папка не найдена: {path}")
            continue
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(".png"):
                pixels = read_png(os.path.join(path, fname))
                data.append({"pixels": pixels, "label": label, "name": fname})
    return data
 
 
# ─────────────────────────────────────────────────────────
#  ВЫВОД В ТЕРМИНАЛ
# ─────────────────────────────────────────────────────────
 
def fitness_bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return f"[{'X' * filled + '.' * (width - filled)}] {value:.0%}"
 
 
def print_image(pixels: list) -> None:
    print("  " + "─" * 19)
    for r in range(9):
        row = "".join("##" if pixels[r * 9 + c] else "  " for c in range(9))
        print(f"  |{row}|")
    print("  " + "─" * 19)
 
 
def print_epoch(trace: EpochTrace) -> None:
    for line in trace.details:
        print(f"  {line}")
    print(f"  Прогресс : {fitness_bar(trace.accuracy)}")
    print()
 
 
# ─────────────────────────────────────────────────────────
#  РЕЖИМ РАСПОЗНАВАНИЯ
# ─────────────────────────────────────────────────────────
 
def recognize_loop(network: NeuralNetwork, class_names: list) -> None:
    print(f"\n{'─' * 46}")
    print("  РЕЖИМ РАСПОЗНАВАНИЯ")
    print("  Введите путь к PNG-файлу (или 'q' для выхода):")
    print(f"{'─' * 46}")
 
    while True:
        path = input("\n  Путь к изображению: ").strip()
        if path.lower() == "q":
            break
        if not os.path.isfile(path):
            print(f"  Файл не найден: {path}")
            continue
 
        pixels = read_png(path)
        proba  = network.predict_proba(pixels)
        pred   = proba.index(max(proba))
 
        print_image(pixels)
        print(f"\n  Результат: «{class_names[pred]}»")
        print()
        print("  Выходы всех нейронов последнего слоя:")
        for i, (cls, val) in enumerate(zip(class_names, proba)):
            marker = " <-- победитель" if i == pred else ""
            print(f"    [{i}] {cls:8s}  {val:.4f}  {fitness_bar(val, 10)}{marker}")
 
 
# ─────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────
 
CLASS_NAMES = ["+", "V", "O", "sq"]
 
 
def main():
    cfg = NetworkConfig()
    base_dir = os.path.dirname(os.path.abspath(__file__))
 
    print("=" * 46)
    print("  НС обратного распространения")
    print("  Классы: " + ", ".join(f"«{c}»" for c in CLASS_NAMES))
    print("=" * 46)
    print(f"\n  Входов           : {cfg.input_size}  (9x9 пикселей)")
    print(f"  Скрытый слой     : {cfg.hidden_size} нейронов")
    print(f"  Выходов          : {cfg.output_size}  (по одному на класс)")
    print(f"  Скорость обучения: {cfg.learning_rate}")
    print(f"  Макс. эпох       : {cfg.epochs}")
    print(f"  Порог ошибки     : {cfg.error_threshold}")
 
    train = load_dataset(os.path.join(base_dir, "train"), CLASS_NAMES)
    print(f"\n  Загружено обучающих примеров: {len(train)}")
    for i, cls in enumerate(CLASS_NAMES):
        count = sum(1 for s in train if s["label"] == i)
        print(f"    «{cls}» : {count} шт.")
 
    network = NeuralNetwork(cfg)
 
    print(f"\n{'─' * 46}")
    print("  РЕЖИМ ОБУЧЕНИЯ")
    print(f"{'─' * 46}\n")
 
    log_every = 50
 
    def on_epoch(trace: EpochTrace) -> None:
        if trace.epoch == 1 or trace.epoch % log_every == 0:
            print_epoch(trace)
 
    history, scores = network.train(train, CLASS_NAMES, progress_callback=on_epoch)
 
    print(f"{'─' * 46}")
    print("  Обучение завершено.")
    print(f"  Финальная точность : {fitness_bar(scores['accuracy'])}")
    print(f"  Правильно          : {scores['correct']}/{scores['total']}")
 
    recognize_loop(network, CLASS_NAMES)
 
 
if __name__ == "__main__":
    random.seed(42)
    main()