import os, math, random
from dataclasses import dataclass
from PIL import Image
#LLM импорт для визуализации
from weight_visualizer import show_inference_path, show_weight_history


@dataclass
class NetworkConfig:
    input_size: int = 81
    hidden_size: int = 20
    output_size: int = 4
    learning_rate: float = 0.01
    epochs: int = 500
    error_min: float = 0.01
    weight_range: float = 0.0   
    bias_range: float = 1.0   


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(output: float) -> float:
    return output * (1.0 - output)


def amount(a: list, b: list) -> float:
    summ = 0.0
    for x,y in zip(a, b):
        summ += x * y
    return summ


class NeuralNetwork:
    def __init__(self, cfg: NetworkConfig):
        self.config = cfg
        random.seed(42)
        #LLM инициализация истории весов и метрик
        self.weight_history = []

        def matrix(r, c):
            _matrix = []
            for i in range(r):
                row = []
                for j in range(c):
                    value = random.uniform(-cfg.weight_range, cfg.weight_range)
                    row.append(value)
                _matrix.append(row)
            return _matrix


        self.weights_hidden = matrix(cfg.hidden_size, cfg.input_size)

        self.bias_hidden = []
        for i in range(cfg.hidden_size):
            value = random.uniform(-cfg.bias_range, cfg.bias_range)
            self.bias_hidden.append(value)

        self.weights_output = matrix(cfg.output_size, cfg.hidden_size)

        self.bias_output = []
        for i in range(cfg.output_size):
            value = random.uniform(-cfg.bias_range, cfg.bias_range)
            self.bias_output.append(value)
        #LLM запись начальных весов и метрик
        self.record_weight_snapshot(0, None, None, None)
    #LLM метод для записи истории весов и метрик
    def record_weight_snapshot(self, epoch: int, avg_error: float, accuracy: float, val_error: float) -> None:
        hidden_copy = [row[:] for row in self.weights_hidden]
        output_copy = [row[:] for row in self.weights_output]
        self.weight_history.append({
            "epoch": epoch,
            "hidden": hidden_copy,
            "output": output_copy,
            "avg_error": avg_error,
            "accuracy": accuracy,
            "val_error": val_error,
        })

    def forward(self, inputs: list) -> tuple:
        hidden = []

        for j in range(self.config.hidden_size):
            net = amount(self.weights_hidden[j], inputs) + self.bias_hidden[j]
            value = sigmoid(net)
            hidden.append(value)

        output = []

        for k in range(self.config.output_size):
            net = amount(self.weights_output[k], hidden) + self.bias_output[k]
            value = sigmoid(net)
            output.append(value)
        
        return hidden, output

    def backward(self, inputs: list, hidden: list, outputs: list, targets: list) -> float:
        lr = self.config.learning_rate
        
        delta_output = []

        for k in range(self.config.output_size):
            error = 2*(targets[k] - outputs[k])
            delta = error * sigmoid_derivative(outputs[k])
            delta_output.append(delta)

        for k in range(self.config.output_size):
            for j in range(self.config.hidden_size):
                self.weights_output[k][j] += lr * delta_output[k] * hidden[j]
            self.bias_output[k] += lr * delta_output[k]

        delta_hidden = []

        for j in range(self.config.hidden_size):
            summ = 0.0
            for k in range(self.config.output_size):
                summ += self.weights_output[k][j] * delta_output[k]
            delta = summ * sigmoid_derivative(hidden[j])
            delta_hidden.append(delta)


        for j in range(self.config.hidden_size):
            for i in range(self.config.input_size):
                self.weights_hidden[j][i] += lr * delta_hidden[j] * inputs[i]
            self.bias_hidden[j] += lr * delta_hidden[j]

        return sum((targets[k] - outputs[k]) ** 2
                   for k in range(self.config.output_size)) / self.config.output_size

    def predict(self, inputs: list) -> int:
        _, outputs = self.forward(inputs)
        return outputs.index(max(outputs))

    def predict_proba(self, inputs: list) -> list:
        _, outputs = self.forward(inputs)
        return outputs

    def train(self, dataset: list, val: list) -> tuple:
        avg_error = 0.0

        for epoch in range(1, self.config.epochs + 1):
            random.shuffle(dataset)
            total_error = 0.0
            correct = 0

            for one in dataset:
                targets = [0.0] * self.config.output_size
                targets[one["label"]] = 1.0
                hidden, outputs = self.forward(one["pixels"])
                total_error += self.backward(one["pixels"], hidden, outputs, targets)
                if outputs.index(max(outputs)) == one["label"]:
                    correct += 1

            avg_error = total_error / len(dataset)
            accuracy  = correct / len(dataset)            
            val_correct = sum(1 for s in val if self.predict(s["pixels"]) == s["label"])
            val_errors  = len(val) - val_correct
            val_error = val_errors / len(val)
            #LLM запись истории весов и метрик
            self.record_weight_snapshot(epoch, avg_error, accuracy, val_error)

            if epoch == 1 or epoch % 10 == 0:
                print(f"Эпоха {epoch:>4} | MSE: {avg_error:.6f} | Точность: {accuracy:.1%} | "
                      f"Ошибка валидации: {val_error:.1%}")

            #if accuracy == 1.0:
            #    print(f"Достигнута 100% точность на эпохе {epoch}!")
            #    break
            if avg_error < self.config.error_min:
                print(f"Достигнут минимум ошибки на эпохе {epoch}!")
                break

        return avg_error


def read_png(filepath: str) -> list:
    img = Image.open(filepath).convert("L").resize((9, 9))
    
    pixels = []

    for r in range(9):
        for c in range(9):
            pixel = img.getpixel((c, r))

            if pixel < 128:
                pixels.append(0.0)
            else:
                pixels.append(1.0)
    return pixels


def load_dataset(folder: str, class_names: list) -> list:
    data = []
    for label, cls in enumerate(class_names):
        path = os.path.join(folder, cls)
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(".png"):
                pixels = read_png(os.path.join(path, fname))
                data.append({"pixels": pixels, "label": label, "name": fname})
    print(f"Правильные ответы: { [s['label'] for s in data] }\n")
    return data


def print_image(pixels: list) -> None:
    print("  " + "─" * 19)
    for r in range(9):
        row = ""
        for c in range(9):
            pixel = pixels[r * 9 + c]
            if pixel:
                row += "  "
            else:
                row += "##"
        print(f"  |{row}|")
    print("  " + "─" * 19)






CLASS_NAMES = ["+", "V", "O", "sq"]
cfg = NetworkConfig()

print(f"Входов : {cfg.input_size}")
print(f"Скрытый слой : {cfg.hidden_size} нейронов")
print(f"Выходов : {cfg.output_size}")
print(f"Скорость обучения: {cfg.learning_rate}")
print(f"Макс. эпох : {cfg.epochs}")
print(f"Мин. ошибка : {cfg.error_min}\n")

train = load_dataset("train", CLASS_NAMES)
demo = load_dataset("demo", CLASS_NAMES)
network = NeuralNetwork(cfg)
lastMSE = network.train(train, demo)
#LLM визуализация весов
show_weight_history(network.weight_history, CLASS_NAMES, input_side=9)

print(f"\nMSE после обучения : {lastMSE:.6f}")

print("\nДЕМОНСТРАЦИЯ")

while True:
    path = input("\nПуть к изображению: ").strip()
    if not os.path.isfile(path):
        print(f"Файл не найден: {path}")
        continue

    pixels = read_png(path)
    hidden, proba = network.forward(pixels)
    pred   = proba.index(max(proba))
    show_inference_path(path, pixels, hidden, proba, CLASS_NAMES, pred)

    print_image(pixels)
    print(f"\nРезультат: {CLASS_NAMES[pred]}")
    print("\nВыходы нейронов последнего слоя:")
    for i, (cls, val) in enumerate(zip(CLASS_NAMES, proba)):
        marker = ""
        if i == pred:
            marker = " <-- предсказание"
        print(f"  [{i}] {cls:8s}  {val:.4f}{marker}")
