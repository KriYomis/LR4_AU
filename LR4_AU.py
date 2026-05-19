import os, math, random
from dataclasses import dataclass
from PIL import Image

@dataclass
class NetworkConfig:
    input_size:int = 81
    hidden_size:int = 20
    output_size:int = 4 
    learning_rate:float = 0.01
    epochs:int = 500 
    error_threshold:float = 0.01

random.seed(42)

@dataclass
class EpochResult:
    epoch:int
    total_error:float
    accuracy:float
 

@dataclass
class EpochTrace:
    epoch:int
    total_error:float
    accuracy:float
    weights_hidden:list 
    weights_output:list
    details:list  
 
 
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
 
 
def sigmoid_derivative(output: float) -> float:
    return output * (1.0 - output)
 
 
def amount(a: list, b: list) -> float:
    for i in range(len(a)):
        for j in range(len(b)):
            total = sum(a[i] * b[i] for i in range(len(a)))
    return total
 
 
class NeuralNetwork:
    def __init__(self, config: NetworkConfig):
        self.config = config
        def rand_matrix(r, c):
            return [[random.uniform(-1.0, 1.0) for _ in range(c)]
                    for _ in range(r)]
 
        self.weights_hidden = rand_matrix(config.hidden_size, config.input_size)
        self.bias_hidden    = [random.uniform(-1.0, 1.0) for _ in range(config.hidden_size)]
        self.weights_output = rand_matrix(config.output_size, config.hidden_size)
        self.bias_output    = [random.uniform(-1.0, 1.0) for _ in range(config.output_size)]
 
    def forward(self, inputs: list) -> tuple:
        hidden = [
            sigmoid(amount(self.weights_hidden[j], inputs) + self.bias_hidden[j])
            for j in range(self.config.hidden_size)
        ]
        output = [
            sigmoid(amount(self.weights_output[k], hidden) + self.bias_output[k])
            for k in range(self.config.output_size)
        ]
        return hidden, output
 
    def backward(self, inputs: list, hidden: list, outputs: list, targets: list) -> float:
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
        _, outputs = self.forward(inputs)
        return outputs.index(max(outputs))
 
    def predict_proba(self, inputs: list) -> list:
        _, outputs = self.forward(inputs)
        return outputs
 
    def snapshot_weights(self) -> tuple:
        return ([row[:] for row in self.weights_hidden],
                [row[:] for row in self.weights_output])
 
    def train(self, dataset: list, class_names: list, progress_callback=None) -> tuple:
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
                print(f"\nДостигнут порог ошибки на эпохе {epoch}!")
                break
 
        scores = self._evaluate(dataset)
        return history, scores
 
    def _evaluate(self, dataset: list) -> dict:
        correct = sum(1 for s in dataset if self.predict(s["pixels"]) == s["label"])
        return {"correct": correct, "total": len(dataset),
                "accuracy": correct / len(dataset)}
 
 
def read_png(filepath: str) -> list:
    img = Image.open(filepath).convert("L").resize((9, 9))
    return [1.0 if img.getpixel((c, r)) < 128 else 0.0
            for r in range(9) for c in range(9)]
 
 
def load_dataset(folder: str, class_names: list) -> list:
    data = []
    for label, cls in enumerate(class_names):
        path = os.path.join(folder, cls)
        if not os.path.isdir(path):
            print(f"Папка не найдена: {path}")
            continue
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(".png"):
                pixels = read_png(os.path.join(path, fname))
                data.append({"pixels": pixels, "label": label, "name": fname})
    return data
 
def print_image(pixels: list) -> None:
    print("  " + "─" * 19)
    for r in range(9):
        row = "".join("##" if pixels[r * 9 + c] else "  " for c in range(9))
        print(f"  |{row}|")
    print("  " + "─" * 19)
 
 
def print_epoch(trace: EpochTrace) -> None:
    for line in trace.details:
        print(f"  {line}")
    print()
 
 
def recognize_loop(network: NeuralNetwork, class_names: list) -> None:
    print("\nДЕМОНСТРАЦИЯ")
 
    while True:
        path = input("\nПуть к изображению: ").strip()
 
        pixels = read_png(path)
        proba  = network.predict_proba(pixels)
        pred   = proba.index(max(proba))
 
        print_image(pixels)
        print(f"\nРезультат: «{class_names[pred]}»")
        print()
        print("Выходы всех нейронов последнего слоя:")
        for i, (cls, val) in enumerate(zip(class_names, proba)):
            marker = " <-- предсказание" if i == pred else ""
            print(f"    [{i}] {cls:8s}  {val:.4f}  {marker}")


random.seed(42)
CLASS_NAMES = ["+", "V", "O", "sq"]
 
cfg = NetworkConfig()
base_dir = os.path.dirname(os.path.abspath(__file__))

print(f"\nВходов           : {cfg.input_size}  (9x9 пикселей)")
print(f"Скрытый слой     : {cfg.hidden_size} нейронов")
print(f"Выходов          : {cfg.output_size}")
print(f"Скорость обучения: {cfg.learning_rate}")
print(f"Макс. эпох       : {cfg.epochs}")
print(f"Порог ошибки     : {cfg.error_threshold}\n")

train = load_dataset(os.path.join(base_dir, "train"), CLASS_NAMES)

network = NeuralNetwork(cfg)

def on_epoch(trace: EpochTrace) -> None:
    if trace.epoch == 1 or trace.epoch % 10 == 0:
        print_epoch(trace)

history, scores = network.train(train, CLASS_NAMES, progress_callback=on_epoch)

print(f"\nMSE после обучения : {history[-1].total_error:.6f}")
print(f"Правильно          : {scores['correct']}/{scores['total']}")

recognize_loop(network, CLASS_NAMES)