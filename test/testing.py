import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from src.read_data import ReadFile

a = ReadFile(
	train_path="/home/aditta/Desktop/trainme/input/train.csv",
	label="price_range",
    task_type="classification",
	compare=False,
	fold="random",
	model_name="SVM",
)

print(a.report())
print(a.train())