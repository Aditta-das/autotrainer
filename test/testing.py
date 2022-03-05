import sys, json
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from src.read_data import ReadFile

a = ReadFile(
	train_path="/home/aditta/Desktop/trainme/input/Breast_cancer_data.csv",
	label="diagnosis",
    task_type="binary_classification",
	compare=False,
	fold="skfold",
	model_name="xgb",
	output_path="new",
	study_name="new_train"
)

print(a.report())
print(a.train())

# with open("/home/aditta/Desktop/trainme/output/features.json") as f:
# 	data = json.load(f)
# 	print(data["label"])