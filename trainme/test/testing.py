import sys, json
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from src.read_data import ReadFile

s = ReadFile(
	train_path="/home/aditta/Desktop/trainme/trainme/input/multi_class_classification.csv",
	test_path="/home/aditta/Desktop/trainme/trainme/input/multi_class_classification_test.csv",
	label="target",
    task_type="multi_classification",
	compare=False,
	fold="skfold",
	model_name="xgb",
	output_path="/media/aditta/NewVolume/amazon",
	study_name="new_train",
	store_file ="out9",
	n_trials=1
)

print(s.report())
print(s.train())

# import pandas as pd
# train_path="/home/aditta/Desktop/trainme/trainme/input/multi_class_classification.csv"
# df = pd.read_csv(train_path)
# print(df)
# print(df["target"].dtype == "object")
# with open("/home/aditta/Desktop/trainme/output/features.json") as f:
# 	data = json.load(f)
# 	print(data["label"])