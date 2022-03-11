import sys, json
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from src.read_data import ReadFile

s = ReadFile(
	train_path="/home/aditta/Desktop/kaggle_comp/train.csv",
	test_path="/home/aditta/Desktop/kaggle_comp/test.csv",
	# submission_path="/home/aditta/Desktop/kaggle_comp/sample_submission.csv",
	drop_col=["Name", "PassengerId", "Cabin"],
	label="Transported",
    task_type="binary_classification",
	compare=False,
	fold="kfold",
	model_name="lgb",
	output_path="/home/aditta/Desktop/kaggle_comp",
	study_name="new_train",
	folder_output ="new_out1",
	n_trials=1,
	kaggle=False
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