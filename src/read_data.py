'''
@aditta_das_nishd

This class will automatically read path and analyze path

read_path: 
1. csv
2. excel
3. json
4. feather
5. sqlite3
'''
import os
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from logger import logger
from IPython.display import display
from .utils import auto_output_folder, normal_data_split, null_checker, reduce_mem_usage
from .visualize import *
from .params import without_compare


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


class ReadFile:
	def __init__(
		self, 
		train_path, 
		test_path=None, 
		submission_path=None, 
		drop_col=None,
		label=None,
		task_type="Classification",
		scaler="standard",
		fold="kfold",
		compare=True,
		model_name="RandomForest",
		fill_value=None,
		output_path="output"
	):
		self.train_path = train_path
		self.test_path = test_path
		self.submission_path = submission_path
		self.drop_col = drop_col
		self.label = label
		self.task_type = task_type
		self.fold = fold
		self.compare = compare
		if self.compare is False:
			self.model_name = model_name
		self.output_path = output_path

	def load_path(self):
		train_file = pd.read_csv(self.train_path)
		train_file = reduce_mem_usage(train_file)
		auto_output_folder(self.output_path)
		logger.info(f"Output folder : {self.output_path} created")
		if self.drop_col is not None:
			train_file.drop(self.drop_col, axis=1, inplace=True)
		return train_file

	def report(self):
		train_file = self.load_path()
		logger.info("Description of data")
		display(
			tabulate([
				[train_file.describe()]
			])
		)
		logger.info("Is there any null values?")

		null_checker(train_file)
		
		if self.task_type == "classification":
			logger.info(f"Label: {self.label} balanced or not?")
			display(
				tabulate([
					[train_file[self.label].value_counts()]
				])
			)
		else: pass # for regression

		train_file.to_csv(f"{os.path.join(os.path.dirname(os.getcwd()), self.output_path)}/reduced_dataset.csv", index=False)
		logger.info("Datset created and saved")	

	def visualize(self):
		pass

	def train(self):
		path = os.path.join(f"{os.path.join(os.path.dirname(os.getcwd()), self.output_path)}")
		df = pd.read_csv(os.path.join(f"{path}/reduced_dataset.csv"))
		if self.fold == "kfold":
			print("Kfold")
		elif self.fold == "skfold":
			print("Stratified Kfold")
		elif self.fold == "random":
			print("Random")
			xtrain, xtest, ytrain, ytest = normal_data_split(df, self.label)

		if self.task_type == "classification":
			if self.compare is True:
				pass
			else:
				scores = []
				for model_name, params in without_compare().items():
					if model_name == self.model_name:
						clf = GridSearchCV(params["model"], params["params"], cv=5, return_train_score=False)
						clf.fit(xtrain, ytrain)
						scores.append({
							"model": self.model_name,
							"best_params": clf.best_params_,
							"best_scores": clf.best_score_,
						})
		return pd.DataFrame(scores)
			# with open(f"{path}/best_params_{self.model_name}.json", "w") as file:
			# 	json.dump(model.get_params(), file)
			# logger.info(f"Saving best params >> best_params_{self.model_name}.json")
			# return model.best_params_


	def prediction(self):
		pass

	def kaggle_submission(self):
		pass

