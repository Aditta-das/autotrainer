# trainme - documentation

## Authors

- [@Aditta-das](https://github.com/Aditta-das)


# kaggle-autolgb

trainer is a combination of xgboost, lightgbm and optuna. I tried to make kaggle monthly competition simple. Its only working with classification problem.

## Installation
```
pip install trainer
``` 
   
## Features

- autotune
- autotrain
- auto submission file generate
- auto prediction 


## Deployment

```
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

```

## License

[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)


