from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy",
    "optuna",
    "pyarrow",
    "joblib",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "logger"
]

with open("README.md") as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="trainme",
        version="0.0.1",
        description="tune with optuna and model",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Aditta Das Nishad",
        author_email="nishad009adi@gmail.com",
        install_requires=INSTALL_REQUIRES,
        platforms=["linux", "unix"],
        python_requires=">=3.6",
        package_dir={"": "trainme"},
        packages=find_packages("trainme"),
        url="https://github.com/Aditta-das/autotrainer"
    )