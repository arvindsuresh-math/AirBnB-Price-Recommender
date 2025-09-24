# Airbnb Project — Initial GitHub Repo Setup (Minimum Viable Setup)

This document outlines the minimal set of files, directories, and configurations that must be created at the start of the project to ensure smooth onboarding and avoid downstream compatibility issues.

---

## ✅ 1. Create `.gitignore` file

Create a file named `.gitignore` in the project root with the following content:

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*.so

# Jupyter Notebook
.ipynb_checkpoints/

# Environment files
.env
.venv/
*.env

# macOS
.DS_Store

# AWS config
.aws/

# Logs
*.log

# Data
data/
```

---

## ✅ 2. Create `environment.yml` for conda (Python + Spark environment)

Create a file named `environment.yml` in the project root with the following content:

```
name: airbnb-project
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pyspark
  - pandas
  - jupyterlab
  - boto3
  - sqlalchemy
  - ipykernel
  - pip
  - pip:
      - great_expectations
```

> ✅ NOTE: This is a *conda environment* specification. Team members will run:
> ```
> conda env create -f environment.yml
> conda activate airbnb-project
> ```

---

## ✅ 3. Create `README.md` in the root

Create a basic `README.md` file with the following boilerplate:

```
# Airbnb Deep Learning Project

This repository contains code and assets for a deep learning project using Airbnb listings and reviews data.

## Project Setup

1. Install Anaconda or Miniconda.
2. Create the environment:

```bash
conda env create -f environment.yml
conda activate airbnb-project
```

3. Launch Jupyter Lab (optional):

```bash
jupyter lab
```

## Directory Structure

- `notebooks/`: Jupyter notebooks for EDA and prototyping.
- `sql/`: SQL scripts for querying and transforming raw data.
- `scripts/`: Python scripts for data access and utilities.
- `src/`: Source code for core logic, utils, and models.
```

---

## ✅ 4. Create basic folder structure

In the root of the repository, create the following **empty directories**:

```
notebooks/
sql/
scripts/
src/
```

> You may add placeholder `.gitkeep` files in each directory to allow Git to track them even if they're empty.

Example command:
```bash
touch notebooks/.gitkeep sql/.gitkeep scripts/.gitkeep src/.gitkeep
```

---

## ✅ 5. Optional: Add `SETUP.md` if not already present

If a `SETUP.md` file does not yet exist, create one. Add team-specific setup instructions, such as Git setup, AWS CLI configuration, or VSCode setup. This will grow over time.

---

## ✅ 6. Initialize git repo (if not already done)

In the project root:
```bash
git init
git add .
git commit -m "Initial commit: base environment and folder structure"
```

Push to GitHub:
```bash
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```

---

## ✅ Summary of Files and Folders

```
airbnb-project/
│
├── .gitignore
├── environment.yml
├── README.md
├── SETUP.md                # Optional, if not already present
│
├── notebooks/
├── sql/
├── scripts/
├── src/
```

> All folders may be initialized with a `.gitkeep` file if empty.
