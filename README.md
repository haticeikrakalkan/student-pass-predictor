# ScholarPath: Student Pass/Fail Predictor

![CI](https://github.com/haticeikrakalkan/student-pass-predictor/actions/workflows/ci.yml/badge.svg)

An end-to-end machine learning project: from data exploration to a trained model served through a **FastAPI** REST API and a simple web interface, packaged with **Docker**, tested with **pytest**, checked by **GitHub Actions** and deployed on **Render**.

Given background information about a student (family, study habits, lifestyle, absences and the **first-period grade G1**), the app predicts whether the student is likely to **pass or fail** the course, together with a probability.

> This is a learning project and a decision-support demo, not a tool for real decisions about students.

## Live demo

**[Open the app](https://student-pass-predictor.onrender.com)** | [API docs](https://student-pass-predictor.onrender.com/docs)

> The demo runs on a free hosting plan. If nobody has used it for a while it goes to sleep, so the first load can take about a minute.

## Project overview

| | |
|---|---|
| **Data** | [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008), Mathematics course, 395 students from two Portuguese schools (GP / MS). Downloaded from Kaggle. |
| **Target** | `Passing Status` = 1 if the final grade `G3 >= 10`, else 0 (265 pass / 130 fail) |
| **Features** | 31 inputs: demographics, family, school, lifestyle, absences, plus the first-period grade `G1` |
| **Excluded** | `G3` (it defines the target) and `G2` (see below) |
| **Final model** | Logistic Regression (`class_weight="balanced"`), with One-Hot Encoding and Standard Scaling |
| **Serving** | FastAPI + Jinja2 templates |
| **MLOps** | Docker, pytest, GitHub Actions (CI), MLflow experiment tracking, Render deployment |

## Key decisions and findings

**1. Avoiding target leakage.**
`G1`, `G2` and `G3` are highly correlated. With `G1` and `G2` the models looked excellent (regression R² of about 0.81), but that mostly reflects "previous grades predict the final grade". I removed `G2` and `G3` so the model works as an **early risk estimate** that only needs information available at the start of the term.

**2. Regression did not work well, so I reframed the problem.**
Without previous grades, the best regression model reached only R² of about 0.26 in 5-fold cross-validation. Predicting *pass/fail* is a more realistic and more useful question than predicting the exact grade.

**3. The first-period grade matters a lot.**

| Features | Accuracy | Recall (fail class) |
|---|---|---|
| No previous grades | about 0.68 (baseline is 0.67) | about 0.50 |
| With `G1` | about 0.73-0.80 | about 0.69-0.77 |

**4. Model selection used cross-validation, not a single split.**
The test set has only 79 students (26 failing), so one student changes recall by about 4 points and single-split rankings are noisy. I compared 9 classifiers with 5-fold stratified cross-validation, using the F1 score of the **fail class** as the metric. Logistic Regression was the best and most stable (mean F1 of about 0.76, std of about 0.03).

**5. Hyperparameter tuning (GridSearchCV).**
Tuning searched `penalty`, `C` and `class_weight`. The best setting was `C=1.0`, `l2` and `class_weight="balanced"`, so the main effect came from class weighting:

| Model | Accuracy | Fail-class recall | Fail-class F1 |
|---|---|---|---|
| Default Logistic Regression | 0.759 | 0.69 (18/26) | 0.65 |
| **Balanced Logistic Regression (final)** | 0.734 | **0.73 (19/26)** | 0.64 |

Accuracy dropped slightly (2 students) but more failing students are caught, which matches the goal of the tool. Overall, tuning did not change the model dramatically.

## Project structure

```
.
├── app.py                 # FastAPI application
├── student_model.pkl      # Saved model + encoder + scaler
├── model_test.py          # Quick script to test the saved model
├── test_model.py          # pytest test for the saved model
├── testdatascaled.csv     # Scaled test data used by the tests
├── Dockerfile             # Container image definition
├── .dockerignore          # Files excluded from the Docker image
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions: runs pytest on every push to main
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # EDA, regression and classification experiments, tuning
│   ├── 02_model_training_pipeline.ipynb     # Final training and model export
│   └── student_data.csv                     # Dataset
├── static/                # CSS / JS
├── templates/             # HTML templates
└── requirements.txt       # Pinned dependencies
```

## Run locally

Python 3.13 is recommended: it is the version used for training, in Docker and in CI.

```bash
# 1. Clone the repository
git clone https://github.com/haticeikrakalkan/student-pass-predictor.git
cd student-pass-predictor

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> for the web interface, or <http://127.0.0.1:8000/docs> for the interactive API docs.

> **Note:** `student_model.pkl` was saved with **scikit-learn 1.7.2**. `requirements.txt` pins this exact version (and the matching pandas and NumPy versions), so the model loads without version warnings.

## Run with Docker

```bash
docker build -t student-predictor .
docker run -p 8000:8000 student-predictor
```

Then open <http://localhost:8000>. The container listens on the port given by the `PORT` environment variable (default `8000`), which is how the hosting platform assigns its own port.

## Tests and CI

The test in `test_model.py` loads the saved model, predicts on the scaled test data and checks that there is one prediction per row and that every prediction is 0 or 1.

```bash
pip install pytest
pytest
```

The same test runs automatically on GitHub Actions on every push to `main` (see `.github/workflows/ci.yml`).

## Experiment tracking and deployment

- **MLflow:** training runs with different `C` values were logged locally (parameters and accuracy) to compare them. The tracking database is not part of this repository.
- **Deployment:** the Docker image is built from this repository and deployed on Render (free plan), which redeploys on every push to `main`.

## API

### `POST /predict`

Send a JSON body with the 31 student features. Example response:

```json
{
  "prediction": 1,
  "passed": true,
  "label": "Pass",
  "probability": 0.78
}
```

`probability` is the model's confidence in the predicted class.

## Limitations

- Small dataset (395 students, 79 in the test set), so results have high variance.
- Data comes from two Portuguese schools and may not generalize to other schools or countries.
- The model needs the first-period grade `G1`. Without it, background features alone predict poorly.
- `class_weight="balanced"` shifts probabilities toward the "fail" class, so probabilities are not calibrated.
- The live demo is on a free plan, so it sleeps when idle and the first request is slow.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, FastAPI, Uvicorn, Jinja2, Docker, pytest, GitHub Actions, MLflow, Render