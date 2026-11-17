# MindTrace NG

Web-based system for early detection of Alzheimer's disease from MRI images using
Random Forest classification. Nigeria-wide case study. Indigo, ivory and saffron theme with a sidebar workspace.

An academic prototype. It is not a medical device and does not diagnose.

## Run it (5 minutes)

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
flask --app wsgi seed-demo        # optional demo login: demo / Demo@1234
python wsgi.py                    # open http://127.0.0.1:5000
```

The trained model is already in `models/`, so the app predicts immediately.
Fonts, Bootstrap and scripts are bundled, so it works with no internet.

### Use MySQL (as in the write-up)

1. Start MySQL (XAMPP or WAMP) and run `database/schema.sql`, or just create an empty database named `mindtrace_ng`.
2. Set the connection before starting the app:
   `set DATABASE_URL=mysql+pymysql://root:@127.0.0.1:3306/mindtrace_ng` (Windows)
   `export DATABASE_URL=...` (macOS/Linux)
3. `python wsgi.py`. Tables are created automatically.

Without `DATABASE_URL` the app stores data in `instance/mindtrace.db` (SQLite), which
is what a free Render instance will use.

### Try it

Sample slices from the held-out test set are in `samples/` (file names show the true class).
`samples/INVALID_*` files demonstrate rejection of bad uploads (TC04).

### Tests

```bash
python -m pytest -q        # 22 tests covering TC01 to TC12, features, privacy, performance
```

### Retrain

```bash
python train_model.py      # downloads the dataset to data/ if missing, about 4 minutes on a laptop
```

## Results on the reserved test set (960 images)

| Metric | Value |
| --- | --- |
| Accuracy | 81.7% |
| Precision (AD) | 78.4% |
| Recall (AD) | 87.5% |
| F1-score | 82.7% |
| Specificity | 75.8% |
| ROC-AUC | 0.906 |
| Confusion matrix | TP 420 · FN 60 · FP 116 · TN 364 |
| 5-fold CV accuracy (training set) | 82.8% ± 1.5 |
| Training / validation accuracy | 100.0% / 83.3% |

Dataset: 6,400 slices (3200 AD, 3200 No AD), split
4480 / 960 / 960.
Best parameters: {'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.33}.

### Limitations to state in Chapter 4 and at the defence

- The dataset has no subject identifiers, so slices from one person may fall in both
  training and test sets. The reported accuracy is probably optimistic.
- The forest fits its training data perfectly but scores about 82% on unseen data. Quote the test figures.
- The model sees one 2D slice. It has not been validated on scans from Nigerian hospitals.
- ADNI (named in the write-up) needs an approved data-use application, so the public
  "Alzheimer MRI" 4-class set was used and mapped to two classes. See `docs/WRITEUP_FIXES.md`.

## Project layout

```
wsgi.py                 entry point
train_model.py          dataset -> features -> Random Forest -> metrics
mindtrace/
  __init__.py           app factory, configuration
  models.py             User, MRIImage, Prediction tables
  routes.py             sign-in, scan upload, report, records, CSV export, performance
  nigeria.py            36 states + FCT by geopolitical zone
  ml/preprocess.py      validation, grayscale, resize, normalise
  ml/features.py        GLCM texture + intensity statistics (NumPy)
  ml/classifier.py      loads the model, predicts, feature importance
models/                 rf_model.joblib, metrics.json
templates/  static/     front-end (Bootstrap grid + mindtrace.css and mindtrace.js)
database/schema.sql     MySQL schema
tests/test_app.py       automated test cases
docs/PLAN.md            algorithm, flow, design and work plan
docs/WRITEUP_FIXES.md   corrections needed in Chapters 1 to 3
```

## Deploy on Render

Push to GitHub, create a Web Service from the repo; `render.yaml` supplies the build and
start commands. Free instances lose uploaded files and the SQLite database on redeploy.
