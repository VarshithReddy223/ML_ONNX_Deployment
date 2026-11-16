# Insurance Charges and Diabetes Risk — PyTorch Models Served with ONNX

Varshith Reddy — machine-learning homework on regression and classification.

Live site: `https://<github-username>.github.io/<repository>/`

## What is here

All files are in one folder, so the repository can be uploaded through GitHub's web page in one step.

* `index.html`, `index.js`: the GitHub Pages site (two tabs, one per model)
* `insurance_charges.onnx`: regression model, input `input1` [batch, 9] raw values, output `output1` [batch, 1] charges in USD
* `diabetes_risk.onnx`: classification model, input `input1` [batch, 8] clinical readings (0 = not measured), output `output1` [batch, 1] probability
* `PartA_Regression_InsuranceCharges.ipynb`, `PartB_Classification_Diabetes.ipynb`: full pipelines, run with outputs saved
* `A_*.png`, `B_*.png`: figures produced by the notebooks
* `regression_results.json`, `classification_results.json`: every cross-validation and test metric
* `insurance.csv`, `diabetes.csv`: the two datasets

## Results

Part A (regression, medical insurance charges, 1,337 people after removing one duplicate). ShallowTanh was chosen by 5-fold cross-validation, with mean R² 0.805. On the 268-person test set it scored R² 0.886, RMSE $4,585 and MAE $2,398.

Part B (classification, Pima Indians diabetes, 768 patients). F1plusF2 was chosen by 5-fold stratified cross-validation, with decision threshold 0.48. On the 154-patient test set it scored F1 0.661, recall 0.741, precision 0.597, accuracy 0.734 and ROC-AUC 0.813.

Both parts compare five PyTorch models: a linear baseline, ShallowTanh, F1plusF2, ResidualNet and DeepBN.

## Publishing on GitHub Pages

1. Make a public repository, open "uploading an existing file", and drag in every file from this folder.
2. Open Settings, then Pages. Choose "Deploy from a branch", then `main` and `/ (root)`, and save.
3. The site appears at `https://<github-username>.github.io/<repository>/` within a couple of minutes.

The `.onnx` files only load over http(s). To try the site locally, run `python -m http.server` in this folder.

## Rerunning

Open a notebook in Colab or Jupyter and run all cells. Each notebook downloads its data, runs cross-validation and writes a new `.onnx` file to `ONNXmodels/`. Copy that file next to `index.html` to update the site.

Data: [insurance.csv](https://github.com/stedy/Machine-Learning-with-R-datasets) and [Pima Indians Diabetes](https://github.com/jbrownlee/Datasets). Deployment pattern: [ConstraintsMLprediction Run10](https://github.com/rcalix1/ConstraintsMLprediction/tree/main/Deployment/RealData/Run10).
