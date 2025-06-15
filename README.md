# Insurance Risk Analytics & Predictive Modeling for ACIS

This project explores and models risk and profitability in insurance data using Exploratory Data Analysis (EDA), statistical reasoning, and version-controlled workflows with DVC (Data Version Control). It follows a reproducible, modular, and scalable data science process to inform insurance pricing and risk decisions.

---

## Task 1: Exploratory Data Analysis (EDA)

### Objective
Gain a foundational understanding of the insurance dataset, assess data quality, and uncover patterns in customer risk and profitability.

### EDA Steps Performed
- **Data Summarization**: Info, descriptive statistics
- **Missing Value Analysis**: Visual and tabular inspection
- **Univariate & Bivariate Analysis**:
  - Histograms and bar plots for key variables
  - Loss Ratio calculated as `TotalClaims / TotalPremium`
  - Grouped Loss Ratio by Province, Gender, and Vehicle Type
- **Temporal Trends**:
  - Claims and premium changes over the 18-month period
- **Vehicle Risk**:
  - Top 10 vehicle makes/models with highest claim amounts
- **Outlier Detection**:
  - Boxplots for `TotalClaims`, `SumInsured`, `CustomValueEstimate`
- **Geographic Analysis**:
  - Loss Ratio visualized by Postal Code (Zip Code)
- **Skewness Review**:
  - Skewness of all financial/numerical variables

 All visualizations are saved in the `plots/` directory, organized by theme.

---

##  Task 2: Data Versioning with DVC

###  Objective
Ensure reproducibility and traceability of the modeling pipeline through version control for datasets using **DVC (Data Version Control)**.

###  Key Steps Completed

- Initialized DVC:
  ```bash
  dvc init
````

* Created and configured a **local DVC remote**:

  ```bash
  mkdir ~/dvc-storage
  dvc remote add -d localstorage ~/dvc-storage
  ```
* Tracked cleaned dataset:

  ```bash
  dvc add data/cleaned_machineLearningRating.csv
  ```
* Updated `.gitignore` to exclude `.csv`, but track `.dvc` files
* Committed changes to Git:

  ```bash
  git add data/cleaned_machineLearningRating.csv.dvc .dvcignore .gitignore .dvc/config
  git commit -m "Track dataset using DVC and configure remote"
  ```
* Pushed data to DVC remote:


  dvc push
  ```

---

###  DVC Functionality & Remote Configuration 


* ✔️ **Remote storage** is fully configured using `dvc remote add -d`
* ✔️ Data tracked using `.dvc` pointer files, not Git
* ✔️ New contributors can pull versioned data using:

  ```bash
  git clone <repo-url>
  cd Insurance-Risk-Analytics---Predictive-Modeling-for-ACIS
  dvc pull
  ```

#### Example to update and push a new dataset version:

```bash
dvc add data/cleaned_machineLearningRating.csv
git add data/cleaned_machineLearningRating.csv.dvc
git commit -m "Add updated cleaned dataset v2"
dvc push
```

 This ensures any future modeling task is fully reproducible and traceable.

---

## Next Step: Task 3 – Predictive Modeling

We are now ready to begin predictive modeling and risk segmentation. See branch `task-3` for model development, evaluation, and deployment prep.

---

## Project Structure Overview

```
Insurance-Risk-Analytics---Predictive-Modeling-for-ACIS/
│
├── .dvc/                          # DVC config & metadata
├── data/                          # Raw and cleaned datasets
│   ├── cleaned_machineLearningRating.csv
│   ├── cleaned_machineLearningRating.csv.dvc
│   └── machineLearningRating_v3.txt
│
├── notebooks/                     # Jupyter notebooks
│   ├── eda.ipynb                  # Task 1 EDA notebook
│   └── import_clean.ipynb        # Data import and cleaning
│
├── plots/                         # All saved plots grouped by type
│   ├── bivariate/
│   ├── geography/
│   ├── multivariate/
│   ├── outliers/
│   ├── time/
│   ├── univariate/
│   └── vehicle_risks/
│
├── reports/                       # Optional: final reports
├── src/                           # Modular Python scripts
│   ├── __init__.py
│   ├── data_load.py
│   ├── eda.py
│   └── utils.py
│
├── tests/                         # Placeholder for unit tests
├── .dvcignore                     # Files ignored by DVC
├── .gitignore                     # Files ignored by Git
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```


## Sample Outputs

![Loss Ratio by Gender](notebooks/plots/bivariate/loss_ratio_by_Gender.png)
![Correlation Heatmap](notebooks/plots/multivariate/correlation_heatmap.png)




## Tools & Technologies

* **Python**: `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Version Control**: Git + GitHub
* **Data Versioning**: DVC (Local Remote)
* **Project Management**: Branch-based workflows (`task-1`, `task-2`, etc.)

---

## Author

Prepared by **Nurye Nigus Mekonen**
---

```

