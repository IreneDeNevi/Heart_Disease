# Heart_Disease - Code Documentation
## Heart Disease Dataset Exploratory Data Analysis
### Overview
The Jupiter notebook `heart_disease_eda.ipynb` performs exploratory data analysis (EDA) and preprocessing on Heart Disease Dataset (link: https://www.kaggle.com/datasets/yasserh/heart-disease-dataset). The goal is to clean, explore, and prepare the data for machine learning tasks.

### Workflow
The first step begins with data acquisition and initial inspection. The dataset is loaded from a CSV file located at `data/heart.csv`.
``` python
file_path = "data/heart.csv"
data = pd.read_csv(file_path)
```
After the first records in the database have been displayed, a check is made for missing values. In this case, the data is already clean and no further action is taken.
The data set is then divided into two primary components: features (X), e.g. the predictor variables, and target (y), e.g. whether a patient is healthy (0) or not (1).

``` python
target_column = "target"
X = data.drop(columns=[target_column])
y = data[target_column]
```

A summary statistics are generated, offering a statistical snaposhot of the dataset. Then  a systematic search for duplicate entrie is performed, in order to avoid bias in a medical prediction. After the removal of redundant entries, the cleaned dataset is saved as `data/heart_cleaned.csv`

``` python
# Check for duplicate rows
duplicates = data.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")

# Drop duplicate rows
data_cleaned = data.drop_duplicates()

# Verify duplicates are removed
duplicates_after = data_cleaned.duplicated().sum()
print(f"Number of duplicate rows after removal: {duplicates_after}")

# Save the cleaned dataset 
data_cleaned.to_csv("data/heart_cleaned.csv", index=False)
print("Cleaned dataset saved as 'data/heart_cleaned.csv'")
```

Visualisation is then performed in the form of a box plot for each numerical column, creating a visual representation of the data distribution. These box plots are ideal for identifying outliers and provide an intuitive understanding of the characteristics of each feature.

Then thought the IQR method, outliers are detected across all features, and printed out. This preprocessing part involves a capping strategy tailored to each feature's medical context.

- Cholesterol levels are capped at a maximum of 400
- Resting blood pressure is limited to 180
- Maximum heart rate has a minimum threshold of 80
- ST depression is capped at 4.0
- The number of major vessels is restricted to a maximum of 3

This approach preserves data while mitigating the impact of "extreme" . Note that the `thal` column is managed differently: zeros values are converted to missing.

This preprocessing part is saved as `data/heart_capped.csv`

```python
data.to_csv("data/heart_capped.csv", index=False)
print("Cleaned dataset saved as 'data/heart_capped.csv'")
```

## Application of OOP principles on ML pipeline
