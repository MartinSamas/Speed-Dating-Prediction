# Speed-Dating-Prediction
In this project we've applied machine learning techniques to predict romantic compatibility. The business objective is to transform standard dating platforms into complex predictive tools that consider variables like shared interests and fields of study, ultimately increasing user retention by suggesting higher-probability matches.

The analysis is structured into two parts: supervised classification and unsupervised clustering.

## Dataset
* **Source:** OpenML Dataset #40536 (Speed dating dataset by Ray Fisman and Sheena Iyengar).
* **Size:** 8,378 rows representing four-minute speed dates from 21 events at Columbia University between 2002 and 2004.
* **Features:** The data covers demographics, partner ratings (1-10 scale across various traits), dating preferences, and lifestyle habits.
* **Target Variable:** The target attribute is `match`, which is a binary variable (1 = both participants said yes, 0 = at least one participant said no).

## Methodology

### Part 1: Classification
* **Instance of Interest:** The analysis focused on the participant with `iid 11` to illustrate model predictions at the row level.
* **Attribute of Interest:** The primary attribute investigated was the attractiveness rating of the partner (`attractive_partner`).
* **Cost Matrix Evaluation:** A custom cost matrix was implemented to evaluate the classifiers based on subjective business reasoning. False Negatives (missing a genuine connection) were assigned a higher penalty cost of 2, while False Positives (wasting a date on a match that doesn't materialize) were assigned a cost of 1.

### Part 2: Clustering
* **Subset Focus:** The dataset was filtered to only include participants whose field of study is "Business" in order to detect specific matching features within this statistically significant group.
* **Instance & Attribute of Interest:** Focused on the participant with `iid 155` and the `field` attribute.
* **Preprocessing:** `SimpleImputer` (using the mean strategy) was applied to handle missing values without losing data, and features were subsequently scaled using `MinMaxScaler`.
* **Modeling:** K-Means clustering was applied to group the business students based on their traits. The optimal number of clusters was determined to be 3 ($k=3$) by utilizing the elbow method.
* **Visualization:** Principal Component Analysis (PCA) was used to project the high-dimensional data into 2D and 3D spaces for cluster visualization.

## Technologies Used
* **Data Manipulation:** Python, Pandas, NumPy
* **Machine Learning:** Scikit-learn (Classification, K-Means Clustering, PCA, SimpleImputer, MinMaxScaler)
* **Visualization:** Matplotlib, Seaborn
