# 📦 Supply Chain Optimization Using Predictive Analytics

A machine learning–based supply chain optimization project that uses historical sales and inventory data to **forecast product demand and generate inventory replenishment recommendations**.

The project combines time-series feature engineering with machine learning models to help answer practical supply-chain questions such as:

* How much demand can be expected for each product?
* How much inventory should be maintained?
* When should a product be reordered?
* How much should be ordered?
* Which products may require replenishment?

---

## 🚀 Project Overview

Effective inventory management requires accurate demand forecasting. Overstocking increases holding costs, while understocking can lead to missed sales and customer dissatisfaction.

This project develops a predictive analytics pipeline that analyzes historical product sales, inventory, pricing, production, and operational information to forecast future demand.

The final pipeline uses a **tuned XGBoost regression model** to generate six-month demand forecasts for individual products. These forecasts are then used to calculate:

* Average monthly demand
* Demand variability
* Safety stock
* Reorder point
* Current inventory
* Stock gap
* Recommended order quantity
* Whether an order is required

The project therefore moves beyond simple demand prediction toward **data-driven inventory planning**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze historical product sales and inventory data.
2. Clean and preprocess the dataset.
3. Engineer time-series and demand-related features.
4. Build machine learning models for demand forecasting.
5. Compare forecasting approaches using evaluation metrics.
6. Tune the XGBoost model using time-series cross-validation.
7. Forecast product demand for the next six months.
8. Calculate safety stock and reorder points.
9. Generate recommended replenishment quantities.
10. Identify products that require inventory replenishment.

---

## 🏗️ Project Workflow

```text
Historical Sales & Inventory Data
              │
              ▼
       Data Preprocessing
              │
              ▼
       Exploratory Analysis
              │
              ▼
       Feature Engineering
              │
              ├── Time Features
              ├── Lag Features
              ├── Rolling Statistics
              └── Product / Inventory Features
              │
              ▼
       Time-Series Train/Test Split
              │
              ▼
      Machine Learning Models
              │
        ┌─────┴─────┐
        ▼           ▼
     XGBoost       LSTM
        │
        ▼
 XGBoost Hyperparameter
       Tuning
        │
        ▼
  Best XGBoost Model
        │
        ▼
 Six-Month Demand Forecast
        │
        ▼
 Inventory Optimization
        │
        ├── Safety Stock
        ├── Reorder Point
        ├── Stock Gap
        └── Recommended Order Quantity
        │
        ▼
   Final Inventory Plan
```

---

## 📊 Dataset

The project uses a shoe sales and inventory dataset containing **500 records**.

The processed dataset contains **23 columns** before additional forecasting features are created.

The historical snapshot dates range from **2015 to 2024**.

### Major Dataset Features

| Feature              | Description                     |
| -------------------- | ------------------------------- |
| `product_name`       | Product identifier              |
| `product_type`       | Product category/type           |
| `total_produced`     | Number of units produced        |
| `total_sold`         | Units sold / demand target      |
| `damaged_units`      | Number of damaged units         |
| `edition_type`       | Product edition category        |
| `unit_price`         | Product selling price           |
| `target_gender`      | Target customer gender category |
| `unsold_inventory`   | Current unsold inventory        |
| `units_received`     | Units received                  |
| `performance_score`  | Product performance indicator   |
| `sneaker_grade`      | Product grade                   |
| `snapshot_date`      | Historical observation date     |
| `price_bucket`       | Price category                  |
| `manufacturing_date` | Manufacturing-related feature   |
| `selling_date`       | Selling-related feature         |

Additional time features such as `year`, `month`, `quarter`, and `week` are generated during preprocessing.

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Standardizing column names
* Converting `snapshot_date` into a datetime format
* Removing records with invalid dates
* Sorting records chronologically
* Removing duplicate records
* Handling missing numerical values using median imputation
* Handling missing categorical values using most-frequent imputation
* Encoding categorical features using `LabelEncoder`
* Creating year, month, and week features
* Removing potential data-leakage features from the forecasting dataset

The final preprocessing validation confirms that the dataset contains no remaining missing values.

---

## 🔍 Exploratory Data Analysis

The notebook performs exploratory analysis to understand demand and product behavior.

Visualizations include:

* Distribution of total units sold
* Top 10 products by total demand
* Feature correlation analysis
* Actual vs predicted demand
* XGBoost feature importance
* Recommended order quantity by product

These analyses help identify demand patterns and the factors influencing product sales.

---

## ⚙️ Feature Engineering

The target variable for demand forecasting is:

```text
total_sold
```

The project creates several groups of predictive features.

### Time Features

```text
year
month
quarter
```

### Lag Features

Historical demand is used to capture previous sales behavior:

```text
lag_1
lag_2
lag_3
lag_6
```

These represent previous demand observations for each product.

### Rolling Features

Rolling statistics are generated using historical demand:

```text
rolling_mean_3
rolling_std_3
rolling_mean_6
rolling_std_6
```

These features capture recent demand trends and demand variability.

### Operational Features

The forecasting model also uses features such as:

```text
total_produced
units_received
unsold_inventory
damaged_units
unit_price
performance_score
```

### Leakage Prevention

The following derived features are removed before forecasting:

```text
estimated_revenue
sell_through_rate
damage_rate
```

This prevents information derived directly from sales from being unintentionally used to predict the sales target.

---

## 🤖 Machine Learning Models

### 1. XGBoost

The primary machine learning model is **XGBoost Regressor**.

The initial model uses parameters including:

```text
n_estimators = 300
learning_rate = 0.05
max_depth = 6
subsample = 0.8
colsample_bytree = 0.8
```

XGBoost is used because it can effectively model nonlinear relationships between product, inventory, pricing, operational, and historical demand features.

---

### 2. LSTM

An LSTM-based neural network is also implemented to explore sequential demand forecasting.

The LSTM pipeline:

* Groups observations by product
* Sorts observations chronologically
* Scales input and target variables using `MinMaxScaler`
* Creates sequences of six historical time steps
* Uses historical demand, inventory, and received-unit information
* Trains an LSTM model using an 80/20 train-test split
* Evaluates predictions using RMSE, MAE, and MAPE

The notebook therefore explores both **tree-based machine learning and deep-learning approaches** for demand prediction.

---

## 🧪 Model Evaluation

The forecasting models are evaluated using:

### RMSE

Root Mean Squared Error measures the magnitude of prediction errors while giving greater weight to larger errors.

```text
RMSE = √(Mean Squared Error)
```

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted demand.

```text
MAE = Mean(|Actual - Predicted|)
```

### MAPE

Mean Absolute Percentage Error measures prediction error as a percentage.

```text
MAPE = Mean(|Actual - Predicted| / Actual) × 100
```

---

## ⏳ Time-Series Data Splitting

Instead of using a random train-test split, the project uses a chronological split.

The latest **six months** are reserved as test data.

```text
Historical Data
─────────────────────────────────────
          Training Data       Test
───────────────────────────────┬─────
                               │
                         Latest 6 Months
```

The notebook also verifies that training data does not contain future observations relative to the test set.

This helps prevent temporal data leakage.

---

## 🔧 XGBoost Hyperparameter Tuning

The XGBoost model is further optimized using:

```text
RandomizedSearchCV
```

with:

```text
TimeSeriesSplit(n_splits=5)
```

The search explores parameters such as:

* Number of estimators
* Maximum tree depth
* Learning rate
* Subsample ratio
* Column sampling ratio
* Minimum child weight

A tuned XGBoost model is then trained using the best parameters discovered during the search.

---

## 📈 Six-Month Demand Forecasting

After selecting the tuned XGBoost model, the project generates future demand forecasts for each product.

The process:

1. Selects the latest available record for every product.
2. Generates the next month's date.
3. Updates the time-based features.
4. Updates lag features using previous predictions.
5. Generates a demand prediction.
6. Uses that prediction to generate the next future observation.
7. Repeats the process for six months.

The output contains:

```text
product_name
forecast_month
predicted_demand
```

This produces a six-month demand forecast for every product.

---

# 📦 Inventory Optimization

The demand forecasts are transformed into an inventory planning strategy.

## Average Monthly Demand

The six-month forecasts are aggregated for each product to calculate:

```text
Average Monthly Demand
```

---

## Demand Variability

The standard deviation of predicted demand is calculated to estimate demand uncertainty.

```text
Demand Standard Deviation
```

---

## 🛡️ Safety Stock

The project uses:

```text
Z = 1.65
```

representing a 95% service-level assumption.

With a one-month lead time:

```text
Safety Stock =
Z × Demand Standard Deviation × √Lead Time
```

---

## 🔄 Reorder Point

The reorder point is calculated as:

```text
Reorder Point =
Average Monthly Demand × Lead Time
+ Safety Stock
```

This determines the inventory level at which replenishment should be triggered.

---

## 📦 Recommended Order Quantity

The project uses a **two-month replenishment coverage** assumption.

If the stock gap indicates that additional inventory is required, the recommended order quantity is calculated based on expected demand coverage and current inventory.

Negative order quantities are prevented by applying a lower bound of zero.

---

## 🚨 Order Recommendation

Each product receives an order recommendation:

```text
YES
```

or

```text
NO
```

An order is required when:

```text
Current Inventory <= Reorder Point
```

---

## 📋 Final Inventory Plan

The final output contains:

| Column                  | Description                                 |
| ----------------------- | ------------------------------------------- |
| `product_name`          | Product identifier                          |
| `current_inventory`     | Current available inventory                 |
| `avg_monthly_demand`    | Forecasted average monthly demand           |
| `safety_stock`          | Additional inventory buffer                 |
| `reorder_point`         | Inventory level that triggers replenishment |
| `recommended_order_qty` | Suggested replenishment quantity            |
| `order_required`        | Whether replenishment is required           |

This transforms the forecasting model into a practical inventory-planning solution.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras
* LSTM

### Data Visualization

* Matplotlib
* Seaborn

### Development Environment

* Google Colab / Jupyter Notebook

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/HaripriyaLogu/SUPPLY-CHAIN-OPTIMIZATION-USING-PREDICTIVE-ANALYTICS.git
cd SUPPLY-CHAIN-OPTIMIZATION-USING-PREDICTIVE-ANALYTICS
```

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost tensorflow
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/HaripriyaLogu/SUPPLY-CHAIN-OPTIMIZATION-USING-PREDICTIVE-ANALYTICS.git
```

### 2. Open the notebook

Open:

```text
SUPPLY_CHAIN_OPTIMIZATION.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 3. Add the dataset

Place the dataset in the expected location and ensure the file is named:

```text
shoe_sales.csv
```

### 4. Run the notebook

Execute the notebook cells sequentially.

The notebook performs:

```text
Data Loading
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
XGBoost
      ↓
LSTM
      ↓
XGBoost Hyperparameter Tuning
      ↓
Six-Month Forecast
      ↓
Inventory Optimization
      ↓
Final Inventory Plan
```

---

## 📁 Repository Structure

```text
SUPPLY-CHAIN-OPTIMIZATION-USING-PREDICTIVE-ANALYTICS/
│
├── SUPPLY_CHAIN_OPTIMIZATION.ipynb
├── shoe_sales.csv
└── README.md
```

> If the dataset is not committed to the repository, download/provide it separately and place it in the expected project location.

---

## 💡 Key Outcomes

The project demonstrates how predictive analytics can be used to move from historical sales data to actionable supply-chain decisions.

The complete pipeline provides:

* Historical demand analysis
* Product-level demand forecasting
* Time-series feature engineering
* Machine learning-based prediction
* Deep-learning experimentation using LSTM
* XGBoost hyperparameter optimization
* Six-month future demand forecasting
* Safety-stock calculation
* Reorder-point calculation
* Inventory-gap analysis
* Recommended order quantities
* Automated order-required classification

---

## 🔮 Future Enhancements

Potential improvements include:

* Deploying the model as a REST API using FastAPI
* Building an interactive Streamlit dashboard
* Adding real-time inventory data
* Integrating external demand drivers such as holidays and promotions
* Adding automated model retraining
* Comparing additional forecasting algorithms
* Implementing product-level model monitoring
* Deploying the solution to Azure or another cloud platform
* Adding automated alerts for products reaching reorder points
* Integrating the prediction pipeline with enterprise inventory systems

---

## 🎓 Project Type

**Academic / Machine Learning / Predictive Analytics / Supply Chain Optimization**

### Domain

```text
Supply Chain
Inventory Management
Demand Forecasting
Predictive Analytics
Machine Learning
```

---
