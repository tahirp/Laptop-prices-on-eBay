"""
## Project Overview: Used Laptop Price Prediction

### Introduction
The used laptop market currently lacks a standardized pricing guide, leading to inefficiencies where sellers either undervalue their products or overprice them, hindering sales and contributing to e-waste. This project aims to address this gap by developing a predictive model that accurately estimates the market price of used laptops based on their specifications.

### Project Definition & Goals

#### 1. Main Goal
To build a robust predictive model that accurately estimates the market price of used laptops based on their specifications, assisting sellers in effective pricing and buyers in avoiding overpayment.

#### 2. Secondary Goals
- **Identify Influential Features**: Determine key laptop specifications (RAM, processor, storage, screen size) impacting market price.
- **Competitive Pricing Tool**: Develop a tool for sellers to price used laptops competitively.
- **Market Insight**: Provide insights into market trends and value depreciation.
- **Reduce E-waste**: Contribute to reducing e-waste by facilitating efficient sales.

#### 3. Key Performance Indicators (KPIs)
- **R-squared (R2)**: Target 0.80 or higher.
- **Mean Absolute Error (MAE)**: Target less than $50.
- **Root Mean Squared Error (RMSE)**: Target less than $75.

#### 4. Project Scope
- **Laptop Types**: Primarily consumer-grade laptops, excluding specialized hardware.
- **Data Sources**: Public datasets, online marketplaces, web scraping (recent sales data, last 2-3 years).
- **Geographical Focus**: Initially US market, with potential future expansion.
- **Exclusions**: Refurbished/heavily damaged and brand-new laptops.

### Data Source
The raw data was obtained from: [https://www.kaggle.com/datasets/elvinrustam/ebay-laptops-and-netbooks-sales?select=EbayPcLaptopsAndNetbooksUnclean.csv](https://www.kaggle.com/datasets/elvinrustam/ebay-laptops-and-netbooks-sales?select=EbayPcLaptopsAndNetbooksUnclean.csv)

### Data Cleaning & Preprocessing Steps

1.  **Initial Data Inspection**: Checked DataFrame structure, data types, and basic statistics.
2.  **Missing Value Analysis**: Identified columns with high missing percentages.
3.  **Column Dropping**: Removed irrelevant columns or those with extremely high missing percentages (`Manufacturer Color`, `Country Region Of Manufacture`, `Rating`, `Ratings Count`, `Release Year`, `Seller Note`, `Features`).
4.  **Duplicate Removal**: Removed duplicate rows to ensure data integrity.
5.  **Price Cleaning**: Cleaned the `Price` column by removing currency symbols, handling price ranges (averaging), and converting to a float.
6.  **Numerical Feature Cleaning & Imputation**: Applied custom functions to 'Hard Drive Capacity', 'Screen Size', 'Processor Speed', 'Ram Size', and 'SSD Capacity' to extract numerical values, convert units to GB, and impute missing values (median for 'Screen Size', 'Processor Speed', 'Ram Size'; 0 for 'SSD Capacity', 'Hard Drive Capacity').
7.  **Categorical Feature Cleaning & Imputation**: Standardized values and imputed 'Unknown' for 'Brand', 'Condition', 'Processor', 'OS', 'Storage Type', 'GPU', 'Type', 'Color', and 'Maximum Resolution'.
8.  **Outlier Handling**: Applied IQR-based capping to numerical features ('Price', 'Screen Size', 'Processor Speed', 'Ram Size', 'SSD Capacity', 'Hard Drive Capacity') to mitigate the impact of extreme values.

### Feature Engineering

New features were engineered to capture more nuanced information and improve model performance:

1.  **`Total_Storage`**: Sum of `SSD Capacity` and `Hard Drive Capacity`.
2.  **`Processor_Family`**: Categorized processors into broader families (e.g., 'Intel Core i', 'AMD Ryzen').
3.  **`Processor_Tier`**: Assigned a numerical tier based on the performance hierarchy of processors.
4.  **`Screen_Size_Category`**: Binned `Screen Size` into 'Small', 'Medium', 'Large' categories.
5.  **`Ram_Category`**: Binned `Ram Size` into 'Entry-level', 'Mid-range', 'High-end' categories.
6.  **`Processor_Ram_Interaction`**: Interaction term calculated by multiplying a mapped `Processor_Family` with `Ram Size`.
7.  **`Resolution_Width`, `Resolution_Height`, `Pixel_Count`, `Aspect_Ratio`**: Extracted from `Maximum Resolution`.
8.  **`Storage_Configuration`**: Categorized laptops based on storage setup (e.g., 'SSD Only', 'HDD Only', 'Hybrid', 'eMMC Only').
9.  **`Screen_Size_Category_Mapped`**: Numerical mapping of `Screen_Size_Category`.
10. **`Ram_Screen_Interaction`**: Interaction term calculated by multiplying `Ram Size` with `Screen_Size_Category_Mapped`.

### Model Training and Evaluation

1.  **Feature Encoding**: All categorical features, including the 'Unknown' categories, were one-hot encoded using `pd.get_dummies` with `drop_first=True` to prevent multicollinearity. Original categorical columns were dropped.
2.  **Feature Scaling**: All numerical features (excluding 'Price') were scaled using `StandardScaler`.
3.  **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets.
4.  **Model Selection & Training**: A Linear Regression model was chosen and trained on the preprocessed training data.
5.  **Model Evaluation**: The model's performance was evaluated on the test set using the defined KPIs.

#### Model Performance Results:

*   **R-squared (R2)**: 0.48 (Target: 0.80 or higher) - **Below Target**
*   **Mean Absolute Error (MAE)**: $112.59 (Target: Less than $50) - **Above Target**
*   **Root Mean Squared Error (RMSE)**: $168.21 (Target: Less than $75) - **Above Target**

**Conclusion**: The Linear Regression model did not meet the predefined KPIs. Further refinement, more complex models, or additional feature engineering are necessary to improve predictive accuracy.

### Feature Importance Analysis

An analysis of the Linear Regression model's coefficients revealed the most influential features:

1.  **`Aspect_Ratio_Unknown` (158.63)**: Highest importance, suggesting missing aspect ratio data strongly impacts price, potentially indicating older/less desirable models.
2.  **`Resolution_Height` (156.30)** & **`Pixel_Count` (140.66)**: Display resolution features are critical drivers, with higher values correlating with higher prices.
3.  **`GPU_Unknown` (154.60)**: Indicates that the absence of specific GPU information significantly influences price, likely associated with lower-end or older models.
4.  **`Storage Type_SSD` (103.14)**: The presence of an SSD is a strong positive predictor, reflecting better performance and modern systems.
5.  **`Storage Type_Unknown` (97.77)** & **`Aspect_Ratio_16:9` (93.66)**: Other significant factors, with 'Unknown' storage type possibly correlating with lower prices and '16:9' aspect ratio representing a common standard.
6.  **`Aspect_Ratio_8:5` (63.38)** and **`OS_Unknown` (59.24)** and **`OS_Windows 10` (55.45)**: These also contribute significantly to price prediction.

**Key Insight**: Display characteristics (resolution, aspect ratio) and storage types are crucial. The high importance of 'Unknown' categories highlights that data completeness and quality are strong predictors, often correlating with lower-priced laptops due to lack of detail.
"""
