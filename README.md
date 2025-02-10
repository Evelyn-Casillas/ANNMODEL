ANN Model for Customer Churn Prediction
This repository contains a Jupyter Notebook that demonstrates a data preprocessing pipeline for predicting customer churn in a telecommunication company. It includes steps like data cleaning, encoding, feature scaling, and exploring correlations among various customer attributes. The target variable is whether a customer has exited the service (Exited), and the goal is to train a model to predict this based on features such as age, balance, and membership status.

Table of Contents
Data Loading and Exploration
Data Preprocessing
Feature Encoding and Scaling
Correlation Analysis
Modeling
Conclusion
1. Data Loading and Exploration
The dataset is loaded from a CSV file (Churn_Modelling.csv) containing information about 10,000 customers.
It includes 14 columns, such as customer demographics (Age, Gender), financial information (Balance, EstimatedSalary), and churn status (Exited).
Data Overview
python
Copy
Edit
churn = pd.read_csv('Churn_Modelling.csv')
churn.shape
(10000, 14)
churn.head(5)
The dataset consists of columns like CreditScore, Geography, Gender, Age, Tenure, etc.
Summary Statistics
python
Copy
Edit
churn.describe()
2. Data Preprocessing
Missing values were not found in the dataset, so no imputation is needed.
The CustomerId and Surname columns were removed since they are not relevant for modeling.
python
Copy
Edit
churn_new = churn.drop(['CustomerId', 'Surname', 'RowNumber'], axis=1)
3. Feature Encoding and Scaling
The Gender column is encoded numerically using LabelEncoder().
The Geography column is one-hot encoded using pd.get_dummies() to create separate binary columns for France, Germany, and Spain.
python
Copy
Edit
churn_new["Gender"] = LabelEncoder().fit_transform(churn["Gender"])
churn_encoded = pd.get_dummies(churn_new, columns=['Geography']).astype(int)
Feature scaling is performed using MinMaxScaler to normalize the feature values between 0 and 1.
python
Copy
Edit
scaler = MinMaxScaler()
df_rescaled = pd.DataFrame(scaler.fit_transform(churn_encoded), columns=churn_encoded.columns)
4. Correlation Analysis
A heatmap is generated to visualize correlations between features. Strong correlations may indicate which variables are most influential in predicting customer churn.

python
Copy
Edit
corr = df_rescaled.corr()
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, cmap='BrBG')
The top positively and negatively correlated features are selected for further modeling.
python
Copy
Edit
top_features1 = ['Age', 'Balance', 'Geography_Germany', 'IsActiveMember', 'Gender']
5. Modeling
In this section, a model (e.g., Artificial Neural Network or any other model) would be trained using the selected features and the target variable Exited. This is currently a placeholder section where you can implement your model of choice.

python
Copy
Edit
X = df_rescaled.drop(columns=['Exited'])
y = df_rescaled['Exited'].values
6. Conclusion
This notebook presents a comprehensive pipeline for preprocessing and preparing a customer churn dataset for machine learning. The steps include data cleaning, feature encoding, scaling, and exploratory data analysis. A model can now be trained using the processed features to predict whether a customer will exit the service.
