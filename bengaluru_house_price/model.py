import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import pickle

# Load the data
df = pd.read_csv('Bengaluru_House_Data.csv')

def preprocess_data(df):
    # Drop null values and duplicates
    df = df.dropna().drop_duplicates()
    
    # Convert size to number of BHK
    df['bhk'] = df['Size (sqft)'].str.extract('(\d+)').astype(float)
    
    # Convert total_sqft to numeric and handle range values
    def convert_sqft_to_num(x):
        if isinstance(x, str) and ' - ' in x:
            return float(x.split(' - ')[1])
        try:
            return float(x)
        except:
            return None
    
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df.dropna()
    
    # Create price per sqft
    df['price_per_sqft'] = df['Price ($)'] * 100000 / df['total_sqft']
    
    # Remove outliers using IQR method
    def remove_outliers(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
        return df
    
    df = remove_outliers(df, ['total_sqft', 'bhk', 'price_per_sqft', 'Bathrooms'])
    
    # Remove properties where bathrooms > bedrooms + 2
    df = df[df['Bathrooms'] <= df['bhk'] + 2]
    
    # Feature engineering
    df['sqft_per_bedroom'] = df['total_sqft'] / df['bhk']
    df['bath_per_bedroom'] = df['Bathrooms'] / df['bhk']
    
    # Group locations by mean price_per_sqft
    location_stats = df.groupby('Location')['price_per_sqft'].mean()
    location_stats_less_than_10 = location_stats[location_stats.values < 10]
    df['Location'] = df['Location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    
    # Select relevant features
    df = df[['Location', 'total_sqft', 'Bathrooms', 'bhk', 'sqft_per_bedroom', 'bath_per_bedroom', 'Price ($)']]
    
    # Encode location using mean price encoding
    location_mean_price = df.groupby('Location')['Price ($)'].mean()
    df['location_price_mean'] = df['Location'].map(location_mean_price)
    
    # Create dummy variables for top locations
    top_locations = df['Location'].value_counts().nlargest(20).index
    for location in top_locations:
        df[f'is_{location}'] = (df['Location'] == location).astype(int)
    
    # Save location names before dropping the Location column
    with open('locations.pkl', 'wb') as f:
        pickle.dump(list(top_locations), f)
    
    df = df.drop('Location', axis=1)
    return df

# Preprocess the data
df_processed = preprocess_data(df)

# Split features and target
X = df_processed.drop('Price ($)', axis=1)
y = df_processed['Price ($)']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using RobustScaler (better for outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'linear': LinearRegression(),
    'rf': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'dt': DecisionTreeRegressor(max_depth=10, random_state=42),
    'svr': SVR(kernel='rbf', C=100, epsilon=0.1)
}

# Train and evaluate models
model_scores = {}
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    model_scores[name] = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'cv_score': np.mean(cross_val_score(model, X_train_scaled, y_train, cv=5))
    }

# Save the best performing models (Linear Regression and Random Forest)
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(models['linear'], f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(models['rf'], f)

with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(models['dt'], f)

with open('svr_model.pkl', 'wb') as f:
    pickle.dump(models['svr'], f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the feature columns for later use
features = X.columns.tolist()
with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

# Save model scores
with open('model_scores.pkl', 'wb') as f:
    pickle.dump(model_scores, f)

print("\nModel Performance:")
for name, scores in model_scores.items():
    print(f"\n{name.upper()} Model:")
    print(f"R² Score: {scores['r2']:.4f}")
    print(f"MAE: {scores['mae']:.4f}")
    print(f"MSE: {scores['mse']:.4f}")
    print(f"CV Score: {scores['cv_score']:.4f}")