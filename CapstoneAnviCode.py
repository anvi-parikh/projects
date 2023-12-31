#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:41:48 2023

@author: anavi
"""

# ---- Imports ----
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error#, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(11476370)

# ---- Reading CSV File ----
data = pd.read_csv('spotify52kData.csv')

# ---- Data Cleaning and Pre-Processing ----
# checking for missing values
missing_data = data.isnull().sum()
print("missing values: ", missing_data[missing_data > 0])

# dropping duplicates
columns_exclude = ['songNumber', 'album_name']
duplicates = data[data.duplicated(data.columns.difference(columns_exclude), keep=False)]
numofdups = duplicates.shape[0]
#printing and dropping the number of duplicates
print("Number of duplicates: ", numofdups)
data = data.drop_duplicates(data.columns.difference(columns_exclude), keep='first')
num_non_duplicates = data.shape[0]
#after dropping, printing the number of non-duplicates
print("Number of Non-Duplicates:", num_non_duplicates)
print()





# ---- Question 1 ----
# Select the 10 song features
selected_features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
fig.suptitle('Distribution of Song Features')

# Plot histograms for each feature 
for i, feature in enumerate(selected_features):
    sns.histplot(data[feature], kde=True, ax=axes[i//5, i%5], bins=30, color='green')
    axes[i//5, i%5].set_title(feature)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



# ---- Question 2 ----
data['duration_minutes'] = data['duration'] / (60 * 1000)
sns.scatterplot(x='duration_minutes', y='popularity', data=data, s=10, color ='green')  
plt.title('Scatter Plot: Song Duration vs Popularity')
plt.xlabel('Song Duration (minutes)')
plt.ylabel('Popularity')
plt.show()

pearson_corr, _ = stats.pearsonr(data['duration'].dropna(), data['popularity'].dropna())
print(f"Pearson Correlation Coefficient (Duration&Popularity): {pearson_corr}")
spearman_corr, _ = stats.spearmanr(data['duration'].dropna(), data['popularity'].dropna())
print(f"Spearman Correlation Coefficient (Duration&Popularity) : {spearman_corr}")
print()



# ---- Question 3 ----
explicit_songs = data[data['explicit'] == True]['popularity'].dropna()
non_explicit_songs = data[data['explicit'] == False]['popularity'].dropna()

# Checking number of explicit and non-explicit songs
explicit_counts = data['explicit'].value_counts()
print("Number of Explicit Songs:", explicit_counts[True] if True in explicit_counts else 0)
print("Number of Non-Explicit Songs:", explicit_counts[False] if False in explicit_counts else 0)

# Mean and Median
print("Mean (Explicit): ", explicit_songs.mean())
print("Mean (Non-Explicit):", non_explicit_songs.mean())
print("Median (Explicit): ", explicit_songs.median())
print("Median (Non-Explicit):", non_explicit_songs.median())

# Perform Mann-Whitney U test
u, p_value = stats.mannwhitneyu(explicit_songs, non_explicit_songs, alternative='greater')

# Output the results
print("P-value: ", p_value)
if p_value < 0.05:
    print("Popularity of explicit and non-explicit songs is significantly different.")

    # Visualize the distributions using histograms
    plt.hist(explicit_songs, bins=30, alpha=0.5, label='Explicit', color='blue')
    plt.hist(non_explicit_songs, bins=30, alpha=0.5, label='Non-Explicit', color='orange')
    plt.axvline(explicit_songs.median(), color='blue', linestyle='dashed', linewidth=2, label='Median (Explicit)')
    plt.axvline(non_explicit_songs.median(), color='orange', linestyle='dashed', linewidth=2, label='Median (Non-Explicit)')
    
    plt.title('Popularity Distribution by Explicitness')
    plt.xlabel('Popularity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

else:
    print("Popularity of explicit and non-explicit songs is not significantly different.")

print()

#drop the assumption that there is no relationship between explicitness of a song and song popularity
    

    
# ---- Question 4 ----
major_key_songs = data[data['mode'] == 1]['popularity'].dropna()
minor_key_songs = data[data['mode'] == 0]['popularity'].dropna()

mode_counts = data['mode'].value_counts()
print("Number of Major Key Songs:", mode_counts[1])
print("Number of Minor Key Songs:", mode_counts[0])

major_key_data = data[data['mode'] == 1]
minor_key_data = data[data['mode'] == 0]

# Calculate mean and median
mean_major = major_key_data['popularity'].mean()
median_major = major_key_data['popularity'].median()

mean_minor = minor_key_data['popularity'].mean()
median_minor = minor_key_data['popularity'].median()

print("Mean (Major Key):", mean_major)
print("Median (Major Key):", median_major)
print("\nMean (Minor Key):", mean_minor)
print("Median (Minor Key):", median_minor)

# Rest of your code...

# Rest of your code...

# Perform Mann-Whitney U test
u, p_value = stats.mannwhitneyu(major_key_songs, minor_key_songs, alternative='greater')
print("P-value: ", p_value)

# Output the results
if p_value < 0.05:
    print("Popularity of major key and minor key songs is significantly different.")
else:
    print("Popularity of major key and minor key songs is not significantly different.")

# Rest of your code...

# Visualize the distributions with histograms (moved outside the if statement)
plt.figure(figsize=(10, 6))
plt.hist(major_key_songs, bins=30, alpha=0.5, label='Major', color='blue')
plt.hist(minor_key_songs, bins=30, alpha=0.5, label='Minor', color='orange')
plt.axvline(major_key_songs.median(), color='blue', linestyle='dashed', linewidth=2, label='Median (Major)')
plt.axvline(minor_key_songs.median(), color='orange', linestyle='dashed', linewidth=2, label='Median (Minor)')

plt.title('Popularity Distribution by Key')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print()

    
    
# ---- Question 5 ----  
sns.scatterplot(x='energy', y='loudness', data=data, s=10) 
plt.title('Scatter Plot: Energy vs Loudness')
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.show()

pearson_corr, _ = stats.pearsonr(data['energy'].dropna(), data['loudness'].dropna())
print(f"Pearson Correlation Coefficient (Energy and Loudness): {pearson_corr}")
spearman_corr, _ = stats.spearmanr(data['energy'].dropna(), data['loudness'].dropna())
print(f"Spearman Correlation Coefficient (Energy and Loudness) : {spearman_corr}")
print()



# ---- Question 6 ----  
selected_features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

#zscore data
selected_data = data[selected_features]
zscoredData = stats.zscore(selected_data)

X = zscoredData.values
y = data['popularity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11476370)

# Iterate through each feature and build a linear regression model
performance = {}

for feature in selected_features:
    X_train_feature = X_train[:, selected_features.index(feature)].reshape(-1, 1)
    X_test_feature = X_test[:, selected_features.index(feature)].reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train_feature, y_train)
    y_pred = model.predict(X_test_feature)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)
    performance[feature] = {'RMSE': rmse, 'R-squared': r_squared}
    
for feature, scores in performance.items():
    print(f"Feature: {feature}")
    print(f"RMSE: {scores['RMSE']}")
    print(f"R-squared: {scores['R-squared']}")
    print()

# Identify the feature with the lowest RMSE and highest r2
best_rmse_feature = min(performance, key=lambda k: performance[k]['RMSE'])
best_r2_feature = max(performance, key=lambda k: performance[k]['R-squared'])

print(f"The feature that best predicts popularity (based on RMSE) is: {best_rmse_feature}")
print(f"The feature that best predicts popularity (based on R^2) is: {best_r2_feature}")
print(f"RMSE for {best_rmse_feature}: {performance[best_rmse_feature]}, R^2 for {best_r2_feature}: {performance[best_r2_feature]}")

#Plot Instrumentalness vs Popularity
X_plot_feature = X_test[:, selected_features.index(best_rmse_feature)].reshape(-1, 1)
plt.scatter(X_plot_feature, y_test, color='green', s=10, label='True Labels')
plt.plot(X_plot_feature, y_pred, color='black', linewidth=1, label='Linear Regression Line')
plt.xlabel(best_rmse_feature)
plt.ylabel('Popularity')
plt.title('Scatter Plot with Linear Regression: Instrumentalness vs Popularity')
plt.legend()
plt.show()

print()



# ---- Question 7 ---- 
X = zscoredData.values
y = data['popularity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11476370)

model_all_features = LinearRegression()
model_all_features.fit(X_train, y_train)

y_pred_all_features = model_all_features.predict(X_test)

rmse_all_features = np.sqrt(mean_squared_error(y_test, y_pred_all_features))

r_squared_all_features = r2_score(y_test, y_pred_all_features)

print(f"RMSE for the model using all features: {rmse_all_features}")
print(f"R-squared for the model using all features: {r_squared_all_features}")

# Access the RMSE value for the best feature
best_rmse_rmse = performance[best_rmse_feature]['RMSE']
best_r2_r2 = performance[best_r2_feature]['R-squared']

# Compare with model from question 6
improvement_in_rmse = best_rmse_rmse - rmse_all_features
print(f"Improvement in RMSE: {improvement_in_rmse}")
improvement_in_r2 = r_squared_all_features - best_r2_r2
print(f"Improvement in R2: {improvement_in_r2}")

# Scatter plot
plt.scatter(y_test, y_pred_all_features, alpha=0.5, s=20)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)

# Labels and title
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Multiple Linear Regression Model: Actual vs Predicted Popularity')

plt.show()

print()



# ---- Question 8 ---- 
#Correlation matrix
corrMatrix = np.corrcoef(selected_data.values, rowvar=False)
plt.imshow(corrMatrix, cmap='viridis', aspect='auto')
plt.xticks(range(len(selected_features)), selected_features, rotation='vertical')
plt.yticks(range(len(selected_features)), selected_features)
plt.xlabel('Song Feature')
plt.ylabel('Song Feature')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Matrix for Selected Song Features')
plt.show()

#Do PCA

#Initialize PCA object and fit to our data
pca = PCA().fit(zscoredData)

#Eigenvalues
eigVals = pca.explained_variance_

#Loadings
loadings = pca.components_

#Rotated Data
rotatedData = pca.fit_transform(zscoredData)

#Eigenvalues in terms of variance explained
varExplained = eigVals/sum(eigVals)*100


#Display this for each factor:
for ii in range(len(varExplained)):     
    print(varExplained[ii].round(3))
    
#Create scree plot
numFeatures = len(selected_features)
x = np.linspace(1, numFeatures, numFeatures)
plt.bar(x, eigVals, color='gray')
plt.plot([0, numFeatures], [1, 1], color='green')  
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot for PCA')
plt.show()

#Criteria - Kaiser
kaiserThreshold = 1
print('Number of principal components selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))

#Interpreting the factors - plotting loadings
for whichPrincipalComponent in range(3):
    plt.bar(x, loadings[whichPrincipalComponent, :] * -1, color='green')
    plt.xlabel('Song Feature')
    plt.ylabel(f'Loading PC:{whichPrincipalComponent + 1}')
    plt.title(f'Loading Plot for Principal Component {whichPrincipalComponent + 1}')
    plt.xticks(range(1, len(selected_features) + 1), selected_features, rotation='vertical')
    plt.show()
    #NAME these according to a summarization
    
#K-means for clustering
rotatedData_3D = rotatedData[:, :3]
k_values = range(2, 7)  


silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=11476370)  # Explicitly set n_init
    labels = kmeans.fit_predict(rotatedData_3D)
    silhouette_avg = silhouette_score(rotatedData_3D, labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for K-means Clustering')
plt.show()

#kMeans clustering
numClusters = 2 



# ---- Question 9 ---- 
mode_counts = data['mode'].value_counts()

print("Number of Major Songs:", mode_counts[1])
print("Number of Minor Songs:", mode_counts[0])


y = data['mode'].values
zscoredValence = stats.zscore(data[['valence']])
X = zscoredValence.values   #note: z-score doesn't affect this model at all

#Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11476370)
logistic_model = LogisticRegression(random_state=11476370)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("Confusion Matrix:") #CMIN, FMAJ, FMIN, CMAJ
print(conf_matrix)

plt.scatter(X_test, y_test, color='black', s=10, label='True Labels')
plt.scatter(X_test, y_pred, color='red', s=10, label='Predicted Labels')
plt.plot(X_test, logistic_model.predict_proba(X_test)[:, 1], color='green', linewidth=3, label='Logistic Regression Curve')
plt.xlabel('Valence')
plt.ylabel('Mode (Major: 1, Minor: 0)')
plt.title('Logistic Regression for Predicting Major or Minor Key')
plt.legend()
plt.show()

y_pred_proba = logistic_model.predict_proba(X_test)[:, 1] #better for binary, consider removing accuracy
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC-ROC Score: {auc_roc}')

#Testing whether there is a better predictor
selected_features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Iterate through each feature and build a logistic regression model
performance_scores = {}

for feature in selected_features:
    # Extract the feature column
    X_feature = stats.zscore(data[[feature]]).values  # z-score normalization for the feature
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=11476370)
    
    # Initialize and fit the logistic regression model
    logistic_model = LogisticRegression(random_state=11476370)
    logistic_model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred_proba = logistic_model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC-ROC score
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the scores
    performance_scores[feature] = {'AUC-ROC': auc_roc, 'Accuracy': accuracy}

# Print scores for each feature
for feature, scores in performance_scores.items():
    print(f"Feature: {feature}")
    print(f"AUC-ROC Score: {scores['AUC-ROC']}")
    print(f"Accuracy: {scores['Accuracy'] * 100:.2f}%")
    print()




# ---- Question 10 ---- 
#Find genres
unique_genres = data['track_genre'].unique()
print("Unique Genres:", unique_genres)

#Label genres
from sklearn.preprocessing import LabelEncoder
genre_encoder = LabelEncoder()
data['genre_encoded'] = genre_encoder.fit_transform(data['track_genre'])
encoded_genres = data['genre_encoded'].unique()
print("Encoded Genres:", encoded_genres)

X = zscoredData.values
y = data['genre_encoded'].values

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=11476370)

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)  

# Display the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average Accuracy:", np.mean(cv_scores))

fig, ax = plt.subplots()
bars = ax.bar(range(1, len(cv_scores) + 1), cv_scores, color='lightgreen')
plt.axhline(np.mean(cv_scores), color='darkblue', linestyle='dashed', linewidth=2, label='Average Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores for Random Forest Classifier')
plt.legend()

for bar, score in zip(bars, cv_scores):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(score, 3), ha='center', va='bottom')

plt.show()


#Visualizing the importance of each feature in predicting the genre
rf_classifier.fit(X, y)
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices], align="center", color='lightgreen')
plt.xticks(range(X.shape[1]), selected_features, rotation='vertical')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Classifier - Feature Importances')
plt.show()

# Create a new column for song title length
data['title_length'] = data['track_name'].apply(len)

print()



































