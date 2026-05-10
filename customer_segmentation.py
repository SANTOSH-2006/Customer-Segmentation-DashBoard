import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.DataFrame({
    'Customer_ID': [1,2,3,4,5,6,7,8,9,10],
    'Age': [21,22,24,42,52,33,38,25,26,41],
    'Annual_Income': [18000,20000,35000,90000,95000,85000,72000,66000,45000,120000],
    'Spending_Score': [38,25,33,74,15,8,45,78,68,54]
})

print(data)

X = data[['Annual_Income', 'Spending_Score']]

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

print("\nClustered Data:\n", data)

plt.scatter(data['Annual_Income'], data['Spending_Score'], c=data['Cluster'])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()

data.to_csv("clustered_customers.csv", index=False)
