import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from fcmeans import FCM


df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled shape:", X_scaled.shape)
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3))


ks = range(2, 11)
inertias, silhouettes = [], []

for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow plot", xlabel="K", ylabel="Inertia")

axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette score")

plt.tight_layout()
plt.show()

metrics_df = pd.DataFrame({
    "K": list(ks),
    "inertia": inertias,
    "silhouette": silhouettes
}).round(3)

print(metrics_df)


k = 5 # replace with the value you can justify
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

print(df.groupby("kmeans_cluster")[features].mean().round(1))
print(df["kmeans_cluster"].value_counts().sort_index())

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2],
hue="kmeans_cluster", palette="tab10")
plt.title("K-means customer segments")
plt.tight_layout()
plt.show()