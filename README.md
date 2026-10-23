This app runs scikit-learn's AgglomerativeClustering and renders a SciPy dendrogram.
By default it computes the full tree (distance_threshold=0, n_clusters=None) to mirror the official example,
then optionally cuts the tree by distance or to a fixed number of clusters. Outputs include cluster labels,
a SciPy-compatible linkage matrix, a base64 PNG dendrogram, and a compact HTML report.