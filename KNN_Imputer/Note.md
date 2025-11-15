# KNNImputer in Scikit-Learn to Handle Missing Data 

KNNimputer is a scikit-learn class used to fill out or predict the missing values in a dataset. It is a more useful method that works on the basic approach of the KNN algorithm rather than the naive approach of filling all the values with the mean or the median. In this approach, we specify a distance from the missing values which is also known as the K parameter. The missing value will be predicted about the mean of the neighbors.


# How Does KNNImputer Work?
The KNNImputer works by finding the k-nearest neighbors (based on a specified distance metric) for the data points with missing values. It then imputes the missing values using the mean or median (depending on the specified strategy) of the neighboring data points. The key advantage of this approach is that it preserves the relationships between features, which can lead to better model performance.

# Working Principle
For every sample (row) that contains one or more missing values:

1. Identify the k-nearest neighbors
* The algorithm finds the k most similar rows based on a distance metric.
* By default, it uses nan_euclidean distance which ignores missing values during distance calculation.

2. Select the neighbors' values for the missing feature
* Only the values from the chosen neighbors for that specific column are considered.

3. Impute the missing data
* The missing value is filled using either:
     * Uniform mean → simple average of neighbor values
     * Distance-weighted mean → closer neighbors contribute more

# Advantages of Using KNNImputer
1. Preserves Relationships: By using the k-nearest neighbors, this method preserves the relationships between features, which can improve model performance.
2. Customizable: The ability to customize the number of neighbors, distance metric, and weighting scheme makes KNNImputer highly versatile and adaptable to different types of data.
3. Handles Different Data Types: KNNImputer can be used with both continuous and categorical data, making it a flexible tool for a wide range of applications.



# Limitations of KNNImputer
1. Computationally Intensive: Finding the k-nearest neighbors for each missing value can be computationally expensive, especially for large datasets with many missing values.
2. Sensitive to Outliers: The method may be influenced by outliers in the dataset, as outliers can distort the imputation by skewing the mean of the neighbors.
3. Requires Sufficient Data: KNNImputer works best when there is sufficient data to find reliable neighbors. In datasets with a high proportion of missing values, this method may not perform as well.


# Example: How KNNImputer Works

Consider this dataset:

| Student | Math    | Physics | Chemistry |
| ------- | ------- | ------- | --------- |
| A       | 85      | 90      | 88        |
| B       | 78      | 75      | 80        |
| C       | **NaN** | 92      | 86        |
| D       | 90      | 88      | 94        |

Student C has a missing Math score.
We will use k = 2 neighbors.

1. Identify Nearest Neighbors of Student C
Use Physics + Chemistry because Math is missing.
Distances (conceptually):
* C ↔ A → small
* C ↔ D → small
* C ↔ B → larger

So, the 2 nearest neighbors are A and D.

2. Collect Neighbor Math Scores
| Neighbor | Math |
| -------- | ---- |
| A        | 85   |
| D        | 90   |

3. Impute Missing Value
If weights = "uniform" (default):
Math(C) = 85 + 90 / 2 = 87.5

Final Dataset After Imputation
| Student | Math     | Physics | Chemistry |
| ------- | -------- | ------- | --------- |
| A       | 85       | 90      | 88        |
| B       | 78       | 75      | 80        |
| C       | **87.5** | 92      | 86        |
| D       | 90       | 88      | 94        |


It is implemented by the KNNimputer() method which contains the following arguments:

n_neighbors: number of data points to include closer to the missing value. metric: the distance metric to be used for searching. values - {nan_euclidean. callable} by default - nan_euclidean weights: to determine on what basis should the neighboring values be treated values -{uniform , distance, callable} by default- uniform.
