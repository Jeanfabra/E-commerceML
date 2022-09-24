# E-commerceML

Machine Learning model development for a transport company, the objective is to predict whether an order will arrive on time or not.

## Problem Description
We are part of a logistics company that works for an important E-Commerce portal, and our Team Leader gives us the task of implementing a model that allows us to predict
whether a shipment will arrive on time or not, according to the 
information contained in the dataset.

## About the dataset

The main dataset is a version of Kaggle [E-Commerce Shipping Data](https://www.kaggle.com/datasets/prachi13/customer-analytics). This dataset contains the following information:

* ID: ID Number of Customers.
* Warehouse block: The Company have big Warehouse which is divided in to block such as A,B,C,D,E.
* Mode of shipment:The Company Ships the products in multiple way such as Ship, Flight and Road.
* Customer care calls: The number of calls made from enquiry for enquiry of the shipment.
* Customer rating: The company has rated from every customer. 1 is the lowest (Worst), 5 is the highest (Best).
* Cost of the product: Cost of the Product in US Dollars.
* Prior purchases: The Number of Prior Purchase.
* Product importance: The company has categorized the product in the various parameter such as low, medium, high.
* Gender: Male and Female.
* Discount offered: Discount offered on that specific product.
* Weight in gms: It is the weight in grams.
* Reached on time: It is the target variable, where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time.

## Metrics to be evaluated

### [Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

Recall of the Confusion Matrix will be used as a method for evaluating model performance. Our main interest is to find those shipments that will not arrive on time. **The recall will answer the question: 
What percentage of shipments that do not arrive on time are we able to identify?**

$$ Recall=\frac{TP}{TP+FN}$$

where $TP$ the true positives and $FN$ the false negatives.

## [Accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

Accuracy is a metric also based on the confusion matrix. In this case we will take this metric to evaluate the classification performance for both class 1 and class
0 in our target variable. Note that in this exercise the primary class will be class 1, i.e. those shipments that do not arrive on time.

$$ Accuracy=\frac{TP + TN}{TP+ TN + FN + FP}$$

where $TP$ the true positives, $TN$ true negatives, $FN$ false negatives, $FP$ false positives.

## General Steps

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. First Modeling Batch (Working with raw data)
4. Second Modeling Batch (Aplying One hot Encoding)
5. Third Modeling Batch (Evaluating StandardScaler)
6. Fourth Modeling Batch (Evaluating Dimension Reduction using PCA)
7. Final model selection and searching for best hyperparameters with GridSearchCV
8. Conclusions

For more deep information please don't hesitate to open the main.ipynb.

## Documentation to highlight

* [Sckit-Learn Documentation](https://scikit-learn.org/stable/index.html#)
* [StandardScaler vs MinMaxScaler](https://stackoverflow.com/questions/61255108/python-numpy-ravel-function-not-flattening-array)
* [Video: Scaling, Normalization and Standardization (Spanish)](https://www.youtube.com/watch?v=-VuR14Qyl7E&lc=UgyGv3R3K4siP3YPgLh4AaABAg.9gDcR4wNAti9gDnlbOEOx4)
* [Video: How to implement One Hot Encoding](https://www.youtube.com/watch?v=InZ0n2knz1E&lc=UgymfF3vTXC8PFTFOZR4AaABAg.9gAv8UJZvWe9gBrEABT8oV)

## Contact

Greetings,
Jean Paul Fabra Ruiz: jeanfabra11@gmail.com 

LinkedIn: https://www.linkedin.com/in/jeanfabra/



