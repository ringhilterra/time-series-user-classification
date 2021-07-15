# time-series-user-classification


Example ML Project for Time Series Classification - User Identification

*"The dataset collects data from an Android smartphone positioned in the chest pocket. Accelerometer Data are collected from 22 participants walking in the wild over a predefined path. The dataset is intended for Activity Recognition research purposes. It provides challenges for identification and authentication of people using motion patterns."*

**Dataset:** https://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity#

<img src="resources/data_view.png" width="600" height="300">

22 different users (classes), can see some class imbalance. Around ~150 total raw sensor samples.

## Notebooks

`full_classify_workflow_rforest.ipynb`
- proper train / test split for time series data
- Feature Engineering with time series windowing
- Exploratory Data Analysis
- Multiclass Classification with Scikit-Learn
- Evaluation Metrics and Plots
- GridSearchCV Example

`full_classify_workflow_rforest_1vsrest.ipynb`
- copy of full_classify_workflow_rforest.ipynb but trying 1 vs all ML classifier

`sktime_rocket_classify.ipynb`
- Time Series classification using sktime library, ROCKET (random convolutional kernels) time series classifer model which has shown state of the art performance in a range of time series classification benchmarks


## Results
Random forest classification model was built from using different size windowing / lag features on dataset.


| Classifier | Test Accuracy  | Balanced Test Accuracy | Weighted Avg. Precision | Weighted Avg. F1 Score |
| ------------- | ------------- |  ------------- | ------------- | ------------- |
| RF Window 2  | 0.39 | 0.37 | 0.37 | 0.37 |
| RF Window 5  | 0.50 | 0.46 | 0.49 | 0.48 |
| RF Window 10  | 0.56 | 0.51 | 0.55 | 0.54 |
| RF Window 25  | 0.60 | 0.54 | 0.60 | 0.58 |
| RF Window 50  | 0.61 | 0.54 | 0.62 | 0.60 |
| ROCKET Window 100 - step size 5 | **0.70** | **0.61** | **0.70** | **0.69** |

Increasing the window size shows decreasing performance gain, not getting higher than ~60% test accuracy.

**ROCKET outperforms the Random Forest classifiers signifcantly, using only the raw x,y,z initial 3 features.**



## Resources on sktime, ROCKET
- https://github.com/alan-turing-institute/sktime/blob/main/examples/02_classification_univariate.ipynb
- https://github.com/alan-turing-institute/sktime/blob/main/examples/03_classification_multivariate.ipynb
- https://github.com/alan-turing-institute/sktime/blob/main/examples/rocket.ipynb
- http://www.timeseriesclassification.com/index.php
- *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.* - https://github.com/angus924/rocket
