# AI-based PD diagnostic predictions

## 1 Overview

This repository contains the data and Python scripts in support of the manuscript: Integrated metabolomics and machine learning approach to identify biomarkers and predict Parkinson's disease severity.  In this work, we develop ML models for PD detection and advanced analysis.

![image](https://github.com/ShiLab-GitHub/PD-metabolomics-AI/blob/main/pic/flow.jpg)

## 2 Dataset

The dataset are illustrated in the table. Plasma data utilized in this study contain numerous metabolites and lipid ratios obtained from the PPMI portal (https://www.ppmi-info.org/). Metabolites and lipids were extracted from plasma samples of PPMI-recruited participants using Bravos Low Volume Plasma Metabolite/Lipid protocols. These markers were organized into the dataset for further analysis. To develop our model effectively, we randomly split the dataset into training and testing sets in an 8:2 ratio and performed 10-fold cross-validation ten times to evaluate the performance of the models.

| Cohort | Number of the samples | Age (mean) | Male/Female | UPDRSⅢ | MoCA |
| ------ | --------------------- | ---------- | ----------- | ------ | ---- |
| PD     | 304                   | 64.5       | 159/145     | 38.4   | 26.8 |
| HC     | 120                   | 63.8       | 67/53       | -      | -    |
| pPD    | 183                   | 65.1       | 90/93       | -      | -    |

Note: The demographic and clinical characteristics of three cohorts - PD, HC, and pPD -including age, gender, UPDRSIII, and MoCA scores. 

UPDRS = Unified Parkinson Disease Rating Scale (a standard assessment tool of motor function in PD); MoCA = Montreal Cognitive Assessment (a validated screening tool for cognitive impairment across multiple domains).

The processed data are available in **Data/Metabolome data.csv**



## 3 System requirements

```
Hardware requirements: 
    main.py requires a computer with enough RAM to support the in-memory operations.
    Operating system：windows 10
    
Code dependencies(environment.txt):
    python == 3.8.7 
    torch == 1.13.0 + cu116
    numpy == 1.21.0
    sklearn == 0.0
    pandas == 1.3.5
    ...
```

4 Quick Setup
-----------

1.Clone the InnerEye-DeepLearning repo by running the following command:

```
git clone {待填充}
```

2.Create and activate your environment:

```
pip install -r environment.txt
```

## 5 Instructions for use

Verify that your installation was successful by running the main.py model

```
  python main.py --model --n_classes
```

**Parameter description**

```
'''
  --model = {"XGB-RFE","RF-RFE","GBM-RFE"}
      --model stands for the AI model used by the program
  
  --n_classes = {"2","3"}
      --n_classes represents the number of categories that belong to the data that is used
'''
```

## 6 Description of the method

In recent years, machine learning has become an effective means of analyzing and processing diverse datasets to gain a comprehensive understanding of the relationships between different components of metabolomics data with differing concentrations and reaction characteristics. By establishing machine learning models to predict the links between specific metabolic pathways/metabolites and diseases, we can discover new biological insights.

### Random Forest

RF is an ensemble learning algorithm based on decision tree [1]. We used the Gini coefficient as a dividing criterion.

### Gradient Boosting Machine

GBM is a model with high performance, using weakly independent learners [2]. The number of weak classifiers (n_estimators) was set to 200 and the maximum depth of the tree (max_depth) to 10.

### eXtreme Gradient Boosting

XGB is an optimized distributed gradient enhancement model. It continuously builds Classification and Regression (CART) Trees. based on the direction of gradient descent of the loss function, and gradually approaches the local minimum of the loss function to finally build a strong classifier with high accuracy [3]. The max_depth was set to 5.

### Recursive Feature Elimination

RFE algorithm obtains the optimal combination of variables that maximizes the model performance by removing specific feature variables [4] In RFE algorithms, we add another layer of resampling process (10-fold cross-validation)  to the outer layer of the aforementioned algorithm in order to better evaluate the performance fluctuations in the feature selection process.

## 7 Results

Prediction performance on the test dataset

| Metrics             | GBM   | RF    | XGBoost |
| ------------------- | ----- | ----- | ------- |
| Accuracy (%)        | 95.87 | 93.71 | 96.47   |
| F1_micro (%)        | 95.87 | 93.71 | 96.47   |
| F1_macro (%)        | 95.58 | 92.91 | 96.39   |
| Precision_micro (%) | 95.87 | 93.71 | 96.47   |
| Precision_macro (%) | 96.05 | 94.27 | 96.79   |
| Recall_micro (%)    | 95.87 | 93.71 | 96.47   |
| Recall_macro (%)    | 95.43 | 92.24 | 96.21   |
| AUC_ovr (%)         | 99.76 | 99.44 | 99.81   |

## 8 License

The model is licensed under the [Apache 2.0 license](LICENSE).

## 9 Contact

If you have any feature requests, or find issues in the code, please create an issue on GitHub.

Please send an email to 1120200076@bit.edu.cn if you would like further information about this project.

## 10 References

>  [1] Elith, J., et al. (2008). "A working guide to boosted regression trees." Journal of animal ecology 77(4): 802-813.[link](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/j.1365-2656.2008.01390.x)

>  [2] Friedman, J. H. (2001). "Greedy function approximation: a gradient boosting machine." Annals of statistics: 1189-1232.[link](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)

>  [3] Chen, T. and C. Guestrin (2016). Xgboost: A scalable tree boosting system. Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining.[link](https://dl.acm.org/doi/10.1145/2939672.2939785)

>  [4] Chen, X.-w. and J. C. Jeong (2007). Enhanced recursive feature elimination. Sixth international conference on machine learning and applications (ICMLA 2007), IEEE.[link](https://ieeexplore.ieee.org/document/4457188)
