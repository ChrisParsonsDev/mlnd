# Capstone Proposal
Chris Parsons
15/11/17

## Proposal

### Domain Background

Credit Card Fraud is a serious problem!

In order to improve the retail experience, and ultimately make more money, both credit card providers and retailers are striving to make the payment process as convenient and simple as possible. Since the 1980's these electronic payment systems have forged their way into our everyday lives. It's estimated that $31,000,000,000 of revenue was generated via these electronic payment systems worldwide in 2016. A staggering 7.3% rise on the year before. <cite>[Nilson Report][1]</cite>

However it's not all good news..

While these payment systems provide us with a simple, consumable and convenient payment platform they are also at risk of nefarious activities - including fraud. In a 5 year period from 2010-2015 fraudulent transactions on credit cards rose from $8,000,000 to $21,000,000.  <cite>[BBC][2]</cite> Not only does this impact consumers who will be hugely inconvenienced by having to report and rectify these errors, or worse be left out of pocket, the retailers lose out too! If you're impacted by a fraudulent transaction on a particular website or in a store this hugely decreases the chances of you shopping there again. Steps must be taken to mitigate the impact these transactions have.

So what can be done to prevent this incredible growth in fraudulent transactions that hurts both consumers and retailers?

### Problem Statement


There's certainly no shortage of data. From a financial services perspective we've got plenty of information about customers and their spending habits.  There's everything transaction frequency and location to what kind of products you're likely to purchase. The problem is time!

While it's possible to spot credit card fraud, given a team of experts and enough time. The challenge is that these transactions happen all over the globe, simultaneously, in the blink of an eye.

So given this abundance of data, and inherent time constraints affiliated with the domain - is it possible to build a system that will recognise fraudulent transactions on the fly? As these transactions take place? Wherever you are in the world?

Such a system, if even only 75% accurate at detecting and halting fraudulent transactions before they complete would save the industry over 15,000,000 dollars annually!

This is the challenge.

Given historical information about fraudulent credit card transactions, can I train a system to be able to reliably detect new instances of fraud as they happen?

### Datasets and Inputs

The dataset being explored as part of this study can be found [here](https://www.kaggle.com/dalpozz/creditcardfraud).

The "Credit Card Fraud" dataset contains transactions from two days in September 2013 made by European credit card holders. There are 284,807 transactions in total and 492 instances of fraud. Plainly this means I am working with a highly unbalanced dataset where the positive class accounts for ~0.2% of the overall dataset - which while representative of the real world, is not ideal for building a Machine Learning model.


For confidentiality reasons the majority of the features have been transformed via PCA and only numerical values remain. Both 'Time' and 'Amount' features are not transformed.

The label for each transaction can be found under the 'Class' feature and is coded as:

1. Positive Class (1): Incident of fraud
2. Negative Class (0): Normal, not fraudulent, transaction


### Solution Statement

To solve the problem caused by credit card fraud, I will attempt to build a system that can automatically predict whether or not a transaction is fraudulent in real time.

The core component of such a system will be a Machine Learning model that is capable of classifying credit card transactions as fraud or not fraud. This system will be trained on historical transaction data such that it is able to correctly determine whether a new transaction contains an instance of fraud.

### Benchmark Model

The simplest solution to this classification problem is to use a Logisitc Regression classifier. This classifier will form the benchmark model for comparison. The success of this benchmark model can be determined by examining the accuracy when classifying against a known standard, or test set.

This accuracy rating for the trivial Logistic Regression solution can then be used to evaluate the performance of subsequent models developed as part of this project.

### Evaluation Metrics

In order to successfully evaluate the performance of both the benchmark model and any subsequent more complicated approaches I will assess the accuracy of the model. Concretely I will evaluate the model's precision and recall.

![precision and recall](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/350px-Precisionrecall.svg.png)

This will examine two things:

1. What percentage of the transactions labeled as fraud by the model were indeed fraud.
2. What percentage of the transactions labeled as fraud were detected by the model.

From this we will be able to accurately infer the suitability of the model.

### Project Design

This section describes a theoretical workflow for the project.

#### Programming Languages and Tooling

1. **Python 2.7** - I have elected to use Python as a programming language for this project primarily because of it's ubiquity within the Data Science space and subsequent documentation. Python is also ideal for this use case because of the way the language interprets and handles large numbers which are abundant in this dataset.
2. **Jupyter Notebooks** - These notebooks provide a suitable platform for developing the model and documenting the workflow. Jupyter notebooks are great for demonstrating the underlying implementation of Machine Learning models which makes them ideally suited for use in this project.
2. **Scikit-Learn** - SKLearn is a Python-based Data Science library. The library comes with tools and functions that allow data manipulation and the implementation some simple Machine Learning models in a compact and consumable way. This library will prove invaluable in rapidly iterating on the design of the final system.
3. **Keras** - Keras provides a 'High Level Neural Network API'. Such an abstraction will vastly improve the development workflow when examining the suitability of Neural Networks when applied to this problem domain.
4. **TensorFlow** - This open source Machine Learning and deep learning framework delivers the back end functionality to Keras, in a sense TensorFlow does all the heavy lifting and number crunching of the Keras model definition.

#### Model Implementation

1. **Benchmark Model** - The benchmark model will be a [Logistic Regression Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). This model will be implemented using the tooling in SKLearn and be used to compare all other models developed as part of this project.
2. **Complex Model** - I will then explore how using models like SVM or [xgboost](https://github.com/dmlc/xgboost) perform when applied to this binary classification problem.
3. **Neural Network** - In order to examine the effectiveness of complex solutions on this non-image domain I will develop a Neural Network architecture that is capable of classifying the instances of fraud. Subsequent analysis of the accuracy of the Neural Network architecture will allow direct comparison between complex and simple techniques in this problem space.
4. **Transfer Learning** - Finally, to compare the accuracy that can be obtained between different NN architectures I will then use Transfer Learning to teach a pre-trained model to recognise fraud based on the input dataset from this project.



-----------

### Acknowledgements

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

[1]:https://www.nilsonreport.com/upload/content_promo/The_Nilson_Report_10-17-2016.pdf
[2]:http://www.bbc.com/capital/story/20170711-credit-card-fraud-what-you-need-to-know
