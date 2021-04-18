<a id="markdown-cousera-machine-learning" name="cousera-machine-learning"></a>
# Cousera Machine Learning 
<img src="https://github.com/tomo-2525/Cousera_ML/blob/master/Cousera_ML.jpg"/>  

<!-- TOC -->

- [Cousera Machine Learning](#cousera-machine-learning)
- [Week1](#week1)
    - [Introduction](#introduction)
        - [What is Machine Learing](#what-is-machine-learing)
            - [Machine Learning difinition:](#machine-learning-difinition)
            - [Supervised Learning (教師あり学習)](#supervised-learning-教師あり学習)
            - [Unsupervised Learning (教師なし学習)](#unsupervised-learning-教師なし学習)
    - [Linear Regression with One Variable (線形回帰,線形回帰)](#linear-regression-with-one-variable-線形回帰線形回帰)
        - [Model](#model)
        - [Parameter Learning](#parameter-learning)
            - [Gradient decent](#gradient-decent)
            - [Gradient decentの動き](#gradient-decentの動き)
    - [Linear Algebra Review](#linear-algebra-review)
- [WEEK 2](#week-2)
    - [Linear Regression with Multiple Variables](#linear-regression-with-multiple-variables)
        - [Multivavariate Linear Regression](#multivavariate-linear-regression)
        - [Computing Parameters Analytically](#computing-parameters-analytically)
    - [Octave/Matlab Tutorial](#octavematlab-tutorial)
- [WEEK 3](#week-3)
    - [Logistic Regression](#logistic-regression)
        - [Classification and Representaion](#classification-and-representaion)
        - [Logistic Regression Model](#logistic-regression-model)
        - [Multiclass Classification](#multiclass-classification)
    - [Regularization](#regularization)
        - [Solving the Problem of Overfitting](#solving-the-problem-of-overfitting)
- [WEEK 4](#week-4)
    - [Neural Networks: Representation](#neural-networks-representation)
        - [Motivations](#motivations)
        - [Neural Networks](#neural-networks)
        - [Applications](#applications)
- [WEEK 5](#week-5)
- [Neural Networks: Learning](#neural-networks-learning)
    - [Cost Function and Backpropagation](#cost-function-and-backpropagation)
    - [Backpropagation in Practice](#backpropagation-in-practice)
    - [Application of Neural Networks](#application-of-neural-networks)
- [WEEK 6](#week-6)
    - [Advice for Applying Machine Learning](#advice-for-applying-machine-learning)
        - [Evaluating a Learning Algorithm](#evaluating-a-learning-algorithm)
        - [Bias vs Variance](#bias-vs-variance)
    - [Machine Learning System Design](#machine-learning-system-design)
        - [Building a Spam Classifier](#building-a-spam-classifier)
        - [Handling Skewed Data](#handling-skewed-data)
        - [Using Large Data Sets](#using-large-data-sets)
- [WEEK 7](#week-7)
    - [Support Vector Machines](#support-vector-machines)
        - [Large Margin Classification](#large-margin-classification)
        - [Kernels](#kernels)
        - [SVMs in Practice](#svms-in-practice)
- [WEEK 8](#week-8)
    - [Unsupervised Learing](#unsupervised-learing)
        - [Clustering](#clustering)
    - [Dimensionality Reduction](#dimensionality-reduction)
        - [Motivation](#motivation)
        - [Principal Component Analysis](#principal-component-analysis)
        - [Applying PCA](#applying-pca)
- [WEEK 9](#week-9)
    - [Anomaly Detection](#anomaly-detection)
        - [Density Estimation](#density-estimation)
        - [Building an Anomaly Detection System](#building-an-anomaly-detection-system)
        - [Multivariate Gaussian Distribution (Optional)](#multivariate-gaussian-distribution-optional)
    - [Recommender Systems](#recommender-systems)
        - [Prediction Movie Ratings](#prediction-movie-ratings)
        - [Collaborative Filtering](#collaborative-filtering)
        - [Low Rank Matrix Factorization](#low-rank-matrix-factorization)
- [WEEK 10](#week-10)
    - [Large Scale Machine Learning](#large-scale-machine-learning)
        - [Gradient Descent with Large Datasets](#gradient-descent-with-large-datasets)
        - [Advanced Topics](#advanced-topics)
- [WEEK 11](#week-11)
    - [Application Example: Photo OCR](#application-example-photo-ocr)
        - [Photo OCR](#photo-ocr)

<!-- /TOC -->


<a id="markdown-week1" name="week1"></a>
# Week1
<a id="markdown-introduction" name="introduction"></a>
## Introduction
<a id="markdown-what-is-machine-learing" name="what-is-machine-learing"></a>
### What is Machine Learing
<a id="markdown-machine-learning-difinition" name="machine-learning-difinition"></a>
#### Machine Learning difinition:  
• ArthurSamuel(1959).MachineLearning:Fieldof study that gives computers the ability to learn without being explicitly programmed.  
• TomMitchell(1998)Well-posedLearning Problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

Example: playing checkers.
E = the experience of playing many games of checkers
T = the task of playing checkers.
P = the probability that the program will win the next game.
In general, any machine learning problem can be assigned to one of two broad classifications: Supervised learning and Unsupervised learning.

<a id="markdown-supervised-learning-教師あり学習" name="supervised-learning-教師あり学習"></a>
#### Supervised Learning (教師あり学習)
right answers given
* Regression (回帰):Predict continuous valued output
Y = f(X) というモデルの時にYが連続である
* classification (分類):Discrete valued output 
Y = f(X) というモデルの時にYが離散（とびとび）である
(例)  
Regression - Given a picture of a person, we have to predict their age on the basis of the given picture  
Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 

<a id="markdown-unsupervised-learning-教師なし学習" name="unsupervised-learning-教師なし学習"></a>
#### Unsupervised Learning (教師なし学習)
right answers 'not' given
 
* Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.

* Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment. (i.e. identifying individual voices and music from a mesh of sounds at a cocktail party).

<a id="markdown-linear-regression-with-one-variable-線形回帰線形回帰" name="linear-regression-with-one-variable-線形回帰線形回帰"></a>
## Linear Regression with One Variable (線形回帰,線形回帰)
* この講義で用いられる用語
![](./img/README_2021-04-18-13-53-07.png) 
 
<a id="markdown-model" name="model"></a>
### Model
![](./img/README_2021-04-18-14-00-23.png)  
When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.  
* Hypothesis Function(仮説関数):yの値を予想してくれる関数のこと。  
![](./img/README_2021-04-18-14-29-12.png)

* Cost Function(目的関数):h(x)とyの差の2乗の平均を2で割ったもの0に近づくほど仮説関数が正確に予測できている。この目的関数は、2乗誤差関数と呼ばれる  
![](./img/README_2021-04-18-14-04-20.png)  

* 仮説関数と目的関数の関係(仮説関数のパラメーターが1つの場合(分かりやすくするため))  
![](./img/README_2021-04-18-14-26-14.png)  
* 仮説関数と目的関数の関係(パラメーターが2つの場合)  
![](./img/README_2021-04-18-14-29-12.png)  
![](./img/README_2021-04-18-14-46-42.png)  
θ1 and θ2 tend to be around 0.12 and 250 respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'. 

<a id="markdown-parameter-learning" name="parameter-learning"></a>
### Parameter Learning
<a id="markdown-gradient-decent" name="gradient-decent"></a>
#### Gradient decent
Gradient decent(最急降下法,勾配降下法)というアルゴリズムを用いて目的化関数を最小化(θの更新)する。（線形回帰以外でも使われる）(これから紹介するアルゴリズムは、Batch Gradient Decentと呼ばれる場合もある（θを更新するときに全てのデータセットをみるから))  
![](./img/README_2021-04-18-15-15-55.png)  
![](./img/README_2021-04-18-15-28-01.png)  
![](./img/README_2021-04-18-15-22-36.png)  

最急降下法を始める場所(θの初期値)によって局所的最小値が異なる  
θの更新は同時に行う  
α:学習率（αが大きすぎると 大きなステップで降下し収束しない、小さすぎると時間がかかる）  
最小値に近づくにつれ、偏微分の値が小さくなるので、αの更新の必要はない
![](./img/README_2021-04-18-15-57-37.png)  

<a id="markdown-gradient-decentの動き" name="gradient-decentの動き"></a>
#### Gradient decentの動き
![](./img/README_2021-04-18-15-52-21.png)  

<a id="markdown-linear-algebra-review" name="linear-algebra-review"></a>
## Linear Algebra Review


<a id="markdown-week-2" name="week-2"></a>
# WEEK 2
<a id="markdown-linear-regression-with-multiple-variables" name="linear-regression-with-multiple-variables"></a>
## Linear Regression with Multiple Variables
<a id="markdown-multivavariate-linear-regression" name="multivavariate-linear-regression"></a>
### Multivavariate Linear Regression
<a id="markdown-computing-parameters-analytically" name="computing-parameters-analytically"></a>
### Computing Parameters Analytically
<a id="markdown-octavematlab-tutorial" name="octavematlab-tutorial"></a>
## Octave/Matlab Tutorial


<a id="markdown-week-3" name="week-3"></a>
# WEEK 3 
<a id="markdown-logistic-regression" name="logistic-regression"></a>
## Logistic Regression
<a id="markdown-classification-and-representaion" name="classification-and-representaion"></a>
### Classification and Representaion
<a id="markdown-logistic-regression-model" name="logistic-regression-model"></a>
### Logistic Regression Model
<a id="markdown-multiclass-classification" name="multiclass-classification"></a>
### Multiclass Classification

<a id="markdown-regularization" name="regularization"></a>
## Regularization
<a id="markdown-solving-the-problem-of-overfitting" name="solving-the-problem-of-overfitting"></a>
### Solving the Problem of Overfitting


<a id="markdown-week-4" name="week-4"></a>
# WEEK 4 

<a id="markdown-neural-networks-representation" name="neural-networks-representation"></a>
## Neural Networks: Representation
<a id="markdown-motivations" name="motivations"></a>
### Motivations
<a id="markdown-neural-networks" name="neural-networks"></a>
### Neural Networks
<a id="markdown-applications" name="applications"></a>
### Applications


<a id="markdown-week-5" name="week-5"></a>
# WEEK 5

<a id="markdown-neural-networks-learning" name="neural-networks-learning"></a>
# Neural Networks: Learning
<a id="markdown-cost-function-and-backpropagation" name="cost-function-and-backpropagation"></a>
## Cost Function and Backpropagation
<a id="markdown-backpropagation-in-practice" name="backpropagation-in-practice"></a>
## Backpropagation in Practice
<a id="markdown-application-of-neural-networks" name="application-of-neural-networks"></a>
## Application of Neural Networks


<a id="markdown-week-6" name="week-6"></a>
# WEEK 6 
<a id="markdown-advice-for-applying-machine-learning" name="advice-for-applying-machine-learning"></a>
## Advice for Applying Machine Learning
<a id="markdown-evaluating-a-learning-algorithm" name="evaluating-a-learning-algorithm"></a>
### Evaluating a Learning Algorithm
<a id="markdown-bias-vs-variance" name="bias-vs-variance"></a>
### Bias vs Variance
<a id="markdown-machine-learning-system-design" name="machine-learning-system-design"></a>
## Machine Learning System Design
<a id="markdown-building-a-spam-classifier" name="building-a-spam-classifier"></a>
### Building a Spam Classifier
<a id="markdown-handling-skewed-data" name="handling-skewed-data"></a>
### Handling Skewed Data
<a id="markdown-using-large-data-sets" name="using-large-data-sets"></a>
### Using Large Data Sets

<a id="markdown-week-7" name="week-7"></a>
# WEEK 7 
<a id="markdown-support-vector-machines" name="support-vector-machines"></a>
## Support Vector Machines
<a id="markdown-large-margin-classification" name="large-margin-classification"></a>
### Large Margin Classification
<a id="markdown-kernels" name="kernels"></a>
### Kernels
<a id="markdown-svms-in-practice" name="svms-in-practice"></a>
### SVMs in Practice


<a id="markdown-week-8" name="week-8"></a>
# WEEK 8 

<a id="markdown-unsupervised-learing" name="unsupervised-learing"></a>
## Unsupervised Learing
<a id="markdown-clustering" name="clustering"></a>
### Clustering
<a id="markdown-dimensionality-reduction" name="dimensionality-reduction"></a>
## Dimensionality Reduction
<a id="markdown-motivation" name="motivation"></a>
### Motivation
<a id="markdown-principal-component-analysis" name="principal-component-analysis"></a>
### Principal Component Analysis
<a id="markdown-applying-pca" name="applying-pca"></a>
### Applying PCA


<a id="markdown-week-9" name="week-9"></a>
# WEEK 9 

<a id="markdown-anomaly-detection" name="anomaly-detection"></a>
## Anomaly Detection
<a id="markdown-density-estimation" name="density-estimation"></a>
### Density Estimation
<a id="markdown-building-an-anomaly-detection-system" name="building-an-anomaly-detection-system"></a>
### Building an Anomaly Detection System
<a id="markdown-multivariate-gaussian-distribution-optional" name="multivariate-gaussian-distribution-optional"></a>
### Multivariate Gaussian Distribution (Optional)
<a id="markdown-recommender-systems" name="recommender-systems"></a>
## Recommender Systems
<a id="markdown-prediction-movie-ratings" name="prediction-movie-ratings"></a>
### Prediction Movie Ratings
<a id="markdown-collaborative-filtering" name="collaborative-filtering"></a>
### Collaborative Filtering
<a id="markdown-low-rank-matrix-factorization" name="low-rank-matrix-factorization"></a>
### Low Rank Matrix Factorization 


<a id="markdown-week-10" name="week-10"></a>
# WEEK 10 
<a id="markdown-large-scale-machine-learning" name="large-scale-machine-learning"></a>
## Large Scale Machine Learning 
<a id="markdown-gradient-descent-with-large-datasets" name="gradient-descent-with-large-datasets"></a>
### Gradient Descent with Large Datasets
<a id="markdown-advanced-topics" name="advanced-topics"></a>
### Advanced Topics


<a id="markdown-week-11" name="week-11"></a>
# WEEK 11 

<a id="markdown-application-example-photo-ocr" name="application-example-photo-ocr"></a>
## Application Example: Photo OCR
<a id="markdown-photo-ocr" name="photo-ocr"></a>
### Photo OCR