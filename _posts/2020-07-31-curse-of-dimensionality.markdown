---
layout: post
title:  "Curse of Dimensionality"
comments: true
description: "What is curse of dimensionality? How to tackle it? How Does it affect the Data and Training Algorithm"
date:   2020-07-31
permalink: /curse-of-dimensionality/
author: Fuzail Palnak
---

### What is Dimensionality in Data?
Dimensionality in statistics refers to how many attributes/features a data point has. For example, real state data, where 
each house is a data point which can be represented with attributes/features 
$$house =  \begin{bmatrix}\text{num of bedrooms} \\\text{num of rooms} \\\text{has a garage} \\\text{...}  \end{bmatrix}$$


### What happens when dimensionality in data is increased?
> "As the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially." <br /> - Charles Isbell, Professor and Senior Associate Dean, School of Interactive Computing, Georgia Tech

Consider 5 data points with just one feature the feature space needed to represent the data is just $5^1$, now, when an additional feature is introduced without 
any new data points the feature space grows to $5^2$, and as no new data point is added the rest of the space is just empty<br/ >
If the addition of features is continued without any new data point the feature space increases exponentially and eventually become sparse. In other words
the volume of feature space grows so quickly that the data cannot keep up and the feature space is just left empty<br/ >



| ![Data In One Dimension](https://fuzailpalnak.github.io/assets/curse/scale_first.png)  | ![Data in Two Dimension with ambient space](https://fuzailpalnak.github.io/assets/curse/scale_second.png) |
|:---:|:---:|
| Data In One Dimension | Data in Two Dimension with ambient space |

In Machine Learning, we want the data to be spread in every part of the region not just in a limited region, for our
machine learning algorithm to perform well, if data points are not in proportion with the features then the model might experience poor performance during testing.

### Data with low dimensional structure

However not always such a huge feature space is required to represent the underlying data, known as intrinsic dimensionality of data. 
Intrinsic dimension for a data set can be thought of as the number of features needed in a minimal representation of the data. 
The true dimensionality of the data can be much lower than its ambient space, i.e the space could be high dimensional
$\Re^n$ but the data might lie in a *sub space* $\Re^d$, Imagine a 2D $\Re^2$ plane embedded in $n$ dimension space $\Re^n$, here the data is $2D$, 
the rest is just ambient space, here the data has $2D$ intrinsic dimensionality. 
![png](https://fuzailpalnak.github.io/assets/curse/subspace.png)
Other case could be the data lies in a *sub dimensional manifold* that is embedded within $\Re^n$
![png](https://fuzailpalnak.github.io/assets/curse/manifold.png)



 

