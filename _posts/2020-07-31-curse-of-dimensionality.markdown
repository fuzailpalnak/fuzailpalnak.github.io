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
each house is a data point which can be represented with attributes/features.
$$house =  \begin{bmatrix}\text{num of bedrooms} \\\text{num of rooms} \\\text{has a garage} \\\text{...}  \end{bmatrix}$$
 The word **feature** and **dimension** will be used interchangeably

### What happens when dimensionality in data is increased?
> "As the number of features or dimensions grows, the amount of data we need to generalize accurately grows exponentially." <br /> - Charles Isbell, Professor and Senior Associate Dean, School of Interactive Computing, Georgia Tech

<ul>
<li>

<b>Ambient/Empty Space</b><br />

Consider 5 data points with just one feature, the feature space needed to represent the data is just $5^1$,
Now, when an additional feature is introduced without 
any new data points the feature space grows to $5^2$, and as no new data point is added the rest of the space is just empty

<div style="padding: 10px;">
<figure class="image">
  <img src="https://fuzailpalnak.github.io/assets/curse/scale_first.png" alt="Data In One Dimension">
  <figcaption>Data In One Dimension</figcaption>
</figure>
</div>


<div style="padding: 10px;">
<figure class="image">
  <img src="https://fuzailpalnak.github.io/assets/curse/scale_second.png" alt="Data in Two Dimension with ambient space">
  <figcaption>Data in Two Dimension with ambient space</figcaption>
</figure>
</div>

If the addition of features is continued without any new data point the feature space increases exponentially and eventually become sparse. In other words
the volume of feature space grows so quickly that the data cannot keep up and the feature space is just left empty<br />

In Machine Learning, we want the data to be spread in every part of the region not just in a limited region, for our
machine learning algorithm to perform well, if data points are not in proportion with the features then the model might experience poor performance during testing.

</li>

<li>

<b>Euclidean Distance in High Dimensional Space</b><br />
The distance between data points in $n$ dimensional space is given as $$\sqrt{ \triangle x^2 +  \triangle y^2 +  \cdots + \triangle n^2} $$The addition of new feature
 will always add a positive value to the overall distance, the distance between two data points increases drastically with their dimensionality.
i.e the data points move further apart as a new feature is added, So in high dimensional space euclidean distance is not an accurate distance metric to consider.


</li>
</ul>

### Data with low dimensional structure

However, not always such a huge feature space is required to represent the underlying data, this representation is known as intrinsic dimensionality of data. 
Intrinsic dimension for a data set can be thought of as the number of features needed in a minimal representation of the data. 
The true dimensionality of the data can be much lower than its ambient space.
<ul>
<li>

<b>Sub Spaces</b><br />

The space could be high dimensional but the data might lie in a *sub space*, a $\Re^d$ dimensional data embedded in $\Re^n$  dimensional space 
<div style="padding: 10px;">
<figure class="image">
  <img src="https://fuzailpalnak.github.io/assets/curse/subspace.png" alt="Sub Space">
  <figcaption>Two Dimensional Data embedded in a 3 Dimensional Space</figcaption>
</figure>
</div>


</li>

<li>

<b>Manifolds</b><br />

Data lies in a *sub dimensional manifold* that is embedded within $\Re^n$

<div style="padding: 10px;">
<figure class="image">
  <img src="https://fuzailpalnak.github.io/assets/curse/manifold.png" alt="Sub Space">
  <figcaption>Two Dimensional Manifold embedded in a 3 Dimensional Space</figcaption>
</figure>
</div>
</li>
</ul>

There are several algorithm which finds sub dimensional manifold and subspaces, algorithm such as [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), 
[LocallyLinearEmbedding](https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf), to name a few ,these algorithm get the data from high-dimensional 
space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the
original data, ideally close to its intrinsic dimension. This process is referred as Dimensionality reduction



 

