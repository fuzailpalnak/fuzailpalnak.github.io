---
layout: post
title:  "Understanding The Perceptron Algorithm"
comments: true
description: "How does Perceptron learns the decision boundary needed to 
classify the data?, What is the learning Rule and Decision for Peceptron?"
date:   2020-07-21
permalink: /perceptron/
author: Fuzail Palnak
---

The Perceptron algorithm is the simplest type of artificial neural network. It is a model of a single neuron that can
be used for two-class classification problems and provides the foundation for later developing much larger networks.


*```Before we start with Perceptron, lets go through few concept that are essential in understanding the Algorithm```*

### What are HyperPlanes
	
> As defined by [Wikipedia](https://en.wikipedia.org/wiki/Hyperplane), a hyperplane is a subspace whose dimension is one less than that of its ambient space. If a space is 
3-dimensional then its hyperplanes are the 2-dimensional planes, while if the space is 2-dimensional,
its hyperplanes are the 1-dimensional lines.

Consider a ***2D*** space, the `standard equation` of hyperplane in a ***2D*** space is defined
as $$ax + by + c = 0$$, If the equation is simplified it results to  $$y = (-a/b) x + (-c/b)$$, which is noting but the
`general equation` of line with slope `-a/b` and intercept `-c/b`, which is a ***1D*** hyperplane in a ***2D*** space,
this validates our definition of hyperplanes to be one dimension less than the ambient space

What are a, b? - they are the components of the vector, this vector has a special name called `normal vector`, 
so any hyperplane can be defined using its normal vector. 
<ul>
$$
n^T * data + intercept = 0
$$
$$
\text{where};
\text{ n} = \begin{bmatrix}a  \\b \end{bmatrix} ;
\text{data} =  \begin{bmatrix}x  \\y \end{bmatrix} ;
\text{intercept = distance from origin}
$$
</ul>


One property of normal vector is, it is always perpendicular to hyperplane.

<ul>
<li>

<b>How to relate hyperplane?</b><br />
Consider the normal vector n <code><3, 1></code>, we can define the hyperplane as $$3x + 1y + c = 0$$
this is equivalent to having a line with slope <code>-3</code> and intercept <code>-c</code>, this forms $$y = (-3) x + (-c)$$

</li>
</ul>


To have a deep dive in hyperplanes and how are hyperplanes formed and defined, have a look at 
[this explanation](https://www.youtube.com/watch?v=-sNDkhE2Vsk&feature=emb_logo)

### Assumptions made by the Algorithm
The assumptions the Perceptron makes is that data is `linearly separable` and the classification problem is `binary`
<ul>
<li>

<b>What Does linearly separable data mean</b><br />

This means that there must exists a hyperplane which separates the data points in way making all the points belonging
positive class lie on one side of hyperplane and the data points belonging to negative class lie on the other side.

</li>
</ul>
![png]({{ site.url }}/assets/perceptron_files/hyperplane.png)

<ul>
<li>

<b>Multiple Hyperplanes</b><br />
Now we have assumed that the data is linearly separable.<br />
<code>How many hyperplanes could exists which separates the data?
Just One? More than One?</code><br />
The answer is more than one, in fact infinite hyperplanes could exists if data is linearly separable, 
and perceptron finds one such hyperplane out of the many hyperplanes that exists

</li>
</ul>
![png]({{ site.url }}/assets/perceptron_files/multiple_hyperplanes.png)



### Perceptron Algorithm

There are two core rules that forms the Algorithm 
<ul>
<li>

<b>Decision Rule</b><br />
This rule checks whether the data point lies on the positive side of the hyperplane or on the negative side, it does so
by checking the <code>dot product</code> of the <code>weight</code> with the <code>data point</code>
<img src="https://fuzailpalnak.github.io/assets/perceptron_files/classifier.png" alt="Classifier">

For Simplicity we eliminate the intercept term from $$w^T * x + b = 0$$ i.e remove the <code>b</code> from the equation, now the
hyperplane will go through origin, so the equation will be 

$$w^T * x = 0$$
<ul>
<li>

<b>How does the dot product tells whether the data point lies on the positive side of the hyper plane or negative side of hyperplane?</b><br />

Lets look at the other representation of dot product
$$
w^T* x = \| w \|  \| x \| cos \theta 
$$

$$
\Theta  =  arcos   \frac{w^{T} * x }{\| w \|  \| x \|} 
$$
<img src="https://fuzailpalnak.github.io/assets/perceptron_files/example2.png" alt="Example 2">

For all the positive points theta is <code><90</code> which will result in a positive value as cos is positive and for all the
negative points theta is <code>>90</code> which will result in a negative value as cos is negative<br />
So if the value of $$w^T* x $$ is positive the algorithm will yield a positive prediction and  when its negative it
will yield a negative prediction

</li>
</ul>
</li>



<li>

<b>Learning Rule</b><br />

Now we know when the data point belong to negative class and when it belongs to positive class, using this information 
we can keep on updating the weight vector <code>w</code> whenever we make a wrong prediction until we find a separating hyperplane<br />
The rule says $$yi*w^T* x <= 0$$ i.e the point has been misclassified hence we update the vector <code>w</code> with the update rule
$$w = w + y * x$$ 

<b>Rule when positive class is miss classified</b><br />

<code>y = 1</code> then <code>w</code> is updated by $$w = w + x$$
This translates to, the algorithm is trying to decrease the <code>theta</code> between <code>w</code> and the <code>data point</code><br />


<b>Rule when negative class is miss classified</b><br />

<code>y = -1</code> then <code>w</code> is updated by $$w = w - x$$
This translates to, the algorithm is trying to increase the <code>theta</code> between <code>w</code> and the <code>data point</code><br />


<img src="https://fuzailpalnak.github.io/assets/perceptron_files/intution.png" alt="Intution">

 
</li>
</ul>

Combining the `Decision Rule` and `Learning Rule`, the perceptron algorithm is derived
```python
while True:
    miss_classified = 0
    for loop over data:
        if decision_rule() <= 0:
            # Miss classified the data point and adjust the weight
            learning_rule()
            miss_classified = miss_classified + 1
    if miss_classified == 0:
        # if no miss classified then the perceptron has converged and found a hyperplane
        break
```