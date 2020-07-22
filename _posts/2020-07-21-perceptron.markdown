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
$$
n^T* data + intercept = 0
"where"
"n =" ((a),(b))
"data = " ((x),(y))
"intercept = distance from origin"
$$
One property of normal vector is, it is always perpendicular to hyperplane.

<ul>
<li>

*How to relate hyperplane?*<br />
Consider the normal vector n <3, 1>, we can define the hyperplane as $$3x + 1y + c = 0$$,
this is equivalent to having a line with slope `-3` and intercept `-c`, this forms $$y = (-3) x + (-c)$$

</li>
</ul>


To have a deep dive in hyperplanes and how are hyperplanes formed and defined, have a look at 
[this explanation](https://www.youtube.com/watch?v=-sNDkhE2Vsk&feature=emb_logo)

### Assumptions made by the Algorithm
The assumptions the Perceptron makes is that data is `linearly separable` and the classification problem is `binary`
<ul>
<li>

*What Does linearly separable data mean*<br />

This means that there must exists a hyperplane which seperates the data points in way making all the points belonging
positive class lie on one side of hyperplane and the data points belonging to negative class lie on the other side.

![png]({{ site.url }}/assets/perceptron_files/hyperplane.png)
</li>
<li>

*Multiple Hyperplanes*<br />
Now we have assumed that the data is linearly seperable.<br />
`How many hyperplanes could exists which seperates the data?
Just One? More than One?`<br />
The answer is more than one, in fact infinite hyperplanes could exists if data is linearly separable, 
and perceptron finds one such hyperplane out of the many hyperplanes that exists

![png]({{ site.url }}/assets/perceptron_files/multiple_hyperplanes.png)
</li>
</ul>



### Perceptron Algorithm

There are two core rules that forms the Algorithm 
<ul>
<li>

*Decision Rule*<br />

For Simplicity we eliminate the intercept term from $$w^T * x + b = 0$$ i.e remove the `b` from the equation, now the
hyperplane will go through origin, so the equation will be 

$$w^T * x = 0$$

This rule checks wheather the data point lies on the positive side of the hyperplane or on the negative side, it does so
by checking the `dot product` of the `weight` with the `data point`

$$w^T * x$$
<ul>
<li>

*How does the dot product tells wheather the data point lies on the positive side of the hyper plane or negative side of hyperplane?*<br />

Lets look at the other representation of dot product
$$
w^T* x = norm(vecw)norm(vecx)"cos"theta
theta = "arcos" w^T* x/norm(vecw)norm(vecx)
$$
![png]({{ site.url }}/assets/perceptron_files/example2.png)

For all the positive points theta is `<90` which will result in a positive value as cos is positive and for all the
negative points theta is `>90` which will result in a negative value as cos is negative<br />
So if the value of $$w^T* x $$ is positive the algorithm will yield a positive prediction and  when its negative it
will yield a negative prediction

</li>
</ul>
</li>



<li>

*Learning Rule*<br />

Now we know when the data point belong to negative class and when it belongs to positive class, using this information 
we can keep on updating the weight vector `w` whenever we make a wrong prediction until we find a seperating hyperplane<br />
The rule says $$yi*w^T* x <= 0$$ we update the vector `w`  $$vecw = vecw + y * vecx$$
When ever the algorithm miss classifies a positive point we add `x` to `w`, this translates to, the algorithm is trying
to decrease the `theta` between `w` and the `data point` when positive point is miss classified and will 
increase the `theta` between `w` and the `data point` when negative point is miss classified 

 
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