---
layout: post
title:  "Perceptron Classifier in Python"
comments: true
description: "Implementing the Perceptron classifier from scratch in python"
date:   2020-07-22
permalink: /perceptron-python/
author: Fuzail Palnak
---

This post will implement the perceptron classifier in python from scratch, this post will cover how to implement the 
Classifier and do not look at the theoretical specifics, have a look at [this](https://fuzailpalnak.github.io/perceptron/) post if you are interested in
understanding how the perceptron classifier works

- Necessary imports 
```python
import numpy as np
import matplotlib.pyplot as plt
```
- Defining Variables
```python
class Perceptron:
    def __init__(self):
        self.w = np.array([0.0, 0.0, 0.0]) # Initial weight value
        self.data = list() # Initialize Empty Data list
        self.labels = list() # Initialize Empty Label list
        self.positive = None # Initialize positive point
        self.negative = None # Initialize negative point
        self.populate_data() # A gui interface to get data from user
```

- Get Positive and Negative Data Points, This block of code will load a Interactive GUI which will wait for user to 
provide input, The bias term b is absolved by the data making the data one dimensional higher than the user provided input, making
$\text{w } = \begin{bmatrix}w_{1}  \\w_{2} \\b  \end{bmatrix}$ and 
$\text{data } = \begin{bmatrix}xcoordinate \\ycoordinate \\1  \end{bmatrix}$
```python

    def populate_data(self):
        self.positive, self.negative = self.get_data(5)

        for i in range(len(self.positive)):
            # Absolving the bias term and setting it to 1
            data = [self.positive[i][0], self.positive[i][1], 1]
            self.data.append(np.array(data))
            self.labels.append(1)
        for i in range(len(self.negative)):
            data = [self.negative[i][0], self.negative[i][1], 1]
            self.data.append(np.array(data))
            self.labels.append(-1)

    def get_data(self, number_of_points):
        plt.clf()
        plt.setp(plt.gca(), autoscale_on=False)
        positive = self.plt_data_t("Positive Class", number_of_points)
        negative = self.plt_data_t("Negative Class", number_of_points)

        plt.title("DATA", fontsize=10)
        plt.scatter(positive[:, 0], positive[:, 1], marker="o")
        plt.scatter(negative[:, 0], negative[:, 1], marker="x")

        plt.draw()
        plt.show()
        return positive, negative
```
- Positive Data Points Input
![png]({{ site.url }}/assets/perceptron_python/positive.png)

- Negative Data Points Input
![png]({{ site.url }}/assets/perceptron_python/negative.png)

- Positive Data Points are circle and Negative Data Points are crosses
![png]({{ site.url }}/assets/perceptron_python/pos_neg.png)

        
- Training the Perceptron classifier, it is combination of two rules `decision rule` and the `learning rule`

    - *Decision Rule :-* $w^T * x$ 
    ```python
        @staticmethod
        def __decision_rule(w, x):
            """
            data points above the hyperplane will be positive as the theta will be [0, 90] with respect to self.w
            and points below the hyperplane will be negative
            
            :param w:
            :param x:
            :return:
            """
            return np.dot(w, x)
    ```

    - *Learning Rule* $$w =\begin{cases}w & y * w^T * x > 0\\w = w + y * x & y * w^T * x <= 0\end{cases}$$  
```python
    @staticmethod
    def __update(w, x, y):
        """
        if Point belong to -1 class then w = w - x
        if point belong to +1 class then w = w + x
        The main objective here is to increase the cos(alpha) between the weight vector and the positive data points
        and decrease the cos(alpha) for negative points

        :param w:
        :param x:
        :param y:
        :return:
        """
        w += y * x
        return w
```
  
- Train the classifier using the `Learning Rule` and `Decision Rule`, the classifier will loop until it finds the
hyperplane
```python
    def train(self):
        step = 0
        while True:
            miss_classified = 0
            for iterator in range(len(self.data)):
                x = self.data[iterator]
                y = self.labels[iterator]

                if self.__decision_rule(self.w, x) * y <= 0:
                    # Miss classified the data point and adjust the weight
                    w_prev = self.w
                    self.w = self.__update(self.w, x, y)
                    miss_classified = miss_classified + 1
                    print(
                        "Adjusting Weight from w: {} to w_new: {}".format(
                            tuple(w_prev), tuple(self.w)
                        )
                    )
            # self.plt_decision_boundary()
            step += 1
            if miss_classified == 0:
                # if no miss classified then the perceptron has converged and found a hyperplane
                print("Perceptron Converged on Step : {}".format(step))
                break
```

- Run
```python
p = Perceptron()
p.train()
p.plt_decision_boundary()
```
![png]({{ site.url }}/assets/perceptron_python/decision.png)







