---
layout: post
title:  "A short explanation on geometric deep learning"
date:   2021-05-06 11:34:12 -0400
categories: programming
---

The last few days, I've been reviewing the new proto-book on geometric deep learning by Michael Bronstrain, Joan Bruna, Taco Cohen, and Petar Veličković. Ignoring the bibliography and other end-matter, the book is about 127 pages, and it leans on a foundation on group theory and functional analysis which make it more dense than many machine learning papers from an analytical perspective. This subfield is very new for me, but I wanted to quickly discuss the motivation for bringing in more abstract mathematical concepts into the domain of machine learning.

For anyone that doesn't want the deep dive of their book, the authors have also written this [blog post](https://towardsdatascience.com/geometric-foundations-of-deep-learning-94cdd45b451d) and given a [keynote](https://www.youtube.com/watch?v=9cxhvQK9ALQ) summarizing the subject. There are also a few earlier papers discussing the topic.
 
## Why geometric deep learning is intriguing

Machine learning typically works out to be some kind of optimization problem. This is true from linear regression to neural networks and the only major difference is what form the model parameters finally take and the steps used to guarantee convergence. However, objects like network structures pose a problem for this formulation since they can't be easily represented in the space that we normally solve these machine learning problems. 

Consider the case where we have a network of banking transactions consisting of a weighted digraph. We would have to find a way to map the following information:

1. The adjacency list of each node.
2. The weighting of each edge.
3. The direction of each edge.
4. The features for each node.

Some methods seek to use a low dimensional representation of the network structure in linear space which can then be used as features for the problem. An example of this would be embedding techniques.  Another alternative is to use the adjacency matrix but this is an inefficient way to code the information since it leads to a very sparse set of features that take up a large amount of memory. 

Instead of transforming and approximating the data, geometric deep learning seeks to redefine the space the data is being used on in a way that preserves its structure. A fantastic example and visualization of what this looks like is provided in this [blog post](https://dawn.cs.stanford.edu/2019/10/10/noneuclidean/) where we can see that we can solve a decision tree as a continuous optimization problem by looking at it as a problem in hyperbolic space! 

One major benefit of considering the geometric space of the application is that it can help avoid the curse of dimensionality. When we consider a problem in Euclidean space, the number of data-points required to avoid sparsity increases exponentially in relation to the number of dimensions of the problem being studied.  

## Libraries/implementations

From my own searches, the main implementation work in geometric deep learning is [Pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/). I also saw that there was an implementation in Julia called [GeometricFlux](https://github.com/FluxML/GeometricFlux.jl) in Julia that is based on that language's Flux library. There isn't any implementation that I could find in R. Pytorch exists in the more mature ecosystem that Python provides, but I find the Flux library appealing because it's only written in Julia which makes understanding the code easier.

I'll add that there are more libraries for specific applications such as [Stellargraph](https://stellargraph.readthedocs.io/en/stable/README.html) for graph deep learning, and [Graphein](https://graphein-graphein.readthedocs-hosted.com/en/latest/) for protein graphs.
