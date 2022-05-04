---
title: "The link between text prediction and classification in social networks"
date: 2021-04-22
---


Social networks like Facebook friend groups, citation rings, or transactions between accounts are notably difficult to imagine and model with conventional machine learning algorithms. Unlike simple classification, or image recognition, the problem we are trying to solve doesn't lend itself to being imagined as a neat table. It's what is known as a non-Euclidean problem as described in [Geometric deep learning: going beyond Euclidean data](https://arxiv.org/abs/1611.08097):

> Many scientific fields study **data with an underlying structure that is a non-Euclidean space.** Some examples include social networks in computational social sciences, sensor networks in communications, functional networks in brain imaging, regulatory networks in genetics, and meshed surfaces in computer graphics.

This brings researchers into a space that's foreign to what much of the theory in statistics was originally designed for. Using conventional machine learning algorithms for these problems demands additional steps to reformulate the problem. Detection of money laundering is one of these problem spaces. However, by borrowing the concept of vector embedding from natural language processing, we can transform network data into a format which is usable for a conventional classification algorithm (logistic regression for instance).

To begin, we'll introduce the concept of word embeddings and Google's word2vec algorithm. Then, we will review how this method can be extended to the space of social networks.

## Word2vec

Consider the machine learning problem of text prediction. One strategy that has become popular since its [introduction](https://arxiv.org/abs/1301.3781) in 2013 uses a neural network to map words and their contexts into a Euclidean feature space. By using this transformation method, the text corpus we are using can be represented as a set of $N$ dimensional vectors where words with similar contexts are grouped closely together in the new feature space.

To quickly map out this methodology, I'll summarize the steps below, but for an in-depth treatment I recommend the original paper linked earlier, the [Tensorflow documentation](https://www.tensorflow.org/tutorials/text/word2vec), and [Word2Vec Explained](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/) which describe the method in more detail. If you want to dive deep into the step-by-step calculation, the spreadsheet based numerical example [here](https://docs.google.com/spreadsheets/d/1mgf82Ue7MmQixMm2ZqnT1oWUucj6pEcd2wDs_JgHmco/edit#gid=0) is a great introduction.

### Step 1: Creating associations between words:

We will begin by defining text that we want to process and analyze which we will call the corpus. Our first step will be breaking up a corpus into windows of length $W$ containing a set of words. For example, if we break up "The quick red fox jumped over the lazy brown dog." into windows of 4, we would get the following:
 

![](https://raw.githubusercontent.com/poc1673/poc1673.github.io/main/embeddings0.png)

This step lets us treat each window as a specific data-point containing a set of words together. We can begin to see how this could be used to create a crude "correlation" based on whether groups of words appear with one another in a specific window.

### Generate one-hot encoders for each word and context:

Next, we need to encode the association of each unique word in our corpus with its respective context words. To do this, we review the window that each word occurs withinin. For instance, let's choose "red" to be the word we examine. The window we will examine is:

|     |     |        |      |
|-----|-----|--------|------|
| red | fox | jumped | over |

In the above example, "red" is the **target word** and the other words that appear in the window are **context words.** We identify the context words that occur with each word in the corpus and then represent them mathematically as one-hot encoder vectors. For this application, each vector will have as many entries as unique words in the corpus. One vector will be used to represent the target word, and another set will be created which will represent the context words.


![](https://raw.githubusercontent.com/poc1673/poc1673.github.io/main/embeddings1.png)


The vectors shown above will be used to train a shallow neural network.

### Train a shallow neural network with word vectors as input

From here, we train a simple neural network one the word representation vectors described above. The network being used to train the association between target words and the context words uses two layers

As a diagram, it appears as follows:

![](https://raw.githubusercontent.com/poc1673/poc1673.github.io/main/embeddings2.png)

Where $v$ is the number of dimensions we are choosing for the vector embedding output.

The strategy that's being used here departs from what we are used to seeing for neural networks in that **we aren't going to use the prediction layer**. Instead, we will use the prediction layer in the process of training the model, and instead extract the weights from the second layer of the neural network. This network will be trained for every target word and the output weightings will be the **vector embedding** of our target word. Aside from providing information on details like what words are likely to occur next, this also has some intriguing properties like coding some elements of the "meaning" of some words together (IE Subtracting the vector for "man" from "king" and adding "woman" might produce a vector which has a cosine distance that is closest to "queen").

To see a detailed step-by-step explanation on how the network is trained, I recommend [Derek Chia's walkthrough](https://docs.google.com/spreadsheets/d/1mgf82Ue7MmQixMm2ZqnT1oWUucj6pEcd2wDs_JgHmco/edit?ouid=104435606536692026625&usp=sheets_home&ths=true) in Google sheets.

## Imagining a graph as a sentence

Word2vec has some very appealing properties, but how can we use it for something as different as graphs? As it turns out, this requires an additional step, but the principle remains very similar. Typically a graph can be imagined to appear like the figure below where a set of nodes where direct relationships are represented by connecting nodes together with lines (edges).:

```{python}
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
karate = nx.karate_club_graph()
club_labels = nx.get_node_attributes(karate,'club')
circ_pos = nx.circular_layout(karate)
karate_communities = list(greedy_modularity_communities(karate))
community_0 = sorted(karate_communities[0])
all_communities =  sorted(karate_communities[1]) + sorted(karate_communities[2])
nx.draw_networkx_nodes(karate, circ_pos, nodelist=community_0, node_color='b', alpha=0.5)
nx.draw_networkx_nodes(karate, circ_pos, nodelist=all_communities, node_color='r', alpha=0.5)
nx.draw_networkx_edges(karate, circ_pos,style='dashed',width = 0.5)
#nx.draw_networkx_labels(karate, circ_pos, club_labels, font_size=9)
plt.show()
```


![](https://raw.githubusercontent.com/poc1673/poc1673.github.io/main/embeddings3.png)

It turns out that we can make a network graph like the one above suitable for use in the Word2Vec algorithm by employing a random walk across the node structure. A simple version of this random walk was introduced in [DeepWalk](https://arxiv.org/abs/1403.6652) where truncated random walks were used to represent the context of each node in the network. [Node2Vec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) came out a few years later and used a more flexible methodology of representing the random walk which allows the user to weight how quickly the random walk moves away from the initial node, and what the likelihood of it stepping backwards within the random walk. This allows the user to tune the hyperparameters for the vector embedding based on how much locality is desired in the graph results. The node lists can then be used in the exact same manner as the target-context work representation was used to create vector embedding of the network.

 
