---
title: "Generating graph embeddings for the Elliptic bitcoin data-set"
date: 2021-04-29
---

# Introduction

Last week I [covered](https://poc1673.github.io/2021/04/22/The-link-between-text-prediction.html)  providing a high level introduction to the concept of node embeddings for network graphs and their origin in natural language processeing. To summarize from the past post, most conventional methods being applied to machine learning rely on the data being in a neat tabular format. However, network data doesn't lend itself to this facile representation. This typifies a space called non-Euclidean problems which are defined [here](https://arxiv.org/abs/1611.08097) as:

> Many scientific fields study data with an underlying structure that is a non-Euclidean space. Some examples include social networks in computational social sciences, sensor networks in communications, functional networks in brain imaging, regulatory networks in genetics, and meshed surfaces in computer graphics.

In this post, I am going to explore the Node2vec algorithm as it's implemented in the Stellargraph library. This will focus on the steps of the implementation described in my last post and end show the resulting features of the algorithm. We'll begin with a trivial example illustrating the method for Zachary's karate club data-set After that, we'll use the same steps on the Elliptic bitcoin data-set which is a classification problem for identifying illicit transactions. The strategies used here to generate the feature vectors will eventually be used to train a machine learning model to identify illicit transactions.

# Zachary's karate club

[Zachary's karate club](http://networkrepository.com/soc-karate.php) data-set fills the same role in graph machine learning that the iris data-set occupies for classification problems. It's a simple, usable data-set which can be used for network analysis. The origin of the data is [An Information Flow Model for Conflict and Fission in Small Groups](http://www1.ind.ku.dk/complexLearning/zachary1977.pdf) where it is used to analyze the breakup of a karate club.

## Generating node embeddings:

For this post, we will be relying on the [Stellargraph](https://www.stellargraph.io/) and [NetworkX](https://networkx.org/) packages to do the bulk of this work. NetworkX has a few native data-sets (including the karate club data-set) as well as useful utilities to turn a dataframe into a graph object. We will begin this warmup by importing the necessary packages and the karate club data:

```python
import matplotlib.pyplot as plt
import networkx as nx
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification
import keras

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from stellargraph import datasets
from IPython.display import display, HTML

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

We can see a brief chart of the relationships below:
![](https://raw.githubusercontent.com/poc1673/poc1673.github.io/main/embeddings3.png)
 
 We can review the network results below:
 
```python
karate_data = StellarGraph.from_networkx(karate)
print(karate_data.info())
```

```python
walk_count = 15
walk_length = 3
karate_walk = BiasedRandomWalk(
    karate_data,
    n=walk_count,
    length=walk_length,
    p=0.5,  
    q=2.0, 
)
unsupervised_samples = UnsupervisedSampler(karate_data, nodes=list(karate_data.nodes()), walker=karate_walk)
batch_size = 50
epochs = 2
generator = Node2VecLinkGenerator(karate_data, batch_size)
emb_size = 5
node2vec = Node2Vec(emb_size, generator=generator)
x_inp, x_out = node2vec.in_out_tensors()
predictions = prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)
```

The results above show the basic coding steps of using this method. From here, let's move to a less trivial and larger scale example.

# Elliptic data-set

From this simple we move to the [Elliptic](https://www.kaggle.com/ellipticco/elliptic-data-set) data-set. This is a data-set available on Kaggle which was originally used in this paper by [Weber et al.](https://arxiv.org/pdf/1908.02591.pdf) paper. It contains three files:

* Classes - the category for each node. There are three outcomes: 1 (illicit transactions), 2 (licit), and unknown (unlabeled). In this exercise, we will only focus on the known transactions. 
* Edgelist - the source and target of each transaction.
* Features - additional details on each transaction for analysis

Since we are only concentrating on the node embedding aspect of this, we will omit additional features from analysis in favor of focusing on only the graph.


```python
classes = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
edges = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
features = pd.read_csv("elliptic_bitcoin_dataset/elliptic_txs_features.csv")
```

We can see a breakdown on the number of edges associated with each node in the table below:

```python
edges
``` 

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>txId1</th>
      <th>txId2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230425980</td>
      <td>5530458</td>
    </tr>
    <tr>
      <th>1</th>
      <td>232022460</td>
      <td>232438397</td>
    </tr>
    <tr>
      <th>2</th>
      <td>230460314</td>
      <td>230459870</td>
    </tr>
    <tr>
      <th>3</th>
      <td>230333930</td>
      <td>230595899</td>
    </tr>
    <tr>
      <th>4</th>
      <td>232013274</td>
      <td>232029206</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>234350</th>
      <td>158365409</td>
      <td>157930723</td>
    </tr>
    <tr>
      <th>234351</th>
      <td>188708874</td>
      <td>188708879</td>
    </tr>
    <tr>
      <th>234352</th>
      <td>157659064</td>
      <td>157659046</td>
    </tr>
    <tr>
      <th>234353</th>
      <td>87414554</td>
      <td>106877725</td>
    </tr>
    <tr>
      <th>234354</th>
      <td>158589452</td>
      <td>158589457</td>
    </tr>
  </tbody>
</table>
<p>234355 rows × 2 columns</p>
</div>


## Generating node embeddings

 We begin by using the NetworkX package to reformat these dataframes into a graph:
 
 
```python
classes["ID"] = classes["txId"]
edges["ID"] = edges["txId1"]
merged_data = pd.merge(edges,classes, how = "left", left_on = "txId1", right_on = "txId")
merged_data = merged_data[["txId1","txId2","class"]][merged_data["class"] != "unknown"]
merged_data["txId1"] = merged_data["txId1"].astype(str)
merged_data["txId2"] = merged_data["txId2"].astype(str)
merged_data = merged_data.rename(columns={"txId1": "source", "txId2": "target", "class": "class"})
G = nx.Graph()
G = nx.from_pandas_edgelist(merged_data, 'source', 'target')
test = StellarGraph.from_networkx(G)
print(test.info())

```

    StellarGraph: Undirected multigraph
     Nodes: 53904, Edges: 53198
    
     Node types:
      default: [53904]
        Features: none
        Edge types: default-default->default
    
     Edge types:
        default-default->default: [53198]
            Weights: all 1 (default)
            Features: none
     
Here, I use a fairly short walk due to the small number of connections being made between each node. I am only generating five walks for each node, and I am setting the embedding size to be output by the model to 4. To These could definitely be optimized (particularly the number of walks being done, and the number of embeddings(20) , but for the purpose of a small-scale numerical example being done on a home laptop, this is acceptable.

From here, we run the results. This particulary example took about an hour.

```python
walk_count = 5
walk_length = 4
karate_walk = BiasedRandomWalk(
    test,
    n=walk_count,
    length=walk_length,
    p=0.5,  
    q=2.0, )

unsupervised_samples = UnsupervisedSampler(test, nodes=list(test.nodes()), walker=karate_walk)
batch_size = 32
epochs = 3
generator = Node2VecLinkGenerator(test, batch_size)
emb_size = 20
node2vec = Node2Vec(emb_size, generator=generator)
x_inp, x_out = node2vec.in_out_tensors()
predictions = prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],)

history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)
```

The next piece is a step we are using to label class for each node ID:

```python
x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
unique_ids = np.unique( list(merged_data["source"]) + 
                     list(merged_data["target"])      )
unique_ids = pd.DataFrame({"Node_ids":unique_ids })
unique_ids = pd.merge(unique_ids,merged_data, how = "left", left_on = "Node_ids", right_on = "source")
unique_ids = pd.merge(unique_ids,merged_data, how = "left", left_on = "Node_ids", right_on = "target")
unique_ids
```

Finally, we output the node embeddings using the Elliptic data-set and the class assignment performed directly above:


```python
source_vals = merged_data[["source","class"]].drop_duplicates()
source_vals.columns = ["Node_ID","class"]
target_vals = merged_data[["target","class"]].drop_duplicates()
target_vals.columns = ["Node_ID","class"]
testing = source_vals.append(target_vals)
testing = testing.drop_duplicates(subset = "Node_ID",keep = "first")
node_gen = Node2VecNodeGenerator(test, batch_size).flow(test.nodes())
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
```
If we glance at the node embeddings, we'll see that we have received the mapped results from the node embedding algorithm we trained above:

```python
pd.DataFrame(node_embeddings).head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.417580</td>
      <td>0.923112</td>
      <td>0.299836</td>
      <td>0.121735</td>
      <td>0.502202</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.616742</td>
      <td>-0.292056</td>
      <td>-1.011988</td>
      <td>-0.769103</td>
      <td>0.002047</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.865584</td>
      <td>-0.215785</td>
      <td>-0.755236</td>
      <td>0.937466</td>
      <td>-0.173979</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.942629</td>
      <td>0.396682</td>
      <td>1.237774</td>
      <td>-1.136209</td>
      <td>0.697967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.197375</td>
      <td>0.557650</td>
      <td>-0.392052</td>
      <td>0.932329</td>
      <td>0.516879</td>
    </tr>
  </tbody>
</table>
</div>



# Is this it?

This was purely a numerical example of how to generate the features but I haven't taken the time to build a model around them yet! There is still work to do:

1. We have a set of features for this problem but we haven't considered the hyperparameters being used to train the node embeddings.
2. We must also consider the feature set that's provided alongside the network information.
3. After this, we still have to approach the modeling methodology that is actually selected in the end and assess its performance.
