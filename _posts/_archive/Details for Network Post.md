```python
import matplotlib.pyplot as plt
import networkx as nx
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification

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


    
![png](output_0_0.png)
    



```python
karate_data = StellarGraph.from_networkx(karate)
print(karate_data.info())
```

    StellarGraph: Undirected multigraph
     Nodes: 34, Edges: 78
    
     Node types:
      default: [34]
        Features: none
        Edge types: default-default->default
    
     Edge types:
        default-default->default: [78]
            Weights: all 1 (default)
            Features: none


The walk length employed for the random walk will be kept small due to the small size of our graph (see above):


```python
import keras
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

    link_classification: using 'dot' method to combine node embeddings into edge embeddings
    Epoch 1/2
    41/41 [==============================] - 3s 8ms/step - loss: 0.7325 - binary_accuracy: 0.4928
    Epoch 2/2
    41/41 [==============================] - 0s 7ms/step - loss: 0.7327 - binary_accuracy: 0.5007



```python
x_inp_src = x_inp[0]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
#node_gen = Node2VecNodeGenerator(karate_data, batch_size).flow(subjects.index)
#node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)



```


```python
transform = TSNE  # PCA

trans = transform(n_components=2)
#node_embeddings_2d = trans.fit_transform(node_embeddings)

```


```python
dataset = datasets.Cora()
display(HTML(dataset.description))

G, subjects = dataset.load(largest_connected_component_only=True)
type(subjects)
```


The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.





    pandas.core.series.Series



# Fraud Data Set


```python
import pandas as pd
```

Import the data:


```python
classes = pd.read_csv("/home/peter/Downloads/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
edges = pd.read_csv("/home/peter/Downloads/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
features = pd.read_csv("/home/peter/Downloads/elliptic_bitcoin_dataset/elliptic_txs_features.csv")
```

This data-set contains three different pieces. The classes represents the data in three different ways:

1. If class = 1, it is illicit.
2. If class = 2, it is licit.
3. If class = unknown, then we don't know the label.

I am going to narrow this data down to just the labeled nodes since this is meant to be a numerical exercise to explore node embeddings.


```python
edges
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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




```python
classes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>txId</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230425980</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5530458</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>2</th>
      <td>232022460</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>3</th>
      <td>232438397</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>230460314</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>203764</th>
      <td>173077460</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>203765</th>
      <td>158577750</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>203766</th>
      <td>158375402</td>
      <td>1</td>
    </tr>
    <tr>
      <th>203767</th>
      <td>158654197</td>
      <td>unknown</td>
    </tr>
    <tr>
      <th>203768</th>
      <td>157597225</td>
      <td>unknown</td>
    </tr>
  </tbody>
</table>
<p>203769 rows × 2 columns</p>
</div>



The features data-frame contains addition information that has been anonymized above each transaction. We will omit the same rows as we did for the original data but our main focus in this work will be concentrating on learning what we can from the local neighborhood, not the transaction details.


```python
classes[classes["class"]!="unknown"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>txId</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>232438397</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>232029206</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>232344069</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>27553029</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3881097</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>203752</th>
      <td>80329479</td>
      <td>2</td>
    </tr>
    <tr>
      <th>203754</th>
      <td>158406298</td>
      <td>2</td>
    </tr>
    <tr>
      <th>203759</th>
      <td>158375075</td>
      <td>1</td>
    </tr>
    <tr>
      <th>203763</th>
      <td>147478192</td>
      <td>2</td>
    </tr>
    <tr>
      <th>203766</th>
      <td>158375402</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>46564 rows × 2 columns</p>
</div>




```python
classes["ID"] = classes["txId"]
edges["ID"] = edges["txId1"]
merged_data = pd.merge(edges,classes, how = "left", left_on = "txId1", right_on = "txId")
merged_data = merged_data[["txId1","txId2","class"]][merged_data["class"] != "unknown"]

```


```python
merged_data["txId1"] = merged_data["txId1"].astype(str)
merged_data["txId2"] = merged_data["txId2"].astype(str)
merged_data = merged_data.rename(columns={"txId1": "source", "txId2": "target", "class": "class"}) 
merged_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>target</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>232344069</td>
      <td>27553029</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3881097</td>
      <td>232457116</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>232051089</td>
      <td>232470704</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>232033533</td>
      <td>230470022</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26</th>
      <td>230473487</td>
      <td>7089694</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Examine the class imbalance for our dataset which shows that the instances of illicit transactions make up about 7% of the categorized data:


```python
merged_data["class"].value_counts()

merged_data[["source","target"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>232344069</td>
      <td>27553029</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3881097</td>
      <td>232457116</td>
    </tr>
    <tr>
      <th>15</th>
      <td>232051089</td>
      <td>232470704</td>
    </tr>
    <tr>
      <th>22</th>
      <td>232033533</td>
      <td>230470022</td>
    </tr>
    <tr>
      <th>26</th>
      <td>230473487</td>
      <td>7089694</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>234338</th>
      <td>194020062</td>
      <td>47521535</td>
    </tr>
    <tr>
      <th>234340</th>
      <td>158574502</td>
      <td>109383451</td>
    </tr>
    <tr>
      <th>234344</th>
      <td>158594124</td>
      <td>157631640</td>
    </tr>
    <tr>
      <th>234347</th>
      <td>157631640</td>
      <td>21644119</td>
    </tr>
    <tr>
      <th>234350</th>
      <td>158365409</td>
      <td>157930723</td>
    </tr>
  </tbody>
</table>
<p>53198 rows × 2 columns</p>
</div>




```python
test =StellarGraph(merged_data[["source","target"]])
print(test.info())
```

    StellarGraph: Undirected multigraph
     Nodes: 53198, Edges: 0
    
     Node types:
      default: [53198]
        Features: float32 vector, length 2
        Edge types: none
    
     Edge types:


### Generate node embeddings:

### Traing model


```python
G = nx.Graph()
G = nx.from_pandas_edgelist(merged_data, 'source', 'target')
leaderboard = {}
for x in G.nodes:
    leaderboard[x] = len(G[x])
s = pd.Series(leaderboard, name='connections')
df2 = s.to_frame().sort_values('connections', ascending=False)
df2

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>connections</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2984918</th>
      <td>473</td>
    </tr>
    <tr>
      <th>89273</th>
      <td>289</td>
    </tr>
    <tr>
      <th>3181</th>
      <td>113</td>
    </tr>
    <tr>
      <th>7952</th>
      <td>100</td>
    </tr>
    <tr>
      <th>565334</th>
      <td>91</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>372696454</th>
      <td>1</td>
    </tr>
    <tr>
      <th>194125075</th>
      <td>1</td>
    </tr>
    <tr>
      <th>313417088</th>
      <td>1</td>
    </tr>
    <tr>
      <th>372726671</th>
      <td>1</td>
    </tr>
    <tr>
      <th>158365409</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>53904 rows × 1 columns</p>
</div>




```python
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


### Model Results


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


```python
source_vals = merged_data[["source","class"]].drop_duplicates()
source_vals.columns = ["Node_ID","class"]
target_vals = merged_data[["target","class"]].drop_duplicates()
target_vals.columns = ["Node_ID","class"]
testing = source_vals.append(target_vals)
testing = testing.drop_duplicates(subset = "Node_ID",keep = "first")
node_gen = Node2VecNodeGenerator(test, batch_size).flow(test.nodes())
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

X = node_embeddings
# y holds the corresponding target values
y = np.array(testing["class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, test_size=None)
print(
    "Array shapes:\n X_train = {}\n y_train = {}\n X_test = {}\n y_test = {}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape
    )
)

```

# Logistic regression


```python
clf = LogisticRegressionCV(
    Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=300
)
clf.fit(X_train, y_train)
```




    LogisticRegressionCV(cv=10, max_iter=300, multi_class='ovr', scoring='accuracy',
                         verbose=False)




```python
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.9308447046213464




```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred)
```




    0.5



## Adaboost implementation:




```python
from sklearn.ensemble import AdaBoostClassifier
test_model = AdaBoostClassifier()
test_model.fit(X_train, y_train)
y_pred = test_model.predict(X_test)
accuracy_score(y_test, y_pred)
```


```python
roc_auc_score(y_test, y_pred)
```




    0.5004129967804765


