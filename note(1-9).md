## Node Embeddings

**Snapshot**

Node embeddings map nodes into an embedding space. Similarity of embeddings between nodes indicates their similarity in the network. Because node embeddings encode network information, it is potentially used for many downstream tasks. 

**Random walk approaches**

Given a graph and a starting point, random walk selects a neighbor at random and move to this neighbor. We keep moving and get a random sequence, which is a random walk on the graph. 

The approach hypothesizes that if two nodes co-occur on a random walk, then their node embeddings should be similar. 

Define vector $z_u$ as the embedding of node $u$. The probability $P(v|z_u)$ is the predicted probability of visiting node $v$ on random walks starting from node $u$. 

The approach maximizes the following log-likelihood: 
$$
\arg \max_z \sum_{u\in V}\sum_{v\in N(u)} \log P(v|z_u)
$$
where $N(u)$ is the neighbor that on the random walk starting from $u$. Equivalently: 
$$
\arg \min_z\sum_{u\in V}\sum_{v\in N(u)} - \log P(v|z_u)
$$
Parameterize $P(v|z_u)$ using softmax: 
$$
P(v|z_u) = \frac{\exp(z_u^T z_v)}{\sum_{n\in V} \exp(z_{u}^T z_n)}
$$
However, doing softmax naively is too expensive. Instead of normalizing with respect to all nodes, just normalize against $k$ random "negative samples" $n_i$ : 
$$
P(v|z_u)\ \approx \ \log(\sigma (z_u^Tz_v)) + \sum_{i=1}^k \log(1-\sigma(z_u^Tz_{n_i}))
$$
where $\sigma$ is the sigmoid function and node $n_i$ is sampled with probability, proportional to its degree. Higher $k$ gives more robust estimates but corresponds to higher bias on negative events. In practice $k = 5\sim 20$. 

Summary: 

- Run short fixed-length random walks starting from each node on the graph. 
- For each node $u$ collect $N(u)$
- Optimize embedding $z$ using gradient descent. 

**node2vec: Biased Walks**

Key observation: flexible notion of network neighbor $N(u)$ leads to rich node embeddings. Use more flexible, biased random walk that can trade of between local and global views of the network. 

Recall two classic walking strategies --- BFS and DFS. BFS will provide a micro-view of neighborhood while DFS provides a macro-view of neighborhood. 

Let $q$ be the ratio of BFS(moving inwards) vs. DFS(moving outwards). Simultaneously, we define $p$ as the probability of returning back to the previous walk: 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240313163330956.png" width = 550>

In the diagram above, $S_1$ is the previous node and $w$ is the current node. We walk to $S_2$ with an unnormalized prob $1$ (BFS, same distance to $S_1$) and walk to $S_3$ or $S_4$ with an unnormalized prob $1/q$ (DFS, away from $S_1$). 

The walk strategy is a replacement for unbiased random walk. But no method wins in all cases. In general, we must choose definition of node similarity that matches your application. 

## Graph Neural Network

**GCN**

GCN (graph convolutional network) averages neighbor message and apply a neural network. The basic update function of GCN goes like this ($N(v)$ is the first-order neighbor of node $v$): 
$$
h_v^{(k+1)} = \sigma(W_k\sum_{u\in N(v)} \frac{h_u^{(k)}}{|N(v)|} + B_kh_v^{(k)})\ , \ \forall \ k\in\{0,...,K-1\}
$$
where $h_v^{(k)}$ is the hidden embedding of $v$ at layer $k$. $W_k$ and $B_k$ are all learnable parameter in the neural network. After $K$ layers of calculation, we will take $h_{v}^{K}$ as the node embeddings. 

Let $D$ be diagonal matrix where $D_{v,v} = \deg(v) = |N(v)|$. The inverse matrix $D^{-1}$ is also diagonal where $D_{v,v}^{-1} = \frac{1}{|N(v)|}$. Rewrite update function in matrix form: 
$$
H^{k+1} = \sigma(D^{-1}AH_kW_k^{T}+H_kB_k^T)\ ,  \ \forall \ k\in\{0,...K-1\}
$$
where $A$ is adjacent matrix. $\widetilde{A} = D^{-1}A$ is sparse, which implies that efficient sparse matrix multiplication can be used. 

Given a node, the GCN that computes its embedding is permutation invariant. That means for any isomorphic graphs, their node embedding set will be the same. 

Here is the implementation of GraphSage. The `propagation` function encapsulates the `message` and `aggregation` function, which can be overloaded: 

```python
class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_l = nn.Linear(self.in_channels, self.out_channels) 
        self.lin_r = nn.Linear(self.in_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        out = None
        # propagate = message + aggregation + update
        prop = self.propagate(edge_index, x = (x, x), size = size) 
        out = self.lin_l(x) + self.lin_r(prop)
        # apply L2 normalization if it is required. 
        if self.normalize: 
            out = F.normalize(out, p = 2)
        return out

    def message(self, x_j):
        # the message from neighbors
        out = x_j
        return out

    def aggregate(self, inputs, index, dim_size = None):
        node_dim = self.node_dim
        # add up inputs along dim = node_dim with respect to index 
        out = torch_scatter.scatter(inputs, index, node_dim, out,
                                    dim_size = dim_size, reduce = 'mean')
        return out
```

**Graph Attention Network**

Not all node's neighbors are equally important. But GCN assigns an equal attention weight $a_{vu}=\frac{1}{|N(v)|}$ to each node (In our following discussion, $N(v)$ includes $v$ itself). 

Inspired by cognitive attention, we can compute attention weight $a_{vu}$ which focuses on the important part of input data and fades out the rest. 

Let $a_{vu}$ be computed as a byproduct of an **attention mechanism $a$**: 
$$
e_{vu} = a(W^{(l)}h_{u}^{(l-1)}, W^{(l)}h_{v}^{(l-1)})\\
a_{vu}  = \frac{\exp(e_{vu})}{\sum_{k\in N(v)} \exp(e_{vk})}
$$
Then compute the weighted sum based on the final attention weight $a_{vu}$: 
$$
h_{v}^{(l)} = \sum_{u\in N(v)} a_{vu} W^{(l)}h_{u}^{(l-1)}
$$
The attention mechanism $a$ is a single-layer feedforward neural network, parameterized by a weight vector $\vec a\in \mathbb R^{F}$ (where $F$ is the size of hidden state): 
$$
e_{vu} = \text {LeakyRelu}(\vec a^{T} [Wh_v || Wh_u])
$$
where $||$ is the concatenate operation. 

Multi-head attention can be applied, stabilizing the learning process of attention mechanism. But different from multi-head attention in transformer, the approach creates multiple attention scores(each replica with a different set of parameters) instead of splitting them into pieces: 
$$
h_{v}' = \sigma\Big(\frac{1}{T}\sum_{t=1}^{T}\sum _{u\in N(v)} a_{vu}^{t}W^{t}h_v\Big)\\
$$
It is a shared edge-wise mechanism and doesn't depend on the global graph structure. Sparse matrix operations do not require more than $O(V+E)$ entries to be stored. 

**Relation GCN**

Heterogeneous graph is graph with node and edge types. Up to now we are all discussing homogeneous graphs. We will extend GCN to handle heterogeneous: 
$$
h_{v}^{(l+1)} = \sigma (\sum_{r\in R}\sum _{u\in N_v^r} \frac{1}{|N_v^r|}W_r^{(l)}h_u^{(l)} + W_{0}^{(l)}h_v^{(l)})
$$
The size of each $W_r^{(l)}$ is $d^{(l+1)}\times d^{(l)}$. Due to rapid growth of the number of parameters with respect to number of relations, overfitting becomes an issue. Here are two approaches to regularize the weights of $W_r^{(l)}$: 

Use block diagonal matrices. If use $B$ low-dimensional matrices, then parameters of $W_r^{(l)}$ can be reduced to $B\times \frac{d^{(l+1)}}{B}\times \frac{d^{(l)}}{B}$ 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240315200727728.png" width = 400>

The second approach is basis learning. Represent the matrix of each relation as a linear  combination of basis transformations: $W_r = \sum_{b=1}^B a_{rb}\cdot V_b$, where $V_b$ is shared across all relations. 

Now each relation only needs to learn $\{a_{rb}\}_{b=1}^B$, which is B scalars. 

## Knowledge Graph

We'll let a knowledge graph $G=(E,S,R)$ consist of the set of entities $E$ (nodes), a set of edges $S$ and a set of possible relationship $R$. 

Edges in knowledge graph are represented as triples $(h,r,t)$ where head $h$ has relation $l$ with tail $t$. Given a triple $(h,r,t)$, the goal is that the embedding of $(h,r)$ should be close to embedding of $t$. 

**TransE**

For a triple $(h,r,t)$, let $\mathbf {h,r,t} \in \mathbb{R}^{d}$ be embedding vectors. If $(h,r,t)\in S$, TransE tries to ensure that $\mathbf {h} + \mathbf {r} \approx \mathbf t$. Simultaneously, TransE tries to make sure $\mathbf h+\mathbf r \not\approx \mathbf t$ when edge $(h,r,t)$ doesn't exist. TransE accomplishes this by minimizing the following loss: 
$$
L = \sum_{(h,r,t)\in S}\Big(\sum_{(h',r,t')\in S'_{(h,r,t)}} \max\big (\gamma + d(h+r,t) - d(h'+r,t'),0\big )\Big)
$$
The form of the loss function is similar to that in SVM. Here, $(h',r,t')$ are "corrupted" triples, where either $h$ or $t$ (not both) is replaced by a **random** entity, and $r$ remains the same. 
$$
S'_{(h,r,t)} = \{(h',r,t) \ \bold | \ h'\in E\} \cup \{(h,r,t') \ \bold | \ t'\in E\}
$$
Additionally, $\gamma$ is a fixed scalar called *margin*. The function $d(\cdot,\cdot)$ is the Euclidean distance. 

Before each training epoch, TransE normalizes every entity embedding to have unit length. Otherwise, the algorithm could trivially minimize the loss by increasing the magnitude of the embeddings indefinitely.

**TransR**

Model entities as vectors in the entity space $\mathbb{R}^{d}$ and model each relation as vector in relation space  $r\in \mathbb{R}^{k}$ with $\mathbf M_r \in \mathbb{R}^{k\times d}$ as the projection matrix. During training: 
$$
\mathbf{h'} = \mathbf{M_rh} \ , \ \mathbf{t'} = \mathbf{M_rt}
$$
**DistMult**

The score function of TransE $d(h+r,t)$ is Euclidean distance, while that in DistMult is cosine similarity between $\mathbf{h\cdot r}$ and $\mathbf{t}$ : 
$$
f_r(h,t) = \sum_i h_i\cdot r_i\cdot t_i
$$
**ComplEx**

Based on DistMult, ComplEx embeds entities and relations in Complex vector space. The score function of ComplEx is slight different from that in DistMult: 
$$
f_r(h,t) = \text{Re}(\sum_i h_i\cdot r_i\cdot \overline{t}_i)
$$

## Reasoning in KG

**Path Queries**

An n-hop path query $q$ can be represented by: 
$$
q = (v_a,(r_1,r_2...,r_n))
$$
where $v_a$ is an anchor entity. Query plan in the path query is a chain: 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240319152038305.png" width = 500>

**Conjunctive Queries**

Conjunctive Queries are path queries with logic conjunction operation. For example:  

> What are medicines that cause Short of Breath and treat diseases associate with protein ESR2. 
>
> Conjunction between (e:ESR2, (r:Assoc, r:TreatedBy)) and (e:Short of Breath, (r:CausedBy)). 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240319154437277.png" width = 500>

**Box embeddings**

Box is hyper-rectangle that enclose all the answer entities. A box is given by: 
$$
\mathbf q = (\text{Cen}(q),\text{Off}(q))\\
\text{Box}_q =\{v\in \mathbb R^{d}: \text{Cen}(q) - \text{Off}(q)\preceq v\preceq \text{Cen}(q)+\text{Off}(q)\}
$$
where $\preceq$ is element-wise inequality. Here is a box in two-dimensional embedding space: 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240319161925715.png" width = 400>

Boxes are a powerful abstraction, as we can project the center and control the offset to model the set of entities enclosed in the box. Overall, the training process consists of following five stages: 

<span style="color:purple">Initial boxes for source nodes.</span> Each source node represents an anchor entity $v\in V$, which we can regard as a set that only contains a single entity. Formally, we set the initial box embedding as $(\bold v,\bold 0)$ where $\bold v$ is the node embedding and $\bold 0$ is a d-dimensional all-zero vector. 

<span style="color:purple">Geometric projection operator.</span> We associate each relation $r\in R$ with relation embedding $\bold r = (\text{Cen}(r),\text{Off}(r))\in \mathbb R^{2d}$ with $\text{Off}(r)\succeq 0$. The operator takes the current box as input and use the relation embedding to project and expand the box: 
$$
\text{Cen}(q') = \text{Cen}(q) + \text{Cen}(r) \\ 
\text{Off}(q') = \text{Off}(q) + \text{Off}(r)
$$
<span style="color:purple">Geometric intersection operator.</span> Intuitively, the center of the new box should be close to the center of the input boxes. Simultaneously, the offset(box size) should be shrink. 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240320121830028.png" width = 250>

The center can be a weighted sum of the input box after applying attention mechanism: 
$$
\text{Cen}(q_{inter}) = \sum w_i \odot \text{Cen}(q_i)  \\ 
w_i = \frac{\exp(f_{cen}(\text{Cen}(q_i)))}{\sum_j \exp(f_{cen}(\text{Cen}(q_j)))}
$$
$w_i\in \mathbb R^{d}$ is calculated by a neural network $f_{cen}$. For offset, we first take minimum of the offset of  the input box. Next, we make the model more expressive by introducing a new function to extract the representation of the input boxes with a sigmoid function to guarantee shrinking: 
$$
\text{Off} (q_{inter}) = \min(\text{Off}(q_1), ...,\text{Off}(q_n)) \odot \sigma(f_{off}(\text{Off}(q_1), ...,\text{Off}(q_n)))
$$
<span style="color:purple">Entity-to-box distance.</span> Given a box $[Math Processing Error]\mathbf {q}\in\mathbb R^{2d}$ and an entity vector $v\in \mathbb R^{d}$, we define their distance as: 
$$
dis(\bold v,\bold q) = dis_{out}(\bold v, \bold q) + \alpha \cdot dis_{in}(\bold v, \bold q)
$$
where $dis_{out}$ is the distance from $\bold v$ to the border of the box(zero if the point is enclosed in the box) and $dis_{in}$ is the corresponds to the distance between the center of the box and its side/corner (or the entity itself if the entity is inside the box).

The key idea is to **downweight** the distance inside the box using $0<\alpha < 1$. This is because if the point is enclosed in the box, we regard it "close enough" to the query center. 

<span style="color:purple">Training objective.</span> Optimize a negative sampling loss to train our model: 
$$
L = -\log\sigma(\gamma - dis(\bold v,\bold q)) - \frac{1}{k}\sum_{i=1}^k \log \sigma(dis(\bold v_i',\bold q) - \gamma)
$$
where $v$ is a positive entity(also a possible answer to query $q$) and $v_i'$ corresponds to a negative entity(non-answer to query $q$). Because the distance is always non-negative, we use a margin $\gamma$ (fixed scalar) to shift it. 