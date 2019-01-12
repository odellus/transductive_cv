#! /usr/bin/env python3.5
# -*- coding: utf-8
# author: Thomas Wood
# email: thomas@synpon.com
# description: common file for data utilities.

# standard library modules
from io import BytesIO
from pprint import pprint
import time
from uuid import uuid4

# Redis modules
import redis
from redisgraph import Node, Edge, Graph
from redisgraph.query_result import QueryResult

# Numerical modules
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

my_thoughts = """
So the idea is:
  1. Get an entire adjacency matrix back from redis_graph.
  2. Store numeric data for the nodes in redis using a common key/index.
  3. Use the adjacency matrix to do differentiable sampling of the features
     inside transductive deep learning algorithms.

First things first, let's just make a random adjacency matrix with random
features and figure out how to do this thing first. Then we can upgrade to using
the same general formula to create the VOC gold standard region graph dataset.
"""

def _gen_key(key_len):
    return uuid4().hex[:key_len]

def _index_key(index_dict, key_len=10):
    index_key = _gen_key(key_len)
    while index_key in index_dict:
        index_key = _gen_key(key_len)
    index_dict[index_key] = 1
    return index_key

def _array2bytes(arr):
    with BytesIO() as b:
        np.save(b, arr)
        return b.getvalue()

def _bytes2array(byts):
    return np.load(BytesIO(byts))

def _add_nodes(r, graph, n_nodes):
    nodes = []
    index_dict = {}
    for k in range(n_nodes):
        index_key = _index_key(index_dict)
        n = Node(label="node", properties={"index_key":index_key})
        graph.add_node(n)
        nodes.append(n)
        feat = np.random.rand(1,100,7,7)
        feat_bytes = _array2bytes(feat)
        r.set(index_key, feat_bytes)
    # graph.commit()
    return nodes

def _add_edges(nodes, graph, edge_prob):
    t = time.time()
    edges = []
    for k, node0 in enumerate(nodes):
        for kk, node1 in enumerate(nodes[:k]):
            if np.random.rand() < edge_prob:
                edge = Edge(node0, "adjacent_to", node1)
                graph.add_edge(edge)
                edges.append(edge)
    # graph.commit()
    print("_add_edges: {}".format(time.time() - t))
    return edges

def _create_random_graph(r, graphname="random", n_nodes=1000, edge_prob=0.1):
    redis_graph = Graph(graphname, r)
    nodes = _add_nodes(r, redis_graph, n_nodes)
    edges = _add_edges(nodes, redis_graph, edge_prob)
    return redis_graph


def _query_redis(r, graphname, query):
    res = r.execute_command("GRAPH.QUERY", graphname, query)
    return QueryResult(res[0], res[1])

def _get_nodes(r, graphname):
    query = "MATCH (n) RETURN (n)"
    ret = _query_redis(r, graphname, query)
    node_list = [x[0] for x in ret.result_set[1:]]
    return node_list

def _get_edges(r, graphname):
    query = "MATCH (n)-[r]->(m) RETURN n,r,m"
    ret = _query_redis(r, graphname, query)
    return ret.result_set[1:]

def _get_graph(r, graphname="random"):
    g = nx.Graph()
    nodes = _get_nodes(r, graphname)
    edges = _get_edges(r, graphname)
    for node in nodes:
        g.add_node(node)
    for edge in edges:
        g.add_edge(edge[0],edge[2])
    return g, _get_index_map(g)

def _get_index_map(g):
    nodelist = list(g.nodes)
    return {x:k for k, x in enumerate(nodelist)}

def _get_adjacency(g):
    return nx.adjacency_matrix(g)

def _sample_adj(adj, idxs):
    adj_p = adj[idxs,:]
    return adj_p[:, idxs]

def _get_features(r, indexes):
    matlist = []
    for index in indexes:
        matlist.append(_bytes2array(r.get(index)))
    return np.stack(matlist, axis=-1)

def _sample_subgraph(r, adj, index_map, indexes):
    # Turn those string index keys into integer position indexes
    idxs = [index_map[x] for x in indexes]
    sampled_adj = _sample_adj(adj, idxs)

    # Now we need to get the feature matrix for this subgraph.
    feat_mat = _get_features(r, indexes)
    return sampled_adj, feat_mat

def _main():
    r = redis.Redis(host='localhost', port=6379)
    g = _create_random_graph(r)
    t = time.time()
    g.commit()
    print("after calling graph.commit: {}".format(time.time() - t))

    # Now load the saved graph and features saved in redis into a nx.Graph
    G, idx_map = _get_graph(r)
    adj = _get_adjacency(G)

    # Select a subgraph.
    n_subgraph = 200
    all_nodes = list(G.nodes)
    subgraph = np.random.choice(all_nodes, size=n_subgraph, replace=False).tolist()

    # These are what we need to hand over to for mini-batch GCN
    adj, features = _sample_subgraph(r, adj, idx_map, subgraph)

    # Show the adjacency matrix and print shape of features
    plt.matshow(adj.todense())
    plt.show()
    print(features.shape)

    # Flush redis so we can do it over again without trouble.
    r.flushall()

if __name__ == "__main__":
    _main()
