#! /usr/bin/env python3.5
# -*- coding: utf-8
# author: Thomas Wood
# email: thomas@synpon.com
# description: common file for data utilities.

# standard library modules
from io import BytesIO
from pprint import pprint
import time

# Redis modules
import redis
from redisgraph import Node, Edge, Graph

# Numerical modules
import numpy as np



"""
########################
### EXAMPLE USAGE!!! ###
########################
r = redis.Redis(host='localhost', port=6379)
redis_graph = Graph('social', r)

john = Node(label='person', properties={'name': 'John Doe', 'age': 33, 'gender': 'male', 'status': 'single'})
redis_graph.add_node(john)

japan = Node(label='country', properties={'name': 'Japan'})
redis_graph.add_node(japan)

edge = Edge(john, 'visited', japan, properties={'purpose': 'pleasure'})
redis_graph.add_edge(edge)

redis_graph.commit()
"""

"""
So the idea is:
  1. Get an entire adjacency matrix back from redis_graph.
  2. Store numeric data for the nodes in redis using a common key/index.
  3. Use the adjacency matrix to do differentiable sampling of the features
     inside transductive deep learning algorithms.

First things first, let's just make a random adjacency matrix with random
features and figure out how to do this thing first. Then we can upgrade to using
the same general formula to create the VOC gold standard region graph dataset.
"""

def _array2bytes(arr):
    with BytesIO() as b:
        np.save(b, arr)
        return b.getvalue()

def _bytes2array(byts):
    return np.load(BytesIO(byts))

def _add_nodes(r, graph, n_nodes):
    nodes = []
    for k in range(n_nodes):
        n = Node(label="node")
        graph.add_node(n)
        nodes.append(n)
        _id = n.alias
        feat = np.random.rand(1,100,7,7)
        feat_bytes = _array2bytes(feat)
        r.set(_id, feat_bytes)
    # graph.commit()
    return nodes

def _add_edges(nodes, graph, edge_prob):
    t = time.time()
    edges = []
    for k, node0 in enumerate(nodes):
        for kk, node1 in enumerate(nodes):
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

def _main():
    r = redis.Redis(host='localhost', port=6379)
    g = _create_random_graph(r)
    t = time.time()
    g.commit()
    print("after calling graph.commit: {}".format(time.time() - t))
    # pprint(g.nodes.keys()[:10])
    # print(len(g.nodes.keys()))
    # g.delete()
    # r.flushall()

if __name__ == "__main__":
    a = time.time()
    _main()
    print(time.time() - a)
