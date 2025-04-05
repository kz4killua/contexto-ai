from typing import List, Optional
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from game import Game


EXPLORATION_PARAMETER = 0.1
CLUSTERS_PER_NODE = 3


class Node:
    """A node is a cluster of word embeddings."""

    def __init__(self, cluster, parent=None, name='0'):
        self.cluster: List[int]  = cluster
        self.parent: Optional[Node] = parent
        self.name: str = f"{self.parent.name}-{name}" if parent else name
        self.children: List[Node] = []
        self.w: int = 0
        self.n: int = 0

    def __repr__(self):
        return f"Node({self.name}: {self.cluster})"
    
    def __str__(self):
        return f"Node({self.name}: {self.cluster})"


class SimpleClusteringAgent:
    """Uses K-means clustering and Monte Carlo tree search to select guesses."""

    def __init__(self, game: Game, words: np.ndarray, embeddings: np.ndarray, metric:str='cosine'):
        self.words = words
        self.embeddings = embeddings
        self.game = game
        self.metric = metric
        self.root = Node(list(range(len(self.words))))
        # Initialize the root node to expand directly
        self.root.n = 1


    def play(self, moves: int, log: bool=False):
        """Play the game for a given number of moves."""

        for _ in range(moves):

            leaf = select(self.root)
            node = expand(leaf, self.embeddings, self.metric)
            guess, rank = simulate(node, self.game, self.words)

            if log:
                print(f"Guess #{self.game.number_of_guesses}: {self.words[guess]}, Rank: {rank}")

            if rank == 0:
                return self.words[guess]

            # Update the Monte Carlo search tree
            if rank is None:
                pass
            elif rank in list(self.game.history.values())[:-1]:
                pass
            else:
                backpropagate(node, rank)

            prune(node, guess)


def select(node: Node) -> Node:
    """Returns a leaf node."""
    if node.children:
        return select(max(node.children, key=uct))
    return node


def expand(node: Node, embeddings: np.ndarray, metric: str) -> Node:
    """Expand the node by adding a child."""

    if node.n == 0:
        return node
    
    if len(node.cluster) == 1:
        return node
    
    # Run K-means clustering to split the cluster into sub-clusters
    n_clusters = min(CLUSTERS_PER_NODE, len(node.cluster))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    X = embeddings[node.cluster]

    if metric == 'euclidean':
        X = normalize(X)
    elif metric != 'cosine':
        raise ValueError(f"Unknown metric: {metric}")

    labels = kmeans.fit_predict(X)
    
    clusters = defaultdict(list)
    for index, label in zip(node.cluster, labels):
        clusters[label].append(index)

    for label, cluster in clusters.items():
        node.children.append(Node(cluster, node, str(label)))
    
    return node.children[0]


def simulate(node: Node, game: Game, words: np.ndarray):
    """Pick a guess from the cluster and get its rank."""
    guess = node.cluster[0]
    word = words[guess]
    rank = game.make_guess(word)
    return guess, rank


def backpropagate(node: Node, rank: int):
    """Update the nodes from the leaf to the root."""
    w = calculate_reward(rank)
    while node:
        node.n += 1
        node.w += w
        node = node.parent


def uct(node):
    """Upper Confidence Bound for Trees."""
    c = EXPLORATION_PARAMETER
    if node.n == 0:
        return float('inf')
    else:
        return (node.w / node.n) + c * np.sqrt(np.log(node.parent.n) / node.n)


def calculate_reward(rank: int) -> int:
    """Calculate the reward based on the rank."""
    return -rank


def prune(leaf: Node, guess: int):
    """Remove a guess from the search tree."""
    node = leaf
    while node:

        # Remove the guess from the cluster
        if guess in node.cluster:
            node.cluster.remove(guess)

        # Remove child nodes with empty clusters
        if len(node.cluster) == 1:
            node.children = []
        node.children = [child for child in node.children if len(child.cluster) > 0]

        node = node.parent
