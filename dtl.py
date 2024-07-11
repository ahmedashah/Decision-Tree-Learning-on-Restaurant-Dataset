import pandas as pd
import numpy as np
from collections import Counter
import csv

class Tree:
    def __init__(self, attribute=None):
        self.attribute = attribute
        self.children = {}
    
    def add_branch(self, value, subtree):
        self.children[value] = subtree
    
    def __repr__(self, level=0):
        ret = "\t"*level + repr(self.attribute) + "\n"
        for key, value in self.children.items():
            ret += "\t"*level + key + " -> " + value.__repr__(level + 1)
        return ret

def entropy(labels):
    """ Calculate the entropy of a list of labels. """
    if len(labels) == 0:
        return 0
    label_counts = Counter(labels)
    total = len(labels)
    probabilities = [count / total for count in label_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def mode(data):
    """ Return the most common class label in the dataset. """
    labels = [row[-1] for row in data]
    return Counter(labels).most_common(1)[0][0]

def choose_attribute(attributes, dataSet):
    base_entropy = entropy([data[-1] for data in dataSet])
    gains = []
    best_gain = -1
    best_attribute = None

    # Calculate information gain for each attribute and store results
    for attribute in attributes:
        attr_index = attributes.index(attribute)
        attribute_values = {}
        for data in dataSet:
            key = data[attr_index]
            attribute_values.setdefault(key, []).append(data[-1])

        weighted_entropy = 0
        total_data_points = len(dataSet)
        for key, subset in attribute_values.items():
            subset_entropy = entropy(subset)
            weight = len(subset) / total_data_points
            weighted_entropy += weight * subset_entropy

        info_gain = base_entropy - weighted_entropy
        gains.append((attribute, info_gain))

    # Print current gains for diagnostic purposes
    print(f"Attributes => {[attr for attr, _ in gains]}")
    print(f"Information Gain => {[round(gain, 2) for _, gain in gains]}")

    # Select the attribute with the highest information gain
    best_gain = max(gains, key=lambda item: item[1])[1]  # Find the highest gain value
    for attr, gain in gains:
        if gain == best_gain:
            best_attribute = attr
            break  # Stop at the first occurrence of the highest gain

    print(f"Highest information gain => {best_attribute} ({round(best_gain, 2)})\n")
    return best_attribute

def DecisionTreeLearning(restaurant_data, attributes, default_class):
    if not restaurant_data:
        return Tree(default_class)
    if all(data[-1] == restaurant_data[0][-1] for data in restaurant_data):
        return Tree(restaurant_data[0][-1])
    if not attributes:
        return Tree(mode(restaurant_data))

    best = choose_attribute(attributes, restaurant_data)
    tree = Tree(best)
    best_attr_values = set(data[attributes.index(best)] for data in restaurant_data)
    for value in best_attr_values:
        subset = [data for data in restaurant_data if data[attributes.index(best)] == value]
        subtree = DecisionTreeLearning(subset, [attr for attr in attributes if attr != best], mode(subset))
        tree.add_branch(value, subtree)
    return tree

def print_decision_tree(tree, level=0, prefix="Root: "):
    indent = "    " * level
    if isinstance(tree.attribute, str) and tree.children:
        print(indent + f"{prefix}{tree.attribute}")
        for value, subtree in tree.children.items():
            print_decision_tree(subtree, level + 1, f"{value} -> ")
    else:
        print(indent + f"{prefix}Leaf: {tree.attribute}")


df = pd.read_csv('restaurant.csv', header=None)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
restaurant_data = df.values.tolist()

attributes = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est"]

# Default class extraction and tree construction
default_class = mode(restaurant_data)
decision_tree = DecisionTreeLearning(restaurant_data, attributes, default_class)

# Print the tree structure
print_decision_tree(decision_tree)