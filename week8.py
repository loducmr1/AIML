import pandas as pd
from collections import Counter
import math
from pprint import pprint

# Entropy calculation function
def entropy(probs):
    return sum(-prob * math.log(prob, 2) for prob in probs if prob > 0)

# Calculate entropy of a list
def entropy_of_list(a_list):
    cnt = Counter(a_list)
    num_instances = len(a_list)
    probs = [s / num_instances for s in cnt.values()]
    return entropy(probs)

# Information gain function
def information_gain(df, split_attribute_name, target_attribute_name):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index) * 1.0

    df_agg_ent = df_split.agg(
        entropy_of_list=(target_attribute_name, entropy_of_list),
        prob=(target_attribute_name, lambda x: len(x) / nobs)
    )

    avg_info = sum(df_agg_ent['entropy_of_list'] * df_agg_ent['prob'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - avg_info

# ID3 Decision Tree function
def id3DT(df, target_attribute_name, attribute_names, default_class=None):
    cnt = Counter(df[target_attribute_name])
    if len(cnt) == 1:
        return next(iter(cnt))
    elif df.empty or not attribute_names:
        return default_class
    else:
        default_class = max(cnt, key=cnt.get)
        gains = [information_gain(df, att, target_attribute_name) for att in attribute_names]
        index_of_max = gains.index(max(gains))
        best_attr = attribute_names[index_of_max]
        tree = {best_attr: {}}
        remaining_attributes = [i for i in attribute_names if i != best_attr]
        
        for attr_val, data_subnet in df.groupby(best_attr):
            subtree = id3DT(data_subnet, target_attribute_name, remaining_attributes, default_class)
            tree[best_attr][attr_val] = subtree
        return tree

# Classification function
def classify(instance, tree, default=None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute]:
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict):
            return classify(instance, result)
        else:
            return result
    else:
        return default

# Load dataset
df = pd.read_csv('C:/Users/student/Downloads/weather.csv')
attribute_names = list(df.columns)
attribute_names.remove('play')

# Build decision tree
tree = id3DT(df, 'play', attribute_names)
print("The Resultant Decision Tree is:")
pprint(tree)

# Test classification with a sample DataFrame
tree1 = {
    'outlook': ['rainy', 'sunny'],
    'temperature': ['mild', 'hot'],
    'humidity': ['normal', 'high'],
    'windy': [True, False]
}

df2 = pd.DataFrame(tree1)
df2['Predicted'] = df2.apply(classify, axis=1, args=(tree, 'No'))
print(df2[['outlook', 'temperature', 'humidity', 'windy', 'Predicted']])