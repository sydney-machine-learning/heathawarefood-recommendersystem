# Health Food recommender system
# Health-aware Food Recommendation System using Graph Attention Network
import os
import pandas as pd
import networkx as nx
import math
import glob
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import torch_geometric.utils as utils
from dgl.nn.pytorch import GATConv
from torch_geometric.utils import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.optim as optim
import dgl.function as fn
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score, ndcg_score, recall_score


# Specify the directory path where the "Food" folder is located
folder_path = r"C:\Food"

# List of files to read
files_to_read = ['RAW_interactions.csv', 'RAW_recipes.csv']

# Create a dictionary to store user_id as key and total score and count as values
user_scores = {}

# Loop through the files and read their contents
for file in files_to_read:
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
        
        # Extract user_id and rating from 'RAW_interactions.csv' file
        if file == 'RAW_interactions.csv':
            user_id = df['user_id']
            recipe_id = df['recipe_id']
            rating = df['rating']
            
            # Iterate through user_id, recipe_id, and rating columns to calculate total score and count for each user_id
            for i in range(len(user_id)):
                uid = user_id[i]
                rid = recipe_id[i]
                r = rating[i]
                if uid not in user_scores:
                    user_scores[uid] = {'total_score': 0, 'count': 0}
                user_scores[uid]['total_score'] += r
                user_scores[uid]['count'] += 1

# Calculate the average score for each user_id
for uid, scores in user_scores.items():
    avg_score = scores['total_score'] / scores['count']

# Extract ingredients and nutrition from 'RAW_recipes.csv' file
df = pd.read_csv(os.path.join(folder_path, 'RAW_recipes.csv'))
recipe_id = df['recipe_id']
ingredients = df['ingredients']
nutrition = df['nutrition']

# -----------------Heterogeneous graph-------------------
# Specify the directory path where the "Food" folder is located
folder_path = r"C:\Food"

# List of files to read
files_to_read = ['RAW_interactions.csv', 'RAW_recipes.csv']

# Create a directed multigraph
G = nx.MultiDiGraph()

# Loop through the files and read their contents
for file in files_to_read:
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, dtype=str)

        # Extract user_id and rating from 'RAW_interactions.csv' file
        if file == 'RAW_interactions.csv':
            user_id = df['user_id']
            recipe_id = df['recipe_id']
            rating = df['rating']

            # Iterate through user_id, recipe_id, and rating columns to add edges with weights to the graph
            for i in range(len(user_id)):
                uid = user_id[i]
                rid = recipe_id[i]
                r = rating[i]

                # Add user_id and recipe_id as nodes
                G.add_node(uid, node_type='user')
                G.add_node(rid, node_type='recipe')

                # Add edge between user_id and recipe_id with weight as the rating
                G.add_edge(uid, rid, weight=r, edge_type='rating')

        # Extract recipe_id, ingredients, and nutrition from 'RAW_recipes.csv' file
        elif file == 'RAW_recipes.csv':
            recipe_id = df['recipe_id']
            ingredients = df['ingredients']
            nutrition = df['nutrition']

            # Iterate through recipe_id, ingredients, and nutrition columns to add edges to the graph
            for i in range(len(recipe_id)):
                rid = recipe_id[i]
                ing = ingredients[i]
                nut = nutrition[i]

                # Add recipe_id as a node
                G.add_node(rid, node_type='recipe')

                # Add edges between recipe_id and ingredients
                if isinstance(ing, str):
                    ingredients_list = ing.split(',')
                    for ingredient in ingredients_list:
                        G.add_node(ingredient.strip(), node_type='ingredient')
                        G.add_edge(rid, ingredient.strip(), edge_type='ingredient')

                # Add edges between recipe_id and nutrition
                if isinstance(nut, str):
                    nutrition_list = nut.split(',')
                    for nutrition_item in nutrition_list:
                        G.add_node(nutrition_item.strip(), node_type='nutrition')
                        G.add_edge(rid, nutrition_item.strip(), edge_type='nutrition')

# Define the meta-paths
meta_paths = [
    ['user', 'rating', 'recipe_id'],
    ['recipe_id', 'ingredient', 'recipe_id'],
    ['recipe_id', 'nutrition', 'recipe_id'],
    ['recipe_id', 'ingredient', 'nutrition']
]

# -------------------------------- Filter HG based the users interisting foods --------------------------
# Create a dictionary to store user_id as key and list of ratings as values
user_ratings = {}

# Loop through the files and read their contents
for file in files_to_read:
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)

        if file == 'RAW_interactions.csv':
            # Extract user_id and rating from 'RAW_interactions.csv' file
            user_id = df['user_id']
            recipe_id = df['recipe_id']
            rating = df['rating']

# Iterate through user_id and rating columns to store ratings for each user_id
for i in range(len(user_id)):
    uid = user_id[i]
    r = rating[i]
    if uid not in user_ratings:
        user_ratings[uid] = []
    user_ratings[uid].append(r)

# Calculate the average rating for each user_id
user_avg_ratings = {}
for uid, ratings in user_ratings.items():
    avg_rating = sum(ratings) / len(ratings)
    user_avg_ratings[uid] = avg_rating

# Extract ingredients and nutrition from 'RAW_recipes.csv' file
df = pd.read_csv(os.path.join(folder_path, 'RAW_recipes.csv'))
recipe_id = df['recipe_id']
ingredients = df['ingredients']
nutrition = df['nutrition']

# Create a new graph for the filtered data
filtered_graph = nx.MultiDiGraph()

# Loop through the recipe data and add nodes and edges to the filtered graph
for i in range(len(recipe_id)):
    rid = recipe_id[i]
    ing = ingredients[i]
    nut = nutrition[i]

    if rid in user_ratings and rid in user_avg_ratings:
        recipe_avg_rating = sum(user_ratings[rid]) / len(user_ratings[rid])
        user_avg_rating = user_avg_ratings[rid]

        if recipe_avg_rating >= user_avg_rating:
            filtered_graph.add_node(str(rid), node_type='recipe')

            if isinstance(ing, str):
                ingredients_list = ing.split(',')
                for ingredient in ingredients_list:
                    filtered_graph.add_node(ingredient.strip(), node_type='ingredient')
                    filtered_graph.add_edge(str(rid), ingredient.strip(), edge_type='ingredient')

            if isinstance(nut, str):
                nutrition_list = nut.split(',')
                for nutrition_item in nutrition_list:
                    filtered_graph.add_node(nutrition_item.strip(), node_type='nutrition')
                    filtered_graph.add_edge(str(rid), nutrition_item.strip(), edge_type='nutrition')

            user_ids = [uid for uid in user_ratings[rid]]
            for uid in user_ids:
                filtered_graph.add_node(str(uid), node_type='user')
                filtered_graph.add_edge(str(uid), str(rid), edge_type='rated')

# Convert node and edge labels to integers using label encoding
label_encoder = LabelEncoder()
for node, data in filtered_graph.nodes(data=True):
    if 'node_type' in data:
        node_type = data['node_type']
        data['node_type'] = label_encoder.fit_transform([node_type])[0]

for source, target, data in filtered_graph.edges(data=True):
    if 'edge_type' in data:
        edge_type = data['edge_type']
        data['edge_type'] = label_encoder.fit_transform([edge_type])[0]

# Convert the filtered graph to a DGLGraph
dgl_graph = dgl.from_networkx(filtered_graph, node_attrs=['node_type'], edge_attrs=['edge_type'])

# Add self-loops to the graph
dgl_graph = dgl.add_self_loop(dgl_graph)

# Loop through the edges in the filtered graph
for edge_id in range(dgl_graph.number_of_edges()):
    src, dst = dgl_graph.find_edges(edge_id)
    edge_type = dgl_graph.edata['edge_type'][edge_id]

# Read the contents of 'RAW_interactions.csv'
interactions_df = pd.read_csv(os.path.join(folder_path, 'RAW_interactions.csv'))
user_id = interactions_df['user_id']
recipe_id = interactions_df['recipe_id']
rating = interactions_df['rating']             

# Read the contents of 'RAW_recipes.csv'
recipes_df = pd.read_csv(os.path.join(folder_path, 'RAW_recipes.csv'))
recipe_id = recipes_df['recipe_id']
ingredients = recipes_df['ingredients']
nutrition = recipes_df['nutrition']

# Merge the data frames based on the common column 'recipe_id'
merged_df = pd.merge(interactions_df, recipes_df, on='recipe_id')

# Create a new graph
G = nx.Graph()

# Add nodes and edges to the graph based on the merged data
for _, row in merged_df.iterrows():
    user = row['user_id']
    rating = row['rating']
    recipe = row['recipe_id']
    ingredient = row['ingredient']
    nutrition = row['nutrition']

    G.add_node(user, node_type='user')
    G.add_node(recipe, node_type='recipe')
    G.add_edge(user, recipe, edge_type='rating')

    if ingredient:
        G.add_node(ingredient, node_type='ingredient')
        G.add_edge(recipe, ingredient, edge_type='ingredient')

    if nutrition:
        G.add_node(nutrition, node_type='nutrition')
        G.add_edge(recipe, nutrition, edge_type='nutrition')

# Define the meta-paths
meta_paths = [
    ['user_id', 'rating', 'recipe_id', 'ingredient'],
    ['user_id', 'rating', 'recipe_id', 'nutrition', 'ingredient']
]

# #-------------- Node Level Attention (NLA) ----------------------

class NodeLevelAttention(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(NodeLevelAttention, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)
        self.attention = nn.Sequential(
            nn.Linear(out_feats, 1),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, g, h):
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        att_scores = self.attention(h).squeeze()
        g.ndata['attention_scores'] = att_scores
        g.update_all(fn.copy_u('attention_scores', 'm'), fn.sum('m', 'attention_sum'))
        att_sum = g.ndata.pop('attention_sum')
        h = h * att_sum.unsqueeze(-1)
        return h

# Define the MSE loss function
loss_fn = nn.MSELoss()

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the DGL graph and its features to the device
dgl_graph = dgl_graph.to(device)

# Initialize the GNN model
gnn = NodeLevelAttention(in_feats=1, hidden_feats=32, out_feats=16).to(device)

# Set up the optimizer
optimizer = optim.Adam(gnn.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    gnn.train()
    optimizer.zero_grad()
    
    # Perform forward propagation
    h = dgl_graph.ndata['node_type'].unsqueeze(-1).float().to(device)
    h = gnn(dgl_graph, h)
    
    # Calculate the loss
    loss = loss_fn(h, torch.zeros_like(h).to(device))
    
    # Perform backward propagation and update parameters
    loss.backward()
    optimizer.step()
    
#------------------------ Semenatic Level Attention (SLA) -----------------------------
# Define the health foods list
health_foods_list = ["Proteins", "Carbohydrates", "Sugars", "Sodium", "Fat", "Saturated_fats", "Fibers"]

class SemanticLevelAttention(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_classes, meta_paths):
        super(SemanticLevelAttention, self).__init__()
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_feats, out_feats)
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.meta_paths = meta_paths

    def forward(self, graph, h):
        x = self.fc1(h)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        attention_weights = self.softmax(x)
        h = h * attention_weights
        return h

    def edge_loss(self, h_sla):
        loss = -torch.log(1 / (1 + torch.exp(h_sla)))
        return loss

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the meta-paths
meta_path1 = [['user_id', 'recipe_id', 'ingredient']] 
meta_path2 = [['user_id', 'recipe_id', 'nutrition']]  
meta_path3 = [['user_id', 'recipe_id', 'nutrition', 'ingredient']]

meta_paths = ['meta_path1', 'meta_path2', 'meta_path3']  # List of meta-paths

# Create an instance of the SemanticLevelAttention model
sla = SemanticLevelAttention(in_feats=1, hidden_feats=32, out_feats=16, num_classes=len(health_foods_list), meta_paths=meta_paths).to(device)

# Set up the optimizer
optimizer = optim.Adam(sla.parameters(), lr=0.001)

# Inside the training loop
for epoch in range(100):
    sla.train()
    optimizer.zero_grad()

    # Perform forward propagation with NLA
    h = dgl_graph.ndata['node_type'].unsqueeze(-1).float().to(device)

    # Perform forward propagation with SLA
    h_sla = sla(dgl_graph, h)  # Pass the graph and node features to the SLA model

    # Calculate the loss
    edge_loss_value = sla.edge_loss(h_sla)
    loss = edge_loss_value.mean()  # Compute the mean loss

    # Perform backward propagation and update parameters
    loss.backward()
    optimizer.step()

# Apply softmax to the final output
h_sla = sla.softmax(h_sla)

# Create a mask for health foods with at least three ingredients from the health_foods_list
mask = (h_sla.sum(dim=1) >= 3).squeeze()
health_food_embeddings = h_sla[mask]

#-----------Combine NLA and SLA Loss Function-------------------------------------------

# Define the MSE loss function for NLA
loss_fn_NLA = nn.MSELoss()

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the DGL graph and its features to the device
dgl_graph = dgl_graph.to(device)

# Initialize the NLA model
nla = NodeLevelAttention(in_feats=1, hidden_feats=32, out_feats=16).to(device)

# Initialize the SLA model
sla = SemanticLevelAttention(in_feats=1, hidden_feats=32, out_feats=16, num_classes=len(health_foods_list), meta_paths=meta_paths).to(device)

# Set up the optimizer
optimizer_nla = optim.Adam(nla.parameters(), lr=0.001)
optimizer_sla = optim.Adam(sla.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    nla.train()
    sla.train()
    optimizer_nla.zero_grad()
    optimizer_sla.zero_grad()

    # Perform forward propagation with NLA
    h_nla = dgl_graph.ndata['node_type'].unsqueeze(-1).float().to(device)
    h_nla = nla(dgl_graph, h_nla)

    # Perform forward propagation with SLA
    h_sla = dgl_graph.ndata['node_type'].unsqueeze(-1).float().to(device)
    h_sla = sla(dgl_graph, h_sla)

    # Calculate the losses separately for NLA and SLA
    loss_nla = loss_fn_NLA(h_nla, torch.zeros_like(h_nla).to(device))
    loss_sla = sla.edge_loss(h_sla).mean()
    
    # Calculate the total loss by adding the individual losses
    lambda_value = 0.5  

# Training loop
for epoch in range(100):
    nla.train()
    sla.train()
    optimizer_nla.zero_grad()
    optimizer_sla.zero_grad()

    # Perform forward propagation with NLA
    h_nla = dgl_graph.ndata['node_type'].unsqueeze(-1).float().to(device)
    h_nla = nla(dgl_graph, h_nla)

    # Perform forward propagation with SLA
    h_sla = dgl_graph.ndata['node_type'].unsqueeze(-1).float().to(device)
    h_sla = sla(dgl_graph, h_sla)

    # Calculate the losses separately for NLA and SLA
    loss_nla = loss_fn_NLA(h_nla, torch.zeros_like(h_nla).to(device))
    loss_sla = sla.edge_loss(h_sla).mean()

    # Calculate the total loss by adding the individual losses
    total_loss = loss_nla + lambda_value * loss_sla

    # Perform backward propagation and update parameters for NLA
    total_loss.backward(retain_graph=True)
    optimizer_nla.step()

    # Clear gradients of SLA optimizer
    optimizer_sla.zero_grad()

    # Perform backward propagation and update parameters for SLA
    loss_sla.backward()
    optimizer_sla.step()
# ----------------- Recommendation System -----------------------------------
# Create empty dictionaries to store ingredient and nutrient links
ingredient_links = {}
nutrient_links = {}

# Loop through the edges in the filtered graph
for edge_id in range(dgl_graph.number_of_edges()):
    src, dst = dgl_graph.find_edges(edge_id)
    edge_type = dgl_graph.edata['edge_type'][edge_id]

    if int(edge_type) == 0:
        source_node = str(src.item())
        target_node = str(dst.item())
        if source_node not in ingredient_links:
            ingredient_links[source_node] = []
        ingredient_links[source_node].append(target_node)
    elif int(edge_type) == 1:
        source_node = str(src.item())
        target_node = str(dst.item())
        if source_node not in nutrient_links:
            nutrient_links[source_node] = []
        nutrient_links[source_node].append(target_node)

# Extracting HF (Healthy Foods), HI (Healthy Food Ingredients), and HN (Healthy Food Nutrients)
HF = health_food_embeddings.detach().cpu().numpy()  # Convert tensor to numpy array
HI = set()  # Set to store unique healthy food ingredients
HN = set()  # Set to store unique healthy food nutrients

for i, embedding in enumerate(HF):
    ingredients = [health_foods_list[j] for j, value in enumerate(embedding) if value > 0]
    nutrients = [health_foods_list[j] for j, value in enumerate(embedding) if value <= 0]
    HI.update(ingredients)
    HN.update(nutrients)

HI = list(HI)  # Convert back to list
HN = list(HN)  # Convert back to list

# Calculate PUI
PUI = 0.0
if len(HI) > 0:
    for I_i in HI:
        count = sum(1 for I in HI if I_i in I)
        PUI += (count / len(HI))
    PUI /= len(HI)

# Calculate PUN
PUN = 0.0
if len(HN) > 0:
    for N_j in HN:
        count = sum(1 for N in HN if N_j in N)
        PUN += (count / len(HN))
    PUN /= len(HN)
else:
    PUN = 0.0

# Create a list to store recommended foods for all users
recommended_foods = []

# Iterate over each user and their recipe ID
for user_id, recipe_id in zip(user_id, recipe_id):
    # Iterate over the health food embeddings and find neighbor links
    neighbor_links = []
    for food_embedding in health_food_embeddings:
        # Extract the ingredients and nutrients associated with the healthy food
        HF_I = [health_foods_list[i] for i, value in enumerate(food_embedding) if value > 0]
        HF_N = [health_foods_list[i] for i, value in enumerate(food_embedding) if value <= 0]

        # Find the neighbor links based on the ingredient and nutrient information
        HF_I_neighbors = []
        for ingredient in HF_I:
            if ingredient in ingredient_links:
                HF_I_neighbors.extend(ingredient_links[ingredient])

        HF_N_neighbors = []
        for nutrient in HF_N:
            if nutrient in nutrient_links:
                HF_N_neighbors.extend(nutrient_links[nutrient])

        neighbor_links.append((HF_I_neighbors, HF_N_neighbors))

    # Recommend foods based on popular ingredients and nutrients
    user_recommended_foods = []
    for HF_I, HF_N in neighbor_links:
        if PUI > 0 and PUN > 0:
            # Recommend foods that contain both popular ingredients and nutrients
            if all(ingredient in HI for ingredient in HF_I) and all(nutrient in HN for nutrient in HF_N):
                user_recommended_foods.append(HF)

        elif PUI > 0:
            # Recommend foods that contain at least one popular ingredient
            if any(ingredient in HI for ingredient in HF_I):
                user_recommended_foods.append(HF)

        elif PUN > 0:
            # Recommend foods that contain at least one popular nutrient
            if any(nutrient in HN for nutrient in HF_N):
                user_recommended_foods.append(HF)

    recommended_foods.append((user_id, recipe_id, user_recommended_foods))

# ------------------- Evaluation Section -------------------------
# Create empty lists to store evaluation results
auc_scores = []
ndcg_scores = []
recall_scores = []
positive_foods = {}

# Extract user_id and rating from 'RAW_interactions.csv' file
interactions_file = 'RAW_interactions.csv'
if interactions_file in files_to_read:
    interactions_df = pd.read_csv(os.path.join(folder_path, interactions_file))
    user_id = interactions_df['user_id']
    recipe_id = interactions_df['recipe_id']
    rating = interactions_df['rating']

    # Calculate the average rating for each user_id
    user_avg_ratings = {}
    for uid, rid, r in zip(user_id, recipe_id, rating):
        if uid not in user_avg_ratings:
            user_avg_ratings[uid] = {'total_rating': 0, 'count': 0}
        user_avg_ratings[uid]['total_rating'] += r
        user_avg_ratings[uid]['count'] += 1

# Split the user data into training and validation sets
train_user_id, val_user_id, train_recipe_id, val_recipe_id, train_rating, val_rating = train_test_split(
    user_id, recipe_id, rating, test_size=0.1, random_state=42)

# Iterate over each user and their recommended foods in the validation set
for uid, rid, recommended_foods_list in zip(val_user_id, val_recipe_id, recommended_foods):
    # Get the ground truth positive foods for the user
    true_foods = positive_foods.get(uid, [])

    # Create an empty list to store the flattened recommended foods
    flatten_recommended_foods = []

    # Check if recommended_foods_list is a list
    if isinstance(recommended_foods_list, list):
        # Flatten the nested list of recommended foods
        for sublist in recommended_foods_list:
            if isinstance(sublist, list):
                flatten_recommended_foods.extend(sublist)
            else:
                flatten_recommended_foods.append(sublist)
    else:
        # If recommended_foods_list is not a list, add it directly to the flatten_recommended_foods list
        flatten_recommended_foods.append(recommended_foods_list)

    # Create arrays for true labels and predicted scores
    true_labels = np.array([1 if food in true_foods else 0 for food in health_foods_list])
    predicted_scores = np.array(flatten_recommended_foods, dtype=object)

    # Check if there are positive labels (foods)
    if np.any(true_labels):
        # Calculate AUC score
        auc_score = roc_auc_score(true_labels, predicted_scores)
        auc_scores.append(auc_score)

        # Calculate NDCG score
        ndcg_score = ndcg_score([true_labels], [predicted_scores])
        ndcg_scores.append(ndcg_score)

        # Calculate Recall score
        recall_score = recall_score(true_labels, predicted_scores.round())
        recall_scores.append(recall_score)

# Check if there are scores available before calculating the average
if auc_scores:
    average_auc = np.mean(auc_scores)
else:
    average_auc = 0.00

if ndcg_scores:
    average_ndcg = np.mean(ndcg_scores)
else:
    average_ndcg = 0.00

if recall_scores:
    average_recall = np.mean(recall_scores)
else:
    average_recall = 0.00

# Calculate AUC@k, NDCG@k, and Recall@k for k values between 1 and 20
k_values = range(1, 21)
auc_k_scores = []
ndcg_k_scores = []
recall_k_scores = []

for k in k_values:
    # Select the top-k predicted scores and corresponding labels
    top_k_predicted_scores = predicted_scores[:k]
    top_k_true_labels = true_labels[:k]

    # Convert predicted scores to probabilities of positive class predictions
    positive_probabilities = [score[1] for score in top_k_predicted_scores]

    # Check if both positive and negative labels are present
    unique_labels = set(top_k_true_labels)
    if len(unique_labels) == 2:
        # Calculate AUC@k
        auc_at_k_score = roc_auc_score(top_k_true_labels, positive_probabilities)
        auc_k_scores.append(auc_at_k_score)

        # Calculate NDCG@k
        ndcg_at_k_score = ndcg_score([top_k_true_labels], [positive_probabilities], k=k)
        ndcg_k_scores.append(ndcg_at_k_score)

        # Calculate Recall@k
        recall_at_k_score = recall_score(top_k_true_labels, [1 if prob > 0.5 else 0 for prob in positive_probabilities])
        recall_k_scores.append(recall_at_k_score)
