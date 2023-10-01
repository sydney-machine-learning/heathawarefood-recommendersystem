#!/usr/bin/env -S python3 -u
#PBS -N Saman
#PBS -l select=1:ncpus=16:mem=256gb
#PBS -l walltime=24:00:00
#PBS -v OMP_NUM_THREADS=16
#PBS -j oe
#PBS -k oed
#PBS -M s.forouzandeh@unsw.edu.au
#PBS -m ae

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
import random
import dgl
import uuid
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.optim as optim
import dgl.function as fn
from dgl.nn import GraphConv
from sklearn.metrics import roc_auc_score, ndcg_score, recall_score
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import zipfile
from sklearn.metrics import ndcg_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize


# Configure the logging system
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

# Now you can use logging to write log messages
logging.info('This is an informational log message.')

# folder_path = '/home/z5318340'
# files_to_read = ['Main_Dataset.zip']  
# file_path = '/home/z5318340/Main_Dataset.zip'

folder_path = r"C:\Food"
files_to_read = ['Food_Dataset.zip']
file_path = r"C:\Food\Food_Dataset.zip"

# Read the file into a pandas DataFrame
df = pd.read_csv(file_path)

def process_data(folder_path, files_to_read):
    # Create a dictionary to store recipe_id as key and total score and count as values
    recipe_scores = {}

    # Loop through the files and read their contents
    for file in files_to_read:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # Read the CSV file
            if file == 'Food_Dataset.zip':
                # Specify the data type for the user_id column as int
                df = pd.read_csv(file_path, dtype={'user_id': int})
                user_id = df['user_id']
                recipe_id = df['recipe_id']
                rating = df['rating']

                # Iterate through user_id, recipe_id, and rating columns to calculate total score and count for each recipe_id
                for i in range(len(user_id)):
                    uid = user_id[i]
                    rid = recipe_id[i]
                    r = rating[i]
                    if rid not in recipe_scores:
                        recipe_scores[rid] = {'total_score': 0, 'count': 0}
                    recipe_scores[rid]['total_score'] += r
                    recipe_scores[rid]['count'] += 1

    # # Calculate the average score for each recipe_id
    # for rid, scores in recipe_scores.items():
    #     avg_score = scores['total_score'] / scores['count']
    #     logging.info(f"Recipe ID: {rid}, Average Score: {avg_score}")

    # Extract ingredients and nutrition from 'Food_Dataset.zip' file
    df = pd.read_csv(os.path.join(folder_path, 'Food_Dataset.zip'))
    recipe_id = df['recipe_id']
    ingredients = df['ingredients']
    nutrition = df['nutrition']

    # Print the first 10 user_ids along with their information
    for i in range(min(3, len(df))):
        uid = df.loc[i, 'user_id']
        rid = df.loc[i, 'recipe_id']
        rat = df.loc[i, 'rating']
        ing = df.loc[i, 'ingredients']
        nut = df.loc[i, 'nutrition']

    # Count the number of unique user_ids
    num_unique_user_ids = df['user_id'].nunique()
    # Count the number of unique recipe_ids
    num_unique_recipe_ids = df['recipe_id'].nunique()
    # Count the number of unique ingredients
    num_unique_ingredients = df['ingredients'].nunique()
    # Count the number of unique combinations of user_id and recipe_id
    num_interactions = df[['user_id', 'recipe_id']].drop_duplicates().shape[0]

    # Print the counts
    print(f"Number of unique user_ids: {num_unique_user_ids}")
    print(f"Number of unique recipe_ids: {num_unique_recipe_ids}")
    print(f"Number of interactions between users and recipes: {num_interactions}")
    print(f"Number of unique ingredients: {num_unique_ingredients}")

def Load_Into_Graph(df):
    """Given a data frame with columns 'user_id', 'recipe_id',
    'ingredients', and 'nutrition', construct a multigraph with the
    following schema:

    Nodes:
    * user: identified with user_id
    * recipe: identified with recipe_id
    * ingredients: identified with ingredient string
    * nutrient: one of nutrients below

    Edges:
    * user -> recipe, if user rated recipe, with the rating as the weight
    * recipe -> ingredients, if recipe contains that ingredient
    * recipe -> nutrient, if recipe contains that nutrient, with the amount as the weight

    Note: Ingredient and nutrient lists are included in the data frame
        as Python-like lists, e.g., "['salt', 'wheat flour', 'rice']"
        for ingredients and [1,.5,0] for nutrients. They are therefore
        decoded.

    """

    logging.info("Loading data into a graph...")
    
    # Create an empty graph
    G = nx.Graph()

    nutrients = ["Proteins", "Carbohydrates", "Sugars",
                "Sodium", "Fat", "Saturated_fats", "Fibers"]
    G.add_nodes_from(nutrients, node_type='nutrition')

    # Iterate through the data and populate the graph
    for uid, rid, r, ing, nut in df[['user_id', 'recipe_id', 'rating', 'ingredients', 'nutrition']].itertuples(False, None):
        # Add user_id, recipe_id
        G.add_node(f"u{uid}", node_type='user')
        G.add_node(f"r{rid}", node_type='recipe')

        # Add edges between user_id and recipe_id
        G.add_edge(f"u{uid}", f"r{rid}", weight=r, edge_type='rating')

        # Add new ingredients as nodes
        if type(ing) is str:
            # Remove brackets and single quotes
            ing = eval(ing)
            G.add_nodes_from(ing, node_type='ingredients')
            # Add edges between recipe_id and ingredients
            for i in ing: 
                G.add_edge(f"r{rid}", i, edge_type='ingredient')

        # Add edges between recipe_id and nutrients
        if type(nut) is str:
            nuts = eval(nut)
            for j, nut in enumerate(nutrients):
                if nuts[j] > 0:
                    G.add_edge(f"r{rid}", nut, weight=nuts[j], edge_type='nutrition')

    logging.info("Finished; resulting graph:")
    logging.info(G)
    return G

def Heterogeneous_Graph(df):
    # Populate the heterogeneous graph
    G = Load_Into_Graph(df)

    # Define the meta-paths
    meta_paths = [
        ['user_id', 'recipe_id', 'nutrition', 'ingredients'],
        ['user_id', 'recipe_id'],
        ['user_id', 'recipe_id', 'ingredients', 'nutrition'],
        ['recipe_id', 'nutrition', 'ingredients'],
        ['recipe_id', 'ingredients', 'nutrition']
    ]

    # Print the edges and their attributes for each meta-path
    for meta_path in meta_paths:
        # logging.info("Meta-Path: %s", " -> ".join(meta_path))
        paths = []

        # Check if the meta-path starts with 'user_id' and ends with 'ingredients'
        if meta_path[0] == 'user_id' and meta_path[-1] == 'ingredients':
            for uid in G.nodes():
                if G.nodes[uid]['node_type'] == 'user':
                    for rid in G.neighbors(uid):
                        if G.nodes[rid]['node_type'] == 'recipe':
                            for ing in G.neighbors(rid):
                                if G.nodes[ing]['node_type'] == 'ingredients':
                                    paths.append([f"{uid}", f"{rid}", ing])

        # Check if the meta-path starts with 'user_id' and ends with 'nutrition'
        elif meta_path[0] == 'user_id' and meta_path[-1] == 'nutrition':
            for uid in G.nodes():
                if G.nodes[uid]['node_type'] == 'user':
                    for rid in G.neighbors(uid):
                        if G.nodes[rid]['node_type'] == 'recipe':
                            for nut in G.neighbors(rid):
                                if G.nodes[nut]['node_type'] == 'nutrition':
                                    for ing in G.neighbors(rid):
                                        if G.nodes[ing]['node_type'] == 'ingredients':
                                            paths.append([f"{uid}", f"{rid}", nut, ing])

        
        # Print only the first 5 paths for each meta-path
        # for i, path in enumerate(paths[:5]):
        #     logging.info("Path:", path)
        #     for j in range(len(path) - 1):
        #         source = path[j]
        #         target = path[j + 1]
        #         edges = G.get_edge_data(source, target)
        #         if edges is not None:
        #             for key, data in edges.items():
        #                 logging.info("Source:", source)
        #                 logging.info("Target:", target)
        #                 logging.info("Edge Data:", data)  # Print all edge data
        #         else:
        #             logging.info("No edges between", source, "and", target)
        #     logging.info()

# Define the NLA class
class NLA(nn.Module):
    def __init__(self, num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths):
        super(NLA, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.nutrition_embedding = nn.Embedding(num_nutrition, embedding_dim)

        # Convert the paths to tensors
        self.paths = paths.clone().detach() if paths is not None else None

    def forward(self, uid, rid, ing, nut):
        user_emb = self.user_embedding(uid)
        recipe_emb = self.recipe_embedding(rid)
        ingredient_emb = self.ingredient_embedding(ing)
        nutrition_emb = self.nutrition_embedding(nut)

        if self.paths is not None:
            path_scores = torch.zeros(uid.size(0), len(self.paths))
            for i, path in enumerate(self.paths):
                path = torch.tensor(path).clone().detach()
                matching_uid = torch.where(uid == path[0])[0]
                matching_rid = torch.where(rid == path[1])[0]
                matching_ing = torch.where(ing == path[2])[0]
                matching_nut = torch.where(nut == path[3])[0]  # Fix this line
                
                # Check if there are any matching indices
                if matching_uid.size(0) > 0 and matching_rid.size(0) > 0 and matching_ing.size(0) > 0 and matching_nut.size(0) > 0:
                    matching_count = min(matching_uid.size(0), matching_rid.size(0), matching_ing.size(0), matching_nut.size(0))
                    matching_indices = torch.stack((matching_uid[:matching_count], matching_rid[:matching_count], matching_ing[:matching_count], matching_nut[:matching_count]))
                    path_scores[matching_indices] += 1
                    
            # Node-Level Attention
            k = 3  # Number of iterations
            node_emb_theta = torch.zeros(user_emb.size(0), user_emb.size(1))
            for i in range(k):
                attention_scores = F.leaky_relu(node_emb_theta, negative_slope=0.01)
                attention_scores = F.softmax(attention_scores, dim=1)
                weighted_attention = attention_scores.unsqueeze(2) * user_emb.unsqueeze(1)

            aggregated_attention = torch.sum(weighted_attention, dim=1)

            # Determine the maximum size along dimension 0
            max_size = max(user_emb.size(0), user_emb.size(0), aggregated_attention.size(0))

            # Pad tensors to match the maximum size along dimension 0
            user_emb = F.pad(user_emb, (0, 0, 0, max_size - user_emb.size(0)))
            recipe_emb = F.pad(recipe_emb, (0, 0, 0, max_size - recipe_emb.size(0)))
            aggregated_attention = F.pad(aggregated_attention, (0, 0, 0, max_size - aggregated_attention.size(0)))

            # Concatenate and return the final embedding
            node_embeddings = torch.cat((user_emb, recipe_emb, aggregated_attention), dim=1)
        else:
            # Concatenate the embeddings without attention
            node_embeddings = torch.cat((user_emb, recipe_emb, ingredient_emb, nutrition_emb), dim=1)

        return node_embeddings

    def train_nla(self, df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder, num_epochs=30):
        criterion_nla = nn.MSELoss()
        optimizer_nla = optim.Adam(self.parameters(), lr=0.01)

        dataset = HeterogeneousDataset(df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            running_loss_nla = 0.0
            for uid, rid, ing, nut, label in data_loader:
                optimizer_nla.zero_grad()

                # Forward pass
                embeddings = self(uid, rid, ing, nut)

                # Modify the target tensor to have the same size as embeddings
                label = label.unsqueeze(1).float()
                label = label.repeat(1, embeddings.size(1))  # Repeat label values to match the size of embeddings

                # Calculate the loss
                loss_nla = criterion_nla(embeddings, label)
                running_loss_nla += loss_nla.item()

                # Backward pass and optimization
                loss_nla.backward()
                optimizer_nla.step()

            avg_loss_nla = running_loss_nla / len(data_loader)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, NLA Loss: {avg_loss_nla:.4f}")
        
        # Return the final NLA loss value
        return avg_loss_nla

    def get_embeddings(self, uid, rid, ing, nut):
        # Forward pass to get embeddings
        with torch.no_grad():
            embeddings = self(uid, rid, ing, nut)
        return embeddings
    
# Define the dataset class
class HeterogeneousDataset(Dataset):
    def __init__(self, df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder):
        self.uids = user_encoder.transform(df['user_id'])
        self.rids = recipe_encoder.transform(df['recipe_id'])
        self.ings = ingredient_encoder.transform(df['ingredients'])
        self.nuts = nutrition_encoder.transform(df['nutrition'])
        self.labels = df['rating'].astype(float).values

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        rid = self.rids[idx]
        ing = self.ings[idx]
        nut = self.nuts[idx]
        label = self.labels[idx]
        return uid, rid, ing, nut, label

def find_paths_users_interests(df):
    # Populate the heterogeneous graph
    G = Load_Into_Graph(df)

    # Calculate the average rating for each recipe_id and create a new column 'avg_rating'
    df['avg_rating'] = df.groupby('recipe_id')['rating'].mean()

    # Print the meta-path
    meta_path = ['user_id', 'recipe_id', 'ingredient', 'nutrition']
    # logging.info("Meta-Path:", " -> ".join(meta_path))

    paths = []
    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
            for rid in user_rated_recipes:
                # Check if there are matching rows in df before accessing by index
                matching_rows = df[df['recipe_id'] == rid]
                if not matching_rows.empty:
                    if matching_rows['rating'].iloc[0] >= matching_rows['avg_rating'].iloc[0]:  # Use 'avg_rating' from matching_rows
                        ingredient_node = []
                        nutrition_node = []

                        for node in G.neighbors(rid):
                            if G.nodes[node]['node_type'] == 'ingredients':
                                ingredient_node.append(node)
                            elif G.nodes[node]['node_type'] == 'nutrition':
                                nutrition_node.append(node)

                        for ing in ingredient_node:
                            for nut in nutrition_node:
                                paths.append([uid, rid, ing, nut])

    # Encode the paths using label encoders
    user_encoder = LabelEncoder()
    recipe_encoder = LabelEncoder()
    ingredient_encoder = LabelEncoder()
    nutrition_encoder = LabelEncoder()
    user_encoder.fit([path[0] for path in paths])
    recipe_encoder.fit([path[1] for path in paths])
    ingredient_encoder.fit([path[2] for path in paths])
    nutrition_encoder.fit([path[3] for path in paths])

    encoded_paths = [[user_encoder.transform([path[0]])[0], recipe_encoder.transform([path[1]])[0], ingredient_encoder.transform([path[2]])[0], nutrition_encoder.transform([path[3]])[0]] for path in paths]

    # Convert paths to tensors
    paths_tensor = torch.tensor(encoded_paths, dtype=torch.long).clone().detach()

    # Print the first 5 filtered paths
    # for i, (path, encoded_path) in enumerate(zip(paths, encoded_paths)):
    #     logging.info("Original Path:", path)
    #     logging.info("Encoded Path:", encoded_path)
    #     if i == 5:
    #         break

    return paths_tensor, meta_path

class SLA(nn.Module):
    def __init__(self, num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths, is_healthy=False):
        super(SLA, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.nutrition_embedding = nn.Embedding(num_nutrition, embedding_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),  # Output size matches embedding_dim
            nn.LeakyReLU(negative_slope=0.01),  
        )
        self.is_healthy = is_healthy  
        self.paths = paths.clone().detach() if paths is not None else None

    def calculate_impression_coefficient(self, source_embedding, destination_embedding):
        # Calculate the impression coefficient using source and destination embeddings
        impression_coefficient = torch.matmul(source_embedding, destination_embedding.T)
        return impression_coefficient

    def calculate_weight(self, impression_coefficient):
        # Calculate the weight using leakyReLU activation
        weight = torch.sum(F.leaky_relu(impression_coefficient, negative_slope=0.01))
        return weight

    def forward(self, uid, rid, ing, nut, is_healthy=None):
        if is_healthy is None:
            is_healthy = self.is_healthy
        else:
            is_healthy = F.leaky_relu(is_healthy, negative_slope=0.01)

        user_emb = self.user_embedding(uid)
        recipe_emb = self.recipe_embedding(rid)
        ingredient_emb = self.ingredient_embedding(ing)
        nutrition_emb = self.nutrition_embedding(nut)

        # Determine the maximum size along dimension 0
        max_size = max(user_emb.size(0), recipe_emb.size(0), ingredient_emb.size(0), nutrition_emb.size(0))

        # Pad tensors to match the maximum size along dimension 0
        user_emb = F.pad(user_emb, (0, 0, 0, max_size - user_emb.size(0)))
        recipe_emb = F.pad(recipe_emb, (0, 0, 0, max_size - recipe_emb.size(0)))
        ingredient_emb = F.pad(ingredient_emb, (0, 0, 0, max_size - ingredient_emb.size(0)))
        nutrition_emb = F.pad(nutrition_emb, (0, 0, 0, max_size - nutrition_emb.size(0)))

        # Concatenate and return the final embedding
        node_embeddings = torch.cat((user_emb, recipe_emb, ingredient_emb, nutrition_emb), dim=1)

        # Calculate the impression coefficient based on the meta-path
        impression_coefficient = self.calculate_impression_coefficient(user_emb, recipe_emb)

        # Calculate the softmax of the impression coefficient
        softmax_impression_coefficient = F.softmax(impression_coefficient, dim=1)

        # Calculate the weight
        weight = self.calculate_weight(softmax_impression_coefficient)

        return node_embeddings, impression_coefficient, weight
    
    # def edge_loss(self, impression_coefficient):
    #     # Calculate the log loss based on the impression coefficient
    #     log_loss = torch.log(1 / (1 + torch.exp(impression_coefficient))) 
    #     # Sum all the log losses
    #     loss = -torch.sum(log_loss)  
    #     return loss
    
    def edge_loss(self, weight):
        loss = -torch.log(1 / (1 + torch.exp(weight)))
        return loss.mean()

    def train_sla(self, uid_tensor, rid_tensor, ing_tensor, nut_tensor, num_epochs_sla=30):
        optimizer_sla = optim.Adam(self.parameters(), lr=0.01)

        for epoch_sla in range(num_epochs_sla):
            optimizer_sla.zero_grad()

            # Forward pass
            node_embeddings, impression_coefficient, weight = self(uid_tensor, rid_tensor, ing_tensor, nut_tensor)

            # Calculate the loss using the edge_loss function
            loss_sla = self.edge_loss(impression_coefficient)  # Use impression_coefficient for loss calculation
            loss_sla.backward()
            optimizer_sla.step()

            # Print the loss for SLA
            logging.info(f"Epoch SLA {epoch_sla + 1}/{num_epochs_sla}, SLA Loss: {loss_sla.item():.4f}")

        # Print the aggregated ingredient embeddings from SLA (for healthy recipes)
        logging.info("Embeddings Vectors (SLA) based Healthy recipes:")
        logging.info(node_embeddings)

# Define the is_healthy function
def is_healthy(food_data):
    fibres = food_data[0]
    fat = food_data[1]
    sugar = food_data[2]
    sodium = food_data[3]
    protein = food_data[4]
    saturated_fat = food_data[5]
    carbohydrates = food_data[6]
    
    conditions_met = 0
    
    if fibres > 10:
        conditions_met += 1
    if 15 <= fat <= 30:
        conditions_met += 1
    if sugar < 10:
        conditions_met += 1
    if sodium < 5:
        conditions_met += 1
    if 10 <= protein <= 15:
        conditions_met += 1
    if saturated_fat < 10:
        conditions_met += 1
    if 55 <= carbohydrates <= 75:
        conditions_met += 1
    
    return conditions_met >= 3

def find_healthy_foods(df):
    # Populate the heterogeneous graph
    G = Load_Into_Graph(df)

    paths = []
    healthy_foods = set()  # Store healthy recipes here

    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
            for rid in user_rated_recipes:
                # Check if there are matching rows in df before accessing by index
                matching_rows = df[df['recipe_id'] == rid]
                if not matching_rows.empty:
                    nutrition_health = [int(token) for token in eval(matching_rows['nutrition'].iloc[0]) if token.strip().isdigit()]
                    is_healthy_food = is_healthy(nutrition_health)
                    ingredient_node = []
                    nutrition_node = []
                    for node in G.neighbors(rid):
                        if G.nodes[node]['node_type'] == 'ingredients':
                            ingredient_node.append(node)
                        elif G.nodes[node]['node_type'] == 'nutrition':
                            nutrition_node.append(node)
                    for ing in ingredient_node:
                        for nut in nutrition_node:
                            paths.append([uid, rid, ing, nut])
                    if is_healthy_food:
                        healthy_foods.add(rid)  # Add the recipe to healthy foods

    # Encode the paths using label encoders
    recipe_encoder = LabelEncoder()
    recipe_encoder.fit(list(healthy_foods))
    encoded_paths = [[path[1]] for path in paths if path[1] in healthy_foods]

    # Convert paths to tensors
    paths_tensor = torch.tensor(encoded_paths, dtype=torch.long)
    
    return paths_tensor

def normalize_summed_embeddings(summed_embeddings):
    # Detach PyTorch tensors
    summed_embeddings = summed_embeddings.detach().numpy()

    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit the scaler on the summed_embeddings and transform them
    normalized_embeddings = scaler.fit_transform(summed_embeddings)

    return normalized_embeddings

def normalize_user_embeddings(summed_embeddings):
    # Detach PyTorch tensors
    summed_embeddings = summed_embeddings.detach().numpy()

    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit the scaler on the summed_embeddings and transform them
    normalized_embeddings = scaler.fit_transform(summed_embeddings)

    # Create a list to store tuples of (user_id, normalized_embedding)
    user_normalized_embeddings = []

    # Iterate through user embeddings and add user_id to each embedding
    for user_id, embedding_row in enumerate(normalized_embeddings):
        user_normalized_embeddings.append((user_id, embedding_row))

    return user_normalized_embeddings

def rate_healthy_recipes_for_user(user_id, df):
    # Filter the data for the specified user_id
    user_data = df[df['user_id'] == user_id]

    # Get the healthy recipes for the user
    user_healthy_recipes = set()
    for index, row in user_data.iterrows():
        recipe_id = row['recipe_id']
        nutrition_health = eval(row['nutrition'])
        
        # Check if the recipe is healthy based on the 'is_healthy' function
        if is_healthy(nutrition_health):
            user_healthy_recipes.add(recipe_id)

    # Print the results
    # print(f"User ID: {user_id}")
    # print("Healthy Recipes:")
    # for recipe_id in user_healthy_recipes:
    #     print(f"Recipe ID: {recipe_id}")

    return user_healthy_recipes

# Recommend_users_for_healthy_recipes function
def recommend_users_for_healthy_recipes(df, normalized_embeddings, similarity_threshold=0.3, top_n_similar=5):
    recommendations = {}
    user_to_embedding = {}  # Mapping from user ID to embedding

    # Calculate cosine similarities between user embeddings
    similarities = cosine_similarity(normalized_embeddings)
    num_rows = len(df)  # Get the number of rows in the DataFrame

    for i, user_embedding in enumerate(normalized_embeddings):
        if i >= num_rows:
            break  # Ensure that i is within the valid range

        user_id = df.iloc[i]['user_id']  # Get the user_id from the DataFrame

        # Find similar users based on cosine similarity and get the top N similar users
        similar_users = sorted(
            [(j, similarity_score) for j, similarity_score in enumerate(similarities[i]) if j != i and similarity_score >= similarity_threshold],
            key=lambda x: x[1],  # Sort by similarity score
            reverse=True  # Sort in descending order
        )[:top_n_similar]

        # Recommend healthy recipes for the user
        recommended_recipes = set()
        for similar_user_index, _ in similar_users:
            if similar_user_index < num_rows:  # Check if the index is within the valid range
                similar_user_id = df.iloc[similar_user_index]['user_id']  # Get similar user's user_id
                # Exclude the user themselves from recommendations
                if similar_user_id != user_id:
                    # Exclude recipes the current user has already rated
                    user_rated_recipes = set(df[df['user_id'] == user_id]['recipe_id'])

                    # Recommend recipes that are not rated by the current user
                    recommended_recipes.update(recipe for recipe in df['recipe_id'] if recipe not in user_rated_recipes)

                    # Limit to the top N recommendations
                    if len(recommended_recipes) >= top_n_similar:
                        break

        # recommendations[user_id] = {
        #     'most_similar_user_ids': [df.iloc[j]['user_id'] for j, _ in similar_users if j < num_rows],
        #     'user_healthy_recipes': recommended_recipes
        # }

    return recommendations

#---------------Evaluation Library- Three Metrics -------------------------------
def load_ground_truth_ratings(files_to_read, folder_path):
    ground_truth_ratings = {}
    ground_truth_labels = set()  # Use a set to ensure unique user IDs
    test_set_users = set()  # Use a set to ensure unique test set user IDs

    for file in files_to_read:
        if file == 'Food_Dataset.zip':
            interactions_df = pd.read_csv(os.path.join(folder_path, file), dtype=str)
            for index, row in interactions_df.iterrows():
                user_id = int(row['user_id'])  # Convert the user_id to an integer
                recipe_id = int(row['recipe_id'])  # Convert the recipe_id to an integer
                rating = int(row['rating'])  # Convert the rating to an integer

                # Check if the user is in the test set and add to the test_set_users set
                if user_id not in test_set_users:
                    test_set_users.add(user_id)

                # Create a dictionary to store ratings for each user and each recipe if not already created
                if user_id not in ground_truth_ratings:
                    ground_truth_ratings[user_id] = {}

                if recipe_id not in ground_truth_ratings[user_id]:
                    ground_truth_ratings[user_id][recipe_id] = {'positive': [], 'negative': []}

                # Classify ratings as positive or negative (you can define your own criteria)
                if rating >= 3:
                    ground_truth_ratings[user_id][recipe_id]['positive'].append(rating)
                else:
                    ground_truth_ratings[user_id][recipe_id]['negative'].append(rating)

    return ground_truth_ratings, ground_truth_labels, test_set_users

def evaluate_recommendation_system(user_normalized_embeddings, ground_truth_ratings, k):
    # Extract user IDs and normalized embeddings
    user_ids, normalized_embeddings = zip(*user_normalized_embeddings)

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(normalized_embeddings, user_ids, test_size=0.2, random_state=42)

    # Compute cosine similarity between user embeddings in the test set
    similarity_matrix = cosine_similarity(X_test, normalized_embeddings)

    # Initialize lists to store evaluation metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for i, test_user_id in enumerate(y_test):
        # Get the cosine similarity scores for the current user
        similarity_scores = similarity_matrix[i]

        # Sort user IDs by similarity (higher similarity first)
        similar_user_ids = [user_id for _, user_id in sorted(zip(similarity_scores, user_ids), reverse=True)]

        # Get the top-k recommendations
        recommended_user_ids = similar_user_ids[:k]

        # Calculate true labels (1 if user is in ground truth ratings, 0 otherwise)
        true_labels = [1 if user_id in ground_truth_ratings else 0 for user_id in recommended_user_ids]

        # Calculate precision, recall, and F1-score using sklearn.metrics
        precision = precision_score(true_labels, [1] * len(true_labels), zero_division=0)
        recall = recall_score(true_labels, [1] * len(true_labels), zero_division=0)
        f1 = f1_score(true_labels, [1] * len(true_labels), zero_division=0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Compute the mean of evaluation metrics
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)

    return mean_precision, mean_recall, mean_f1

#---------------Evaluation based-AUC -------------------------------

def AUC_ground_truth_ratings(files_to_read, folder_path):
    A_ground_truth_ratings = []  # A list to store ground truth ratings

    for file in files_to_read:
        if file == 'Food_Dataset.zip':
            interactions_df = pd.read_csv(os.path.join(folder_path, file), dtype=str)
            ratings = interactions_df['rating'].astype(int).tolist()
            A_ground_truth_ratings.extend(ratings)

    return A_ground_truth_ratings

def AUC_evaluate_recommendation_system(user_normalized_embeddings, A_ground_truth_ratings):
    # Extract user IDs and normalized embeddings
    user_ids, normalized_embeddings = zip(*user_normalized_embeddings)

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(normalized_embeddings, user_ids, test_size=0.2, random_state=42)

    # Compute cosine similarity between user embeddings in the test set
    similarity_matrix = cosine_similarity(X_test, normalized_embeddings)

    # Initialize a list to store AUC scores
    auc_scores = []

    for i, test_user_id in enumerate(y_test):
        # Get the cosine similarity scores for the current user
        similarity_scores = similarity_matrix[i]

        # Sort user IDs by similarity (higher similarity first)
        similar_user_ids = [user_id for _, user_id in sorted(zip(similarity_scores, user_ids), reverse=True)]

        # Get all recommended user IDs
        recommended_user_ids = similar_user_ids

        # Get the ratings for the recommended recipes for the test user
        recommended_ratings = [A_ground_truth_ratings[recipe_id] for recipe_id in recommended_user_ids]

        # Calculate AUC for this user
        auc = roc_auc_score([1 if rating >= 2 else 0 for rating in recommended_ratings], similarity_scores)
        auc_scores.append(auc)

    # Compute the mean AUC score
    mean_auc = np.nanmean(auc_scores)

    return mean_auc

#------------Evaluation based-NDCG-------------------------------------------------

def NDCG_ground_truth_ratings(files_to_read, folder_path):
    ND_ground_truth_ratings = {}  # Dictionary to store ground truth ratings

    for file in files_to_read:
        if file == 'Food_Dataset.zip':
            interactions_df = pd.read_csv(os.path.join(folder_path, file), dtype=str)
            for index, row in interactions_df.iterrows():
                user_id = int(row['user_id'])  # Convert the user_id to an integer
                recipe_id = int(row['recipe_id'])  # Convert the recipe_id to an integer
                rating = int(row['rating'])  # Convert the rating to an integer

                # Create a dictionary to store ratings for each user if not already created
                if user_id not in ND_ground_truth_ratings:
                    ND_ground_truth_ratings[user_id] = []

                # Append the rating to the user's list of ground truth ratings
                ND_ground_truth_ratings[user_id].append(rating)

    return ND_ground_truth_ratings

def calculate_ndcg(true_labels, scores, k):
    # Sort the true labels and scores by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    true_labels = np.array(true_labels)[sorted_indices]
    scores = np.array(scores)[sorted_indices]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = np.sum((2 ** true_labels - 1) / np.log2(np.arange(2, k + 2)))

    # Calculate Ideal Discounted Cumulative Gain (IDCG) for perfect ranking
    idcg = np.sum((2 ** np.sort(true_labels)[::-1] - 1) / np.log2(np.arange(2, k + 2)))

    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg

def NDCG_Evaluation(user_normalized_embeddings, ND_ground_truth_ratings, similarity_threshold, k):
    # Extract user IDs and normalized embeddings
    user_ids, normalized_embeddings = zip(*user_normalized_embeddings)

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(normalized_embeddings, user_ids, test_size=0.2, random_state=42)

    # Compute cosine similarity between user embeddings in the test set
    similarity_matrix = cosine_similarity(X_test, normalized_embeddings)

    # Initialize lists to store NDCG scores
    ndcg_scores = []

    for i, test_user_id in enumerate(y_test):
        # Get the cosine similarity scores for the current user
        similarity_scores = similarity_matrix[i]

        # Determine the top-k recommendations based on similarity scores and threshold
        recommended_user_ids = [user_id for user_id, similarity in zip(user_ids, similarity_scores) if similarity >= similarity_threshold][:k]

        # Check if there are recommendations
        if recommended_user_ids:
            # Calculate NDCG for this user using the calculate_ndcg function
            ndcg = calculate_ndcg([1] * len(recommended_user_ids), similarity_scores[:k], k)
            ndcg_scores.append(ndcg)

    # Compute the mean NDCG score
    mean_ndcg = np.mean(ndcg_scores)

    return mean_ndcg

#----------------------------------------------------
def main():

    # Call the process_data function
    process_data(folder_path, files_to_read)

    # Call the Heterogeneous_Graph function
    Heterogeneous_Graph(df)
    
   # Call the find_paths_users_interests function
    paths_tensor, meta_path = find_paths_users_interests(df)
    
    # Get the unique node counts
    num_users = len(df['user_id'].unique())
    num_recipes = len(df['recipe_id'].unique())
    num_ingredients = len(df['ingredients'].unique())
    num_nutrition = len(df['nutrition'].unique())

    logging.info("P4) Get the unique node counts")

    # Initialize the label encoders and fit them with the data
    user_encoder = LabelEncoder()
    recipe_encoder = LabelEncoder()
    ingredient_encoder = LabelEncoder()
    nutrition_encoder= LabelEncoder()
    user_encoder.fit(df['user_id'])
    recipe_encoder.fit(df['recipe_id'])
    ingredient_encoder.fit(df['ingredients'])
    nutrition_encoder.fit(df['nutrition'])
    
    # Common embedding dimension
    embedding_dim = 256

    # Initialize the NLA model with the common dimension
    nla_model = NLA(num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths_tensor)

    # Train the NLA model
    NLA_loss = nla_model.train_nla(df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder, num_epochs=10)
    
    # Get and print the embeddings
    uid_tensor = torch.LongTensor(list(range(num_users)))
    rid_tensor = torch.LongTensor(list(range(num_recipes)))
    ing_tensor = torch.LongTensor(list(range(num_ingredients)))
    nut_tensor = torch.LongTensor(list(range(num_nutrition)))
    embeddings_nla = nla_model.get_embeddings(uid_tensor, rid_tensor, ing_tensor, nut_tensor)

    logging.info("Embedding Vectors (NLA):")
    logging.info(embeddings_nla)
    
    # Create an SLA instance for healthy foods with the same common dimension
    sla_for_healthy_foods = SLA(num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths_tensor, is_healthy=True)
    
    # Train the SLA model for healthy foods
    sla_for_healthy_foods.train_sla(uid_tensor, rid_tensor, ing_tensor, nut_tensor, num_epochs_sla=30)

    # Train the SLA model for healthy foods
    embeddings_for_healthy_foods, _, _ = sla_for_healthy_foods(uid_tensor, rid_tensor, ing_tensor, nut_tensor)

    # Find the smaller size between the two tensors' number of rows
    min_size = min(embeddings_nla.shape[0], embeddings_for_healthy_foods.shape[0])

    # Resize both tensors to the same size (number of rows)
    embeddings_nla = embeddings_nla[:min_size]
    embeddings_for_healthy_foods = embeddings_for_healthy_foods[:min_size]

    # Find the larger dimension between the two tensors' embedding dimensions
    embedding_dim = max(embeddings_nla.shape[1], embeddings_for_healthy_foods.shape[1])

    # Pad both tensors with zeros along dimension 1 to match the larger dimension
    padding_dim_nla = embedding_dim - embeddings_nla.shape[1]
    padding_dim_healthy = embedding_dim - embeddings_for_healthy_foods.shape[1]

    zero_padding_nla = torch.zeros(embeddings_nla.shape[0], padding_dim_nla)
    zero_padding_healthy = torch.zeros(embeddings_for_healthy_foods.shape[0], padding_dim_healthy)

    embeddings_nla = torch.cat((embeddings_nla, zero_padding_nla), dim=1)
    embeddings_for_healthy_foods = torch.cat((embeddings_for_healthy_foods, zero_padding_healthy), dim=1)

    # Now both embeddings have the same size and dimensions
    summed_embeddings = embeddings_nla + embeddings_for_healthy_foods

    # normalized_Embeddings vectors of summed_embeddings
    normalized_embeddings = normalize_summed_embeddings(summed_embeddings)
    normalize_user_id_embeddings = normalize_user_embeddings(summed_embeddings)
   
    # Define a list of user IDs for which you want to find healthy recipes (e.g., the first five users)
    user_ids_to_rate = df['user_id'].unique()[:5]

    for user_id in user_ids_to_rate:
        # Call the function to rate healthy recipes for each user
        user_healthy_recipes = rate_healthy_recipes_for_user(user_id, df)

    recommendations = recommend_users_for_healthy_recipes(df, normalized_embeddings, similarity_threshold= 0.3, top_n_similar=5)

    # # Print the top 5 similar users for the first 5 users
    # user_ids = list(recommendations.keys())[:5]  # Get the first 5 user IDs
    # for user_id in user_ids:
    #     print(f"User ID: {user_id}")
    #     print("Top 5 Similar Users:")
    #     for similar_user_id in recommendations[user_id]['most_similar_user_ids']:
    #         print(f"Similar User ID: {similar_user_id}")
    #     print("\n")

    # Call the function to load and process the data
    ground_truth_ratings, ground_truth_labels, test_set_users = load_ground_truth_ratings(files_to_read, folder_path)
    
    # Library Metris Precision, Recall and F1-score
    k_values = list(range(1, 21)) + [30, 40, 50]
    print("Results for Different Values of Library Metris based k:")
    print("=" * 55)
    print(f"{'k':<5}{'Mean Precision':<20}{'Mean Recall':<20}{'Mean F1-score':<20}")
    print("=" * 55)

    for k in k_values:
        mean_precision, mean_recall, mean_f1 = evaluate_recommendation_system(normalize_user_id_embeddings, ground_truth_ratings, k)
        print(f"{k:<5}{mean_precision:.4f}{' ':<5}{mean_recall:.4f}{' ':<5}{mean_f1:.4f}")

#--------Evaluation based-AUC -------------------

    # Call AUC_ground_truth_ratings to construct ground truth ratings dictionary
    A_ground_truth_ratings = AUC_ground_truth_ratings(files_to_read, folder_path)
    
    # Call AUC_evaluate_recommendation_system to calculate the mean AUC score
    mean_auc = AUC_evaluate_recommendation_system (normalize_user_id_embeddings, A_ground_truth_ratings)

    print("=====================================")
    print(f"Mean AUC Score: {mean_auc:.4f}")
    print("=====================================")
    
    #--------------------------------------------------------------------
    # Call the NDCG_ground_truth_ratings function
    ND_ground_truth_ratings = NDCG_ground_truth_ratings(files_to_read, folder_path)

    # Calculate and print NDCG scores for different values of k
    similarity_threshold = 0.3  # Set your desired similarity threshold
    for k_NDCG in range(1, 11):
        mean_ndcg = NDCG_Evaluation(normalize_user_id_embeddings, ND_ground_truth_ratings, similarity_threshold, k_NDCG)
        
        print(f'NDCG Score for k={k_NDCG}: {mean_ndcg:.4f}')

main()