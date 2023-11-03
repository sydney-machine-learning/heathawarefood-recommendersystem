#!/usr/bin/env -S python3 -u
#PBS -N Saman
#PBS -l select=1:ncpus=64:mem=512gb
#PBS -l walltime=72:00:00
#PBS -v OMP_NUM_THREADS=64
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
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


# Configure the logging system
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

# Now you can use logging to write log messages
logging.info('This is an informational log message.')

folder_path = '/home/z5318340'
files_to_read = ['Food_Dataset_S.zip']  
file_path = '/home/z5318340/Food_Dataset_S.zip'

# folder_path = r"C:\Food"
# files_to_read = ['Food_Dataset.zip']
# file_path = r"C:\Food\Food_Dataset.zip"

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
            if file == 'Food_Dataset_S.zip':
                # Specify the data type for the user_id column as int
                df = pd.read_csv(file_path, dtype={'user_id': int})
                user_id = df['user_id']
                recipe_id = df['recipe_id']
                rating = df['rating']
                ingredients = df['ingredients']
                nutrition = df['nutrition']

                # Iterate through user_id, recipe_id, and rating columns to calculate total score and count for each recipe_id
                for i in range(len(user_id)):
                    uid = user_id[i]
                    rid = recipe_id[i]
                    r = rating[i]
                    if rid not in recipe_scores:
                        recipe_scores[rid] = {'total_score': 0, 'count': 0}
                    recipe_scores[rid]['total_score'] += r
                    recipe_scores[rid]['count'] += 1

    # Count the number of unique user_ids
    num_unique_user_ids = df['user_id'].nunique()
    # Count the number of unique recipe_ids
    num_unique_recipe_ids = df['recipe_id'].nunique()
    # Count the number of unique ingredients
    num_unique_ingredients = df['ingredients'].nunique()
    # Count the number of unique combinations of user_id and recipe_id
    num_interactions = df[['user_id', 'recipe_id']].drop_duplicates().shape[0]
    sparsity = (1 - (num_interactions / (num_unique_user_ids * num_unique_recipe_ids))) * 100
       
    # Group the data by 'user_id' and count the number of unique 'recipe_id' each user has rated
    user_rating_counts = df.groupby('user_id')['recipe_id'].nunique()

    # Now, you can create a histogram based on the number of users who have rated a certain number of items
    user_count_histogram = user_rating_counts.value_counts().sort_index()

    
    # Print the counts
    print(f"Number of unique user_ids: {num_unique_user_ids}")
    print(f"Number of unique recipe_ids: {num_unique_recipe_ids}")
    print(f"Number of interactions between users and recipes: {num_interactions}")
    print(f"Number of unique ingredients: {num_unique_ingredients}")
    print("************************************")
    print(f"Sparsity of the dataset: {sparsity:.2f}%")
    print("************************************")
    print("Number of Users per Rating Count:")
    for count, users in user_count_histogram.items():
        print(f" [{count}] {users} users have {count} rated recipes")
    print("************************************")

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
            # Node-Level Attention
            weighted_attention = user_emb.unsqueeze(1) / user_emb.size(1)

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

    def train_nla(self, df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder, num_epochs):
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
    
    def edge_loss(self, weight):
        loss = -torch.log(1 / (1 + torch.exp(weight)))
        return loss.mean()

    def train_sla(self, uid_tensor, rid_tensor, ing_tensor, nut_tensor, num_epochs_sla):
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

    return user_healthy_recipes

def load_ground_truth_ratings(files_to_read, folder_path):
    ground_truth_ratings = []

    for file in files_to_read:
        if file == 'Food_Dataset_S.zip':
            interactions_df = pd.read_csv(os.path.join(folder_path, file), dtype=str)
            ground_truth_ratings.extend(
                [
                    (int(row['user_id']), int(row['recipe_id']), int(row['rating']))
                    for _, row in interactions_df.iterrows()
                ]
            )

    return ground_truth_ratings

# def Recommendation_healthy_recipes(df, normalized_embeddings, similarity_threshold, top_k_values, top_n_popular):
#     recommendations = {}
#     actual_ratings = {}  # Dictionary to store actual ratings for each user

#     # Calculate cosine similarities between user embeddings
#     similarities = cosine_similarity(normalized_embeddings)

#     for i, user_embedding in enumerate(normalized_embeddings):
#         user_id = df.iloc[i]['user_id']  # Get the user_id from the DataFrame

#         # Find similar users based on cosine similarity and the similarity threshold
#         similar_users = [
#             (j, similarity_score) for j, similarity_score in enumerate(similarities[i])
#             if j != i and similarity_score >= similarity_threshold
#         ]

#         # Sort similar users by similarity score in descending order
#         similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

#         # Get the value of k for this user
#         k = top_k_values[i] if i < len(top_k_values) else k

#         # Collect recipes from similar users and calculate their popularity
#         similar_user_rated_recipes = set()
#         for similar_user_index, _ in similar_users:
#             similar_user_id = df.iloc[similar_user_index]['user_id']
#             if similar_user_id != user_id:  # Exclude the user themselves from recommendations
#                 similar_user_rated_recipes.update(df[df['user_id'] == similar_user_id]['recipe_id'])

#         # Calculate popularity scores for recipes in the pool of similar users
#         recipe_popularity = {}
#         for recipe_id in similar_user_rated_recipes:
#             recipe_popularity[recipe_id] = len(df[df['recipe_id'] == recipe_id])

#         # Recommend the top N popular recipes from the pool
#         popular_recipes = sorted(recipe_popularity.items(), key=lambda x: x[1], reverse=True)[:top_n_popular]
#         recommended_recipes = {recipe_id: popularity for recipe_id, popularity in popular_recipes}

#         recommendations[user_id] = recommended_recipes

#         # Store actual ratings for this user
#         actual_ratings[user_id] = {recipe_id: rating for recipe_id, rating in zip(df[df['user_id'] == user_id]['recipe_id'], df[df['user_id'] == user_id]['rating'])}

#     return recommendations, actual_ratings

def Recommendation_healthy_recipes(df, normalized_embeddings, similarity_threshold, top_k_values, top_n_popular):
    recommendations = {}
    actual_ratings = {}  # Dictionary to store actual ratings for each user

    # Calculate cosine similarities between user embeddings
    similarities = cosine_similarity(normalized_embeddings)

    for i, user_embedding in enumerate(normalized_embeddings):
        user_id = df.iloc[i]['user_id']  # Get the user_id from the DataFrame

        # Find similar users based on cosine similarity and the similarity threshold
        similar_users = [
            (j, similarity_score) for j, similarity_score in enumerate(similarities[i])
            if j != i and similarity_score >= similarity_threshold
        ]

        # Sort similar users by similarity score in descending order
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

        # Get the value of k for this user
        k = top_k_values[i] if i < len(top_k_values) else k

        # Collect recipes from similar users and calculate their popularity
        similar_user_rated_recipes = set()
        for similar_user_index, _ in similar_users:
            similar_user_id = df.iloc[similar_user_index]['user_id']
            if similar_user_id != user_id:  # Exclude the user themselves from recommendations
                similar_user_rated_recipes.update(df[df['user_id'] == similar_user_id]['recipe_id'])

        # Print similar_user_rated_recipes for the current user, limited to the first 5 users
        # if i < 5:
        #     print(f"User {user_id}'s similar_user_rated_recipes: {similar_user_rated_recipes}")

        # Calculate popularity scores for recipes in the pool of similar users
        recipe_popularity = {}
        for recipe_id in similar_user_rated_recipes:
            recipe_popularity[recipe_id] = len(df[df['recipe_id'] == recipe_id])

        # Recommend the top N popular recipes from the pool
        popular_recipes = sorted(recipe_popularity.items(), key=lambda x: x[1], reverse=True)[:top_n_popular]
        recommended_recipes = {recipe_id: popularity for recipe_id, popularity in popular_recipes}

        recommendations[user_id] = recommended_recipes

        # Store actual ratings for this user
        actual_ratings[user_id] = {recipe_id: rating for recipe_id, rating in zip(df[df['user_id'] == user_id]['recipe_id'], df[df['user_id'] == user_id]['rating'])}

        # Print recommendations and popularity counts for this user, limited to the first 5 users
        # if i < 5:
        #     print(f"User {user_id} Recommendations:")
        #     for recipe_id, popularity in recommended_recipes.items():
        #         print(f"  Recipe ID: {recipe_id}, Popularity Count: {popularity}")
        #     print()

    return recommendations, actual_ratings

# def calculate_ndcg(df, user_id, recommendations, k):
#     # Extract the true relevance scores from user interactions for recommended items
#     user_ratings = df[(df['user_id'] == user_id) & df['recipe_id'].isin(recommendations)][['recipe_id', 'rating']]
    
#     # Create a dictionary mapping recipe_id to its rating
#     ratings_dict = dict(zip(user_ratings['recipe_id'], user_ratings['rating']))
    
#     # Calculate binary relevance scores for recommended items
#     y_true = [1 if recipe_id in ratings_dict and ratings_dict[recipe_id] > 0 else 0 for recipe_id in recommendations]

#     # Calculate NDCG using sklearn's ndcg_score function
#     y_score = [1 if y_true[i] == 1 else 0 for i in range(len(y_true))]
#     ndcg = ndcg_score([y_true], [y_score], k=k)

#     return ndcg

def calculate_ndcg(df, user_id, recommendations, k):
    # Extract the true relevance scores from user interactions for recommended items
    user_ratings = df[(df['user_id'] == user_id) & df['recipe_id'].isin(recommendations)][['recipe_id', 'rating']]
    
    # Create a dictionary mapping recipe_id to its rating
    ratings_dict = dict(zip(user_ratings['recipe_id'], user_ratings['rating']))
    
    # Calculate binary relevance scores for recommended items
    y_true = [1 if recipe_id in ratings_dict and ratings_dict[recipe_id] > 0 else 0 for recipe_id in recommendations]

    # Convert y_true and y_score to numpy arrays
    y_true = np.array(y_true).reshape(1, -1)  # Convert to a row vector
    y_score = np.array(y_true)  # Copy y_true to y_score

    # Calculate NDCG using sklearn's ndcg_score function
    ndcg = ndcg_score(y_true, y_score, k=k)

    return ndcg

def evaluate_recommendations_with_tp_fp_fn(df, normalized_embeddings, similarity_threshold, test_size=0.2, top_k_values=[]):
    results = {}  # Initialize results dictionary
    top_n_popular = 10  # You can adjust this value based on your preferences
    # Generate recommendations using the provided function
    recommendations, actual_ratings = Recommendation_healthy_recipes(df, normalized_embeddings, similarity_threshold, top_k_values, top_n_popular)

    # Split users into training and test sets
    train_users, test_users = train_test_split(list(recommendations.keys()), test_size=test_size, random_state=42)

    # Create a list to store users with at least one true positive
    users_with_true_positives = []  # Create a list to store users with at least one true positive
    users_with_false_positives = []  # Create a list to store users with false positives
    users_with_false_negatives = []  # Create a list to store users with false negatives

    for k in top_k_values:
        # Initialize dictionaries to store tp, fp, and fn for this value of 'k'
        tp_dict = {}
        fp_dict = {}
        fn_dict = {}

        for user in test_users:
            true_positives = 0
            false_positives_count = 0
            false_negatives_count = 0

            user_actual_ratings = df[df['user_id'] == user]['recipe_id']  # Get user's actual rated recipe_ids
            user_actual_ratings = set(user_actual_ratings)

            recommended_recipe_ids = set(recommendations[user])

            # Calculate true positives
            for recipe_id in recommended_recipe_ids:
                if recipe_id in user_actual_ratings:
                    true_positives += 1

            # If a user has at least one true positive, add them to the list
            if true_positives >= 1:
                users_with_true_positives.append(user)

            # Calculate false positives (recommended but not rated)
            false_positives_count = len(recommended_recipe_ids - user_actual_ratings)
            if false_positives_count > 0:
                users_with_false_positives.append(user)

            # Calculate false negatives (rated but not recommended)
            false_negatives_count = len(user_actual_ratings - recommended_recipe_ids)
            if false_negatives_count > 0:
                users_with_false_negatives.append(user)

            # Store tp, fp, and fn values for this user and this value of 'k'
            tp_dict[user] = true_positives
            fp_dict[user] = false_positives_count
            fn_dict[user] = false_negatives_count

        # Rest of the code to calculate precision, recall, f1-score, and NDCG (if needed)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ndcg_scores = []

        for i, user in enumerate(test_users):
            tp = tp_dict[user]
            fp = fp_dict[user]
            fn = fn_dict[user]

            # Calculate precision
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0

            # Calculate recall
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0

            # Calculate F1-score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            # Calculate NDCG (if needed)
            ndcg = calculate_ndcg(df, user, recommendations[user], k)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            ndcg_scores.append(ndcg)

        average_precision = sum(precision_scores) / len(precision_scores)
        average_recall = sum(recall_scores) / len(recall_scores)
        average_f1 = sum(f1_scores) / len(f1_scores)
        average_ndcg = sum(ndcg_scores) / len(ndcg_scores)

        results[k] = {
            'Precision': average_precision,
            'Recall': average_recall,
            'F1-score': average_f1,
            'NDCG': average_ndcg
        }

    # Calculate the count of users with true positives >= 1
    count_users_with_true_positives = len(users_with_true_positives)
    # Calculate the count of users with false negatives >= 1
    count_users_with_false_negatives = len(users_with_false_negatives)
    Users_test=len(test_users)
    
    print(f"Number of Users: {Users_test}")
    print(f"Number of Users with True Positives >= 1: {count_users_with_true_positives}")
    print(f"Number of Users with Fales Negatives >= 1: {count_users_with_false_negatives}")

    return results

def extract_recommendations_and_ratings(df, normalized_embeddings, similarity_threshold, AUC_popular):
    recommendations = {}
    actual_ratings = {}

    # Calculate cosine similarities between user embeddings
    similarities = cosine_similarity(normalized_embeddings)

    for i, user_embedding in enumerate(normalized_embeddings):
        user_id = df.iloc[i]['user_id']  # Get the user_id from the DataFrame

        # Find similar users based on cosine similarity and the similarity threshold
        similar_users = [
            (j, similarity_score) for j, similarity_score in enumerate(similarities[i])
            if j != i and similarity_score >= similarity_threshold
        ]

        # Sort similar users by similarity score in descending order and limit to the top N
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

        # Store actual user ratings
        actual_ratings[user_id] = set(df[df['user_id'] == user_id]['recipe_id'])

        # Collect recipes from similar users and calculate their popularity
        similar_user_rated_recipes = set()
        for similar_user_index, _ in similar_users:
            similar_user_id = df.iloc[similar_user_index]['user_id']
            if similar_user_id != user_id:  # Exclude the user themselves from recommendations
                similar_user_rated_recipes.update(df[df['user_id'] == similar_user_id]['recipe_id'])

        # Calculate popularity scores for recipes in the pool of similar users
        recipe_popularity = {}
        for recipe_id in similar_user_rated_recipes:
            recipe_popularity[recipe_id] = len(df[df['recipe_id'] == recipe_id])

        # Recommend the top N popular recipes from the pool
        popular_recipes = sorted(recipe_popularity.items(), key=lambda x: x[1], reverse=True)[:AUC_popular]
        recommended_recipes = [recipe_id for recipe_id, _ in popular_recipes]

        # Combine similarity-based and popular recipe-based recommendations
        recommendations[user_id] = recommended_recipes

    return recommendations, actual_ratings

# def calculate_Recommendation_AUC(predictions, actual_ratings, test_size=0.2):
#     true_labels = []  # To store true labels (1 for positive) for each recipe
#     predicted_scores = []  # To store predicted scores (e.g., reverse rank position) for each recipe

#     # Compare recommended rankings with actual ratings for each user
#     for user_id, recommended_recipes in predictions.items():
#         rated_recipes = actual_ratings.get(user_id, [])

#         # Change the random_state parameter when splitting the data
#         train_set, test_set = train_test_split(recommended_recipes, test_size=test_size, random_state=42)

#         # Iterate through the recommended recipes
#         # for recipe_id in recommended_recipes:
#         for recipe_id in test_set:
#             if recipe_id in rated_recipes:
#                 true_labels.append(1)  # True Positive: Recipe is recommended and present in actual ratings
#             else:
#                 true_labels.append(0)  # False Positive: Recipe is recommended but not in actual ratings
#             predicted_scores.append(len(recommended_recipes) - recommended_recipes.index(recipe_id))

#     # Calculate AUC
#     auc = roc_auc_score(true_labels, predicted_scores)

#     return auc

def calculate_Recommendation_AUC(predictions, actual_ratings, test_size=0.2):
    true_labels = []  # To store true labels (1 for positive) for each recipe
    predicted_scores = []  # To store predicted scores (e.g., reverse rank position) for each recipe

    # Compare recommended rankings with actual ratings for each user
    for user_id, recommended_recipes in predictions.items():
        rated_recipes = actual_ratings.get(user_id, [])

        # Ensure there are items to split
        if recommended_recipes:
            # Change the random_state parameter when splitting the data
            train_set, test_set = train_test_split(recommended_recipes, test_size=test_size, random_state=42)

            # Iterate through the recommended recipes in the test set
            for recipe_id in test_set:
                if recipe_id in rated_recipes:
                    true_labels.append(1)  # True Positive: Recipe is recommended and present in actual ratings
                else:
                    true_labels.append(0)  # False Positive: Recipe is recommended but not in actual ratings
                predicted_scores.append(len(recommended_recipes) - recommended_recipes.index(recipe_id))

    # Calculate AUC
    auc = roc_auc_score(true_labels, predicted_scores)

    return auc

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
    embedding_dim = 64

    # Initialize the NLA model with the common dimension
    nla_model = NLA(num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths_tensor)

    # Train the NLA model
    NLA_loss = nla_model.train_nla(df, user_encoder, recipe_encoder, ingredient_encoder, nutrition_encoder, num_epochs=50)
    
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
    sla_for_healthy_foods.train_sla(uid_tensor, rid_tensor, ing_tensor, nut_tensor, num_epochs_sla=50)

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

    # # Call the function to load and process the data
    ground_truth_ratings = load_ground_truth_ratings(files_to_read, folder_path)
       
    #*****************************

    # Call the Recommendation_healthy_recipes function to get recommendations
    similarity_threshold = 0.9  # Adjust the similarity threshold as needed     
    
    # Define the top_k_values you want to evaluate
    top_k_values = [10]
    top_n_popular = 10  # You can adjust this value based on your preferences
    test_size=0.3
    for top_k in top_k_values:
        # Call the Recommendation_healthy_recipes function
        recommendations, actual_ratings = Recommendation_healthy_recipes(df, normalized_embeddings, similarity_threshold, top_k_values, top_n_popular)
    
    # Call the evaluation function
    results = evaluate_recommendations_with_tp_fp_fn(df, normalized_embeddings, similarity_threshold, test_size, top_k_values=top_k_values)

    # # Print or analyze the results for each k
    # for k, metrics in results.items():
    #     print(f"Results for k = {k}:")
    #     print("================================")
    #     print("Precision:", metrics['Precision'])
    #     print("Recall:", metrics['Recall'])
    #     print("F1-score:", metrics['F1-score'])
    #     print("NDCG:", metrics['NDCG'])
    #     print("================================")
        
    # Define the k values and print the header of the table
    k_values = list(results.keys())
    print("Results for Different Values of k:")
    print("=" * 60)
    print(f"{'k':<5}{'Precision':<15}{'Recall':<15}{'F1-score':<15}{'NDCG':<15}")
    print("=" * 60)

    # Print the results in the table format with increased spacing
    for k in k_values:
        metrics = results[k]
        precision = metrics['Precision']
        recall = metrics['Recall']
        f1_score = metrics['F1-score']
        ndcg = metrics['NDCG']
        print(f"{k:<5}{' ':<5}{precision:.4f}{' ':<5}{recall:.4f}{' ':<5}{f1_score:.4f}{' ':<5}{ndcg:.4f}")     
        
    # Calculate AUC_Recipe
    AUC_popular=20
    recommendations, actual_ratings = extract_recommendations_and_ratings(df, normalized_embeddings, similarity_threshold, AUC_popular)
    auc_score = calculate_Recommendation_AUC(recommendations, actual_ratings, test_size)

    # Print the AUC score
    print(f"AUC Score: {auc_score}")
  
main()
