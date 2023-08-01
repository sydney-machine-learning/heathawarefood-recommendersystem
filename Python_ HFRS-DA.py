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
                df = pd.read_csv(file_path)
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

    # Calculate the average score for each recipe_id
    for rid, scores in recipe_scores.items():
        avg_score = scores['total_score'] / scores['count']
        print(f"Recipe ID: {rid}, Average Score: {avg_score}")

    # Extract ingredients and nutrition from 'Food_Dataset.zip' file
    df = pd.read_csv(os.path.join(folder_path, 'Food_Dataset.zip'))
    recipe_id = df['recipe_id']
    ingredients = df['ingredients']
    nutrition = df['nutrition']
    ingredient_tokens = df['ingredient_tokens']

    # Print the first 10 user_ids along with their information
    for i in range(min(3, len(df))):
        uid = df.loc[i, 'user_id']
        rid = df.loc[i, 'recipe_id']
        rat = df.loc[i, 'rating']
        ing = df.loc[i, 'ingredients']
        nut = df.loc[i, 'nutrition']
        ing_t = df.loc[i, 'ingredient_tokens']

        print(f"User ID: {uid}")
        print(f"Recipe ID: {rid}")
        print(f"Rating: {rat}")
        print(f"Ingredients: {ing}")
        print(f"Nutrition: {nut}")
        print(f"Ingredient Tokens: {ing_t}")
        print()
        
def Heterogeneous_Graph(df):
    # Create an empty graph
    G = nx.MultiGraph()

    # Iterate through the data and populate the graph
    for i in range(len(df)):
        uid = df.loc[i, 'user_id']
        rid = df.loc[i, 'recipe_id']
        ing = df.loc[i, 'ingredients']
        nut = df.loc[i, 'nutrition']

        # Add user_id, recipe_id, ingredients, and nutrition as nodes
        G.add_node(uid, node_type='user')
        G.add_node(rid, node_type='recipe')
        G.add_node(ing, node_type='ingredients')
        G.add_node(nut, node_type='nutrition')

        # Add edges between user_id and recipe_id
        G.add_edge(uid, rid, edge_type='rating')

        # Add edges between recipe_id and ingredients
        if isinstance(ing, str):
            ingredients_list = ing.split(',')
            for ingredient in ingredients_list:
                ingredient = ingredient.strip()
                G.add_node(ingredient, node_type='ingredients')
                G.add_edge(rid, ingredient, edge_type='ingredient')

        # Add edges between recipe_id and nutrition
        if isinstance(nut, str):
            nutrition_list = nut.split(',')
            for nutrition_item in nutrition_list:
                nutrition_item = nutrition_item.strip()
                G.add_node(nutrition_item, node_type='nutrition')
                G.add_edge(rid, nutrition_item, edge_type='nutrition')

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
        print("Meta-Path:", " -> ".join(meta_path))
        paths = []
        
        # Check if the meta-path starts with 'user_id' and ends with 'ingredients'
        if meta_path[0] == 'user_id' and meta_path[-1] == 'ingredients':
            for uid in G.nodes():
                if G.nodes[uid]['node_type'] == 'user':
                    for rid in G.neighbors(uid):
                        if G.nodes[rid]['node_type'] == 'recipe':
                            for ing in G.neighbors(rid):
                                if G.nodes[ing]['node_type'] == 'ingredients':
                                    paths.append([uid, rid, ing])
        
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
                                            paths.append([uid, rid, nut, ing])
        
        # Print only the first 5 paths for each meta-path
        for i, path in enumerate(paths[:5]):
            print("Path:", path)
            for j in range(len(path) - 1):
                source = path[j]
                target = path[j + 1]
                edges = G.get_edge_data(source, target)
                if edges is not None:
                    for key, data in edges.items():
                        print("Source:", source)
                        print("Target:", target)
                        if 'edge_type' in data:
                            print("Edge Type:", data['edge_type'])
                        else:
                            print("Edge Type: N/A")
                else:
                    print("No edges between", source, "and", target)
            print()
        
# Define the NLA class
class NLA(nn.Module):
    def __init__(self, num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths):
        super(NLA, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.nutrition_embedding = nn.Embedding(num_nutrition, embedding_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Softmax(dim=1)
        )

        # Convert the paths to tensors
        self.paths = paths.clone().detach() if paths is not None else None

    def forward(self, uid, rid, ing):
        user_emb = self.user_embedding(uid)
        recipe_emb = self.recipe_embedding(rid)
        ingredient_emb = self.ingredient_embedding(ing)

        if self.paths is not None:
            path_scores = torch.zeros(uid.size(0), len(self.paths))
            for i, path in enumerate(self.paths):
                path = torch.tensor(path)
                matching_uid = torch.where(uid == path[0])[0]
                matching_rid = torch.where(rid == path[1])[0]
                matching_ing = torch.where(ing == path[2])[0]
                
                # Check if there are any matching indices
                if matching_uid.size(0) > 0 and matching_rid.size(0) > 0 and matching_ing.size(0) > 0:
                    matching_count = min(matching_uid.size(0), matching_rid.size(0), matching_ing.size(0))
                    matching_indices = torch.stack((matching_uid[:matching_count], matching_rid[:matching_count], matching_ing[:matching_count]))
                    path_scores[matching_indices] += 1

            # Apply attention to ingredient embeddings
            attention_scores = self.attention(ingredient_emb)
            attention_scores = attention_scores.view(attention_scores.size(0), attention_scores.size(1), 1)
            weighted_ingredients = attention_scores * ingredient_emb
            aggregated_ingredients = torch.sum(weighted_ingredients, dim=1)

            # Determine the maximum size along dimension 0
            max_size = max(user_emb.size(0), recipe_emb.size(0), aggregated_ingredients.size(0))

            # Pad tensors to match the maximum size along dimension 0
            user_emb = F.pad(user_emb, (0, 0, 0, max_size - user_emb.size(0)))
            recipe_emb = F.pad(recipe_emb, (0, 0, 0, max_size - recipe_emb.size(0)))
            aggregated_ingredients = F.pad(aggregated_ingredients, (0, 0, 0, max_size - aggregated_ingredients.size(0)))

            # Concatenate and return the final embedding
            node_embeddings = torch.cat((user_emb, recipe_emb, aggregated_ingredients), dim=1)
        else:
            # Concatenate the embeddings without attention
            node_embeddings = torch.cat((user_emb, recipe_emb, ingredient_emb), dim=1)

        return node_embeddings

# Define the dataset class
class HeterogeneousDataset(Dataset):
    def __init__(self, df, user_encoder, recipe_encoder, ingredient_encoder):
        self.uids = user_encoder.transform(df['user_id'])
        self.rids = recipe_encoder.transform(df['recipe_id'])
        self.ings = ingredient_encoder.transform(df['ingredients'])
        self.labels = df['rating'].astype(float).values

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        rid = self.rids[idx]
        ing = self.ings[idx]
        label = self.labels[idx]
        return uid, rid, ing, label

# Define the find_paths_users_interests function
def find_paths_users_interests(df):
    # Create an empty graph
    G = nx.MultiGraph()

    # Iterate through the data and populate the graph
    for i in range(len(df)):
        uid = df.loc[i, 'user_id']
        rid = df.loc[i, 'recipe_id']
        r = df.loc[i, 'rating']
        ing = df.loc[i, 'ingredients']
        nut = df.loc[i, 'nutrition']
        
        # Add user_id, recipe_id, ingredients, and nutrition as nodes
        G.add_node(uid, node_type='user')
        G.add_node(rid, node_type='recipe')
        G.add_node(ing, node_type='ingredients')
        G.add_node(nut, node_type='nutrition')

        # Add edges between user_id and recipe_id
        G.add_edge(uid, rid, weight=float(r), edge_type='rating')

        # Add edges between recipe_id and ingredients
        if isinstance(ing, str):
            ingredients_list = ing.split(',')
            for ingredient in ingredients_list:
                ingredient = ingredient.strip()
                G.add_node(ingredient, node_type='ingredients')
                G.add_edge(rid, ingredient, edge_type='ingredient')

        # Add edges between recipe_id and nutrition
        if isinstance(nut, str):
            nutrition_list = nut.split(',')
            for nutrition_item in nutrition_list:
                nutrition_item = nutrition_item.strip()
                G.add_node(nutrition_item, node_type='nutrition')
                G.add_edge(rid, nutrition_item, edge_type='nutrition')

    # Calculate the average rating for each recipe_id and create a new column 'avg_rating'
    df['avg_rating'] = df.groupby('recipe_id')['rating'].transform(lambda x: math.floor(x.mean()))

    # Print the meta-path
    meta_path = ['user_id', 'recipe_id', 'ingredient', 'nutrition']
    print("Meta-Path:", " -> ".join(meta_path))
       
    paths = []
    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
            has_rated_recipe = any(rid in df['recipe_id'].values.tolist() for rid in user_rated_recipes)
            if has_rated_recipe:
                for rid in user_rated_recipes:
                    if df.loc[df['recipe_id'] == rid, 'rating'].iloc[0] >= df.loc[df['recipe_id'] == rid, 'avg_rating'].iloc[0]:
                        ingredient_node = None
                        nutrition_node = None
                        
                        for node in G.neighbors(rid):
                            if G.nodes[node]['node_type'] == 'ingredients':
                                ingredient_node = node
                            elif G.nodes[node]['node_type'] == 'nutrition':
                                nutrition_node = node
                        
                        if ingredient_node and nutrition_node:
                            paths.append([uid, rid, ingredient_node, nutrition_node])

    # Encode the paths using label encoders
    user_encoder = LabelEncoder()
    recipe_encoder = LabelEncoder()
    ingredient_encoder = LabelEncoder()
    user_encoder.fit([path[0] for path in paths])
    recipe_encoder.fit([path[1] for path in paths])
    ingredient_encoder.fit([path[2] for path in paths])

    encoded_paths = [[user_encoder.transform([path[0]])[0], recipe_encoder.transform([path[1]])[0], ingredient_encoder.transform([path[2]])[0]] for path in paths]

    # Convert paths to tensors
    paths_tensor = torch.tensor(encoded_paths, dtype=torch.long)

    # Print the first 5 filtered paths
    for i, (path, encoded_path) in enumerate(zip(paths, encoded_paths)):
        print("Original Path:", path)
        print("Encoded Path:", encoded_path)
        if i == 5:  
            break

    return paths_tensor, meta_path

# Define the NLA class
class NLA(nn.Module):
    def __init__(self, num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths):
        super(NLA, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.recipe_embedding = nn.Embedding(num_recipes, embedding_dim)
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.nutrition_embedding = nn.Embedding(num_nutrition, embedding_dim)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Softmax(dim=1)
        )

        # Convert the paths to tensors
        self.paths = torch.tensor(paths) if paths is not None else None

    def forward(self, uid, rid, ing):
        user_emb = self.user_embedding(uid)
        recipe_emb = self.recipe_embedding(rid)
        ingredient_emb = self.ingredient_embedding(ing)

        if self.paths is not None:
            path_scores = torch.zeros(uid.size(0), len(self.paths))
            for i, path in enumerate(self.paths):
                path = torch.tensor(path)
                matching_uid = torch.where(uid == path[0])[0]
                matching_rid = torch.where(rid == path[1])[0]
                matching_ing = torch.where(ing == path[2])[0]
                
                # Check if there are any matching indices
                if matching_uid.size(0) > 0 and matching_rid.size(0) > 0 and matching_ing.size(0) > 0:
                    matching_count = min(matching_uid.size(0), matching_rid.size(0), matching_ing.size(0))
                    matching_indices = torch.stack((matching_uid[:matching_count], matching_rid[:matching_count], matching_ing[:matching_count]))
                    path_scores[matching_indices] += 1

            # Apply attention to ingredient embeddings
            attention_scores = self.attention(ingredient_emb)
            attention_scores = attention_scores.view(attention_scores.size(0), attention_scores.size(1), 1)
            weighted_ingredients = attention_scores * ingredient_emb
            aggregated_ingredients = torch.sum(weighted_ingredients, dim=1)

            # Determine the maximum size along dimension 0
            max_size = max(user_emb.size(0), recipe_emb.size(0), aggregated_ingredients.size(0))

            # Pad tensors to match the maximum size along dimension 0
            user_emb = F.pad(user_emb, (0, 0, 0, max_size - user_emb.size(0)))
            recipe_emb = F.pad(recipe_emb, (0, 0, 0, max_size - recipe_emb.size(0)))
            aggregated_ingredients = F.pad(aggregated_ingredients, (0, 0, 0, max_size - aggregated_ingredients.size(0)))

            # Concatenate and return the final embedding
            node_embeddings = torch.cat((user_emb, recipe_emb, aggregated_ingredients), dim=1)
        else:
            # Concatenate the embeddings without attention
            node_embeddings = torch.cat((user_emb, recipe_emb, ingredient_emb), dim=1)

        return node_embeddings

# Define the SLA class
class SLA(nn.Module):
    def __init__(self, num_ingredients, embedding_dim):
        super(SLA, self).__init__()
        self.ingredient_embedding = nn.Embedding(num_ingredients, embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, ing):
        ingredient_emb = self.ingredient_embedding(ing)
        attention_scores = self.attention(ingredient_emb)
        attention_scores = attention_scores.view(attention_scores.size(0), attention_scores.size(1), 1)
        weighted_ingredients = attention_scores * ingredient_emb
        aggregated_ingredients = torch.sum(weighted_ingredients, dim=1)
        return aggregated_ingredients

# Define the health foods list
health_foods_list = {
    "Proteins": 5867,
    "Carbohydrates": 1454,
    "Sugars": 8780,
    "Sodium": 4914,
    "Fat": 3035,
    "Saturated_fats": 119590,
    "Fibers": 10734
}

def find_healthy_foods(df):
    # Create an empty graph
    G = nx.MultiGraph()

    # Iterate through the data and populate the graph
    for i in range(len(df)):
        uid = df.loc[i, 'user_id']
        rid = df.loc[i, 'recipe_id']
        r = df.loc[i, 'rating']
        ing = df.loc[i, 'ingredients']
        nut = df.loc[i, 'nutrition']
        ing_t = df.loc[i, 'ingredient_tokens']

        # Add user_id, recipe_id, ingredients, and nutrition as nodes
        G.add_node(uid, node_type='user')
        G.add_node(rid, node_type='recipe')
        G.add_node(ing, node_type='ingredients')
        G.add_node(nut, node_type='nutrition')

        # Add weighted edge between user_id and recipe_id with weight as the rating
        G.add_edge(uid, rid, weight=float(r), edge_type='rating')

        # Add edges between recipe_id and ingredients
        if isinstance(ing, str):
            ingredients_list = ing.split(',')
            for ingredient in ingredients_list:
                ingredient = ingredient.strip()
                G.add_node(ingredient, node_type='ingredients')
                G.add_edge(rid, ingredient, edge_type='ingredient')

        # Add edges between recipe_id and nutrition
        if isinstance(nut, str):
            nutrition_list = nut.split(',')
            for nutrition_item in nutrition_list:
                nutrition_item = nutrition_item.strip()
                G.add_node(nutrition_item, node_type='nutrition')
                G.add_edge(rid, nutrition_item, edge_type='nutrition')

    # Calculate the average rating for each recipe_id and create a new column 'avg_rating'
    df['avg_rating'] = df.groupby('recipe_id')['rating'].transform(lambda x: math.floor(x.mean()))

    # Print the meta-path
    meta_path = ['user_id', 'recipe_id', 'ingredient', 'nutrition']
    print("Meta-Path:", " -> ".join(meta_path))

    paths = []
    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
            has_rated_recipe = any(rid in df['recipe_id'].values.tolist() for rid in user_rated_recipes)
            if has_rated_recipe:
                for rid in user_rated_recipes:
                    if df.loc[df['recipe_id'] == rid, 'rating'].iloc[0] >= df.loc[df['recipe_id'] == rid, 'avg_rating'].iloc[0]:
                        ingredient_node = None
                        nutrition_node = None

                        for node in G.neighbors(rid):
                            if G.nodes[node]['node_type'] == 'ingredients':
                                ingredient_node = node
                            elif G.nodes[node]['node_type'] == 'nutrition':
                                nutrition_node = node

                        if ingredient_node and nutrition_node:
                            paths.append([uid, rid, ingredient_node, nutrition_node])

    healthy_foods = set()

    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            # Check if user_id has an average rating
            if uid in df['user_id'].values.tolist():
                avg_rating = df.loc[df['user_id'] == uid, 'avg_rating'].iloc[0]
                user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
                for rid in user_rated_recipes:
                    if G.get_edge_data(uid, rid)[0]['weight'] >= avg_rating:
                        ingredients_tokens = [int(token) for token in ing_t.split(',') if token.strip().isdigit()]
                        num_health_foods = sum(1 for food_num in health_foods_list.values() if food_num in ingredients_tokens)
                        if num_health_foods >= 3:
                            healthy_foods.add(rid)

    # Encode the paths using label encoders
    recipe_encoder = LabelEncoder()
    recipe_encoder.fit(list(healthy_foods))

    encoded_paths = [[path[1]] for path in paths if path[1] in healthy_foods]

    # Convert paths to tensors
    paths_tensor = torch.tensor(encoded_paths, dtype=torch.long)
    
    # Print the filtered paths
    for path, encoded_path in zip(paths, encoded_paths):
        print("Health foods Path:", path)
        print("Encoded Path:", encoded_path)

    return paths_tensor

def rate_healthy_recipes_for_user(user_id, df):
    # Filter the data for the specified user_id
    user_data = df[df['user_id'] == user_id]

    # Get the healthy recipes for the user
    user_healthy_recipes = set()
    for rid in user_data['recipe_id'].unique():
        avg_rating = user_data[user_data['recipe_id'] == rid]['avg_rating'].iloc[0]
        rating = user_data[user_data['recipe_id'] == rid]['rating'].iloc[0]
        if rating >= avg_rating:
            ingredients_tokens = [int(token) for token in user_data[user_data['recipe_id'] == rid]['ingredient_tokens'].iloc[0].split(',') if token.strip().isdigit()]
            num_health_foods = sum(1 for food_num in health_foods_list.values() if food_num in ingredients_tokens)
            if num_health_foods >= 3:
                user_healthy_recipes.add(rid)

    return user_healthy_recipes
    
def recommend_users(sla_model, user_embeddings):
    # Calculate the cosine similarity between all pairs of user embeddings
    all_similarities = cosine_similarity(user_embeddings, user_embeddings)

    recommendations = {}
    index_to_user_id = {}

    for i, user_embedding in enumerate(user_embeddings):
        similarities = all_similarities[i]
        similar_users_indices = torch.argsort(similarities, descending=True)  # Get indices of most similar users

        # Create a list to store similar user IDs
        similar_users = []
        for index in similar_users_indices:
            if index != i:
                similar_user_id = index_to_user_id[index.item()]
                similar_users.append(similar_user_id)

        user_id = index_to_user_id[i]
        recommendations[user_id] = {
            'most_similar_user_id': similar_users[0],  # Retrieve the most similar user ID
            'similar_users': similar_users[1:]  # Retrieve other similar user IDs
        }

        # Find users who share ingredients based on embedding vectors
        ingredients_tokens = df.loc[df['user_id'] == user_id, 'ingredient_tokens'].iloc[0].split(',')
        shared_users = []

        for other_uid, other_embedding in zip(index_to_user_id.values(), user_embeddings):
            if other_uid != user_id:
                other_tokens = df.loc[df['user_id'] == other_uid, 'ingredient_tokens'].iloc[0].split(',')
                common_tokens = set(ingredients_tokens) & set(other_tokens)
                if len(common_tokens) > 0:
                    shared_users.append(other_uid)

        recommendations[user_id]['users_with_shared_ingredients'] = shared_users

        # Find users who share nutrition based on embedding vectors
        nutrition_values = df.loc[df['user_id'] == user_id, 'nutrition'].iloc[0].split(',')
        shared_users = []

        for other_uid, other_embedding in zip(index_to_user_id.values(), user_embeddings):
            if other_uid != user_id:
                other_nutrition = df.loc[df['user_id'] == other_uid, 'nutrition'].iloc[0].split(',')
                common_nutrition = set(nutrition_values) & set(other_nutrition)
                if len(common_nutrition) > 0:
                    shared_users.append(other_uid)

        recommendations[user_id]['users_with_shared_nutrition'] = shared_users

    return recommendations

def evaluate_recommendations(recommendations, ground_truth_ratings, validation_size=0.1, test_size=0.2):
    true_ratings = []
    recommended_ratings = []

    for user_id, recommended_user_ids in recommendations.items():
        if user_id in ground_truth_ratings:
            true_rating = float(ground_truth_ratings[user_id]['rating'])
            true_ratings.append(true_rating)

            for recommended_user_id in recommended_user_ids:
                if recommended_user_id in ground_truth_ratings:
                    recommended_rating = float(ground_truth_ratings[recommended_user_id]['rating'])
                    recommended_ratings.append(recommended_rating)

    if len(true_ratings) == 0 or len(recommended_ratings) == 0:
        print("Insufficient data for evaluation.")
        return None

    # Randomly shuffle the data
    combined_data = list(zip(true_ratings, recommended_ratings))
    random.shuffle(combined_data)
    true_ratings, recommended_ratings = zip(*combined_data)

    # Calculate the indices to split the data into test, validation, and training sets
    num_samples = len(true_ratings)
    validation_split_index = int(num_samples * (1 - validation_size))
    test_split_index = int(num_samples * (1 - test_size))

    # Split the data into training, validation, and test sets
    true_ratings_train = true_ratings[:validation_split_index]
    recommended_ratings_train = recommended_ratings[:validation_split_index]

    true_ratings_validation = true_ratings[validation_split_index:test_split_index]
    recommended_ratings_validation = recommended_ratings[validation_split_index:test_split_index]

    true_ratings_test = true_ratings[test_split_index:]
    recommended_ratings_test = recommended_ratings[test_split_index:]

    if len(true_ratings_validation) > 0 and len(recommended_ratings_validation) > 0:
        auc_score_validation = roc_auc_score(true_ratings_validation, recommended_ratings_validation)
        ndcg_score_validation = ndcg_score([true_ratings_validation], [recommended_ratings_validation])
        recall_score_validation = np.mean(np.equal(true_ratings_validation, recommended_ratings_validation))

        print("Validation Evaluation Scores:")
        print("AUC Score:", auc_score_validation)
        print("NDCG Score:", ndcg_score_validation)
        print("Recall Score:", recall_score_validation)

    if len(true_ratings_test) > 0 and len(recommended_ratings_test) > 0:
        auc_score_test = roc_auc_score(true_ratings_test, recommended_ratings_test)
        ndcg_score_test = ndcg_score([true_ratings_test], [recommended_ratings_test])
        recall_score_test = np.mean(np.equal(true_ratings_test, recommended_ratings_test))

        print("Testing Evaluation Scores:")
        print("AUC Score:", auc_score_test)
        print("NDCG Score:", ndcg_score_test)
        print("Recall Score:", recall_score_test)

    return auc_score_validation, ndcg_score_validation, recall_score_validation, auc_score_test, ndcg_score_test, recall_score_test

def main():
        
    # Call the process_data function
    process_data(folder_path, files_to_read)
    
    # Call the Heterogeneous_Graph function
    Heterogeneous_Graph(df)

   # Call the find_paths_users_interests function
    paths_tensor, meta_path = find_paths_users_interests(df)

    # Print the filtered meta-path
    print("Filtered Meta-Path:", " -> ".join(meta_path))
   
    health_foods_paths = find_healthy_foods(df)
    print("Healthy Foods Paths:", health_foods_paths)
    
    # Get the unique user IDs
    unique_user_ids = df['user_id'].unique()

    for user_id in unique_user_ids:
        # Rate healthy recipes for the user
        user_healthy_recipes = rate_healthy_recipes_for_user(user_id, df)
        print(f"Healthy recipes rated by user {user_id}: {user_healthy_recipes}")
           
    # Call the SLA class
    num_ingredients = len(df['ingredients'].unique())
    embedding_dim = 64
    sla = SLA(num_ingredients, embedding_dim)

    # Get the unique node counts
    num_users = len(df['user_id'].unique())
    num_recipes = len(df['recipe_id'].unique())
    num_ingredients = len(df['ingredients'].unique())
    num_nutrition = len(df['nutrition'].unique())

    # Initialize the label encoders and fit them with the data
    user_encoder = LabelEncoder()
    recipe_encoder = LabelEncoder()
    ingredient_encoder = LabelEncoder()
    user_encoder.fit(df['user_id'])
    recipe_encoder.fit(df['recipe_id'])
    ingredient_encoder.fit(df['ingredients'])

    # Initialize the NLA model
    embedding_dim = 64
    model = NLA(num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths_tensor)

    # Define the loss function and optimizer for NLA
    criterion_nla = nn.MSELoss()
    optimizer_nla = optim.Adam(model.parameters(), lr=0.01)

    # Create the dataset and data loader for NLA
    dataset = HeterogeneousDataset(df, user_encoder, recipe_encoder, ingredient_encoder)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop for NLA
    num_epochs = 100
    for epoch in range(num_epochs):
        running_loss_nla = 0.0
        for uid, rid, ing, label in data_loader:
            optimizer_nla.zero_grad()

            # Forward pass
            embeddings = model(uid, rid, ing)

            # Convert labels to the same data type as model output
            label = label.unsqueeze(1).float()

            # Calculate the loss
            loss_nla = criterion_nla(embeddings, label)
            running_loss_nla += loss_nla.item()

            # Backward pass and optimization
            loss_nla.backward()
            optimizer_nla.step()

        # Print the average loss for the epoch
        avg_loss_nla = running_loss_nla / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, NLA Loss: {avg_loss_nla:.4f}")

    # Print the generated embedding vectors from NLA
    with torch.no_grad():
        uid_tensor = torch.LongTensor(list(range(num_users)))
        rid_tensor = torch.LongTensor(list(range(num_recipes)))
        ing_tensor = torch.LongTensor(list(range(num_ingredients)))
        embeddings_nla = model(uid_tensor, rid_tensor, ing_tensor)
        print("Embedding Vectors (NLA):")
        print(embeddings_nla)

    # Get the aggregated ingredient embeddings from SLA
    ingredient_tensor = torch.LongTensor(list(range(num_ingredients)))
    aggregated_ingredients = sla(ingredient_tensor)
    print("Embeddings Vectors (SLA):")
    print(aggregated_ingredients)
    
    # Define the loss function for SLA
    def edge_loss(h_sla):
        loss = -torch.log(1 / (1 + torch.exp(h_sla)))
        return loss.mean()  # Take the mean of the loss tensor

    # Define the optimizer for SLA
    optimizer_sla = optim.Adam(sla.parameters(), lr=0.01)

    # Training loop for SLA
    num_epochs_sla = 100
    for epoch_sla in range(num_epochs_sla):
        optimizer_sla.zero_grad()

        # Forward pass
        aggregated_ingredients = sla(ingredient_tensor)

        # Calculate the loss
        loss_sla = edge_loss(aggregated_ingredients)
        loss_sla.backward()
        optimizer_sla.step()

        # Print the loss for SLA
        print(f"Epoch SLA {epoch_sla + 1}/{num_epochs_sla}, SLA Loss: {loss_sla.item():.4f}")

    # Print the aggregated ingredient embeddings from SLA
    print("Embeddings Vectors (SLA):")
    print(aggregated_ingredients)

    # Print the losses for NLA and SLA
    print(f"NLA Loss: {avg_loss_nla:.4f}")
    print(f"SLA Loss: {loss_sla:.4f}")

    # Calculate the total loss by summing NLA and SLA losses
    total_loss = avg_loss_nla + loss_sla.item()

    # Print the total loss
    print("Total Loss:", total_loss)
    
    top_k = 10  # Number of top similar users to recommend

    # Recommendation step using SLA embeddings
    index_to_user_id = {index: user_id for index, user_id in enumerate(user_encoder.classes_)}

    recommendations = {}
    count_users = 0

    for user_index, user_id in index_to_user_id.items():
        user_embedding = embeddings_nla[user_index]

        # Calculate cosine similarities between the user and all other users
        similarities = torch.cosine_similarity(user_embedding.unsqueeze(0), embeddings_nla)

        # Get the indices of the top-k most similar users
        top_k_indices = torch.argsort(similarities, descending=True)[1:top_k + 1]

        # Map the indices to user IDs
        recommended_user_ids = [index_to_user_id[index.item()] for index in top_k_indices]
        recommendations[user_id] = recommended_user_ids

        count_users += 1
        if count_users == 5:
            break

    # Print the recommendations for each user
    for user_id, recommended_user_ids in recommendations.items():
        print(f"Recommended users for {user_id}:")
        for recommended_user_id in recommended_user_ids:
            print(recommended_user_id)

                            
    # Read the ground truth ratings into a dictionary
    ground_truth_ratings = {}
    for file in files_to_read:
        if file == 'Food_Dataset.zip':
            interactions_df = pd.read_csv(os.path.join(folder_path, file), dtype=str)
            for index, row in interactions_df.iterrows():
                user_id = row['user_id']
                rating = row['rating']
                ground_truth_ratings[user_id] = {'rating': rating}
       
    # Call the evaluate_recommendations function
    validation_size = 0.1  # Set the proportion of the training set for validation
    test_size = 0.2  # Set the proportion of the data for testing
    result = evaluate_recommendations(recommendations, ground_truth_ratings, validation_size, test_size)
                        
if __name__ == '__main__':
    main()