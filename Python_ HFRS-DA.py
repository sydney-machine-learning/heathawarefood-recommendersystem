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
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

folder_path = r"C:\Food"
files_to_read = ['RAW_interactions.csv', 'RAW_recipes.csv', 'PP_recipes.csv']
file_path = r"C:\Food\RAW_recipes.csv"

# Read the file into a pandas DataFrame
df = pd.read_csv(file_path, dtype=str)

def process_data(folder_path, files_to_read):
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
        print(f"User ID: {uid}, Average Score: {avg_score}")

    # Extract ingredients and nutrition from 'RAW_recipes.csv' file
    df = pd.read_csv(os.path.join(folder_path, 'RAW_recipes.csv'))
    recipe_id = df['recipe_id']
    ingredients = df['ingredients']
    nutrition = df['nutrition']
    ingredient_tokens= df['ingredient_tokens']

    # Print ingredients and nutrition for each recipe_id
    for i in range(len(recipe_id)):
        rid = recipe_id[i]
        ing = ingredients[i]
        nut = nutrition[i]
        ing_t=ingredient_tokens[i]

        print(f"Recipe ID: {rid}")
        print(f"Ingredients: {ing}")
        print(f"Nutrition: {nut}")
        print(f"ingredient_tokens: {ing_t}")
        print()
        
def Heterogeneous_Graph(df):
        
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

    # Define the meta-paths
    meta_paths = [
        ['user_id', 'rating', 'recipe_id', 'nutrition', 'ingredient'],
        ['user_id', 'rating', 'recipe_id'],
        ['user_id', 'recipe_id', 'ingredient', 'nutrition'],
        ['recipe_id', 'nutrition', 'ingredient'],
        ['recipe_id', 'ingredient', 'nutrition']
    ]

    # Print the edges and their attributes for each meta-path
    for meta_path in meta_paths:
        print("Meta-Path:", " -> ".join(meta_path))
        paths = []
        
        # Check if the meta-path starts with 'user_id' and ends with 'ingredient'
        if meta_path[0] == 'user_id' and meta_path[-1] == 'ingredient':
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
        
        # Print the paths and their edge attributes
        for path in paths:
            print("Path:", path)
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                edges = G.edges(source, target)
                for edge in edges:
                    data = G.get_edge_data(*edge)
                    print("Source:", source)
                    print("Target:", target)
                    if data is not None and 'edge_type' in data:
                        print("Edge Type:", data['edge_type'])
                    else:
                        print("Edge Type: N/A")
                    if data is not None and 'weight' in data:
                        print("Weight:", data['weight'])
                    else:
                        print("Weight: N/A")
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

    # Calculate the average rating for each user_id
    user_avg_ratings = df.groupby('user_id')['rating'].mean()

    # Specify the meta-path
    meta_path = ['user_id', 'rating', 'recipe_id', 'ingredient', 'nutrition']

    # Print the meta-path
    print("Meta-Path:", " -> ".join(meta_path))

    paths = []
    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            # Check if user_id has an average rating
            if uid in user_avg_ratings:
                avg_rating = user_avg_ratings[uid]
                user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
                for rid in user_rated_recipes:
                    if G.get_edge_data(uid, rid)[0]['weight'] >= avg_rating:
                        for ing in G.neighbors(rid):
                            if G.nodes[ing]['node_type'] == 'ingredients':
                                paths.append([uid, rid, ing])
                        for nut in G.neighbors(rid):
                            if G.nodes[nut]['node_type'] == 'nutrition':
                                paths.append([uid, rid, nut])

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

    # Print the filtered paths
    for path, encoded_path in zip(paths, encoded_paths):
        print("Original Path:", path)
        print("Encoded Path:", encoded_path)

    return paths_tensor

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

    # Calculate the average rating for each user_id
    user_avg_ratings = df.groupby('user_id')['rating'].mean()

    # Specify the meta-path
    meta_path = ['user_id', 'rating', 'recipe_id', 'ingredient', 'nutrition']

    # Print the meta-path
    print("Meta-Path:", " -> ".join(meta_path))
    
    paths = []
    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            # Check if user_id has an average rating
            if uid in user_avg_ratings:
                avg_rating = user_avg_ratings[uid]
                user_rated_recipes = [rid for rid in G.neighbors(uid) if G.nodes[rid]['node_type'] == 'recipe']
                for rid in user_rated_recipes:
                    if G.get_edge_data(uid, rid)[0]['weight'] >= avg_rating:
                        for ing in G.neighbors(rid):
                            if G.nodes[ing]['node_type'] == 'ingredients':
                                paths.append([uid, rid, ing])
                        for nut in G.neighbors(rid):
                            if G.nodes[nut]['node_type'] == 'nutrition':
                                paths.append([uid, rid, nut])

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

    healthy_foods = set()

    for uid in G.nodes():
        if G.nodes[uid]['node_type'] == 'user':
            # Check if user_id has an average rating
            if uid in user_avg_ratings:
                avg_rating = user_avg_ratings[uid]
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

def recommend_users(sla_model, user_embeddings):
        
    # Iterate through the data and populate the graph
    recommendations = {}
    index_to_user_id = {}
    user_id_to_index = {}
    health_food_embeddings = None
    health_foods_list = None
    user_id_to_recipe_ids = {}  # New dictionary to store recipe IDs for each user
    for i in range(len(df)):
        uid = df.loc[i, 'user_id']
        rid = df.loc[i, 'recipe_id']
        r = df.loc[i, 'rating']
        ing = df.loc[i, 'ingredients']
        nut = df.loc[i, 'nutrition']
        ing_t = df.loc[i, 'ingredient_tokens']

        if uid not in user_id_to_index:
            index = len(user_id_to_index)
            user_id_to_index[uid] = index
            index_to_user_id[index] = uid

        # Store recipe IDs for each user
        if uid not in user_id_to_recipe_ids:
            user_id_to_recipe_ids[uid] = []
        user_id_to_recipe_ids[uid].append(rid)
       
    # Extracting HF (Healthy Foods), HI (Healthy Food Ingredients), and HN (Healthy Food Nutrients)
    if health_food_embeddings is not None:
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

    for i, user_embedding in enumerate(user_embeddings):
        similarities = cosine_similarity(user_embedding.unsqueeze(0), user_embeddings)
        similarities_tensor = torch.from_numpy(similarities)  # Convert similarities to a tensor
        most_similar_index = torch.argmax(similarities_tensor)
        most_similar_user_id = index_to_user_id[most_similar_index.item()]  # Convert tensor to Python integer
        user_id = index_to_user_id[i]
        recommendations[user_id] = {
            'most_similar_user_id': most_similar_user_id,
            'recipe_ids': user_id_to_recipe_ids[most_similar_user_id]  # Retrieve recipe IDs for the most similar user
        }
    return recommendations

def evaluate_recommendations(recommendations, ground_truth_ratings, test_size=0.1):
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

    # Perform train-test split   
    true_ratings_train, true_ratings_test, recommended_ratings_train, recommended_ratings_test = train_test_split(
        true_ratings, recommended_ratings, test_size=test_size, random_state=42)

    if len(true_ratings_train) > 0 and len(recommended_ratings_train) > 0:
        auc_score_train = roc_auc_score(true_ratings_train, recommended_ratings_train)
        ndcg_score_train = ndcg_score([true_ratings_train], [recommended_ratings_train])
        recall_score_train = np.mean(np.equal(true_ratings_train, recommended_ratings_train))

        print("Training Evaluation Scores:")
        print("AUC Score:", auc_score_train)
        print("NDCG Score:", ndcg_score_train)
        print("Recall Score:", recall_score_train)

    if len(true_ratings_test) > 0 and len(recommended_ratings_test) > 0:
        auc_score_test = roc_auc_score(true_ratings_test, recommended_ratings_test)
        ndcg_score_test = ndcg_score([true_ratings_test], [recommended_ratings_test])
        recall_score_test = np.mean(np.equal(true_ratings_test, recommended_ratings_test))

        print("Testing Evaluation Scores:")
        print("AUC Score:", auc_score_test)
        print("NDCG Score:", ndcg_score_test)
        print("Recall Score:", recall_score_test)

    return auc_score_train, ndcg_score_train, recall_score_train, auc_score_test, ndcg_score_test, recall_score_test

def main():
        
    # Call the process_data function
    process_data(folder_path, files_to_read)
    
    # Call the Heterogeneous_Graph function
    Heterogeneous_Graph(df)

    # Call the find_paths_users_interests function
    paths = find_paths_users_interests(df)
    
    # Call the find_healthy_foods function
    paths_tensor = find_healthy_foods(df)

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
    model = NLA(num_users, num_recipes, num_ingredients, num_nutrition, embedding_dim, paths)

    # Define the loss function and optimizer for NLA
    criterion_nla = nn.MSELoss()
    optimizer_nla = optim.Adam(model.parameters(), lr=0.001)

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
    optimizer_sla = optim.Adam(sla.parameters(), lr=0.001)

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
    
    top_k = 5  # Number of top similar users to recommend

    # Recommendation step using SLA embeddings
    index_to_user_id = {index: user_id for index, user_id in enumerate(user_encoder.classes_)}

    recommendations = {}
    for user_index, user_id in index_to_user_id.items():
        user_embedding = embeddings_nla[user_index]

        # Calculate cosine similarities between the user and all other users
        similarities = torch.cosine_similarity(user_embedding.unsqueeze(0), embeddings_nla)

        # Get the indices of the top-k most similar users
        top_k_indices = torch.argsort(similarities, descending=True)[1:top_k + 1]

        # Map the indices to user IDs
        recommended_user_ids = [index_to_user_id[index.item()] for index in top_k_indices]
        recommendations[user_id] = recommended_user_ids

    # Print the recommendations for each user
    for user_id, recommended_user_ids in recommendations.items():
        print(f"Recommended users for {user_id}:")
        for recommended_user_id in recommended_user_ids:
            print(recommended_user_id)
    
    # Define the necessary variables
    recommendations = recommend_users(sla, embeddings_nla)
                
    # Read the ground truth ratings into a dictionary
    ground_truth_ratings = {}
    for file in files_to_read:
        if file == 'RAW_interactions.csv':
            interactions_df = pd.read_csv(os.path.join(folder_path, file), dtype=str)
            for index, row in interactions_df.iterrows():
                user_id = row['user_id']
                rating = row['rating']
                ground_truth_ratings[user_id] = {'rating': rating}

    # Call the evaluate_recommendations function to evaluate recommendations
    evaluation_scores = evaluate_recommendations(recommendations, ground_truth_ratings)

    if evaluation_scores:
        auc, ndcg, recall = evaluation_scores
        print("AUC Score:", auc)
        print("NDCG Score:", ndcg)
        print("Recall Score:", recall)
    else:
        print("Insufficient data for evaluation.")
                        
if __name__ == '__main__':
    main()