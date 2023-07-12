HFRS-DA: Health-aware Food Recommendation System with Dual Attention in Heterogeneous Graphs
This repository contains the code for the paper "Health-aware Food Recommendation System with Dual Attention in Heterogeneous Graphs (HFRS-DA)". The code is organized into several sections, as described below:

1. Dataset
The dataset consists of three files:

RAW_interactions.csv: This file contains user_id, recipe_id, and rating information.
RAW_recipes.csv: This file includes recipe_id, nutrition details, and ingredients.
PP_recipes.csv: This file contains recipe_id and corresponding ingredient_tokens.
2. process_data function
The process_data function reads the dataset files and extracts relevant information, such as user_id, rating, recipe_id, and the associated ingredients and nutrition. It utilizes these data points to construct a heterogeneous graph representation.

3. Heterogeneous_Graph function
The Heterogeneous_Graph function generates a heterogeneous graph based on the extracted data. The graph consists of four types of nodes: user_id, recipe_id, ingredients, and nutrition. Edges are created between user_id and recipe_id, as well as between recipe_id and their corresponding ingredients and nutrition. This structure forms the basis for the generation of meta-paths between nodes in the graph.

4. NLA class
The NLA class implements the node level attention (NLA) mechanism. It includes the find_paths_users_interests function, which filters the meta-paths based on users' interests in recipes, ingredients, and nutrition. The class also utilizes a graph attention network (GAT) to generate embedding vectors that capture users' interests in foods.

5. SLA class
The SLA class defines health foods based on the criteria outlined in the paper. It generates embedding vectors using the find_paths_users_interests function and the NLA class. The class also calculates the final loss function, which combines the SLA loss and NLA loss components.

6. recommend_users function
The recommend_users function generates food recommendations for users based on the embedding vectors obtained from the SLA class, as described in the paper.

7. evaluate_recommendations function
The evaluate_recommendations function evaluates the performance of the recommend_users function using evaluation metrics such as the Area Under the ROC Curve (AUC), Normalized Discounted Cumulative Gain (NDCG), and Recall.

Main function
The main function orchestrates the execution of all the defined functions. It runs the data processing, graph construction, attention mechanism, recommendation generation, and evaluation steps. The results of the recommendation system, including the evaluation scores, are presented.

Please refer to the paper for detailed explanations of the algorithms and methodologies employed in this c[Uploading Python_ HFRS-DA.py…]()
ode repository.

For any inquiries or further information, please contact the authors.
