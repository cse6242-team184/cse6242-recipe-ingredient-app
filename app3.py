import streamlit as st
import pandas as pd
from recipe_scrapers import scrape_me
from ingredient_parser import parse_ingredient
 
import pyspark
 
from pyspark.sql import DataFrame, SparkSession, SQLContext
from typing import List
import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

 

# !apt-get install openjdk-8-jdk-headless -qq > /dev/null
# #Check this site for the latest download link https://www.apache.org/dyn/closer.lua/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
# !wget -q https://dlcdn.apache.org/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
# !tar xf spark-3.2.1-bin-hadoop3.2.tgz
# !pip install -q findspark
# !pip install pyspark
# !pip install py4j

# import os
# # import sys
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# # os.environ["SPARK_HOME"] = "/content/spark-3.2.1-bin-hadoop3.2"


# import findspark
# findspark.init()
# findspark.find()

# import pyspark

# from pyspark.sql import DataFrame, SparkSession
# from typing import List
# import pyspark.sql.types as T
# import pyspark.sql.functions as F

spark= SparkSession \
       .builder \
       .appName("Our First Spark Example") \
       .getOrCreate()
 
#  sc = pyspark.SparkContext(appName="HW3-Q1")
#Get Data:

# sc = pyspark.SparkContext(appName="HW3-Q1")
# sqlContext = SQLContext(sc)


def clean_text(text):
    """Clean and normalize text for matching"""
    if pd.isna(text):
        return ""
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    """Tokenize text into words, removing stopwords"""
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'of', 'for', 'on', 'at', 'from', 'by'}
    tokens = clean_text(text).split()
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def calculate_token_match_score(ingredient_tokens, product_tokens, product_name_original):
    """
    Calculate a relevance score based on token matching
    Higher score = better match
    """
    if not ingredient_tokens or not product_tokens:
        return 0.0
    
    score = 0.0
    product_name_lower = product_name_original.lower()
    
    # 1. Exact token matches (high weight)
    matching_tokens = set(ingredient_tokens) & set(product_tokens)
    score += len(matching_tokens) * 10.0
    
    # 2. Check if ALL ingredient tokens appear in product (very important)
    all_tokens_present = all(token in product_tokens for token in ingredient_tokens)
    if all_tokens_present:
        score += 20.0
    
    # 3. Position weighting - tokens at the beginning are more relevant
    for i, token in enumerate(product_tokens[:5]):  # Check first 5 tokens
        if token in ingredient_tokens:
            position_weight = 5.0 / (i + 1)  # Earlier position = higher weight
            score += position_weight
    
    # 4. Partial word matching (substring matching with penalty)
    for ing_token in ingredient_tokens:
        for prod_token in product_tokens:
            if ing_token in prod_token or prod_token in ing_token:
                if ing_token != prod_token:  # Don't double-count exact matches
                    score += 2.0
    
    # 5. Check if ingredient appears as a phrase in product name
    ingredient_phrase = ' '.join(ingredient_tokens)
    if ingredient_phrase in product_name_lower:
        score += 15.0
    
    # 6. Penalize products with too many extra words (likely wrong category)
    extra_words = len(product_tokens) - len(ingredient_tokens)
    if extra_words > 5:
        score -= (extra_words - 5) * 0.5
    
    return score


def calculate_tfidf_similarity(ingredient, product_names):
    """
    Calculate TF-IDF cosine similarity between ingredient and product names
    Returns similarity scores for all products
    """
    if len(product_names) == 0:
        return np.array([])
    
    # Combine ingredient with all product names
    documents = [ingredient] + list(product_names)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Use both single words and bigrams
        min_df=1,
        lowercase=True,
        token_pattern=r'(?u)\b\w+\b'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate cosine similarity between ingredient and all products
        ingredient_vector = tfidf_matrix[0:1]
        product_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(ingredient_vector, product_vectors)[0]
        return similarities
    except:
        return np.zeros(len(product_names))


def match_ingredient_to_products(ingredient, df, top_n=50):
    """
    Advanced matching algorithm combining multiple strategies
    
    Args:
        ingredient: The ingredient name to search for
        df: DataFrame with product data
        top_n: Number of top matches to return
    
    Returns:
        Filtered DataFrame with most relevant products
    """
    if df.empty:
        return df
    
    # Clean ingredient
    ingredient_clean = clean_text(ingredient)
    ingredient_tokens = tokenize(ingredient)
    
    # Calculate token-based scores
    df['product_name_clean'] = df['product_name'].apply(clean_text)
    df['product_tokens'] = df['product_name_clean'].apply(lambda x: tokenize(x))
    df['token_score'] = df.apply(
        lambda row: calculate_token_match_score(
            ingredient_tokens, 
            row['product_tokens'], 
            row['product_name']
        ), 
        axis=1
    )
    
    # Filter to products with at least some token match
    df_filtered = df[df['token_score'] > 0].copy()
    
    if df_filtered.empty:
        # Fallback to simple substring search if no matches
        df_filtered = df[df['product_name_clean'].str.contains(ingredient_clean, na=False)].copy()
        df_filtered['token_score'] = 5.0
    
    if df_filtered.empty:
        return df_filtered
    
    # Calculate TF-IDF similarity for remaining products
    tfidf_scores = calculate_tfidf_similarity(
        ingredient_clean, 
        df_filtered['product_name_clean'].values
    )
    df_filtered['tfidf_score'] = tfidf_scores * 100  # Scale to similar range as token score
    
    # Combined score (weighted average)
    df_filtered['relevance_score'] = (
        df_filtered['token_score'] * 0.7 +  # Token matching is primary
        df_filtered['tfidf_score'] * 0.3     # TF-IDF adds semantic similarity
    )
    
    # Sort by relevance score and return top N
    df_filtered = df_filtered.sort_values('relevance_score', ascending=False).head(top_n)
    
    # Clean up temporary columns
    df_filtered = df_filtered.drop(['product_name_clean', 'product_tokens', 'token_score', 'tfidf_score'], axis=1)
    
    return df_filtered

 
@st.cache_data
def load_data(file_path):
    # Simulate a long-running data loading process
    time.sleep(2)
    df = spark.read.csv(
        file_path,
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"',
        sep='\t'
    )

    df = df.filter(F.lower("product_name").isNotNull())
    
    
    # Create a new dataframe with a subset of columns
    selected_columns = ['product_name', 'url', 'nutrition_grade_fr', 'ingredients_text', 'fat_100g',
                        'proteins_100g', 'carbohydrates_100g', 'sugars_100g', 'sodium_100g',
                        'fiber_100g', 'additives_n', 'countries'] # Replace with the list of columns you want to keep
    
    filtered_df = df[[selected_columns]]

    #filtered_df = df.dropna(subset=selected_columns)
 
    
    #US only:
    filtered_df = filtered_df[filtered_df['countries'].contains('United States')]
    ### Add allergy columns to food dataset:
    
    #https://www.foodallergy.org/living-food-allergies/food-allergy-essentials/common-allergens
    
    milk = ["milk", "butter", "casein", "cheese",
            "cream", "curd", "custard", "ghee",
            "half-and-half", "lactose", "lactulose", "whey", "tagatose", "yogurt"]
    
    egg = ["egg", "Albumin", "Albumen", "Apovitellin", "Avidin globulin", "Lysozyme", "Mayonnaise",
          "Meringue", "Ovalbumin", "Ovomucoid", "Ovomucin", "Ovovitellin", "Surimi", "Vitellin"]
    
    peanut = ["peanut", "Arachis oil", "Artificial nuts", "Beer nuts", "Ground nuts", "Lupin",
              "Lupine", "Mandelonas", "Mixed nuts", "Monkey nuts", "Nut meat", "nut meal", "Nut pieces", "nut"]
    
    soy = ["soy", "edamame", "miso", "natto", "okara", "Shoyu", "Soya", "Tamari",
          "Tempeh", "Textured vegetable protein", "TVP", "tofu"]
    
    wheat = ["Bread crumbs", "Bulgur", "Cereal extract",
            "Couscous", "Cracker meal", "Durum", "Einkorn", "Emmer",
            "Farina", "Farro", "Flour", "Freekeh", "Hydrolyzed wheat protein",
            "Matzoh", "matzo", "matzah", "matza", "Pasta", "Seitan", "Semolina",
            "Spelt", "Triticale", "Vital wheat gluten", "Wheat"]
    
    treenut = ["tree nut", "Almond", "nut", "Cashew", "Filbert", "Gianduja", "Litchi", "lichee",
    "lychee", "Macadamia","Marzipan", "walnut", "Pecan", "Pesto", "Pili nut",
    "Pine nut", "Pistachio", "Praline", "Shea nut"]
    
    shellfish = ["shellfish", "Barnacle", "Crab", "Crawfish", "crawdad", "crayfish",
                "ecrevisse", "Krill", "Lobster", "langouste", "langoustine",
                "Moreton bay bugs", "scampi", "tomalley", "Prawns", "Shrimp"]
    
    fish = ["fish", "Anchovies", "Bass", "Catfish", "Cod", "Flounder", "Grouper", "Haddock",
            "Hake", "Halibut", "Herring", "Mahi mahi", "Perch", "Pike", "Pollock",
            "Salmon", "Scrod", "Sole", "Snapper", "Swordfish", "Tilapia", "Trout", "Tuna"]
    
    sesame = ["sesame", "Benne", "benniseed", "Gingelly",
              "gingelly oil", "Gomasio", "Halvah", "Sesamol", "Sesamum indicum",
              "Sesemolina", "Sim sim", "Tahini", "Tahina", "Tehina", "Til"]
    
    filtered_df = filtered_df.withColumn("Contains_Milk", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Egg", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Peanut", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Soy", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Wheat", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Treenut", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Shellfish", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Fish", lit(0))
    filtered_df = filtered_df.withColumn("Contains_Sesame", lit(0))
    
    
    filtered_df = filtered_df.toPandas()
    
    for idx, row in filtered_df.iterrows():
        ingredients = row['ingredients_text']
        for item in milk:
            if item.upper() in str(ingredients).upper().replace("COCONUT MILK", "").replace("ALMOND MILK", "").replace("OAT MILK", "").replace("SOY MILK", ""):
                filtered_df.loc[idx, 'Contains_Milk'] = 1
        for item in egg:
            if item.upper() in str(ingredients).upper():
                filtered_df.loc[idx, 'Contains_Egg'] = 1
        for item in peanut:
            if item.upper() in str(ingredients).upper().replace("COCONUT", ""):
                filtered_df.loc[idx, 'Contains_Peanut'] = 1
        for item in soy:
            if item.upper() in str(ingredients).upper():
                filtered_df.loc[idx, 'Contains_Soy'] = 1
        for item in wheat:
            if item.upper() in str(ingredients).upper().replace("ALMOND FLOUR", "").replace("CHICKPEA FLOUR", "").replace("COCONUT FLOUR", ""):
                filtered_df.loc[idx, 'Contains_Wheat'] = 1
        for item in treenut:
            if item.upper() in str(ingredients).upper().replace("COCONUT", ""):
                filtered_df.loc[idx, 'Contains_Treenut'] = 1
        for item in shellfish:
            if item.upper() in str(ingredients).upper():
                filtered_df.loc[idx, 'Contains_Shellfish'] = 1
        for item in fish:
            if item.upper() in str(ingredients).upper():
                filtered_df.loc[idx, 'Contains_Fish'] = 1
        for item in sesame:
            if item.upper() in str(ingredients).upper():
                filtered_df.loc[idx, 'Contains_Sesame'] = 1
    
    #filtered_df = spark.createDataFrame(filtered_df)


    return filtered_df


st.set_page_config(
    page_title="Ingredients App",
    layout="centered",
    initial_sidebar_state="auto"
)
 
# Add a title to your app
st.title("Welcome to the Ingredients App!")
 
# Add some text
st.write("Select a recipe below to get started.")
 
recipe_list = ["https://www.spendwithpennies.com/easy-homemade-lasagna/", "https://www.allrecipes.com/recipe/68461/buffalo-chicken-dip/",
               "https://www.spendwithpennies.com/homemade-brownies/#wprm-recipe-container-250452", "https://www.acozykitchen.com/broccoli-cheddar-cheese-soup#wprm-recipe-container-38077",
               "https://chefsavvy.com/5-ingredient-peanut-butter-energy-bites/#recipe", "https://goodfooddiscoveries.com/green-goddess-soup-with-harissa-grilled-cheese/#wpzoom-premium-recipe-card",
               ]
 
recipes = pd.DataFrame(columns = ['recipe', 'ingredient_raw'])
 
for r in recipe_list:
    scraper = scrape_me(r)
    recipe_title = scraper.title()
    recipe_ingredients = scraper.ingredients()
 
    temp = pd.DataFrame({
        'recipe': [recipe_title] * len(recipe_ingredients),
        'ingredient_raw': recipe_ingredients
    })
 
 
    recipes = pd.concat([recipes, temp], ignore_index=True)
 
processed_recipes = pd.DataFrame(columns= ['RECIPE', 'INGREDIENT', 'QUANTITY', 'UNIT', 'PREP'])
 
for i, row in recipes.iterrows():
  parse_recipe = parse_ingredient(row['ingredient_raw'])
 
  parsed_ingredient = parse_recipe.name[0].text if parse_recipe.name else None
  parsed_quantity  = parse_recipe.amount[0].quantity if parse_recipe.amount else None
  parsed_unit      = parse_recipe.amount[0].unit if parse_recipe.amount else None
  parsed_prep      = parse_recipe.preparation if parse_recipe.preparation else None
 
  processed_recipes.loc[len(processed_recipes)] = {
        'RECIPE': row['recipe'],
        'INGREDIENT': parsed_ingredient,
        'QUANTITY': parsed_quantity,
        'UNIT': parsed_unit,
        'PREP': parsed_prep.text if hasattr(parsed_prep, "text") else parsed_prep
    }
 
selected_recipe = st.selectbox("Choose a recipe:", processed_recipes['RECIPE'].unique())
 
st.write(f"You selected: {selected_recipe}")
 
allergy_list = ['Milk', 'Egg', 'Peanut', 'Soy', 'Wheat', 'Treenut', 'Shellfish', 'Fish', 'Sesame']
 
selected_allergies = st.multiselect("Do you have any allergies:", allergy_list)
st.write(f"You selected: {selected_allergies}")
preference_list = ['Low Sugar', 'Low Sodium', 'Best Health Score', 'Low Fat', 'High Protein', 'High Fiber']
 
selected_preferences = st.multiselect("Do you have any dietary preferences:", preference_list)
st.write(f"You selected: {selected_preferences}")
#filtered selected recipes on the selected recipe:
processed_recipes_filtered = processed_recipes[processed_recipes['RECIPE'] == selected_recipe]
 
selected_ingredient = st.selectbox("Select an ingredient to explore options for:", processed_recipes_filtered['INGREDIENT'].unique())

import altair as alt

if st.button('Submit'):
    
  file_path = './en.openfoodfacts.org.products.tsv'
  df = load_data(file_path)
  #df = spark.createDataFrame(df)

  for a in selected_allergies:
    if a == 'Milk':
      df = df[df['Contains_Milk'] == 0]
    elif a == 'Egg':
      df = df[df['Contains_Egg'] == 0]
    elif a == 'Peanut':
      df = df[df['Contains_Peanut'] == 0]
    elif a == 'Soy':
      df = df[df['Contains_Soy'] == 0]
    elif a == 'Wheat':
      df = df[df['Contains_Wheat'] == 0]
    elif a == 'Treenut':
      df = df[df['Contains_Treenut'] == 0]
    elif a == 'Shellfish':
      df = df[df['Contains_Shellfish'] == 0]
    elif a == 'Fish':
      df = df[df['Contains_Fish'] == 0]
    elif a == 'Sesame':
      df = df[df['Contains_Sesame'] == 0]
    
  # Use advanced matching algorithm instead of simple substring search
  df = match_ingredient_to_products(selected_ingredient, df, top_n=100)
  
  if df.empty:
    st.warning(f"No products found matching '{selected_ingredient}' with your allergy restrictions.")
  else:
    # Build sorting parameters based on preferences
    sort_columns = []
    sort_ascending = []
    
    for p in selected_preferences:
      if p == 'Low Sugar':
        sort_columns.append('sugars_100g')
        sort_ascending.append(True)
      elif p == 'Best Health Score':
        sort_columns.append('nutrition_grade_fr')
        sort_ascending.append(True)
      elif p == 'Low Fat':
        sort_columns.append('fat_100g')
        sort_ascending.append(True)
      elif p == 'High Protein':
        sort_columns.append('proteins_100g')
        sort_ascending.append(False)
      elif p == 'High Fiber':
        sort_columns.append('fiber_100g')
        sort_ascending.append(False)
      elif p == 'Low Sodium':
        sort_columns.append('sodium_100g')
        sort_ascending.append(True)
    
    # Apply all sorts at once, but keep relevance_score as primary sort if no preferences
    if sort_columns:
      # Add relevance score as a tie-breaker
      sort_columns.append('relevance_score')
      sort_ascending.append(False)
      df = df.sort_values(by=sort_columns, ascending=sort_ascending)
    else:
      # If no preferences, sort by relevance score only
      df = df.sort_values(by='relevance_score', ascending=False)

    #df = df.toPandas()
    df = df.head(3)
    
    st.write("Here are the top 3 suggested products based on your selections:")
    
    # Show relevance score in debug mode (optional - can comment out)
    st.dataframe(df[['product_name','nutrition_grade_fr', 'relevance_score']],use_container_width=True,hide_index=True)
    
  # Sample Data

   # st.bar_chart(df, x='product_name', y='sugars_100g')
   
    chart_sugar = alt.Chart(df).mark_bar().encode(
        x='product_name', y='sugars_100g',
        color='product_name' # Color bars by category
    ).properties(
        title='Compare Sugar Content per 100g'
    )
    st.altair_chart(chart_sugar, use_container_width=True)

    chart_fat = alt.Chart(df).mark_bar().encode(
        x='product_name', y='fat_100g',
        color='product_name' # Color bars by category
    ).properties(
        title='Compare Fat Content per 100g'
    )
    st.altair_chart(chart_fat, use_container_width=True)

    chart_protein = alt.Chart(df).mark_bar().encode(
        x='product_name', y='proteins_100g',
        color='product_name' # Color bars by category
    ).properties(
        title='Compare Protein Content per 100g'
    )
    st.altair_chart(chart_protein, use_container_width=True)

    chart_sodium = alt.Chart(df).mark_bar().encode(
        x='product_name', y='sodium_100g',
        color='product_name' # Color bars by category
    ).properties(
        title='Compare Sodium Content per 100g'
    )
    st.altair_chart(chart_sodium, use_container_width=True)
 
 
# sort based on preferences and show top 3 best options 
# show graphs comparing them