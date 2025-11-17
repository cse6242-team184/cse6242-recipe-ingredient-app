import streamlit as st
import pandas as pd
import altair as alt
import time
import csv
import re
from recipe_scrapers import scrape_me
from ingredient_parser import parse_ingredient

url = "https://github.com/jordanavery92-javery3/cse6242-recipe-ingredient-app/releases/download/v1.0/en.openfoodfacts.org.products.tsv"

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

#spark= SparkSession \
#       .builder \
#       .appName("Our First Spark Example") \
#       .getOrCreate()
 
#  sc = pyspark.SparkContext(appName="HW3-Q1")
#Get Data:

# sc = pyspark.SparkContext(appName="HW3-Q1")
# sqlContext = SQLContext(sc)

@st.cache_data
def load_data(url):
    # 1. Load a manageable chunk of data (Fixes "Oh no" crash)
    # Increasing nrows=30000 might hit the memory limit, testing required.
    try:
        df = pd.read_csv(
            url,
            sep='\t',
            quoting=csv.QUOTE_NONE,
            usecols=['product_name', 'ingredients_text', 'nutrition_grade_fr', 'fat_100g',
                     'proteins_100g', 'carbohydrates_100g', 'sugars_100g', 'sodium_100g',
                     'fiber_100g', 'countries'],
            low_memory=False,
            on_bad_lines='skip',
            nrows=30000  # <--- CRITICAL FIX: Limit rows to fit in RAM
        )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    # 2. Basic cleanup
    df = df[df['product_name'].notna()]
    df = df[df['countries'].str.contains('United States', na=False, case=False)]
    
    # Ensure ingredients are strings and uppercase for searching
    df['ingredients_text'] = df['ingredients_text'].astype(str).str.upper()

    # 3. Fast Vectorized Allergen Detection (Replaces the slow loop)
    # This logic runs in milliseconds instead of minutes.
    
    def tag_allergen(df, col_name, allergens, exclusions=[]):
        # Create a temporary working column
        temp_ing = df['ingredients_text']
        
        # Remove exclusions (e.g., remove "COCONUT MILK" before checking for "MILK")
        for excl in exclusions:
            temp_ing = temp_ing.str.replace(excl.upper(), "", regex=False)
        
        # Join all allergens into a single search pattern: "MILK|BUTTER|CHEESE"
        # re.escape ensures special characters like "half-and-half" don't break regex
        pattern = '|'.join([re.escape(x.upper()) for x in allergens])
        
        # Check efficiently
        df[col_name] = temp_ing.str.contains(pattern, regex=True).astype(int)
        return df

    # Define your lists (copied from your code)
    milk = ["milk", "butter", "casein", "cheese", "cream", "curd", "custard", "ghee", "half-and-half", "lactose", "lactulose", "whey", "tagatose", "yogurt"]
    egg = ["egg", "Albumin", "Albumen", "Apovitellin", "Avidin globulin", "Lysozyme", "Mayonnaise", "Meringue", "Ovalbumin", "Ovomucoid", "Ovomucin", "Ovovitellin", "Surimi", "Vitellin"]
    peanut = ["peanut", "Arachis oil", "Artificial nuts", "Beer nuts", "Ground nuts", "Lupin", "Lupine", "Mandelonas", "Mixed nuts", "Monkey nuts", "Nut meat", "nut meal", "Nut pieces", "nut"]
    soy = ["soy", "edamame", "miso", "natto", "okara", "Shoyu", "Soya", "Tamari", "Tempeh", "Textured vegetable protein", "TVP", "tofu"]
    wheat = ["Bread crumbs", "Bulgur", "Cereal extract", "Couscous", "Cracker meal", "Durum", "Einkorn", "Emmer", "Farina", "Farro", "Flour", "Freekeh", "Hydrolyzed wheat protein", "Matzoh", "matzo", "matzah", "matza", "Pasta", "Seitan", "Semolina", "Spelt", "Triticale", "Vital wheat gluten", "Wheat"]
    treenut = ["tree nut", "Almond", "nut", "Cashew", "Filbert", "Gianduja", "Litchi", "lichee", "lychee", "Macadamia","Marzipan", "walnut", "Pecan", "Pesto", "Pili nut", "Pine nut", "Pistachio", "Praline", "Shea nut"]
    shellfish = ["shellfish", "Barnacle", "Crab", "Crawfish", "crawdad", "crayfish", "ecrevisse", "Krill", "Lobster", "langouste", "langoustine", "Moreton bay bugs", "scampi", "tomalley", "Prawns", "Shrimp"]
    fish = ["fish", "Anchovies", "Bass", "Catfish", "Cod", "Flounder", "Grouper", "Haddock", "Hake", "Halibut", "Herring", "Mahi mahi", "Perch", "Pike", "Pollock", "Salmon", "Scrod", "Sole", "Snapper", "Swordfish", "Tilapia", "Trout", "Tuna"]
    sesame = ["sesame", "Benne", "benniseed", "Gingelly", "gingelly oil", "Gomasio", "Halvah", "Sesamol", "Sesamum indicum", "Sesemolina", "Sim sim", "Tahini", "Tahina", "Tehina", "Til"]

    # Apply the fast tagging
    df = tag_allergen(df, 'Contains_Milk', milk, exclusions=["COCONUT MILK", "ALMOND MILK", "OAT MILK", "SOY MILK"])
    df = tag_allergen(df, 'Contains_Egg', egg)
    df = tag_allergen(df, 'Contains_Peanut', peanut, exclusions=["COCONUT"])
    df = tag_allergen(df, 'Contains_Soy', soy)
    df = tag_allergen(df, 'Contains_Wheat', wheat, exclusions=["ALMOND FLOUR", "CHICKPEA FLOUR", "COCONUT FLOUR"])
    df = tag_allergen(df, 'Contains_Treenut', treenut, exclusions=["COCONUT"])
    df = tag_allergen(df, 'Contains_Shellfish', shellfish)
    df = tag_allergen(df, 'Contains_Fish', fish)
    df = tag_allergen(df, 'Contains_Sesame', sesame)

    return df


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
    
  df = load_data(url)
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
    
    #after user selection --> filter for ingredient
    #search_term = f"product_name LIKE '%{selected_ingredient.lower()}%'"

   # df = df[df['product_name'] == selected_ingredient]

    for p in selected_preferences:
      if p == 'Low Sugar':
        df = df.sort_values(by="sugars_100g", ascending = True)
      elif p == 'Best Health Score':
        df = df.sort_values(by="nutrition_grade_fr", ascending = True)
      elif p == 'Low Fat':
        df = df.sort_values(by="fat_100g", ascending = True)
      elif p == 'High Protein':
        df = df.sort_values(by="proteins_100g", ascending = False)
      elif p == 'High Fiber':
        df = df.sort_values(by="fiber_100g", ascending = False)
      elif p == 'Low Sodium':
        df = df.sort_values(by="sodium_100g", ascending = False)

    #df = df.toPandas()
  df = df.head(3)
  st.write("Here are the top 3 suggested products based on your selections:")
  st.dataframe(df[['product_name','nutrition_grade_fr']],use_container_width=True,hide_index=True)
  
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

 




