import streamlit as st
import pandas as pd
from recipe_scrapers import scrape_me
from ingredient_parser import parse_ingredient
 
#import pyspark
 
#from pyspark.sql import DataFrame, SparkSession, SQLContext
from typing import List
#import pyspark.sql.types as T
#import pyspark.sql.functions as F
#from pyspark.sql.functions import lit
import time

 

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

 # No more pyspark, findspark, or os.environ imports needed!

@st.cache_data
def load_data(file_path):
    # Simulate a long-running data loading process
    time.sleep(2)
    
    # Define columns to keep to save memory
    selected_columns = ['product_name', 'url', 'nutrition_grade_fr', 'ingredients_text', 'fat_100g',
                        'proteins_100g', 'carbohydrates_100g', 'sugars_100g', 'sodium_100g',
                        'fiber_100g', 'additives_n', 'countries']
    
    # Read with Pandas instead of Spark
    df = pd.read_csv(
        file_path,
        sep='\t',
        escapechar='"',
        usecols=selected_columns, # Only load columns we need
        low_memory=False,
        on_bad_lines='skip' # Handles potential multiline issues
    )

    # Filter with Pandas
    filtered_df = df[df['product_name'].notna()].copy()
    
    # US only (make sure to handle NaNs in string operations)
    filtered_df = filtered_df[filtered_df['countries'].str.contains('United States', na=False)]
    
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
    
    # Add columns (Pandas-style)
    filtered_df["Contains_Milk"] = 0
    filtered_df["Contains_Egg"] = 0
    filtered_df["Contains_Peanut"] = 0
    filtered_df["Contains_Soy"] = 0
    filtered_df["Contains_Wheat"] = 0
    filtered_df["Contains_Treenut"] = 0
    filtered_df["Contains_Shellfish"] = 0
    filtered_df["Contains_Fish"] = 0
    filtered_df["Contains_Sesame"] = 0
    
    # This loop is already Pandas-based and correct
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
    
  file_path = '6242/en.openfoodfacts.org.products.tsv'
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
 