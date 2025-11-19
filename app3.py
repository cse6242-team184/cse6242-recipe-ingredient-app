import streamlit as st
import pandas as pd
from recipe_scrapers import scrape_me
from ingredient_parser import parse_ingredient
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
    """
    Load data in chunks to avoid memory issues with large files.
    Only keeps US products with valid product names to minimize memory usage.
    """
    # Define columns we need
    selected_columns = ['product_name', 'url', 'nutrition_grade_fr', 'ingredients_text', 'fat_100g',
                        'proteins_100g', 'carbohydrates_100g', 'sugars_100g', 'sodium_100g',
                        'fiber_100g', 'additives_n', 'countries']
    
    # Define allergen lists
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
    
    # Helper function to detect allergens
    def detect_allergen(ingredients_str, allergen_list):
        if pd.isna(ingredients_str):
            return 0
        ingredients_upper = str(ingredients_str).upper()
        return int(any(item.upper() in ingredients_upper for item in allergen_list))
    
    # Read file in chunks and filter
    chunk_list = []
    chunk_size = 50000  # Process 50k rows at a time
    
    with pd.read_csv(file_path, sep='\t', encoding='utf-8', 
                     chunksize=chunk_size, usecols=selected_columns,
                     on_bad_lines='skip') as reader:
        
        for chunk in reader:
            # Filter: product_name must exist
            chunk = chunk[chunk['product_name'].notna()]
            
            # Filter: US only
            chunk = chunk[chunk['countries'].fillna('').str.contains('United States', case=False, na=False)]
            
            # Only keep this chunk if it has data
            if len(chunk) > 0:
                chunk_list.append(chunk)
    
    # Combine all filtered chunks
    if not chunk_list:
        st.error("No US products found in the dataset!")
        return pd.DataFrame()
    
    filtered_df = pd.concat(chunk_list, ignore_index=True)
    
    # Now detect allergens on the filtered dataset (much smaller than original)
    ingredients_upper = filtered_df['ingredients_text'].fillna('').str.upper()
    
    # Milk detection (exclude plant-based milks)
    milk_clean = ingredients_upper.str.replace("COCONUT MILK", "", regex=False)\
                                   .str.replace("ALMOND MILK", "", regex=False)\
                                   .str.replace("OAT MILK", "", regex=False)\
                                   .str.replace("SOY MILK", "", regex=False)
    filtered_df['Contains_Milk'] = milk_clean.apply(lambda x: detect_allergen(x, milk))
    
    # Egg detection
    filtered_df['Contains_Egg'] = ingredients_upper.apply(lambda x: detect_allergen(x, egg))
    
    # Peanut detection (exclude coconut)
    peanut_clean = ingredients_upper.str.replace("COCONUT", "", regex=False)
    filtered_df['Contains_Peanut'] = peanut_clean.apply(lambda x: detect_allergen(x, peanut))
    
    # Soy detection
    filtered_df['Contains_Soy'] = ingredients_upper.apply(lambda x: detect_allergen(x, soy))
    
    # Wheat detection (exclude alternative flours)
    wheat_clean = ingredients_upper.str.replace("ALMOND FLOUR", "", regex=False)\
                                    .str.replace("CHICKPEA FLOUR", "", regex=False)\
                                    .str.replace("COCONUT FLOUR", "", regex=False)
    filtered_df['Contains_Wheat'] = wheat_clean.apply(lambda x: detect_allergen(x, wheat))
    
    # Tree nut detection (exclude coconut)
    treenut_clean = ingredients_upper.str.replace("COCONUT", "", regex=False)
    filtered_df['Contains_Treenut'] = treenut_clean.apply(lambda x: detect_allergen(x, treenut))
    
    # Shellfish detection
    filtered_df['Contains_Shellfish'] = ingredients_upper.apply(lambda x: detect_allergen(x, shellfish))
    
    # Fish detection
    filtered_df['Contains_Fish'] = ingredients_upper.apply(lambda x: detect_allergen(x, fish))
    
    # Sesame detection
    filtered_df['Contains_Sesame'] = ingredients_upper.apply(lambda x: detect_allergen(x, sesame))

    return filtered_df


st.set_page_config(
    page_title="Smart Recipe & Ingredient Finder",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🍳"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main background and layout */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Title styling */
    .stTitle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        font-size: 1.4rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Success messages */
    .element-container div[data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 0.9rem 3rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        transform: translateY(-3px);
    }
    
    /* Selectbox styling */
    div[data-baseweb="select"] {
        border-radius: 10px;
    }
    
    /* Multiselect styling */
    div[data-baseweb="tag"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #667eea;
        font-weight: 700;
    }
    
    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f9fafb;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    }
    </style>
    """, unsafe_allow_html=True)
 
# Add a title to your app
st.markdown('<h1 class="stTitle">🍳 Smart Recipe & Ingredient Finder</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover healthier ingredient alternatives tailored to your dietary needs!</p>', unsafe_allow_html=True)

# Add divider
st.divider()
 
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
 
 
# Recipe selection section
st.markdown('<div class="section-header">Step 1: Choose Your Recipe</div>', unsafe_allow_html=True)

selected_recipe = st.selectbox(
    "Select a recipe:",
    processed_recipes['RECIPE'].unique(),
    help="Choose from our curated collection of delicious recipes",
    label_visibility="collapsed",
    placeholder="Choose a recipe..."
)

# Show ingredient count
recipe_count = len(processed_recipes[processed_recipes['RECIPE'] == selected_recipe])
st.markdown(f"<div style='background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); padding: 1rem; border-radius: 10px; margin-top: 1rem; border-left: 4px solid #3B82F6;'><strong>Recipe Details:</strong> This recipe contains <strong>{recipe_count} ingredients</strong></div>", unsafe_allow_html=True)

st.divider()

# Dietary preferences section
st.markdown('<div class="section-header">Step 2: Set Your Dietary Preferences</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Allergies & Restrictions**")
    allergy_list = ['Milk', 'Egg', 'Peanut', 'Soy', 'Wheat', 'Treenut', 'Shellfish', 'Fish', 'Sesame']
    selected_allergies = st.multiselect(
        "Select any allergies:",
        allergy_list,
        help="We'll filter out products containing these allergens",
        label_visibility="collapsed",
        placeholder="Choose allergens to exclude..."
    )
    if selected_allergies:
        st.markdown(f"<div style='background: #FEE2E2; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #EF4444; margin-top: 0.5rem;'><strong>✓ Excluding:</strong> {', '.join(selected_allergies)}</div>", unsafe_allow_html=True)

with col2:
    st.markdown("**Dietary Preferences**")
    preference_list = ['Low Sugar', 'Low Sodium', 'Best Health Score', 'Low Fat', 'High Protein', 'High Fiber']
    selected_preferences = st.multiselect(
        "Select your preferences:",
        preference_list,
        help="Products will be sorted based on these preferences",
        label_visibility="collapsed",
        placeholder="Choose dietary preferences..."
    )
    if selected_preferences:
        st.markdown(f"<div style='background: #D1FAE5; padding: 0.8rem; border-radius: 8px; border-left: 4px solid #10B981; margin-top: 0.5rem;'><strong>✓ Prioritizing:</strong> {', '.join(selected_preferences)}</div>", unsafe_allow_html=True)

st.divider()

# Ingredient selection section
st.markdown('<div class="section-header">Step 3: Select an Ingredient to Explore</div>', unsafe_allow_html=True)

# Initialize session state for selected ingredient
if 'selected_ingredient' not in st.session_state:
    st.session_state.selected_ingredient = None

#filtered selected recipes on the selected recipe:
processed_recipes_filtered = processed_recipes[processed_recipes['RECIPE'] == selected_recipe]

# Prepare dataframe with index for selection - convert Fraction to string for display
ingredients_df = processed_recipes_filtered[['INGREDIENT', 'QUANTITY', 'UNIT']].copy().reset_index(drop=True)
ingredients_df.columns = ['Ingredient', 'Quantity', 'Unit']
# Convert Quantity column to string to handle Fraction objects
ingredients_df['Quantity'] = ingredients_df['Quantity'].apply(lambda x: str(x) if x is not None else '')
ingredients_df['Unit'] = ingredients_df['Unit'].apply(lambda x: str(x) if x is not None else '')

# Display the table (read-only)
st.markdown("**Ingredients in this recipe:**")
st.dataframe(
    ingredients_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ingredient": st.column_config.TextColumn(disabled=True),
        "Quantity": st.column_config.TextColumn(disabled=True),
        "Unit": st.column_config.TextColumn(disabled=True)
    }
)

st.markdown("---")

# Use selectbox for ingredient selection
st.markdown("**Select an ingredient to find alternatives:**")
selected_ingredient = st.selectbox(
    "Choose an ingredient:",
    options=ingredients_df['Ingredient'].tolist(),
    key="ingredient_selector",
    label_visibility="collapsed",
    index=None,
    placeholder="Choose an ingredient..."
)

if selected_ingredient:
    st.success(f"✓ Selected: **{selected_ingredient}**")
    st.session_state.selected_ingredient = selected_ingredient
elif st.session_state.selected_ingredient:
    selected_ingredient = st.session_state.selected_ingredient
    st.info(f"Currently selected: **{selected_ingredient}**")

st.markdown("")  # Spacing

import altair as alt

# Center the submit button
st.markdown("---")

# Only show submit button if ingredient is selected, otherwise show message
if selected_ingredient is None:
    st.info("👆 Please complete all steps above and select an ingredient to continue")
    submit_button = False
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button('🔍 Find Best Products', use_container_width=True)

if submit_button and selected_ingredient:
    
    with st.spinner('🔄 Analyzing products and applying your preferences...'):
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
    
    st.divider()
    
    if df.empty:
        st.error(f"😞 No products found matching '{selected_ingredient}' with your allergy restrictions.")
        st.info("💡 Try adjusting your allergy selections or choosing a different ingredient.")
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
        
        st.markdown('<div class="section-header">Top 3 Product Recommendations</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 1.1rem; color: #6B7280; margin-bottom: 1.5rem;'>Showing best matches for: <strong style='color: #667eea;'>{selected_ingredient}</strong></div>", unsafe_allow_html=True)
        
        # Display products in expandable cards with better styling
        for idx, (i, row) in enumerate(df.iterrows(), 1):
            # Color scheme for each rank
            rank_icons = {1: '🥇', 2: '🥈', 3: '🥉'}
            
            with st.expander(f"{rank_icons[idx]} #{idx} - {row['product_name']}", expanded=(idx == 1)):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown("**Product Information**")
                    if pd.notna(row['url']):
                        st.markdown(f"[View Product on Open Food Facts]({row['url']})")
                    else:
                        st.markdown("*Product link not available*")
                    
                with col2:
                    grade = row['nutrition_grade_fr'] if pd.notna(row['nutrition_grade_fr']) else 'N/A'
                    st.metric("Nutrition Grade", grade.upper() if grade != 'N/A' else grade)
                
                with col3:
                    score = f"{row['relevance_score']:.1f}" if pd.notna(row['relevance_score']) else 'N/A'
                    st.metric("Match Score", score)
                
                st.markdown("---")
                st.markdown("**Nutritional Information (per 100g)**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    sugar = f"{row['sugars_100g']:.1f}g" if pd.notna(row['sugars_100g']) else 'N/A'
                    st.metric("🍬 Sugar", sugar)
                with col2:
                    fat = f"{row['fat_100g']:.1f}g" if pd.notna(row['fat_100g']) else 'N/A'
                    st.metric("🧈 Fat", fat)
                with col3:
                    protein = f"{row['proteins_100g']:.1f}g" if pd.notna(row['proteins_100g']) else 'N/A'
                    st.metric("💪 Protein", protein)
                with col4:
                    sodium = f"{row['sodium_100g']:.1f}g" if pd.notna(row['sodium_100g']) else 'N/A'
                    st.metric("🧂 Sodium", sodium)
        
        st.divider()
        
        # Comparison charts section
        st.markdown('<div class="section-header">Nutritional Comparison</div>', unsafe_allow_html=True)
        
        # Create tabs for different charts
        tab1, tab2, tab3, tab4 = st.tabs(["🍬 Sugar", "🧈 Fat", "💪 Protein", "🧂 Sodium"])
        
        with tab1:
            chart_sugar = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8,
                size=60
            ).encode(
                x=alt.X('product_name:N', 
                       title=None,
                       axis=alt.Axis(labelAngle=0, labelFontSize=12, labelLimit=200)),
                y=alt.Y('sugars_100g:Q', 
                       title='Sugar (g per 100g)',
                       axis=alt.Axis(titleFontSize=13, labelFontSize=11),
                       scale=alt.Scale(domain=[0, df['sugars_100g'].max() * 1.1])),
                color=alt.Color('product_name:N', 
                               legend=None, 
                               scale=alt.Scale(range=['#34A853', '#93C47D', '#B6D7A8'])),
                tooltip=[
                    alt.Tooltip('product_name:N', title='Product'),
                    alt.Tooltip('sugars_100g:Q', title='Sugar (g)', format='.2f')
                ]
            ).properties(
                height=350
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=True,
                gridColor='#E8E8E8',
                domainColor='#E8E8E8'
            )
            st.altair_chart(chart_sugar, use_container_width=True)

        with tab2:
            chart_fat = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8,
                size=60
            ).encode(
                x=alt.X('product_name:N', 
                       title=None,
                       axis=alt.Axis(labelAngle=0, labelFontSize=12, labelLimit=200)),
                y=alt.Y('fat_100g:Q', 
                       title='Fat (g per 100g)',
                       axis=alt.Axis(titleFontSize=13, labelFontSize=11),
                       scale=alt.Scale(domain=[0, df['fat_100g'].max() * 1.1])),
                color=alt.Color('product_name:N', 
                               legend=None, 
                               scale=alt.Scale(range=['#FBBC04', '#F9CB9C', '#FCE5CD'])),
                tooltip=[
                    alt.Tooltip('product_name:N', title='Product'),
                    alt.Tooltip('fat_100g:Q', title='Fat (g)', format='.2f')
                ]
            ).properties(
                height=350
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=True,
                gridColor='#E8E8E8',
                domainColor='#E8E8E8'
            )
            st.altair_chart(chart_fat, use_container_width=True)

        with tab3:
            chart_protein = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8,
                size=60
            ).encode(
                x=alt.X('product_name:N', 
                       title=None,
                       axis=alt.Axis(labelAngle=0, labelFontSize=12, labelLimit=200)),
                y=alt.Y('proteins_100g:Q', 
                       title='Protein (g per 100g)',
                       axis=alt.Axis(titleFontSize=13, labelFontSize=11),
                       scale=alt.Scale(domain=[0, df['proteins_100g'].max() * 1.1])),
                color=alt.Color('product_name:N', 
                               legend=None, 
                               scale=alt.Scale(range=['#4285F4', '#6FA8DC', '#9FC5E8'])),
                tooltip=[
                    alt.Tooltip('product_name:N', title='Product'),
                    alt.Tooltip('proteins_100g:Q', title='Protein (g)', format='.2f')
                ]
            ).properties(
                height=350
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=True,
                gridColor='#E8E8E8',
                domainColor='#E8E8E8'
            )
            st.altair_chart(chart_protein, use_container_width=True)

        with tab4:
            chart_sodium = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=8,
                cornerRadiusTopRight=8,
                size=60
            ).encode(
                x=alt.X('product_name:N', 
                       title=None,
                       axis=alt.Axis(labelAngle=0, labelFontSize=12, labelLimit=200)),
                y=alt.Y('sodium_100g:Q', 
                       title='Sodium (g per 100g)',
                       axis=alt.Axis(titleFontSize=13, labelFontSize=11),
                       scale=alt.Scale(domain=[0, df['sodium_100g'].max() * 1.1])),
                color=alt.Color('product_name:N', 
                               legend=None, 
                               scale=alt.Scale(range=['#EA4335', '#E06666', '#EA9999'])),
                tooltip=[
                    alt.Tooltip('product_name:N', title='Product'),
                    alt.Tooltip('sodium_100g:Q', title='Sodium (g)', format='.2f')
                ]
            ).properties(
                height=350
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=True,
                gridColor='#E8E8E8',
                domainColor='#E8E8E8'
            )
            st.altair_chart(chart_sodium, use_container_width=True)
        
        # Add summary insights
        st.markdown("<hr style='margin: 2rem 0; border: none; height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent);'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #667eea; margin-bottom: 1.5rem;'>Quick Insights</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not df['sugars_100g'].isna().all() and len(df[df['sugars_100g'].notna()]) > 0:
                best_sugar_idx = df['sugars_100g'].idxmin()
                best_sugar = df.loc[best_sugar_idx, 'product_name']
                st.markdown(f"<div style='background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); padding: 1.2rem; border-radius: 12px; text-align: center; border: 2px solid #10B981;'><div style='font-size: 2rem; margin-bottom: 0.5rem;'>🍬</div><div style='font-weight: 600; color: #065F46; margin-bottom: 0.3rem;'>Lowest Sugar</div><div style='font-size: 0.9rem; color: #047857;'>{best_sugar}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background: #F3F4F6; padding: 1.2rem; border-radius: 12px; text-align: center;'><div style='font-size: 2rem; margin-bottom: 0.5rem;'>🍬</div><div style='font-weight: 600; color: #6B7280;'>Lowest Sugar</div><div style='font-size: 0.9rem; color: #9CA3AF;'>Data not available</div></div>", unsafe_allow_html=True)
        
        with col2:
            if not df['proteins_100g'].isna().all() and len(df[df['proteins_100g'].notna()]) > 0:
                best_protein_idx = df['proteins_100g'].idxmax()
                best_protein = df.loc[best_protein_idx, 'product_name']
                st.markdown(f"<div style='background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); padding: 1.2rem; border-radius: 12px; text-align: center; border: 2px solid #3B82F6;'><div style='font-size: 2rem; margin-bottom: 0.5rem;'>💪</div><div style='font-weight: 600; color: #1E40AF; margin-bottom: 0.3rem;'>Highest Protein</div><div style='font-size: 0.9rem; color: #1D4ED8;'>{best_protein}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background: #F3F4F6; padding: 1.2rem; border-radius: 12px; text-align: center;'><div style='font-size: 2rem; margin-bottom: 0.5rem;'>💪</div><div style='font-weight: 600; color: #6B7280;'>Highest Protein</div><div style='font-size: 0.9rem; color: #9CA3AF;'>Data not available</div></div>", unsafe_allow_html=True)
        
        with col3:
            # Handle nutrition grade which is a string (a, b, c, d, e)
            if not df['nutrition_grade_fr'].isna().all() and len(df[df['nutrition_grade_fr'].notna()]) > 0:
                # Filter out NaN values and find the best grade
                df_with_grades = df[df['nutrition_grade_fr'].notna()].copy()
                if len(df_with_grades) > 0:
                    # Convert to string and get the minimum (a < b < c, etc.)
                    df_with_grades['grade_lower'] = df_with_grades['nutrition_grade_fr'].astype(str).str.lower()
                    best_grade_idx = df_with_grades['grade_lower'].idxmin()
                    best_grade = df_with_grades.loc[best_grade_idx, 'product_name']
                    st.markdown(f"<div style='background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); padding: 1.2rem; border-radius: 12px; text-align: center; border: 2px solid #F59E0B;'><div style='font-size: 2rem; margin-bottom: 0.5rem;'>⭐</div><div style='font-weight: 600; color: #92400E; margin-bottom: 0.3rem;'>Best Grade</div><div style='font-size: 0.9rem; color: #B45309;'>{best_grade}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background: #F3F4F6; padding: 1.2rem; border-radius: 12px; text-align: center;'><div style='font-size: 2rem; margin-bottom: 0.5rem;'>⭐</div><div style='font-weight: 600; color: #6B7280;'>Best Grade</div><div style='font-size: 0.9rem; color: #9CA3AF;'>Data not available</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='background: #F3F4F6; padding: 1.2rem; border-radius: 12px; text-align: center;'><div style='font-size: 2rem; margin-bottom: 0.5rem;'>⭐</div><div style='font-weight: 600; color: #6B7280;'>Best Grade</div><div style='font-size: 0.9rem; color: #9CA3AF;'>Data not available</div></div>", unsafe_allow_html=True)
 
 
# Add footer
st.markdown("<hr style='margin: 3rem 0 2rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent, #e5e7eb, transparent);'>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #9CA3AF; padding: 1.5rem 0;'>
        <p style='margin: 0; font-size: 0.95rem;'> Data sourced from <a href='https://world.openfoodfacts.org/' style='color: #667eea; text-decoration: none;'>Open Food Facts</a></p>
    </div>
    """, unsafe_allow_html=True)
# sort based on preferences and show top 3 best options 
# show graphs comparing them