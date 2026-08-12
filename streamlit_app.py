import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st

@st.cache_data
def load_data(path = "tags-20210226.csv"):
    return pd.read_csv(path)

@st.cache_data
def plot_type(type, title="type", figsize=(8,8), offset = 0):
    fig= plt.figure(figsize=figsize)
    plt.plot(type['name'], type['cached_count'], marker='o', linestyle='-', color='black')
    peak = type["cached_count"].idxmax()
    peak_x = type.loc[peak, 'name']
    peak_y = type.loc[peak, 'cached_count']
        
    plt.scatter(peak_x, peak_y, color='red', s=100, zorder=5)
    plt.annotate(
    f'{peak_x}: {peak_y}',
    xy=(peak_x, peak_y),
    xytext=(peak_x, peak_y + offset),
    
)

    
    plt.xlabel('Name of Tag')
    plt.ylabel('Count')
    plt.title(f"Count of Fan Works Based on {title}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

########################################################################################################################################################

df_tags = load_data()
df_tags_clean = df_tags.dropna(subset=['name'])

st.title("What makes a Fan Fiction Popular regardless of what fandom it belongs to?")
st.write("AO3 is a non-profit website for fan work collection where fans either publish their works or engage with published works. " \
"Fan works are usually written works, each work have certain informations available such as **hits** (how many people viewed the work), **kudos** (how many people liked the work), **comments** and **bookmarks** (how many people bookmarked the work), date of publishing and date of last update, **rating tag** (ex. general audience), **category tags** (ex. F/F), **warning tags**, fandom tags, **freeform tags**, along with a summary (if available), author notes and text of the work. " )
st.write("Across Fandoms you will find certain works that are well known among fans I want to know what makes them popular, " \
"I created a proxy for popularity by calculating hits per fandom size for each fan work." \
"I trained different models to predict popularity based on **engagement features** (kudos, comments, bookmarks), **size of fanfiction** (ex. short, long), **rating tags**, **category tags**, **warning tags** and **freeform tags**. " )
st.write("Freeform tags where clustered into 50 clusters to be able to encode them. " \
"Independent of fandom (as certain fandoms are much more poplar than others) what makes a fanfiction work popular what are the important features contributing to its popularity:" \
"I assusme that engagment features play an important role as people can filter works based on how many kudos it has for example, size of work is also important, the themes in the fan work usually summarized well as Freeform, Category, Rating and Warning tags, and completion status.")

st.subheader("Popular Tags")
type_sel = st.sidebar.selectbox("Select Tag Type", ['Media', 'Rating',
       'Fandom', 'Relationship'])

type = df_tags_clean[df_tags_clean["type"] == type_sel]



if type_sel == "Media":
    plot_type(type, "Media", offset=23)
elif type_sel == "Rating":
    type["cached_count"] = type["cached_count"] / 1000000
    plot_type(type, "Rating in Millions", offset= 0.04)
elif type_sel == "Fandom":
    fandom = type.sort_values(by='cached_count', ascending=False)
    top = st.sidebar.slider("Top ##", 5,20)
    fandom_top = fandom.head(top)
    plot_type(fandom_top, f"Fandom (Top {top})",  offset=1000)
elif type_sel == "Relationship":
    relationship = type.sort_values(by='cached_count', ascending=False)
    top = st.sidebar.slider("Top ##", 5,20)
    relationship_top = relationship.head(top)
    plot_type(relationship_top, f"Relationship (Top {top})", offset=1000)

X = load_data(path = "features_final.csv")
y = load_data(path = "y_final.csv")

st.subheader("Models")
st.write("**1. Linear Regression**")

X_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_const).fit(cov_type='HC3')
st.write("Summary")
st.write(f"R2     = {model_ols.rsquared: .4f}") # percentage of var in coef
st.write(f"Adj R2     = {model_ols.rsquared_adj: .4f}")
st.write()


sig = model_ols.pvalues[model_ols.pvalues < 0.05].sort_values()
st.write("sig predictors are:")
rows = []
for var, pv in sig.items():
  coef = model_ols.params[var]
  rows.append(f"{var:<35s}, coef = {coef:.4f}, p = {pv:.4f}")
#   st.write(f"{var:<35s}, coef = {coef:.4f}, p = {pv:.4f}")
df = pd.DataFrame(rows)
st.dataframe(df)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
st.sidebar.write("**XGBRegressor Parameters**")
st.sidebar.write("Sets maximum tree depth")
max_depth = st.sidebar.slider("max_depth", 3,9, 6)
st.sidebar.write("Controls each tree's contribution")
learning_rate = st.sidebar.slider("learning_rate", 0.01,0.2, 0.1)
st.sidebar.write("Fraction of data used per tree")
subsample = st.sidebar.slider("subsample", 0.1,1.0, 1.0)
st.sidebar.write("Fraction of features per tree")
colsample_bytree = st.sidebar.slider("colsample_bytree", 0.1,1.0, 0.8)


xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=123, max_depth=max_depth,
learning_rate=learning_rate,
subsample=subsample,
colsample_bytree=colsample_bytree)
xgb_regressor.fit(X_train, y_train)

# Make predictions
predictions = xgb_regressor.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
st.write("**2. XGBRegressor**")
st.write(f"RMSE: {rmse}")
st.write(f'R²: {r2:.3f}')

importance = xgb_regressor.get_booster().get_score(importance_type='weight')

importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values(by='Importance', ascending=False)

st.sidebar.write("Feature Importance")
top_n = st.sidebar.slider("Top Feature Importance", 1,30, 10)
# top_n = 20
fig = plt.figure(figsize=(10, 8))
plt.barh(
    importance_df['Feature'].head(top_n)[::-1],
    importance_df['Importance'].head(top_n)[::-1],
    color='skyblue'
)
plt.xlabel('Importance Score')
plt.title(f'Top {top_n} Feature Importance')
plt.tight_layout()
st.pyplot(fig)

st.write("After Hyperparameter Tuning, the model now can explain 8.7% of the variance which is double what the linear regression can predict. However this model explains less than 10% of the variance in data and still barely outpreforms a model that predict the mean every time where the model RMSE is 0.44 and popularity variable std = 0.46. " \
"What this tells us is that the relationships between features and dependent variable aren't a simple linear relationship however these variables such as engagement variables, tags, summary words count and completion status aren't the only variables that can predict popularity some pontential variables that are missing here is author's subscribers count which is not available on AO3 website, if the fan work was promoted on other websites and the visibility it gained from it and if the time the work was published coincides with a fandom trend, event or the time the fandom was more active.   ")
