# Import libararies
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# ----- Cleaning -----
# Read dataset
df = pd.read_csv("all_activity_log.csv")

# Convert date columns to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df['updated_at'] = pd.to_datetime(df['updated_at'])

# Flatten the dictionary inside `properties` column 
df['properties'] = df['properties'].apply(ast.literal_eval)
df['campaign_id'] = df['properties'].apply(
    lambda x: x.get('campaign_id') if isinstance(x, dict) else None
    ) # Exact 'campaign_id' inside the properties column

# Dropping rows
open_click_df = df[df['subject_type'].isin(['opens', 'clicks'])].copy()

# Dropping columns
cleaned_df = open_click_df.drop(columns=[
                                'id',
                                'log_name',
                                'subject_id',
                                'properties',
                                'updated_at'
                                ])

# Filter out the NaN in `campaign_id`
cleaned_df = cleaned_df[cleaned_df['campaign_id'].notna()]

# Rename columns
cleaned_df = cleaned_df.rename(columns={
    'created_at' : 'event_time',
    'campaign_id' : 'email_id'
})

# ----- Feature Engineering -----
# Filter to clicks-only df and opens-only df
clicks = cleaned_df[cleaned_df['subject_type'] == 'clicks']
opens = cleaned_df[cleaned_df['subject_type'] == 'opens']

# First click event per session 
first_click = clicks.groupby(['subscriber_id', 'email_id'])['event_time'].min().rename('first_clk_time') # rename column

# First open event per session
first_open = opens.groupby(['subscriber_id', 'email_id'])['event_time'].min().rename('first_opn_time') # rename column

# Concat `first_click` and `first_open`
session_df = pd.concat([first_click, first_open], axis=1)

# Feature 1 -- Compute the difference
session_df['clks_b4_opn'] = (
    (session_df['first_clk_time'] < session_df['first_opn_time']) | # checks click before open
    (session_df['first_clk_time'].notna() & session_df['first_opn_time'].isna()) # Checks for clicks but no opn
).astype(int) # change Tures and Falses into 1|0

# Feature 2 -- Compute latency between email opened and first click and convert to seconds
session_df['opn_to_first_clk_gap'] = (session_df['first_clk_time'] - session_df['first_opn_time']).dt.total_seconds()
session_df.loc[session_df['opn_to_first_clk_gap'] < 0, 'opn_to_first_clk_gap'] = np.nan # Replacing negative values with NaN


# Feature 3 -- Compute the minimum time gap between two consecutive clicks in each email
# Sort by event_time, calculate the difference between each click timestamps within each email session
# Then convert the result to seconds, add the subscriber_id and email_id back to the df after performing .diff()
# take the minimum per email session
click_sorted = clicks.sort_values(['subscriber_id', 'email_id', 'event_time'])
time_diffs = click_sorted.groupby(['subscriber_id', 'email_id'])['event_time'].diff().dt.total_seconds()
time_diffs = time_diffs.groupby([click_sorted['subscriber_id'], click_sorted['email_id']]).min().rename('min_clk_time_gap')

# Feature 4 -- Count all clicks per email
clicks_per_email = clicks.groupby(['subscriber_id', 'email_id']).size().rename('click_counts')

# Merge all the df and features into one df
session_df_v2 = pd.concat([session_df, time_diffs, clicks_per_email], axis=1).reset_index().copy()

# Aggregate all 4 features to subscriber level
sub_lv_act_agg = session_df_v2.groupby('subscriber_id').agg(
    clks_b4_opn_prop = ('clks_b4_opn', 'mean'), # operationally, it's simply the average because it's only 1's and 0's
    opn_to_first_clk_gap_avg = ('opn_to_first_clk_gap', 'mean'),
    min_clk_sec_gap_avg = ('min_clk_time_gap', 'mean'),
    click_counts_avg = ('click_counts', 'mean')
)

# Replacing NaN values 
sub_lv_act_agg['opn_to_first_clk_gap_avg'] = sub_lv_act_agg['opn_to_first_clk_gap_avg'].fillna(sub_lv_act_agg['opn_to_first_clk_gap_avg'].median())
sub_lv_act_agg['click_counts_avg'] = sub_lv_act_agg['click_counts_avg'].fillna(0)
# Fill with 99th percentile
sub_lv_act_agg['min_clk_sec_gap_avg'] = sub_lv_act_agg['min_clk_sec_gap_avg'].fillna(sub_lv_act_agg['min_clk_sec_gap_avg'].quantile(.99))

# ----- Stratification -----
sub_df = pd.read_csv('active_subscriber_raw.csv')
sub_df['ctor'] = (sub_df['clicks_count'] / sub_df['opens_count']) * 100
sub_act_merge = sub_lv_act_agg.merge(sub_df[['id', 'ctor']], left_on='subscriber_id', right_on='id', how='left')

# Stratified sample
low = sub_act_merge[sub_act_merge['ctor'] == 0].sample(300, random_state=42)
mid = sub_act_merge[(sub_act_merge['ctor'] > 0) & (sub_act_merge['ctor'] < 65)].sample(300, random_state=42)
high = sub_act_merge[sub_act_merge['ctor'] >= 65].sample(400, random_state=42)
sample_df = pd.concat([low, mid, high], ignore_index=True)

# ----- Modeling -----
# Initialize and fir the scaler
scaler = RobustScaler()
scaled_fe = scaler.fit_transform(sample_df[['clks_b4_opn_prop', 'min_clk_sec_gap_avg', 'click_counts_avg']])

# K-Means
kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10) # Fit final model
kmeans2.fit(scaled_fe)

# Assign cluster labels
sample_df['kmeans_bot'] = (kmeans2.labels_ == 0).astype(int) # subject to change in every run
cluster_profile = sample_df.groupby('kmeans_bot')[['clks_b4_opn_prop', 'opn_to_first_clk_gap_avg', 
                      'min_clk_sec_gap_avg', 'click_counts_avg']].mean()
print(cluster_profile.round(4))
print(sample_df['kmeans_bot'].value_counts())  


# Rule-based labeling
# Defining the four thresholds
rule_clks_b4_opn = sample_df['clks_b4_opn_prop'] > 0
rule_opn_to_clk = sample_df['opn_to_first_clk_gap_avg'] < 2
rule_min_clk_gap = sample_df['min_clk_sec_gap_avg'] < 5
rule_click_counts = sample_df['click_counts_avg'] > 4

# Applying thresholds
sample_df['rule_bot'] = (
    rule_clks_b4_opn |
    rule_opn_to_clk |
    rule_min_clk_gap |
    rule_click_counts
).astype(int)

# Version B lebel -- rule-based labels only
sample_df['ver_b_bot'] = sample_df['rule_bot'].copy() # All subscribers flagged by rule-based thresholds

# Version A label -- Combined Label
# Only subscribers flagged by both K-Means and rule-based
combined_mask = (sample_df['kmeans_bot'] == 1) & (sample_df['rule_bot'] == 1)
human_mask = (sample_df['kmeans_bot'] == 0) & (sample_df['rule_bot'] == 0)
sample_df['ver_a_bot'] = np.nan               # initialize entire column as NaN (uncertain)
sample_df.loc[combined_mask, 'ver_a_bot'] = 1 # where both agree - bot
sample_df.loc[human_mask, 'ver_a_bot'] = 0    # where both agree - human

# Defining features for the model input
features = ['clks_b4_opn_prop', 'opn_to_first_clk_gap_avg', 
            'min_clk_sec_gap_avg', 'click_counts_avg']

# Version A -- training set
sample_df_A = sample_df.dropna(subset=['ver_a_bot']).copy() # drop NaN rows
X_A = sample_df_A[features]
y_A = sample_df_A['ver_a_bot']
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X_A, y_A, test_size=0.2, random_state=42, stratify=y_A
)
rf_label_a = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_label_a.fit(X_train_A, y_train_A)

# Evaluate on the 20% test data
y_pred_A = rf_label_a.predict(X_test_A)
y_train_A_pred = rf_label_a.predict(X_train_A)
print(f"Version A training data F1: {f1_score(y_train_A, y_train_A_pred)}")
print(f"Version A Precision:        {precision_score(y_test_A, y_pred_A):.4f}")
print(f"Version A Recall:           {recall_score(y_test_A, y_pred_A):.4f}")
print(f"Version A F1:               {f1_score(y_test_A, y_pred_A):.4f}")
print(f"Version A Confusion Matrix: {(confusion_matrix(y_test_A, y_pred_A))}")

# Version B
X = sample_df[features] # the input
y = sample_df['ver_b_bot'] # the labels we are predicting

# Train/test split 
# stratify preserves class ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,   # hold out 20% of the data for evaluation
    random_state=42, # seeds the randomness so results are reproducible
    stratify=y       # ensures bot/human ratio is preserved in both train and test sets
)

# Fit Random Forest
rf_label_b = RandomForestClassifier(
    n_estimators=100,        # setting the model to run 100 decision trees
    class_weight='balanced', # upweights bot class to compensate for imbalance
    random_state=42          # seeds bootstrap sampling and feature selection for reproducibility
)

# Train the model
rf_label_b.fit(X_train, y_train)

# Evaluate on the 20% test data
y_pred = rf_label_b.predict(X_test)
y_train_pred = rf_label_b.predict(X_train)
print(f"Version B training data F1: {f1_score(y_train, y_train_pred)}")
print(f"Version B Precision:        {precision_score(y_test, y_pred):.4f}")
print(f"Version B Recall:           {recall_score(y_test, y_pred):.4f}")
print(f"Version B F1:               {f1_score(y_test, y_pred):.4f}")
print(f"Version B Confusion Matrix: {(confusion_matrix(y_test, y_pred))}")

# ----- Deployment -----
# Score full subscriber dataset using trained RF
X_deploy = sub_act_merge[features]
sub_act_merge['bot_score'] = rf_label_b.predict(X_deploy)
sub_act_merge['bot_probability'] = rf_label_b.predict_proba(X_deploy)[:, 1]

print(sub_act_merge['bot_score'].value_counts())
print(f"Total flagged bots: {sub_act_merge['bot_score'].sum()}")
print(f"Bot rate: {sub_act_merge['bot_score'].mean():.2%}")

sub_act_merge.to_csv('bot_scores_full.csv', index=False)
print("Done. bot_scores_full.csv saved.")