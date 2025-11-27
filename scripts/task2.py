import pandas as pd 
import tarfile 
import os 
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

rng = np.random.RandomState(31)
output_dir = "../results"

def extract_data(file_path, extract_to="../data/extracted_data"):
    with tarfile.open(file_path, "r:xz") as tar:
        tar.extractall(path=extract_to)
    print(f"Files extracted to: {extract_to}")

    data_frames = {}
    for file_name in os.listdir("../data/extracted_data/visits"):
        if file_name.endswith(".csv"):
            df_name = os.path.splitext(file_name)[0]
            data_frames[df_name] = pd.read_csv(os.path.join("../data/extracted_data/visits", file_name))
            print(f"Loaded: {df_name}")
    return data_frames

def combine_data(data_frames):
    combined_df = pd.concat(data_frames.values(), ignore_index=True)
    print("Combined DataFrame:")
    return combined_df

def correct_typos_ASF(combined_df): 
    expected_values_CDR_map = {'midl': "mild", 
                           'very miId' : 'very mild', 
                           'very midl' : 'very mild', 
                           'vry mild' : 'very mild'}
    combined_df["CDR"] = combined_df['CDR'].map(expected_values_CDR_map).fillna(combined_df["CDR"])
    return combined_df

def clean_data(df):

    df = correct_typos_ASF(df)

    combined_df["sex"] = combined_df["sex"].replace({"M": 1, "F":0})
    df['sex'] = df['sex'].astype('category')
    df['CDR'] = df['CDR'].astype('category')
    df['ASF'] = pd.to_numeric(df['ASF'], errors='coerce')
    df['ASF'] = df['ASF'].replace(103.0, np.nan) #handling outlier

    df['ASF'] = df['ASF'].fillna(df['ASF'].median())
    df['SES'] = df['SES'].fillna(df['SES'].median())
    df['MMSE'] = df['MMSE'].fillna(df['MMSE'].mean())
    df['eTIV'] = df['eTIV'].fillna(df['eTIV'].mean())

    print("Cleaned DataFrame:")
    return df

def handle_nominal_and_ordinal(df):
    cdr_order = ['none', 'very mild', 'mild', 'moderate', 'severe']
    ses_order = [1, 2, 3, 4, 5]

    df['CDR'] = pd.Categorical(df['CDR'], categories=cdr_order, ordered=True)

    df['SES'] = pd.Categorical(df['SES'], categories=ses_order, ordered=True)

    print("Updated DataFrame with Nominal and Ordinal Attributes:")
    return df 

def create_target_variable(df):
    def parse_cdr(val):
        mapping = {
            'none': 0.0,
            'very mild': 0.5,
            'mild': 1.0,
            'moderate': 2.0,
            'severe': 3.0
        }
        return mapping.get(str(val).lower(), np.nan)

    combined_df['CDR_numeric'] = combined_df['CDR'].apply(parse_cdr)
    baseline_cdr = combined_df[combined_df['visit'] == 1][['ID', 'CDR_numeric']].rename(columns={'CDR_numeric': 'baseline_CDR'})
    max_cdr = combined_df.groupby('ID')['CDR_numeric'].max().reset_index().rename(columns={'CDR_numeric': 'max_CDR'})
    target_df = pd.merge(baseline_cdr, max_cdr, on='ID')
    target_df['worsened'] = (target_df['max_CDR'] > target_df['baseline_CDR']).astype(int)
    df_final = pd.merge(combined_df, target_df[['ID', 'worsened']], on='ID', how='left')
    return df_final

def plot_disease_progression(df_final):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_final, x='visit', y='CDR_numeric')
    plt.title('Disease Progression Over Time')
    plt.xlabel('Visit')
    plt.ylabel('CDR_numeric')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disease_progression.png"), dpi=300)
    plt.close()
    print("Disease progression plot saved.")


def prepare_data(df_final):
    df_final = df_final[df_final["visit"] == 1] 
    x = df_final.drop(columns=["ID", "MRI_ID", "hand", "visit", "delay", "CDR", "worsened"])
    y = df_final["worsened"]
    return x, y

def logisitic_regression_without_penalty(x, y, rng): 
    results = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)  
    K = cv.n_splits
    df = K - 1
    t_critical = stats.t.ppf(0.975, df)
    accuracy_scores = []
    f1_scores = []
    for train_index, val_index in cv.split(x, y):
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        scaler = StandardScaler()
        scaler.fit(x_train) 
        x_train_scaled = scaler.transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        lr = LogisticRegression(solver='lbfgs', random_state=rng, max_iter=1000, C=1.0, penalty=None)
        lr.fit(x_train_scaled, y_train)
        y_pred_val = lr.predict(x_val_scaled)
        fold_accuracy = accuracy_score(y_val, y_pred_val)
        fold_f1 = f1_score(y_val, y_pred_val)
        accuracy_scores.append(fold_accuracy)
        f1_scores.append(fold_f1)
    
    # 2. Calculate final summary statistics
    cv_accuracy_mean = np.mean(accuracy_scores)
    cv_accuracy_std = np.std(accuracy_scores)
    cv_f1_mean = np.mean(f1_scores)
    
    sem_accuracy = cv_accuracy_std / np.sqrt(K)
    margin_of_error = t_critical * sem_accuracy
    ci_lower = cv_accuracy_mean - margin_of_error
    ci_upper = cv_accuracy_mean + margin_of_error

    # --- Store Results ---
    results.append({
        'CV_Mean_Accuracy': f"{cv_accuracy_mean:.4f}",
        'CV_Mean_f1': f"{cv_f1_mean:.4f}",
        'CI_Lower_95': f"{ci_lower:.4f}",
        'CI_Upper_95': f"{ci_upper:.4f}"
    })

    return results

def logisitic_regression_with_penalty(X, y,rng):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)  
    results = []
    
    penalties_config = [
        {'penalty': 'l2', 'solver': 'lbfgs', 'name': 'L2 (Ridge)'},
        {'penalty': 'l1', 'solver': 'liblinear', 'name': 'L1 (Lasso)'}
    ]

    for config in penalties_config:
        accuracy_scores = []
        f1_scores = []
        
        penalty = config['penalty']
        solver = config['solver']
        name = config['name']

        K = cv.n_splits
        df = K - 1
        t_critical = stats.t.ppf(0.975, df)
        
        for train_index, val_index in cv.split(X, y):
            
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            lr = LogisticRegression(
                solver=solver, 
                penalty=penalty, 
                C=1.0, 
                random_state=rng, 
                max_iter=1000
            )
            
            lr.fit(X_train_scaled, y_train)
            y_pred_val = lr.predict(X_val_scaled)
         
            accuracy_scores.append(accuracy_score(y_val, y_pred_val))
            f1_scores.append(f1_score(y_val, y_pred_val))
      
        cv_accuracy_mean = np.mean(accuracy_scores)
        cv_accuracy_std = np.std(accuracy_scores)
        cv_f1_mean = np.mean(f1_scores)

        sem_accuracy = cv_accuracy_std / np.sqrt(K)
        margin_of_error = t_critical * sem_accuracy
        ci_lower = cv_accuracy_mean - margin_of_error
        ci_upper = cv_accuracy_mean + margin_of_error

        # --- Store Results ---
        results.append({
            'Penalty': name,
            'CV_Mean_Accuracy': f"{cv_accuracy_mean:.4f}",
            'CV_Mean_f1': f"{cv_f1_mean:.4f}",
            'CI_Lower_95': f"{ci_lower:.4f}",
            'CI_Upper_95': f"{ci_upper:.4f}"
        })

    return results


if __name__ == "__main__": 
    dataframes = extract_data("../data/visits.tar.xz")
    combined_df = combine_data(data_frames=dataframes)
    cleaned_data = clean_data(combined_df)
    data = handle_nominal_and_ordinal(cleaned_data)
    target_df = create_target_variable(data)

    plot_disease_progression(target_df)

    x,y = prepare_data(target_df)

    results = logisitic_regression_without_penalty(x, y, rng)
    print(results)

    results_with_penalty = logisitic_regression_with_penalty(x,y, rng)
    print(results_with_penalty)