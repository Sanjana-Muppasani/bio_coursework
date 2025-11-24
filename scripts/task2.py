import pandas as pd 
import tarfile 
import os 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
    df_final = df_final[df_final["visit"]==1] #only visit 1 for prediction 
    x = df_final.drop(columns=["ID", "MRI_ID", "hand", "visit", "delay", "CDR", "worsened"])
    y = df_final["worsened"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=rng)

    scaler = StandardScaler()
    scaler.fit(x_train)
    data_train_scaled = scaler.transform(x_train)
    data_test_scaled = scaler.transform(x_test)
    return data_train_scaled, data_test_scaled, y_train, y_test



def logisitic_regression_without_penalty(data_train_scaled, data_test_scaled, y_train, y_test):
    lr = LogisticRegression(solver='lbfgs', random_state=rng, max_iter=1000, C=1.0, penalty=None)
    lr.fit(data_train_scaled, y_train)
    y_pred = lr.predict(data_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    n = len(y_test)
    ci = 1.96 * np.sqrt((accuracy * (1 - accuracy)) / n)

    ci_lwr = (accuracy - ci) 
    ci_hi= accuracy + ci

    return accuracy, ci_lwr, ci_hi

def logisitic_regression_with_penallty(data_train_scaled, data_test_scaled, y_train, y_test):
    results = []

    # Define the penalties and their required solvers
    penalties_config = [
        {'penalty': 'l2', 'solver': 'lbfgs', 'name': 'L2 (Ridge)'},
        {'penalty': 'l1', 'solver': 'liblinear', 'name': 'L1 (Lasso)'}
    ]

    for config in penalties_config:
        penalty = config['penalty']
        solver = config['solver']
        name = config['name']
        lr = LogisticRegression(
            solver=solver, 
            penalty=penalty, 
            C=1.0, 
            random_state=rng, 
            max_iter=1000
        )
        lr.fit(data_train_scaled, y_train)
        y_pred = lr.predict(data_test_scaled)
        
        # 4. Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        n = len(y_test)
        
        # Calculate 95% Confidence Interval (Normal Approximation)
        ci_half_width = 1.96 * np.sqrt((accuracy * (1 - accuracy)) / n)
        ci_lower = accuracy - ci_half_width
        ci_upper = accuracy + ci_half_width

        results.append({
            'Penalty': name,
            'Accuracy': f"{accuracy:.4f}",
            'CI_Lower': f"{ci_lower:.4f}",
            'CI_Upper': f"{ci_upper:.4f}"
        })

    return results

if __name__ == "__main__": 
    dataframes = extract_data("../data/visits.tar.xz")
    combined_df = combine_data(data_frames=dataframes)
    cleaned_data = clean_data(combined_df)
    data = handle_nominal_and_ordinal(cleaned_data)
    target_df = create_target_variable(data)

    plot_disease_progression(target_df)

    data_train_scaled, data_test_scaled, y_train, y_test = prepare_data(target_df)
    accuracy, lwr, hi = logisitic_regression_without_penalty(data_train_scaled, data_test_scaled, y_train, y_test)

    print("\n--- Summary Table (without penalty) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"95% CI: {lwr:.4f} - {hi:.4f}")


    results_with_penalty = logisitic_regression_with_penallty(data_train_scaled, data_test_scaled, y_train, y_test)

    print("\n--- Summary Table (with penalty) ---")
    for r in results_with_penalty:
        print(f"| {r['Penalty']:<15} | Acc: {r['Accuracy']} | 95% CI: {r['CI_Lower']} - {r['CI_Upper']} |")