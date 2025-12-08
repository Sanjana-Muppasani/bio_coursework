import tarfile 
import pandas as pd 
import os 
import numpy as np 
import lzma
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing, pipeline, model_selection
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

#directory initialization 
results_dir = "../results"
filepath = "../data/appointments.tar.xz"
output_dir = "../data/extracted_data/appointments"
os.makedirs(output_dir, exist_ok=True)

def load_data(filepath, output_dir): 
    with lzma.open(filepath, "rb") as xz_file:
        with tarfile.open(fileobj=xz_file, mode="r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".txt"):
                    extracted_file = tar.extractfile(member)
                    if extracted_file is not None:
                        # Saves to project/processed/appointments/...
                        output_path = os.path.join(output_dir, os.path.basename(member.name))
                        with open(output_path, "wb") as output_file:
                            output_file.write(extracted_file.read())
                        print(f"File extracted and saved to: {output_path}")
                        
    appointments_path = os.path.join(output_dir, "appointments.txt")
    participants_path = os.path.join(output_dir, "participants.txt")
    
    appointments_df = pd.read_csv(appointments_path, sep=r'\s+')
    participants_df = pd.read_csv(participants_path, sep=r'\s+')

    return appointments_df, participants_df

def merge_data(appointments_df, participants_df): 
    combined_df = pd.merge(appointments_df, participants_df, on="participant", how="inner")
    filtered_df = combined_df[combined_df["count"] >= 5]
    return filtered_df

def data_process_pipeline(filtered_df): 
    filtered_df["status"] = filtered_df["status"].replace({"fullfilled":0, "no-show":1})
    categorical_features = ['sms_received', 'weekday', 'sex', 'hipertension', 'diabetes', 'alcoholism']
    numerical_features = ['advance', 'day', 'month', 'age', 'count']

    categorical_transformer = preprocessing.OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = preprocessing.StandardScaler() 

    # 4. Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])

    preprocessing_pipeline = pipeline.Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    # df_transformed_array = preprocessing_pipeline.fit_transform(filtered_df)
    # feature_names = (
    #     preprocessing_pipeline.named_steps['preprocessor']
    #     .named_transformers_['cat']
    #     .get_feature_names_out(categorical_features)
    # )
    # all_feature_names = list(feature_names) + numerical_features 
    # df_transformed_scaled = pd.DataFrame(df_transformed_array, columns=all_feature_names)
    # df_transformed_scaled["status"] = filtered_df["status"].reset_index(drop=True)

    return preprocessing_pipeline, filtered_df, categorical_features, numerical_features

def initialize_constants(df_transformed_scaled):
    rng = np.random.RandomState(31)
    cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=rng)  
    y = df_transformed_scaled["status"]
    x = df_transformed_scaled.drop(columns=["status"])

    return rng, cv, x, y

def random_forest_grid_search(rng, cv, x, y, preprocessor): 
    rf = RandomForestClassifier(random_state=rng) 
    param_grid_rf = {
    'n_estimators': [100, 200],        
    'max_depth': [3, 4, 5],              
    'criterion': ["gini", "entropy"],      
    'max_features': [4, 6]       
    }

    full_pipeline = pipeline.Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    scoring_metrics = {
        'F1_Score': 'f1',
        'Accuracy': 'accuracy'
    }
    refit_metric = "F1_Score"

    grid_search_rf = model_selection.GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid_rf,
        scoring=scoring_metrics,
        cv=cv,
        refit=refit_metric, 
    )

    print(f"Starting Grid Search for Random Forest. Testing {len(param_grid_rf['n_estimators']) * len(param_grid_rf['max_depth']) * len(param_grid_rf['criterion']) * len(param_grid_rf['max_features'])} combinations...")


    grid_search_rf.fit(x, y) 

    print(f"\n--- Random Forest Grid Search Results ---")

    best_f1_score_rf = grid_search_rf.best_score_
    print(f"Best Model Selected using {refit_metric}: {grid_search_rf.best_params_}")
    print(f"Best {refit_metric} achieved (5-fold CV mean): {best_f1_score_rf:.4f}")


    results_df = pd.DataFrame(grid_search_rf.cv_results_)
    best_index = grid_search_rf.best_index_

    mean_accuracy_best = results_df.loc[best_index, 'mean_test_Accuracy']
    mean_f1_best = results_df.loc[best_index, 'mean_test_F1_Score']

    print(f"Mean Accuracy of the Best RF Model (5-fold CV mean): {mean_accuracy_best:.4f}")
    print(f"Mean F1 Score of the Best RF Model (5-fold CV mean): {mean_f1_best:.4f}")

def gradient_boosted_grid_search(rng, cv, x, y, preprocessor): 
    param_grid_gb = {
    'n_estimators': [100, 200],         
    'learning_rate': [0.1, 0.5, 1.0],      
    'max_depth': [4],                 
    }
    full_pipeline = pipeline.Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', gb)
    ])
    gb = GradientBoostingClassifier(random_state=rng)

    grid_search_gb = model_selection.GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid_gb,
        scoring=['f1',"accuracy"], 
        cv=cv,
    )

    grid_search_gb.fit(x, y)

    best_f1_score_gb = grid_search_gb.best_score_
    print(f"Best F1 Score found by Grid Search - GB: {best_f1_score_gb:.4f}")
    best_params_gb = grid_search_gb.best_params_
    print(f"Best parameters - GB: {best_params_gb}")

def knn_grid_search(cv, x, y, preprocessor): 
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': list(range(1, 7)),
        'weights': ['uniform', 'distance']
    }
    full_pipeline = pipeline.Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', knn)
    ])
    scoring_metrics = {
        'F1_Score': 'f1',        
        'Accuracy': 'accuracy'   
    }

    refit_metric = "F1_Score"

    grid_search = model_selection.GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        scoring=scoring_metrics, 
        cv=cv,
        refit = refit_metric
    )


    grid_search.fit(x, y)
    print(f"--- GB Grid Search Results ---")
    print(f"Best Model Selected using {refit_metric}: {grid_search.best_params_}")
    print(f"Best {refit_metric} achieved: {grid_search.best_score_:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)

    best_index = grid_search.best_index_

    mean_accuracy_best = results_df.loc[best_index, 'mean_test_Accuracy']
    mean_f1_best = results_df.loc[best_index, 'mean_test_F1_Score']

    print(f"Mean Accuracy of the Best KNN Model: {mean_accuracy_best:.4f}")
    print(f"Mean F1 Score of the Best KNN Model: {mean_f1_best:.4f}")

def best_hyperparam_run(rng, cv, x, y, preprocessor): 

    results = []
    scoring_metrics = ["f1", "accuracy", "precision", "recall"]
    models_to_test = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=rng, criterion="gini", max_features=4
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.5, max_depth=4, random_state=rng
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }
    
    for model_name, classifier in models_to_test.items():
        full_pipeline = pipeline.Pipeline(steps=[
            ('preprocessor', preprocessor), 
            ('classifier', classifier)
        ])
    
        scores = model_selection.cross_validate(
            full_pipeline, x, y, scoring=scoring_metrics, cv=cv,n_jobs=-1)

        if model_name == "K-Nearest Neighbors":
            params_str = f"n_neighbors={classifier.n_neighbors}"
        else:
            params_str = f"n_estimators={classifier.n_estimators}, max_depth={classifier.max_depth}, ..."
        
        results.append({
            "Model": model_name,
            "Parameters": params_str,
            "Mean F1 Score": scores["test_f1"].mean(),
            "Mean Accuracy": scores["test_accuracy"].mean(),
            "Mean Precision": scores["test_precision"].mean(),
            "Mean Recall": scores["test_recall"].mean()
        })

    print("\n--- Model Comparison Summary (10-Fold CV) ---")
    
    results_df = pd.DataFrame(results)
    return results_df

def plot_pr_curves_cross_val(rng, cv, x, y, preprocessor):
    
    models_to_test = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=rng, criterion="gini", max_features=4
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.5, max_depth=4, random_state=rng
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    plt.figure(figsize=(10, 8))
    
    print("\n--- Generating PR Curves (this may take a moment) ---")
    
    for name, classifier in models_to_test.items():
        full_pipeline = pipeline.Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        y_probas = model_selection.cross_val_predict(full_pipeline, x, y, cv=cv, method='predict_proba', n_jobs=-1)
        y_scores = y_probas[:, 1]
        precision, recall, _ = precision_recall_curve(y, y_scores)
        ap_score = average_precision_score(y, y_scores)
        plt.plot(recall, precision, lw=2, label=f'{name} (AP = {ap_score:.2f})')

    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (10-Fold Cross-Validation)")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    
    
    plot_path = os.path.join(results_dir, "pr_curve_comparison.png")
    plt.savefig(plot_path)
    print(f"PR Curve plot saved to: {plot_path}")

def plot_feature_impact(df, preprocessor, cat_cols, num_cols):
    
    y = df["status"]
    X_raw = df.drop(columns=["status"])
    
    # Fit the pipeline
    X_processed = preprocessor.fit_transform(X_raw)
        
    # Now access named_transformers_ on the correct object
    cat_names = preprocessor.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = list(cat_names) + num_cols
    
    # Train Best Model
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=4, random_state=31)
    clf.fit(X_processed, y)
    
    # Calculate Impact & Direction
    importances = clf.feature_importances_
    X_dense = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    correlations = [np.corrcoef(X_dense[:, i], y)[0, 1] for i in range(X_dense.shape[1])]
    
    impact_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Correlation': correlations
    })
    
    # Filter: Only keep features that DRIVE No-Shows (Positive Correlation)
    impact_df = impact_df[impact_df['Correlation'] > 0].sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(impact_df['Feature'], impact_df['Importance'])
    
    plt.title("Top Risk Factors for No-Shows (Positive Drivers Only)", fontsize=14)
    plt.xlabel("Feature Importance (Gini)", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "feature_impact_noshow_only.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

def plot_feature_importance_cv(df, preprocessor, cat_cols, num_cols):
    
    y = df["status"]
    X_raw = df.drop(columns=["status"])

    skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=31)

    fold_results = []

    print("Starting Cross-Validation for Feature Importance...")

    for fold, (train_index, val_index) in enumerate(skf.split(X_raw, y)):
        # Split Data
        X_train_raw, X_val_raw = X_raw.iloc[train_index], X_raw.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Clone and Fit Preprocessor on Training Data Only (Prevents Leakage)
        fold_preprocessor = clone(preprocessor)
        X_train_processed = fold_preprocessor.fit_transform(X_train_raw)
        
        # Retrieve Feature Names (Logic adapted from your snippet)
        # Note: Ensure the access path matches your specific pipeline structure
        try:
            # If preprocessor is a Pipeline containing a step named 'preprocessor'
            cat_names = fold_preprocessor.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols)
        except AttributeError:
            # Fallback: If preprocessor is directly a ColumnTransformer
            cat_names = fold_preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
            
        feature_names = list(cat_names) + num_cols
        
        # Train Model
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=4, random_state=31)
        clf.fit(X_train_processed, y_train)
        
        # Calculate Impact & Direction
        importances = clf.feature_importances_
        
        # Convert to dense for correlation calculation
        X_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
        
        # Calculate correlation between feature and target for this fold
        correlations = [np.corrcoef(X_dense[:, i], y_train)[0, 1] for i in range(X_dense.shape[1])]
        
        # Store results for this fold
        for feat, imp, corr in zip(feature_names, importances, correlations):
            fold_results.append({
                'Fold': fold,
                'Feature': feat,
                'Importance': imp,
                'Correlation': corr
            })

    # Create DataFrame from all folds
    impact_df = pd.DataFrame(fold_results)

    # --- Filtering Strategy ---
    # 1. Group by Feature to get Mean Correlation
    # 2. Keep features where the Average Correlation is Positive (Drivers of No-Show)
    avg_stats = impact_df.groupby('Feature').agg({'Importance':'mean', 'Correlation':'mean'}).reset_index()
    positive_drivers = avg_stats[avg_stats['Correlation'] > 0]['Feature'].tolist()

    # Filter the main DF to only include these positive drivers
    plot_df = impact_df[impact_df['Feature'].isin(positive_drivers)]

    # Sort by Mean Importance for clean plotting
    order = plot_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).index

    # --- Plotting ---
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=plot_df, x='Importance', y='Feature', order=order, palette='magma', showfliers=False)

    plt.title("Feature Importance Stability (5-Fold CV)\nTop Risk Factors for No-Shows", fontsize=16)
    plt.xlabel("Feature Importance (Gini)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "feature_impact_per_fold.png")
    plt.savefig(plot_path)
    print(f"CV Plot saved to: {plot_path}")

def run_ensemble_improvements(rng, cv, x, y, preprocessor):
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=rng, criterion="gini")
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=4, random_state=rng)
    

    # Check Correlation first 
    print("--- Step 1: Checking Model Correlation ---")
    
    pipe_rf = pipeline.Pipeline([('prep', preprocessor), ('clf', rf)])
    pipe_gb = pipeline.Pipeline([('prep', preprocessor), ('clf', gb)])
    
    # Get out-of-fold probability predictions
    print("Generating predictions to check correlation (this takes a moment)...")
    pred_rf = model_selection.cross_val_predict(pipe_rf, x, y, cv=5, method='predict_proba')[:, 1]
    pred_gb = model_selection.cross_val_predict(pipe_gb, x, y, cv=5, method='predict_proba')[:, 1]
    
    correlation = np.corrcoef(pred_rf, pred_gb)[0, 1]
    print(f"Correlation between RF and GB predictions: {correlation:.4f}")
    if correlation > 0.90:
        print(">> WARNING: High correlation. Ensembling might have diminishing returns.")
    else:
        print(">> Good diversity. Ensembling should help.")


    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[1, 2] 
    )
    
    stacking_clf = StackingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        final_estimator=LogisticRegression(),
        passthrough=True,
        cv=5
    )

    models = {
        "Standalone GB (Baseline)": gb,
        "Weighted Soft Voting": voting_clf,
        "Stacking (Passthrough)": stacking_clf
    }
    results = []
    print("\n--- Step 2: Evaluating Models (10-Fold CV) ---")

    for name, model in models.items():
        full_pipeline = pipeline.Pipeline([('prep', preprocessor), ('clf', model)])
        scores = model_selection.cross_validate(full_pipeline, x, y, cv=cv, 
                                scoring=['average_precision', 'roc_auc', 'f1'], 
                                n_jobs=-1)
        
        results.append({
            "Model": name,
            "Avg Precision (AP)": scores['test_average_precision'].mean(),
            "ROC AUC": scores['test_roc_auc'].mean(),
            "F1 Score": scores['test_f1'].mean()
        })
        print(f"Finished evaluating: {name}")

    return pd.DataFrame(results)
    
if __name__ == "__main__": 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define base directories
    data_dir = os.path.join(script_dir, '..', 'data')
    processed_dir = os.path.join(script_dir, '..', 'processed') 
    results_dir = os.path.join(script_dir, '..', 'results')
    filepath = os.path.join(data_dir, "appointments.tar.xz")
    output_dir = os.path.join(processed_dir, "appointments")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True) # Creates project/processed/appointments
    
    print(f"Extraction target: {output_dir}")

    print(f"Results will be saved to: {results_dir}")

    appointments_df, participants_df = load_data(filepath, output_dir)
    combined_df = merge_data(appointments_df, participants_df)
    
    # Get the preprocessor definition and the raw DataFrame
    preprocessor, combined_df_processed, cat_cols, num_cols = data_process_pipeline(combined_df) 
    rng, cv, x, y = initialize_constants(combined_df_processed)

    # Note: Only run the grid search if needed as it is computationally expensive and takes times
    # The optimal hyperparameters have already been determined and are used in the best_hyperparam_run function
    # random_forest_grid_search(rng, cv, x, y, preprocessor)
    # gradient_boosted_grid_search(rng, cv, x, y, preprocessor)
    # knn_grid_search(cv, x, y, preprocessor)

    # results_df = best_hyperparam_run(rng, cv, x, y, preprocessor)

    # pd.set_option('display.float_format', lambda x: '%.4f' % x)
    # print(results_df.sort_values(by="Mean F1 Score", ascending=False).to_markdown(index=False))

    plot_pr_curves_cross_val(rng, cv, x, y, preprocessor)
    plot_feature_importance_cv(combined_df_processed, preprocessor, cat_cols, num_cols)

    # results_df_ensemble = run_ensemble_improvements(rng, cv, x, y, preprocessor)
    
    # print("\n--- Final Results ---")
    # print(results_df_ensemble.sort_values(by="Avg Precision (AP)", ascending=False).to_markdown(index=False))