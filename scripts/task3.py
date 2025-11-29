import tarfile 
import pandas as pd 
import os 
import numpy as np 
import lzma
import matplotlib.pyplot as plt 
from sklearn import preprocessing, pipeline, model_selection
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay


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
                        output_path = os.path.join(output_dir, os.path.basename(member.name))
                        with open(output_path, "wb") as output_file:
                            output_file.write(extracted_file.read())
                        print(f"File extracted and saved to: {output_path}")

    appointments_path = "../data/extracted_data/appointments/appointments.txt"
    participants_path = "../data/extracted_data/appointments/participants.txt"
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
        n_jobs=-1
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
        n_jobs=-1 
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
        refit = refit_metric,
        n_jobs=-1 
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
    
    # --- FIX FOR ATTRIBUTE ERROR ---
    # Extract the ColumnTransformer from the Pipeline if necessary
    if hasattr(preprocessor, 'named_steps'):
        # Access the step named 'preprocessor'
        transformer_step = preprocessor.named_steps['preprocessor']
    else:
        # It is already the transformer
        transformer_step = preprocessor
        
    # Now access named_transformers_ on the correct object
    cat_names = transformer_step.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = list(cat_names) + num_cols
    # -------------------------------
    
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
if __name__ == "__main__": 
    appointments_df, participants_df = load_data(filepath, output_dir)
    combined_df = merge_data(appointments_df, participants_df)
    
    # Get the preprocessor definition and the raw DataFrame
    preprocessor, combined_df_processed, cat_cols, num_cols = data_process_pipeline(combined_df) 
    rng, cv, x, y = initialize_constants(combined_df_processed)

    # Note: You now pass the preprocessor to the grid search functions
    # random_forest_grid_search(rng, cv, x, y, preprocessor)
    # gradient_boosted_grid_search(rng, cv, x, y, preprocessor)
    # knn_grid_search(cv, x, y, preprocessor)

    results_df = best_hyperparam_run(rng, cv, x, y, preprocessor)

    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(results_df.sort_values(by="Mean F1 Score", ascending=False).to_markdown(index=False))

    plot_pr_curves_cross_val(rng, cv, x, y, preprocessor)
    plot_feature_impact(combined_df_processed, preprocessor, cat_cols, num_cols)