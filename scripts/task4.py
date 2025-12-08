import pandas as pd 
import numpy as np 
import keras 
import tarfile
import os
import lzma
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Bidirectional, LSTM, GRU, SimpleRNN
from sklearn.model_selection import StratifiedKFold, train_test_split
from scikeras.wrappers import KerasClassifier
from keras.backend import clear_session
from sklearn.metrics import f1_score
from scipy import stats

def load_merge_data(filepath, FN_DOMAIN_LIST, FN_SF_NAMES, FN_SEQ_S60, output_dir): 
    with lzma.open(filepath) as xz_file:
        with tarfile.open(fileobj=xz_file) as tar:
            tar.extractall(path=output_dir)

    sequences = []
    current_id = None
    current_seq = []


    with open(FN_SEQ_S60, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences.append({
                        'domain_id': current_id, 
                        'sequence': "".join(current_seq)
                    })
            
                parts = line[1:].split('|')
                
                if len(parts) >= 3:
                    full_code = parts[2]
                    current_id = full_code.split('/')[0]
                else:
                    current_id = line[1:].split()[0]

                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last entry
        if current_id:
            sequences.append({'domain_id': current_id, 'sequence': "".join(current_seq)})
                
    df_seq =  pd.DataFrame(sequences)

    col_names = ['domain_id', 'C', 'A', 'T', 'H', 'S', 'O', 'L', 'I', 'D', 'len', 'res']
    df_domains = pd.read_csv(
        FN_DOMAIN_LIST, 
        sep=r'\s+', 
        comment='#', 
        header=None,
        names=col_names,
        usecols=['domain_id', 'C', 'A', 'T', 'H','S', 'O', 'L', 'I', 'D', 'len', 'res']
    )

    # Create Superfamily ID (C.A.T.H)
    df_domains['superfamily_id'] = df_domains.apply(
        lambda x: f"{x['C']}.{x['A']}.{x['T']}.{x['H']}", axis=1
    )

    print("Loading superfamily names...")
    sf_names = {}
    with open(FN_SF_NAMES, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                sf_names[parts[0]] = parts[1]

    # D. Merge and Filter
    print("Merging data and filtering...")
    # Join sequences with classification
    df_merged= pd.merge(df_seq, df_domains, on='domain_id', how='inner')
    return df_merged

def filter_df(df_merged):
    sf_counts = df_merged['superfamily_id'].value_counts()
    small_sfs = sf_counts[sf_counts < 1000]
    top_5_sfs = small_sfs.nlargest(5).index.tolist()
    final_data = df_merged[df_merged['superfamily_id'].isin(top_5_sfs)].astype(str).copy()
    final_data = final_data[["sequence", "superfamily_id"]].reset_index(drop=True)

    print(f"Selected Superfamilies: {top_5_sfs}")
    print(f"Number of domains selected: {len(final_data)}")
     
    max_len = int(final_data.sequence.str.len().max())

    return final_data, max_len

def vectorize_sequences_encode_labels(final_data, max_len):
    #vectorization and padding 
    vectorizer = keras.layers.TextVectorization(split="character", output_sequence_length=max_len)
    vectorizer.adapt(final_data.sequence)
    x = vectorizer(final_data.sequence)

    #target variable encoding
    le = LabelEncoder()
    y = le.fit_transform(final_data.superfamily_id)
    return x , y 

def cnn_tuning(TOKENS, DIMENSIONS, CLASSES, SIZE, x_numpy,y_numpy): 

    # 0. DATA SPLIT (Matches your example)
    x_train, x_val, y_train, y_val = train_test_split(
        x_numpy, y_numpy, test_size=0.2, stratify=y, random_state=rng
    )

    # --- 1. DEFINE CNN MODEL ---
    def build_cnn(hp):
        # Tunable parameters
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        dropout = hp.Choice('dropout', values=[0.2, 0.5])
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

        model = keras.Sequential([
            keras.Input(shape=(max_len,)),
            Embedding(TOKENS, DIMENSIONS, mask_zero=False), # mask_zero=False is better for CNNs

            # CNN Layer: Uses tuned 'units' for filters
            # We use padding='same' to ensure Flatten works safely
            Conv1D(filters=units, kernel_size=SIZE, activation="relu", padding='same'),
            MaxPooling1D(pool_size=SIZE),
            
            Flatten(),

            Dense(units, activation="relu"),
            Dropout(dropout),

            Dense(CLASSES, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # --- 2. SETUP THE TUNER ---
    cnn_tuner = kt.RandomSearch(
        build_cnn,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='my_dir',
        project_name='cnn_tuning_v1', # Unique name
        overwrite=True                # CRITICAL: Deletes old logs to prevent errors
    )

    # --- 3. RUN SEARCH ---
    print("Starting CNN Tuner Search...")
    cnn_tuner.search(
        x_train, y_train,
        epochs=50,
        validation_data=(x_val, y_val),
        batch_size=32,
        verbose=1
    )

    # --- 4. GET RESULTS ---
    best_hps = cnn_tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps

def lstm_tuning(TOKENS, DIMENSIONS, CLASSES, x_numpy,y_numpy): 

    x_train, x_val, y_train, y_val = train_test_split(x_numpy, y_numpy, test_size=0.2, stratify=y, random_state=rng)
    # --- 1. DEFINE MODEL WITH HYPERPARAMETERS ---
    def build_model(hp):
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        dropout = hp.Choice('dropout', values=[0.2, 0.5])
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

        model = keras.Sequential([
            keras.Input(shape=(max_len,)),
            Embedding(TOKENS, DIMENSIONS, mask_zero=True),
            
            Bidirectional(LSTM(units)),
            
            Dense(units, activation="relu"),
            Dropout(dropout),
            
            Dense(CLASSES, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5, 
        executions_per_trial=1,
        directory='my_dir',
        project_name='cath_tuning'
    )

    print("Starting Keras Tuner Search...")

    tuner.search(
        x_train, y_train,             
        validation_data=(x_val, y_val), 
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. 
    Best units: {best_hps.get('units')}
    Best learning rate: {best_hps.get('learning_rate')}
    Best dropout: {best_hps.get('dropout')}
    """)

    return best_hps

def gru_tuning(TOKENS, CLASSES, x_numpy ,y_numpy):
    
# --- CONFIGURATION ---
    EPOCHS_TUNING = 50
    MAX_TRIALS = 5 

    x_tune_train, x_tune_val, y_tune_train, y_tune_val = train_test_split(
        x_numpy, y_numpy, test_size=0.2, stratify=y, random_state=rng
    )

    # --- 2. DEFINE THE HYPERMODEL (GRU + RNN) ---
    def build_gru_rnn(hp):
        # --- Tunable Hyperparameters ---
        
        # 1. NEW: Tune Embedding Dimensions
        # We try 8, 16, 32, and 64 to see how "wide" the vector needs to be.
        embed_dim = hp.Choice('embedding_dim', values=[8, 16, 32, 64])

        # 2. Tune Units (GRU)
        units = hp.Int('units', min_value=32, max_value=128, step=32)
        
        # 3. Tune Dropout
        dropout = hp.Choice('dropout', values=[0.2, 0.5])
        
        # 4. Tune Learning Rate
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model = keras.Sequential([
            keras.Input(shape=(max_len,)),
            
            # Pass the tuned 'embed_dim' here instead of a fixed constant
            Embedding(input_dim=TOKENS, output_dim=embed_dim, mask_zero=True),

            # Layer 1: GRU
            GRU(units, return_sequences=True),
            
            # Layer 2: SimpleRNN (Half the size of the GRU)
            SimpleRNN(units // 2),

            Dense(units, activation="relu"),
            Dropout(dropout),

            Dense(CLASSES, activation="softmax")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # --- 3. PHASE 1: HYPERPARAMETER TUNING ---
    print(f"--- Phase 1: Tuning Hyperparameters (Max Trials: {MAX_TRIALS}) ---")

    tuner = kt.RandomSearch(
        build_gru_rnn,
        objective='val_accuracy',
        max_trials=MAX_TRIALS,
        executions_per_trial=1,
        directory='my_dir',
        project_name='gru_rnn_tuning_v2', # Changed name to avoid conflict with previous runs
        overwrite=True
    )

    # Run the search
    tuner.search(
        x_tune_train, y_tune_train,
        epochs=EPOCHS_TUNING,
        validation_data=(x_tune_val, y_tune_val),
        verbose=1
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps

def best_cnn_nestedcv_earlystopping(x_numpy, y_numpy, early_stopping = "yes"): 
    
    NUM_OUTER_FOLDS = 5
    MAX_TRIALS = 5  
    EPOCHS = 50
    BATCH_SIZE = 32

    # TOGGLE EARLY STOPPING HERE ('yes' or 'no')
    USE_EARLY_STOPPING = early_stopping 

    # --- HELPER FUNCTION FOR CONFIDENCE INTERVALS ---
    def compute_95_ci(data):
        """Calculates Mean and 95% Confidence Interval margin."""
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
        return mean, h

    def build_cnn(hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=(max_len,)))
        model.add(Embedding(TOKENS, DIMENSIONS, mask_zero=False))

        hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(Conv1D(filters=hp_units, kernel_size=SIZE, activation="relu", padding='same'))
        model.add(MaxPooling1D(pool_size=SIZE))
        
        model.add(Flatten())
        model.add(Dense(hp_units, activation="relu"))
        
        hp_dropout = hp.Choice('dropout', values=[0.2, 0.5])
        model.add(Dropout(hp_dropout))

        model.add(Dense(CLASSES, activation="softmax"))

        hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # --- STORAGE FOR RESULTS ---
    outer_acc = []
    outer_f1 = []
    outer_loss = []
    history_list = []

    fold_no = 1
    skf = StratifiedKFold(n_splits=NUM_OUTER_FOLDS, shuffle=True, random_state=rng)

    print(f"Starting {NUM_OUTER_FOLDS}-Fold Nested Cross-Validation...")
    print(f"Early Stopping Enabled: {USE_EARLY_STOPPING}")

    # --- OUTER LOOP ---
    for train_index, test_index in skf.split(x_numpy, y_numpy):
        print(f"\n--- Processing Outer Fold {fold_no}/{NUM_OUTER_FOLDS} ---")
        
        x_outer_train, x_outer_test = x_numpy[train_index], x_numpy[test_index]
        y_outer_train, y_outer_test = y_numpy[train_index], y_numpy[test_index]
        
        # Inner Split (for Tuner)
        x_inner_train, x_inner_val, y_inner_train, y_inner_val = train_test_split(
            x_outer_train, y_outer_train, test_size=0.2, stratify=y_outer_train, random_state=rng
        )

        tuner = kt.RandomSearch(
            build_cnn,
            objective='val_accuracy',
            max_trials=MAX_TRIALS,
            executions_per_trial=1,
            directory='nested_cv_results',
            project_name=f'tuning_fold_{fold_no}', 
            overwrite=True
        )

        # Search (Inner Loop)
        tuner.search(
            x_inner_train, y_inner_train,
            epochs=EPOCHS,
            validation_data=(x_inner_val, y_inner_val),
            # You can keep early stopping in the tuner to speed up search, 
            # regardless of the outer loop setting.
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0 
        )
        
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)

        # --- CONDITIONAL EARLY STOPPING LOGIC ---
        final_callbacks = [] # Default is empty list
        
        if USE_EARLY_STOPPING.lower() == 'yes':
            print(" > Early Stopping Active for this fold.")
            final_callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',  # Recommended: monitor accuracy to keep scores high
                    patience=12,             # Generous patience
                    restore_best_weights=True
                )
            ]
        
        # Refit Best Model
        history = best_model.fit(
            x_outer_train, y_outer_train,
            validation_data=(x_outer_test, y_outer_test), 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=final_callbacks # Pass the conditional list here
        )
        
        history_list.append(history.history)

        # Evaluation
        eval_results = best_model.evaluate(x_outer_test, y_outer_test, verbose=0)
        fold_loss = eval_results[0]
        fold_acc = eval_results[1]
        
        y_pred_probs = best_model.predict(x_outer_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        fold_f1 = f1_score(y_outer_test, y_pred_classes, average='macro')
        
        print(f"  > Fold {fold_no} Results -> Loss: {fold_loss:.4f} | Acc: {fold_acc:.4f} | F1: {fold_f1:.4f}")
        
        outer_loss.append(fold_loss)
        outer_acc.append(fold_acc)
        outer_f1.append(fold_f1)
        
        clear_session()
        fold_no += 1

    # --- FINAL STATISTICS ---
    mean_acc, ci_acc = compute_95_ci(outer_acc)
    mean_f1, ci_f1 = compute_95_ci(outer_f1)
    mean_loss, ci_loss = compute_95_ci(outer_loss)

    return mean_acc, ci_acc, mean_f1, ci_f1, mean_loss, ci_loss, history_list

def plot_early_stopping_history(history_list, save_path=None, max_epochs=50):
    acc_matrix = np.full((len(history_list), max_epochs), np.nan)
    val_acc_matrix = np.full((len(history_list), max_epochs), np.nan)
    loss_matrix = np.full((len(history_list), max_epochs), np.nan)
    val_loss_matrix = np.full((len(history_list), max_epochs), np.nan)

    for i, h in enumerate(history_list):
        num_epochs = len(h['accuracy'])
        
        # 1. Fill the actual data
        acc_matrix[i, :num_epochs] = h['accuracy']
        val_acc_matrix[i, :num_epochs] = h['val_accuracy']
        loss_matrix[i, :num_epochs] = h['loss']
        val_loss_matrix[i, :num_epochs] = h['val_loss']
        
        # 2. Pad the rest with the FINAL value (Horizontal Line effect)
        # This represents "if we kept the model frozen, how would it look?"
        acc_matrix[i, num_epochs:] = h['accuracy'][-1]
        val_acc_matrix[i, num_epochs:] = h['val_accuracy'][-1]
        loss_matrix[i, num_epochs:] = h['loss'][-1]
        val_loss_matrix[i, num_epochs:] = h['val_loss'][-1]

    # Calculate Mean across folds
    mean_acc = np.nanmean(acc_matrix, axis=0)
    mean_val_acc = np.nanmean(val_acc_matrix, axis=0)
    mean_loss = np.nanmean(loss_matrix, axis=0)
    mean_val_loss = np.nanmean(val_loss_matrix, axis=0)

    epochs_range = range(1, max_epochs + 1)

    # --- PLOTTING ---
    plt.figure(figsize=(14, 5))

    # ACCURACY
    plt.subplot(1, 2, 1)
    # Plot individual folds faintly to show variance
    for i in range(len(history_list)):
        plt.plot(acc_matrix[i], color='blue', alpha=0.1)
        plt.plot(val_acc_matrix[i], color='red', alpha=0.1)
        
    plt.plot(epochs_range, mean_acc, 'b-', linewidth=2, label='Training Acc (Avg)')
    plt.plot(epochs_range, mean_val_acc, 'r--', linewidth=2, label='Validation Acc (Avg)')
    plt.title('Average Accuracy (Early Stopping Padded)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # LOSS
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mean_loss, 'b-', linewidth=2, label='Training Loss (Avg)')
    plt.plot(epochs_range, mean_val_loss, 'r--', linewidth=2, label='Validation Loss (Avg)')
    plt.title('Average Loss (Early Stopping Padded)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")
    plt.show()
    plt.close()

def plot_cv_history(history_list, save_path=None): 
    train_acc = np.mean([h['accuracy'] for h in history_list], axis=0)
    val_acc   = np.mean([h['val_accuracy'] for h in history_list], axis=0)
    train_loss = np.mean([h['loss'] for h in history_list], axis=0)
    val_loss  = np.mean([h['val_loss'] for h in history_list], axis=0)
    
    epochs = range(1, len(train_acc) + 1)

    # 2. Create Plot
    plt.figure(figsize=(14, 5))

    # --- Accuracy Subplot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Acc (Avg)')
    plt.plot(epochs, val_acc, 'r--', label='Validation/Test Acc (Avg)')
    plt.title('Average Accuracy over 5 Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- Loss Subplot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss (Avg)')
    plt.plot(epochs, val_loss, 'r--', label='Validation/Test Loss (Avg)')
    plt.title('Average Loss over 5 Folds')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")
    plt.show()
    plt.close()


if __name__ == "__main__": 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    processed_dir = os.path.join(script_dir, '..', 'processed') 
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    filepath = os.path.join(data_dir, "cath.tar.xz")
    output_dir = os.path.join(processed_dir, "cath")
    os.makedirs(output_dir, exist_ok=True)
    proteins_dir = os.path.join(output_dir, 'proteins')
    
    FN_DOMAIN_LIST = os.path.join(proteins_dir, 'domain_classification.txt')
    FN_SF_NAMES = os.path.join(proteins_dir, 'superfamily_names.txt')
    FN_SEQ_S60 = os.path.join(proteins_dir, 'seqs_S60.fa')
    
    # For the commented out S40 file

    df_merged = load_merge_data(filepath, FN_DOMAIN_LIST, FN_SF_NAMES, FN_SEQ_S60, output_dir)
    final_data, max_len = filter_df(df_merged)
    x , y = vectorize_sequences_encode_labels(final_data, max_len)

    x_numpy = x.numpy() if hasattr(x, 'numpy') else np.array(x)
    y_numpy = y.numpy() if hasattr(y, 'numpy') else np.array(y)
    y_numpy = y_numpy.astype(int)
    rng = np.random.RandomState(42)

    TOKENS = 24
    DIMENSIONS = 16
    CLASSES = 5
    SIZE = 4
    UNITS = 128
    #Metrics from best hyper parameters from CNN 
    SIZE = 4
    DROPOUT_RATE = 0.5
    LEEARNING_RATE = 0.001

    # Run only if needed, takes a lot of time (CNN - 15mins, LSTM-63mins, GRU-90mins)
    # best_cnn_hps = cnn_tuning(TOKENS, DIMENSIONS, CLASSES, SIZE, x_numpy, y_numpy)
    # print(f"""
    # CNN Search Complete.
    # Best units: {best_cnn_hps.get('units')}
    # Best learning rate: {best_cnn_hps.get('learning_rate')}
    # Best dropout: {best_cnn_hps.get('dropout')}
    # """)

    # best_LSTM_hps = lstm_tuning(TOKENS, DIMENSIONS, CLASSES, SIZE, x_numpy, y_numpy)
    # print(f"""
    # LSTM Search Complete.
    # Best units: {best_LSTM_hps.get('units')}
    # Best learning rate: {best_LSTM_hps.get('learning_rate')}
    # Best dropout: {best_LSTM_hps.get('dropout')}
    # """)

    # best_GRU_hps = gru_tuning(TOKENS, DIMENSIONS, x_numpy, y_numpy)
    # print(f"""
    # GRU + RNN Search Complete.
    # Best units: {best_GRU_hps.get('units')}
    # Best learning rate: {best_GRU_hps.get('learning_rate')}
    # Best dropout: {best_GRU_hps.get('dropout')}
    # """)

    mean_acc, ci_acc, mean_f1, ci_f1, mean_loss, ci_loss, history_list = best_cnn_nestedcv_earlystopping(x_numpy, y_numpy, early_stopping = "yes")
    
    print("\n============================================================")
    print("FINAL NESTED CV RESULTS")
    print("============================================================")
    print(f"Accuracy:  {mean_acc:.4f}  (+/- {ci_acc:.4f})")
    print(f"Macro F1:  {mean_f1:.4f}  (+/- {ci_f1:.4f})")
    print(f"Loss:      {mean_loss:.4f}  (+/- {ci_loss:.4f})")
    print("============================================================")
    print(f"Report format: {mean_f1*100:.2f}% ± {ci_f1*100:.2f}% (95% CI)")
    print("============================================================")

    s60_plot_path = os.path.join(results_dir, "S60_CNN_EarlyStopping_History.png")
    plot_early_stopping_history(history_list, save_path=s60_plot_path)

    # #Uncomment to check for S40 sequence data @
    # FN_SEQ_S40 = '../data/extracted_data/cath/proteins/seqs_nonredundant_S40.fa'

    # df_merged = load_merge_data(filepath, FN_DOMAIN_LIST, FN_SF_NAMES, FN_SEQ_S40, output_dir)
    # final_data, max_len = filter_df(df_merged)
    # x , y = vectorize_sequences_encode_labels(final_data, max_len)

    # x_numpy = x.numpy() if hasattr(x, 'numpy') else np.array(x)
    # y_numpy = y.numpy() if hasattr(y, 'numpy') else np.array(y)
    # y_numpy = y_numpy.astype(int)
    # rng = np.random.RandomState(42)

    # TOKENS = 24
    # DIMENSIONS = 16
    # CLASSES = 5
    # SIZE = 4
    # UNITS = 128
    # #Metrics from best hyper parameters from CNN 
    # SIZE = 4
    # DROPOUT_RATE = 0.5
    # LEEARNING_RATE = 0.001

    # mean_acc, ci_acc, mean_f1, ci_f1, mean_loss, ci_loss, history_list = best_cnn_nestedcv_earlystopping(x_numpy, y_numpy, early_stopping = "yes")
    
    # print("\n============================================================")
    # print("FINAL NESTED CV RESULTS")
    # print("============================================================")
    # print(f"Accuracy:  {mean_acc:.4f}  (+/- {ci_acc:.4f})")
    # print(f"Macro F1:  {mean_f1:.4f}  (+/- {ci_f1:.4f})")
    # print(f"Loss:      {mean_loss:.4f}  (+/- {ci_loss:.4f})")
    # print("============================================================")
    # print(f"Report format: {mean_f1*100:.2f}% ± {ci_f1*100:.2f}% (95% CI)")
    # print("============================================================")
    # s40_plot_path = os.path.join(results_dir, "S40_CNN_EarlyStopping_History.png")
    # plot_early_stopping_history(history_list, save_path=s40_plot_path)
    # plot_early_stopping_history(history_list)