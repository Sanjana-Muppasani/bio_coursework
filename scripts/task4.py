import pandas as pd 
import numpy as np 
import keras 
import tarfile
import os
import lzma
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from sklearn.model_selection import StratifiedKFold, train_test_split
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
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

def cnn_build_evaluate(x, y):

    rng = np.random.RandomState(42)

    x_numpy = x.numpy() if hasattr(x, 'numpy') else np.array(x)
    y_numpy = y.numpy() if hasattr(y, 'numpy') else np.array(y)
    y_numpy = y_numpy.astype(int)

    x_train_cv, x_test, y_train_cv, y_test = train_test_split(
        x_numpy, y_numpy, test_size=0.2, stratify=y_numpy, random_state=rng
    )

    print(f"Data Split: {len(x_train_cv)} training/validation samples, {len(x_test)} hold-out test samples.")

    TOKENS = 24
    CLASSES = 5
    DIMENSIONS = 16
    UNITS = 32
    SIZE = 4
    DROPOUT_RATE = 0.2

    def build_cnn():
        model = keras.Sequential([
            keras.Input(shape=(max_len,)),
            Embedding(TOKENS, DIMENSIONS, mask_zero=False),

            Conv1D(UNITS, SIZE, activation="relu"),
            MaxPooling1D(SIZE),
            Dropout(DROPOUT_RATE),

            # Conv1D(UNITS, SIZE, activation="relu"),
            # MaxPooling1D(SIZE),
            # Dropout(DROPOUT_RATE),

            Flatten(),
            Dense(UNITS, activation="relu"),
            Dense(CLASSES, activation="softmax")
        ])

        model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
        return model

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)

    fold_no = 1

    f1_per_fold = [] 
    loss_per_fold = []

    all_train_acc = []
    all_val_acc = []
    all_train_loss = []
    all_val_loss = []

    print(f"\nStarting 5-Fold Stratified Cross-Validation on TRAINING set...")

    for train_index, val_index in skf.split(x_train_cv, y_train_cv):
        # Important: Clear Keras session to prevent state leakage (Feedback: "starting from scratch")
        clear_session()
        
        x_train_fold, x_val_fold = x_train_cv[train_index], x_train_cv[val_index]
        y_train_fold, y_val_fold = y_train_cv[train_index], y_train_cv[val_index]
        
        model = build_cnn()
        
        print(f"Training for Fold {fold_no} ...")
        
        history = model.fit(
            x_train_fold, y_train_fold,
            epochs=50,
            batch_size=32,
            validation_data=(x_val_fold, y_val_fold),
            verbose=0 
        )
        
        y_pred_probs = model.predict(x_val_fold, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        

        fold_f1 = f1_score(y_val_fold, y_pred_classes, average='macro')
        

        scores = model.evaluate(x_val_fold, y_val_fold, verbose=0)
        
        print(f"Fold {fold_no}: Loss {scores[0]:.4f}; Macro F1 {fold_f1:.4f}")
        
        f1_per_fold.append(fold_f1)
        loss_per_fold.append(scores[0])
        
        all_train_acc.append(history.history['accuracy'])
        all_val_acc.append(history.history['val_accuracy'])
        all_train_loss.append(history.history['loss'])
        all_val_loss.append(history.history['val_loss'])
        
        fold_no += 1

    mean_f1 = np.mean(f1_per_fold)
    std_error = stats.sem(f1_per_fold)
    ci = std_error * stats.t.ppf((1 + 0.95) / 2., len(f1_per_fold)-1)

    print("\n------------------------------------------------------------------------")
    print("Cross-Validation Results (Macro F1):")
    print(f"> Mean F1: {mean_f1:.4f}")
    print(f"> 95% Confidence Interval: +/- {ci:.4f} ({mean_f1 - ci:.4f} - {mean_f1 + ci:.4f})")
    print("------------------------------------------------------------------------")


    print("\nRetraining on full CV data and evaluating on Hold-out Test Set...")
    clear_session()
    final_model = build_cnn()
    final_model.fit(x_train_cv, y_train_cv, epochs=50, batch_size=32, verbose=0)

    test_pred_probs = final_model.predict(x_test, verbose=0)
    test_pred_classes = np.argmax(test_pred_probs, axis=1)
    test_f1 = f1_score(y_test, test_pred_classes, average='macro')

    print(f"Final Hold-out Test Set Macro F1: {test_f1:.4f}")

    # --- Plotting (Averaged over folds) ---
    avg_train_acc = np.mean(all_train_acc, axis=0)
    avg_val_acc = np.mean(all_val_acc, axis=0)
    avg_train_loss = np.mean(all_train_loss, axis=0)
    avg_val_loss = np.mean(all_val_loss, axis=0)
    epochs_range = range(1, len(avg_train_acc) + 1)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, avg_train_acc, label='Training Acc')
    plt.plot(epochs_range, avg_val_acc, label='Validation Acc')
    plt.title('Average Accuracy ')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, avg_train_loss, label='Training Loss')
    plt.plot(epochs_range, avg_val_loss, label='Validation Loss')
    plt.title('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_path = os.path.join("../results", "Accuracy over Epochs.png")
    plt.savefig(plot_path)
    print(f"PR Curve plot saved to: {plot_path}")

if __name__ == "__main__": 

    filepath = "../data/cath.tar.xz"
    output_dir = "../data/extracted_data/cath"
    os.makedirs(output_dir, exist_ok=True)
    FN_DOMAIN_LIST = '../data/extracted_data/cath/proteins/domain_classification.txt'
    FN_SF_NAMES = '../data/extracted_data/cath/proteins/superfamily_names.txt'
    FN_SEQ_S60 = '../data/extracted_data/cath/proteins/seqs_S60.fa'

    df_merged = load_merge_data(filepath, FN_DOMAIN_LIST, FN_SF_NAMES, FN_SEQ_S60, output_dir)
    final_data, max_len = filter_df(df_merged)
    x , y = vectorize_sequences_encode_labels(final_data, max_len)
    cnn_build_evaluate(x,y)
