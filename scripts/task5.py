import os
import zipfile
import numpy as np
import keras
import keras_tuner as kt
from keras import layers
import matplotlib.pyplot as plt
from keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import shap 
from keras.applications import MobileNetV2
from keras.optimizers import Adam


def extract_zip(zip_path, extract_to): 
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print(f"Created directory {os.path.abspath(extract_to)} for extraction.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted all files from {os.path.abspath(zip_path)} to {os.path.abspath(extract_to)}.")

def load_data(normalization="no", preprocessing="no"):
    IMG_SIZE =  (150, 150)

    train_ds = keras.utils.image_dataset_from_directory(
        '../data/extracted_data/train_sample2',
        image_size= IMG_SIZE,
        batch_size = batch_size,
        label_mode="categorical"
    )

    test_ds = keras.utils.image_dataset_from_directory(
        '../data/extracted_data/test',
        image_size= IMG_SIZE,
        batch_size = batch_size,
        label_mode="categorical",
        shuffle = False
    )

    val_ds = keras.utils.image_dataset_from_directory(
        '../data/extracted_data/val',
        image_size= IMG_SIZE,
        batch_size = batch_size,
        label_mode="categorical"
    )


    #for custom CNN
    if normalization == "yes":
        normalization_layer = layers.Rescaling(1./255)

        train_ds = train_ds.map(lambda x, y:  (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y:  (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y:  (normalization_layer(x), y))

    #for TL model
    if preprocessing == "yes": 
        def preprocess_data(images, labels):
            return preprocess_input(images), labels

        train_ds = train_ds.map(preprocess_data)
        val_ds = val_ds.map(preprocess_data)
        test_ds = test_ds.map(preprocess_data)

    return train_ds, val_ds, test_ds

def custom_cnn(): 
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #layers.BatchNormalization(),
        # layers.Dropout(0.4),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        #layers.BatchNormalization(),
        # layers.Dropout(0.4),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model 

def tune_cnn(train_ds, val_ds): 
    def build_model(hp):
        hp_filters_1 = hp.Int('filters_1', min_value=32, max_value=128, step=32)
        hp_filters_2 = hp.Int('filters_2', min_value=64, max_value=256, step=64)

  
        hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)


        hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])


        model = keras.Sequential([
            keras.Input(shape=(150, 150, 3)),
            layers.Rescaling(1./255),
            layers.Conv2D(hp_filters_1, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(hp_filters_2, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(hp_dropout),

            layers.Dense(4, activation="softmax"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=15,             # Max epochs per trial
        factor=3,                  # Reduction factor
        directory='my_dir',        # Where to save results
        project_name='oct_tuning'
    )

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    print("Starting Hyperparameter Search...")
    tuner.search(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=[stop_early]
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete.
    The optimal number of filters in layer 1 is {best_hps.get('filters_1')}.
    The optimal learning rate is {best_hps.get('learning_rate')}.
    """)

def train_evaluate_cnn(model, train_ds, test_ds, val_ds): 
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,             # Stop if no improvement for 5 epochs
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="best_custom_cnn.keras",
            save_best_only=True,
            monitor="val_loss"
        )
    ]

    # Train the model
    epochs = 30  # Set high, EarlyStopping will cut it short if needed

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # 2. Make Predictions on the Test Set
    print("Generating predictions...")
    y_pred_probs = model.predict(test_ds)

    # 3. Convert Probabilities to Class Integers (e.g., [0, 0, 1] -> 2)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = []
    for images, labels in test_ds:
        y_true.append(np.argmax(labels.numpy(), axis=1))
    y_true = np.concatenate(y_true)

    # 4. Calculate Macro F1 Score
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # --- Final Output ---
    print("-" * 30)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("-" * 30)

    # Optional: Detailed Report
    print("\nDetailed Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']))

    cm = confusion_matrix(y_true, y_pred)

    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("../results/T5_CNN_CM.png", dpi=300)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("../results/T5_CNN_Learning_Curves.png", dpi=300)

def train_tune_evaluate_tl(train_ds, test_ds, val_ds): 
    def build_transfer_model(input_shape, num_classes):
    # Load MobileNetV2 without the top classification layer
    # alpha=0.5 makes the model smaller/faster (optional, can be removed for higher accuracy)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            alpha=1.0 # Changed to 1.0 for standard performance, use 0.5 for speed
        )

        # Freeze the base model initially
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

# # Initialize
# input_shape = (IMG_SHAPE[0], IMG_SHAPE[1], 3)
# num_classes = 4

    model = build_transfer_model(input_shape, num_classes)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks
    )

    base_model_layer = None
    for layer in model.layers:
        # Check if the layer is a Model/Functional layer (like MobileNetV2)
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            base_model_layer = layer
            break

    if base_model_layer is None:
        print("Error: Could not find the base model! Check model.summary().")
    else:
        print(f"Base model found: {base_model_layer.name}")

        # 2. Unfreeze the base model
        base_model_layer.trainable = True

        # 3. Freeze all but the last 30 layers of the base model
        for layer in base_model_layer.layers[:-30]:
            layer.trainable = False

        # 4. Recompile with LOW Learning Rate (Crucial!)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5), # Very low LR to prevent destroying weights
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Starting Fine-Tuning...")
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=callbacks
        )

    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc:.4f}")

    # Get predictions
    predictions = model.predict(test_ds)
    y_pred = np.argmax(predictions, axis=1)

    # Get true labels
    # Since we didn't shuffle test_ds, we can iterate to get labels
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)

    print("\nDetailed Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']))

    cm = confusion_matrix(y_true, y_pred)
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    # 5. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("../results/T5_TL_CM.png", dpi=300)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig("../results/T5_TL_Learning.png", dpi=300)

    return model 

def shap_explain(model): 
    for images, labels in test_ds.take(1):
        test_batch = images.numpy()

    images_to_explain = test_batch[:5]  # Explain first 5 images

    # 2. Define the Masker
    # This tells SHAP: "When you want to hide a part of the image, use this method."
    # "inpaint_telea" blurs the hidden parts. 
    # We pass the shape of a SINGLE image (e.g., 150, 150, 3).
    masker = shap.maskers.Image("inpaint_telea", images_to_explain[0].shape)

    # 3. Create the Explainer
    # We pass the MODEL and the MASKER.
    # This automatically selects 'PartitionExplainer', which is optimized for images.
    explainer = shap.Explainer(model, masker)

    # 4. Calculate SHAP values
    # This generates the explanation object.
    print("Calculating SHAP values...")
    shap_values = explainer(images_to_explain, max_evals=500, batch_size=50)

    # 5. Visualize
    # Note: The syntax for plotting this specific output is slightly different.
    shap.image_plot(shap_values, show=False)
    plt.savefig('shap_explanation.png', bbox_inches="tight", dpi=300)

if __name__ == "__main__": 

    extracted_data_dir = '../data/extracted_data'
    zip_path = "../data/retinal-oct-sample.zip"

    batch_size = 128
    epochs = 15
    num_classes = 4
    input_shape = (150, 150, 3)

    extract_zip(zip_path, extracted_data_dir)
    train_ds, val_ds, test_ds = load_data(normalization="yes", preprocessing="no") #for custom CNN
    cnn_model = custom_cnn()

    # tune_cnn(train_ds, val_ds) #uncomment and run to find the best combination of hyper params 

    train_evaluate_cnn(cnn_model, train_ds, test_ds, val_ds)


    train_ds_tl, val_ds_tl, test_ds_tl = load_data(normalization="no", preprocessing="yes")
    model_tl = train_tune_evaluate_tl(train_ds_tl,test_ds_tl, val_ds_tl)
    shap_explain(model_tl)