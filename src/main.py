import tensorflow as tf, numpy as np, pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import random, os

train_df    = pd.read_csv('../dataset/train.csv')
val_df      = pd.read_csv('../dataset/val.csv')
test_df     = pd.read_csv('../dataset/test.csv')

class_labels = train_df['Classes'].unique()
class_labels.sort()  # Ensure consistent ordering

class_indices = {label: idx for idx, label in enumerate(class_labels)}
indices_to_classes = {v: k for k, v in class_indices.items()}

train_df['Labels'] = train_df['Classes'].map(class_indices)

datagen_train = ImageDataGenerator(rescale=1./255)
datagen_val = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train.flow_from_dataframe(
    dataframe=train_df,
    directory='../dataset/images/',
    x_col='filename',
    y_col='Classes',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=class_indices 
)

val_generator = datagen_val.flow_from_dataframe(
    dataframe=val_df,
    directory='../dataset/images/',
    x_col='filename',
    y_col='Classes',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=class_indices 
)

test_generator = datagen_val.flow_from_dataframe(
    dataframe=test_df,
    directory='../dataset/images/',
    x_col='filename',
    y_col='Classes',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

model.save('10epochs.h5')

plt.figure(figsize=(12, 5))

# Précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

class_indices = train_generator.class_indices
label_to_name = {v: k for k, v in class_indices.items()}  # Inverser le dictionnaire

# Fonction pour afficher 6 images aléatoires avec leurs prédictions et labels
plt.figure(figsize=(15, 10))

# Sélectionner 6 images aléatoires
random_indices = random.sample(range(len(test_df)), 6)
for i, idx in enumerate(random_indices):
    # Charger l'image et la convertir pour la prédiction
    img_path = os.path.join('../dataset/images/', test_df.iloc[idx]['filename'])
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normaliser l'image
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

    # Prédiction du modèle
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = indices_to_classes[predicted_index]

    # Obtenir les noms des classes depuis le DataFrame
    true_name = test_df.iloc[idx]['Classes']

    # Afficher l'image
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_class}\nTrue: {true_name}")

plt.tight_layout()
plt.show()