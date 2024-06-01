from src.preprocess.preprocess import Preprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from src.preprocess.pipeline import Pipeline
import random
import time

ORIGIN_SAMPLE_RATE = 48_000
FINAL_SAMPLE_RATE = 16_000
MAX_CLIENT_ID_AMOUNT = 2500
MIN_CLIP_DURATION_MS = 6000
SET_SIZE = 24_000
BATCH_SIZE = 64

Preprocess.initialize(
    batch_size=BATCH_SIZE,
    max_client_id_amount=MAX_CLIENT_ID_AMOUNT,
    min_clip_duration_ms=MIN_CLIP_DURATION_MS,
    set_size=SET_SIZE,
    origin_sample_rate=ORIGIN_SAMPLE_RATE,
    final_sample_rate=FINAL_SAMPLE_RATE,
)

print(f"Original sample rate: {Preprocess.ORIGIN_SAMPLE_RATE}")
print(f"Final sample rate: {Preprocess.SAMPLE_RATE}")
print(f"Max client id amount: {Preprocess.MAX_CLIENT_ID_AMOUNT}")
print(f"Min clip duration in milliseconds: {Preprocess.MIN_CLIP_DURATION_MS}")
print(f"Set size: {Preprocess.SET_SIZE}")
print(f"One language set size: {Preprocess.ONE_LANG_SET_SIZE}")

print("-"*25 + "SETS SIZES" + "-"*25)
print(f"Training set size: {Preprocess.SET_HPARAMS.train_size}")
print(f"Val set size: {Preprocess.SET_HPARAMS.val_size}")
print(f"Test set size: {Preprocess.SET_HPARAMS.test_size}")
print(f"Is test + val + train is equal one lang set size: "
      f"\n{Preprocess.SET_HPARAMS.test_size + Preprocess.SET_HPARAMS.val_size + Preprocess.SET_HPARAMS.train_size} <= {Preprocess.ONE_LANG_SET_SIZE}")

start = time.time()
if_save = True
Preprocess.preprocess_probes(save=if_save)
end = time.time()
print(f'Time taken: {end - start}')


if if_save:
    train = Preprocess.load_set('train')
    val = Preprocess.load_set('val')
else:
    train = Preprocess.TRAIN_FILENAMES
    val = Preprocess.VAL_FILENAMES

random.shuffle(train)
random.shuffle(val)


train_dataset = Pipeline.create_pipeline(train, augment=True)
val_dataset = Pipeline.create_pipeline(val)


input_shape = val_dataset.as_numpy_iterator().next()[0].shape[1:]

print(train_dataset.element_spec)

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Normalization(axis=-1),
    layers.Conv2D(16, (7, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3, 2)),
    layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')
])
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    'model1.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=15
)

history = model.fit(train_dataset, epochs=30, validation_data=val_dataset, batch_size=BATCH_SIZE, callbacks=[early_stopping, checkpoint_callback])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

