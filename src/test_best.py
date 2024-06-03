from src.preprocess.preprocess import Preprocess
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from src.preprocess.pipeline import Pipeline
import random

ORIGIN_SAMPLE_RATE = 48_000
FINAL_SAMPLE_RATE = 16_000
MAX_CLIENT_ID_AMOUNT = 2500
MIN_CLIP_DURATION_MS = 6000
SET_SIZE = 27000
BATCH_SIZE = 64

Preprocess.initialize(
    batch_size=BATCH_SIZE,
    max_client_id_amount=MAX_CLIENT_ID_AMOUNT,
    min_clip_duration_ms=MIN_CLIP_DURATION_MS,
    set_size=SET_SIZE,
    origin_sample_rate=ORIGIN_SAMPLE_RATE,
    final_sample_rate=FINAL_SAMPLE_RATE,
)
#
# test = Preprocess.load_set('test')
# random.shuffle(test)
# test_dataset = Pipeline.create_pipeline(test)
#

# best_model = tf.keras.models.load_model('model10.keras')
# test_loss, test_acc = best_model.evaluate(test_dataset, verbose=2)
# print("-"*50)
# print(f'\nTest accuracy: {test_acc}')
# print("-"*50)


test = Preprocess.load_set('test')
random.shuffle(test)
test_dataset = Pipeline.create_pipeline(test)

best_model = tf.keras.models.load_model('model10.keras')
test_loss, test_acc = best_model.evaluate(test_dataset, verbose=2)
print("-"*50)
print(f'\nTest accuracy: {test_acc}')
print("-"*50)


y_true = []
y_pred = []

for x, y in test_dataset:
    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(best_model.predict(x), axis=1))

conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["German", "French", "Polish"])
disp.plot(cmap=plt.cm.Blues)


plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

print("Confusion matrix saved as 'confusion_matrix.png'")
