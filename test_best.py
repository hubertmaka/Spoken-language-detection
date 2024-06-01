from src.preprocess.preprocess import Preprocess
import tensorflow as tf
from src.preprocess.pipeline import Pipeline
import random

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

test = Preprocess.load_set('test')
random.shuffle(test)
test_dataset = Pipeline.create_pipeline(test)


best_model = tf.keras.models.load_model('model1.keras')
test_loss, test_acc = best_model.evaluate(test_dataset, verbose=2)
print("-"*50)
print(f'\nTest accuracy: {test_acc}')
print("-"*50)

