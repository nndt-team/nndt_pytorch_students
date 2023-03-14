import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import codecs, json 

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

class MnistModel:
    def train(self):
        (ds_train, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        ds_train = ds_train.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_test = ds_test.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        self.model.fit(
            ds_train,
            epochs=6,
            validation_data=ds_test,
        )

    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
    
    def predict(self, path_to_img):
        img = tf.keras.utils.load_img(
            path_to_img, target_size=(28, 28), color_mode="grayscale")
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        self.pred = self.model.predict(img_array)
        self.pred += abs(np.min(self.pred))
        self.pred /= np.max(self.pred)
    
    def save_pred_json(self, path):
        tmp = {'pred':self.pred.tolist()[0]}
        json.dump(tmp, codecs.open('pred.json', 'w', encoding='utf-8'))


model = MnistModel()
model.train()
model.save('model.h5')
model.load('model.h5')
model.predict('test.png')
model.save_pred_json('pred.json')