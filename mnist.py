# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:03:04 2021

@author: romal
"""
# =============================================================================
# Каждая цифра представлена в виде черно-белой картинки (оттенок представлен числами от 0 до 255).
# Число входов сети будет равно общему количеству пикселей в картинке (например, 50*50=250 входных узлов).
# Входные узлы это один большой столбец, составленный из столбцов пикселей картинки, т.е. первый столбец, ниже второй и т.д.
# Сеть будет использовать softmax как функцию активации, следовательно, на выходе будет вектор вероятностей.
# Для данной сети достаточно создать два скрытых слоя.
# =============================================================================

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Преобразовываем значения цветов из шкалы 0-255 в 0-1
def scale(image, lable):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, lable

def main():
    # Загружаем набор данных (70 000)
    mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    
    # Делим все данные на тренировочные (60 000) и тестовые (10 000)
    mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']
    
    # Здаем данные для валидации (10%)
    num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
    # Преобразуем в тензор
    num_validation_samples = tf.cast(num_validation_samples, tf.int64)
    
    # Кол-во тестовых переменных
    num_test_samples = mnist_info.splits['test'].num_examples
    num_test_samples = tf.cast(num_test_samples, tf.int64)
    
    # Сохраняем в новой переменной обработанные данные
    scaled_train_and_validation_data = mnist_train.map(scale)
    test_data = mnist_test.map(scale)
    
    
    BUFFER_SIZE = 10000
    
    # Перемешиваем данные по частям
    shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
    
    # Создаем validation dataset
    validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
    train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
    
# =============================================================================
#     Для оптимизации будем пользоваться градиентным спуском
#     Веса и смещение будут обновляться каждые BATCH_SIZE раз
# =============================================================================
    BATCH_SIZE = 100
    
    # Разбиваем все данные на части по BATCH_SIZE в каждом
    train_data = train_data.batch(BATCH_SIZE)
    
    validation_data = validation_data.batch(num_validation_samples)
    test_data = test_data.batch(num_test_samples)
    
    # Задаем validation_inputs и validation_targets
    validation_inputs, validation_targets = next(iter(validation_data))
    
    # Создаем нейросеть
    input_size = 784
    output_size = 10
    hidden_layer_size = 50
    
    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(output_size, activation='softmax')                               
                                ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    
    
    NUM_EPOCHS = 5
    
    model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose=2)
    print('done')
    
    # Тестирование
    test_loss, test_accuracy = model.evaluate(test_data)
    print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
       
if __name__ == "__main__":
    main()
