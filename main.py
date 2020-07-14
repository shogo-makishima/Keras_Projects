import keras

modelPath = "Models\\MNIST_Model\\model"

batch_size = 128 # количество обучающих образцов, обрабатываемых одновременно за одну итерацию алгоритма градиентного спуска;
num_epochs = 1 # количество итераций обучающего алгоритма по всему обучающему множеству;
hidden_size = 512 # количество нейронов в каждом из двух скрытых слоев MLP.

num_train = 60000
num_test = 10000

height, width, depth = 28, 28, 1
num_classes = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(num_train, height * width)
x_test = x_test.reshape(num_test, height * width)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = (x_train / 255 * 0.99) + 0.01
x_test = (x_test / 255 * 0.99) + 0.01

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputLayer = keras.layers.Input(shape=(height * width))
hiddenLayer_1 = keras.layers.Dense(hidden_size, activation='relu')(inputLayer)
# hiddenLayer_2 = keras.layers.Dense(hidden_size, activation='relu')(hiddenLayer_1)
outLayer = keras.layers.Dense(num_classes, activation='softmax')(hiddenLayer_1)

try:
    model: keras.Model = keras.models.load_model(modelPath)
    model.predict(x_test, verbose=1)
except Exception as exception:
    print(exception)

    model = keras.Model(inputLayer, outLayer)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1)
    model.evaluate(x_test, y_test, verbose=1)

    model.save(modelPath)

