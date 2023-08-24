#Exiba a imagem
from matplotlib import pyplot
from matplotlib.image import imread
import tensorflow as tf

training_damaged_image = "train/damage/image (1).jpeg"

# carregue os pixels da imagem
image = imread(training_damaged_image)

pyplot.title("danificado: Imagem 1")

# plote dados brutos de pixel
pyplot.imshow(image)

# exiba a imagem
pyplot.show()


model = tf.keras.models.Sequential([
    
    # Primeira camada de Convolução e Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Segunda camada de Convolução e Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Terceira camada de Convolução e Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # Quarta camada de Convolução e Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),

    # Camada de classificação
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(training_augmented_images, 
            epochs=20, validation_data = validation_augmented_images, verbose=True)


model.Save("Hurricane_damage.H5")
#model1.save("Hurricane_damage.h5")
#model.save("Hurricane_damage.h5")
#model1.Save("Hurricane_damage.H5")
