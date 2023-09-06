

## importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

##loading the MNIST dataset
mnist=tf.keras.datasets.mnist

#splitting the data into training and testing datasets
(x_train_full,y_train_full),(x_test,y_test)=mnist.load_data()

## let us check the shapes of the training and testing datasets
x_train_full.shape

x_test.shape

## As we know that the digits are stored in form of the pixels.
## each digit has 28 pixels
## All pixels ranges from 0 to 255 grey levels

# Creating a validation data set from the training data set
x_valid,x_train=x_train_full[:5000],x_train_full[5000:]

# Now we will scale the data between 0 to 1 by dividing it by 255
x_valid=x_valid/255

x_train=x_train/255

x_test=x_test/255

y_valid,y_train=y_train_full[:5000],y_train_full[5000:]

# Now let us visualize how the MNIST data looks like
plt.imshow(x_train[0],cmap="binary")
plt.show()

# To visualize it in at grey levels
plt.figure(figsize=(15,15))
sns.heatmap(x_train[0],annot=True,cmap="binary")

# Now we will create a Artificial neural network with some hidden layers to build a model that predicts the written digit
Layers=[tf.keras.layers.Flatten(input_shape=[28,28],name="inputlayer"),
        tf.keras.layers.Dense(300,activation="relu",name="hiddenlayer1"),
        tf.keras.layers.Dense(100,activation="relu",name="hiddenlayer2"),
        tf.keras.layers.Dense(10,activation="softmax",name="outputlayer")]

# Now we will buila a Sequential model
model_clf=tf.keras.models.Sequential(Layers)

model_clf.layers

# Let us see the summary of the model
model_clf.summary()

weights,biases=model_clf.layers[1].get_weights()

# Defining the parameters to train the model
LOSS_FUNCTION="sparse_categorical_crossentropy"
OPTIMIZER="SGD"
METRICS=["accuracy"]

model_clf.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=METRICS)

EPOCHS=30
VALIDATION_SET=(x_valid,y_valid)

history=model_clf.fit(x_train,y_train,epochs=EPOCHS,
                      validation_data=VALIDATION_SET,
                      batch_size=32)

history.params

pd.DataFrame(history.history).plot()

model_clf.evaluate(x_test,y_test)

x_new=x_test[:3]
actual=y_test[:3]
y_prob=model_clf.predict(x_new)
y_pred=np.argmax(y_prob,axis=-1)

for i,j,k in zip(x_new,y_pred,actual):
  plt.imshow(i,cmap="binary")
  plt.title(f"predicted {j} and actual is {k}")
  plt.axis("off")
  plt.show()
  print('##########################')
  
  import gradio as gd
  def makePred(img):
    img_3d=img.reshape(-1,28,28)
    im_resize=img_3d/255.0
    predict=model_clf.predict(im_resize)
    pred=np.argmax(predict)
    return str(pred)

demo = gd.Interface(makePred, inputs='sketchpad', outputs='label')
demo.launch(debug='True')

