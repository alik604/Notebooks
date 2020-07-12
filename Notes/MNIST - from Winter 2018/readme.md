# My first successful attempt on MNIST

[link](https://colab.research.google.com/drive/1wP3MdJO04ozfOV5HgDT4DrHcjNiLFq0n) 



### Below I highlight my three final configurations

###  

Attempt n-2



```
model = Sequential() 

model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
```

Test loss: 0.0616

Test accuracy: 0.9826

--------------------

Attempt n-1



```
model = Sequential() 

model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
```

Test loss: 0.03674

Test accuracy: 0.9895

----------------

Attempt n



```
model = Sequential() 

model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
```

Test loss: 0.0310
Test accuracy: 0.9924

-----------

[Sample code which I learned from](https://colab.research.google.com/drive/1iTGGKIfxRRrhWpQQUqHdfVMrfqkdE0un#scrollTo=R6U6JC6Z4Scj)



```
model = Sequential() 

model.add(Conv2D(32, kernel_size=(3, 3),          activation='relu',input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation=Activation(tf.nn.softmax)))


```

Test loss: 0.06937230257001109
Test accuracy: 0.9912



----

###  Attempt n's accuracy was 0.9924, I beats the accuracy of the sample code, which was 0.9912







--------

sources: 

* [keras docs](https://keras.io/)

* [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/)
  + ![img](https://adeshpande3.github.io/assets/Table.png)
* [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/#layerpat)







```python
model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation='softmax'))
# 98%
--------------------------------------------------
model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=100, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
# Test loss: 0.030517288568560615
# Test accuracy: 0.9901
```





