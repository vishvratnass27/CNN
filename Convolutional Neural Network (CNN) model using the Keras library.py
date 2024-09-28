#!/usr/bin/env python
# coding: utf-8

# # Let's break down this deep learning code, which involves building a Convolutional Neural Network (CNN) model using the Keras library (a high-level neural networks API).

# Import Libraries:
# 
# Convolution2D: This is used for adding convolution layers in the model.
# 
# Sequential: A linear stack of layers that you can easily add to build your model.

# In[3]:


from keras.layers import Convolution2D


# In[4]:


from keras.models import Sequential


# Model Initialization:
# 
# model = Sequential(): Creates a sequential model where each layer is stacked on top of the previous one.

# In[5]:


model = Sequential()


# Adding a Convolutional Layer:

# In[6]:


model.add(  
    Convolution2D( 
        kernel_size=(7,7) , 
        filters=32, 
        activation='relu' ,
        strides=(1,1),
        input_shape=(100,100,3)
        
    )  
)


# kernel_size=(7,7): This defines the filter size (7x7) that slides over the input image.
# 
# filters=32: The number of filters (or feature detectors) used in the convolution. In this case, 32 filters are applied.
# 
# activation='relu': ReLU (Rectified Linear Unit) activation is applied to introduce non-linearity.
# 
# strides=(1,1): The number of pixels by which we shift the filter over the input image.
# 
# input_shape=(100,100,3): This is the shape of the input image. Here, it's 100x100 pixels with 3 color channels (RGB).

# Model Configuration and Summary:
# 
# model.get_config(): Retrieves the model configuration.
# 
# model.summary(): Prints a summary of the model architecture, including the layer types, output shapes, and number of parameters.

# In[7]:


model.get_config()


# In[8]:


model.summary()


# In[9]:


from keras.layers import MaxPool2D


# Adding a Max Pooling Layer:

# In[10]:


model.add(  MaxPool2D()  )


# This layer performs down-sampling (reducing the size of the input) by taking the maximum value in a window of the feature map, which helps in reducing the spatial dimensions and computational load

# In[11]:


model.get_config()


# In[12]:


from keras.layers import Flatten


# Flatten Layer:

# In[13]:


model.add( Flatten()
         )


# This layer flattens the 2D feature maps into a 1D vector, preparing the data for the fully connected layers (Dense layers).

# In[14]:


from keras.layers import Dense


# Dense Layers (Fully Connected Layers):

# In[15]:


model.add(  Dense(units=20, activation='relu')  )


# In[16]:


model.add(  Dense(units=10, activation='relu')  )


# In[17]:


model.add(  Dense(units=1, activation='sigmoid')  )


# units=20 and units=10: These are fully connected layers with 20 and 10 neurons, respectively, using ReLU activation.
#     
# The final layer has units=1 with sigmoid activation, typically used for binary classification tasks, where the output will be a probability (0 or 1).

# In[18]:


model.summary()


# The model.summary() at the end will give a complete overview of the layers, their parameters, and the overall structure of the neural network.

# This code defines a Convolutional Neural Network (CNN) using Keras. It starts by applying convolutional layers for feature extraction, followed by pooling to reduce dimensionality. The features are then flattened and passed through fully connected (Dense) layers. The final layer uses a sigmoid activation for binary classification.
