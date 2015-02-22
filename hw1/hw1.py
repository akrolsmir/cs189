
# coding: utf-8

# In[45]:

# from http://stackoverflow.com/q/4601373/1222351
def shuffle_both(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)


# In[46]:

import scipy.io
mat = scipy.io.loadmat('data/digit-dataset/train.mat')


# In[47]:

import numpy
images = numpy.reshape(mat["train_images"], (1, -1, 60000))[0].T
numpy.shape(images)


# In[48]:

labels = numpy.ravel(mat["train_labels"])
numpy.shape(labels)


# In[49]:

shuffle_both(images, labels)
labels


# In[50]:

from sklearn import svm
s = svm.SVC()
s.fit(images[0:100], numpy.ravel(labels)[0:100])


# In[51]:

s.predict(images[102])


# In[52]:

labels[102]


# In[ ]:



