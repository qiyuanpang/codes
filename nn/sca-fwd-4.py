# Butterfly middle

# Working 2D example
import tensorflow as tf
import scipy.integrate as integrate
import numpy as np
import keras.layers as layers
# np.random.seed(24)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Permute, \
    BatchNormalization, Add, Multiply, LocallyConnected1D, Conv1D, Reshape, AveragePooling1D, \
    LocallyConnected2D, Conv2D, ZeroPadding2D
from keras.utils import np_utils
from keras.datasets import mnist
import keras.optimizers as optimizers
import keras.initializers as ini
from keras import regularizers
# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer, InputSpec
#import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.layers.advanced_activations import LeakyReLU, PReLU
import math
import keras.backend as K
import scipy.io as sio
import scipy
import h5py
import os

import numpy.linalg as la

# print(os.getcwd())
if 1:
    data = h5py.File('data/scafull2.h5', 'r')
else:
    data = h5py.File('scafull2.h5', 'r')
# print(data.keys())

xall   = np.array(data['Input'])
yall   = np.array(data['Output'])
yallim = np.array(data['Output2'])
A      = np.array(data['Adjoint'])
A2     = np.array(data['Adjoint2'])


xall   = xall.reshape((xall.shape[0],xall.shape[1],xall.shape[2],1))
#yallim = yallim.reshape((yallim.shape[0],yallim.shape[1],yallim.shape[2],1))
yall   = yall.reshape((yall.shape[0],yall.shape[1],yall.shape[2],1))
yallim = yallim.reshape((yallim.shape[0],yallim.shape[1],yallim.shape[2],1))



xall = xall + 100
yall = 100*np.concatenate((yall,yallim),axis=3)
#yall = np.concatenate((yall,yallim),axis=3)

xall = xall[:,0:80,0:80,:]
yall = yall[:,0:80,0:80,:]


A  = A.reshape((A.shape[0],A.shape[1],A.shape[2],1))
A2 = A2.reshape((A2.shape[0],A2.shape[1],A2.shape[2],1))
A  = np.concatenate((A,A2),axis=3)

# Parameters
L2 = yall.shape[1]
L1x = xall.shape[1]
L1y = xall.shape[2]
Nw2 = 20
Nb2 = L2 // Nw2
Nw1x = 10
Nw1y = 10
Nb1x = L1x // Nw1x
Nb1y = L1y // Nw1y

w = Nw1x
r  = 4  # rank
rc = 1

Nsample = xall.shape[0]
Ntrain = int(xall.shape[0] // 2)
Ntest = Nsample - Ntrain
xtrain = xall[0:Ntrain, :, :, :]
ytrain = yall[0:Ntrain, :, :, :]
Atrain = A[0:Ntrain, :, :]
xtest = xall[Ntrain:-1, :, :, :]
ytest = yall[Ntrain:-1, :, :, :]
Atest = A[Ntrain:-1, :, :]

class DMLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], input_shape[2], self.output_dim))
        # print(shape)
        self.kernel = self.add_weight(name='kernel', shape=shape, initializer='uniform', trainable=True)
        self.bias   = self.add_weight(name='bias', shape=(1,input_shape[1],self.output_dim), initializer='uniform', trainable=True)
        super(DMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print(x.shape)
        print(self.kernel.shape)
        x1 = K.permute_dimensions(x, (1, 0, 2));
        b1 = K.batch_dot(x1, self.kernel, axes=(2, 1));
        b = K.permute_dimensions(b1, (1, 0, 2));
        b = b + self.bias
        return b;

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)



I = Input(shape=(L1x, L1y, xall.shape[3]))

D0 = Reshape((L1x*L1y,1))(I)
D0 = DMLayer(2)(D0)
D0 = Reshape((L1x,L1y,2))(D0)
D1 = Conv2D(6*r, kernel_size=(w,w), padding='same', strides=(1,1), activation='relu')(D0)
D2 = Conv2D(6*r, kernel_size=(w,w), padding='same', strides=(1,1), activation='relu')(D1)
D3 = Conv2D(6*r, kernel_size=(w,w), padding='same', strides=(1,1), activation='relu')(D2)
D4 = Conv2D(2, kernel_size=(w,w), padding='same', strides=(1,1), activation='linear')(D3)

V1 = Reshape((Nb1x, Nw1x, Nb1y, Nw1y, 2))(D4)
V2 = Permute((1, 3, 2, 4, 5))(V1)
V3 = Reshape((Nb1x * Nb1y, Nw1x * Nw1y*2))(V2)
V4 = DMLayer(Nb2**2 * r)(V3)
V5 = Reshape((Nb1x*Nb1y, Nb2**2, r))(V4)
V6 = Permute((2, 1, 3))(V5)

S0 = Reshape((Nb2 **2 * Nb1x * Nb1y, r))(V6)
S1 = DMLayer(r)(S0)
S2 = Reshape((Nb2**2, Nb1x * Nb1y * r))(S1)


U1 = DMLayer(2*Nw2**2)(S2)
U2 = Reshape((Nb2, Nb2, Nw2, Nw2, 2))(U1)
U3 = Permute((1, 3, 2, 4, 5))(U2)
U4 = Reshape((Nb2 * Nw2, Nb2 * Nw2, yall.shape[3]))(U3)




butterfly = Model(inputs=I, outputs=U4)
butterfly.summary()

weights = butterfly.layers[2].get_weights()
weights[0] = np.ones_like(weights[0])
weights[1] = np.zeros_like(weights[1])
butterfly.layers[2].set_weights(weights)


def test_data(X, Y, string):
    Yhat = butterfly.predict(X)
    dY = Yhat - Y
    errs = np.zeros((X.shape[0]));
    err_img = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3]))
    for i in range(0, X.shape[0]):
        errs[i] = np.linalg.norm(dY[i, :, :,:]) / np.linalg.norm(Y[i, :, :,:])
        err_img[i,:,:,:] = np.squeeze(np.sqrt(np.square(dY[i, :, :, :])))

    print("max/ave error of %s data:\t %.1e %.1e\t" % (string, np.amax(errs), np.mean(errs)));
    idx = np.where(errs == max(errs))
    return errs, idx , Yhat, err_img





def smoothing(L,r):
    x = np.linspace(0,1,L)
    x, y = np.meshgrid(x,x)
    tmp  = np.zeros((L,L,2))
    tmp[:,:,0] = x
    tmp[:,:,1] = y
    pset = tmp.reshape((L**2,2))
    ker = np.zeros((L**2,L**2))
    for i in range(L**2):
        tmp = np.zeros((L**2))
        idx = np.where(np.sum(np.square(pset[i,:]-pset),axis=1)<r**2)
        tmp[idx] = 1
        ker[:,i] = tmp

    k = ker.sum(axis=0)
    ker = ker.dot(np.diag(1/k))
    return ker




# Start training
Nadam = tf.keras.optimizers.Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, decay=0.0005)
butterfly.compile(loss='mean_squared_error', optimizer=Nadam)


#butterfly.load_weights('sca_fwd_4.h5')
butterfly.fit(xtrain, ytrain, batch_size=128, epochs=100, verbose=1)
#butterfly.save_weights('sca_fwd_4.h5')


errs, idx , Yhat, err_img = test_data(xtest[0:500,:,:,:], ytest[0:500,:,:,:], 'test')
#test_data(xtrain, ytrain, 'train')




