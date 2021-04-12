# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:44:00 2020

@author: Gregory
"""
import os, shutil, numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from keras.models import Sequential
from keras.optimizers import RMSprop
#from keras.layers import Embedding, SimpleRNN, Dense
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib.pyplot as plt



fname = os.path.join( "D:\\", "Python", "NN", "JenaTemp", "jena_climate_2009_2016.csv" )

f = open( fname )
data = f.read()
f.close()

lines = data.split('\n')
header = lines[ 0 ].split( ',' )
lines = lines[ 1: ]

print( header )
print( len( lines ) )


### 6.29
float_data = np.zeros( ( len( lines ), len( header ) - 1 ) )
for i, line in enumerate( lines ):
    values = [ float(x) for x in line.split(',')[ 1: ] ]
    float_data[ i, : ] = values 

temp = float_data[ :, 1 ]
plt.plot( range( len( temp ) ), temp )
plt.show()

plt.plot( range( 1440 ), temp[ :1440 ] )
plt.show()


mean = float_data[ :200000 ].mean( axis = 0 )
float_data -= mean
std = float_data[ :200000 ].std( axis = 0 )
float_data /= std

### 6.32.
"""
data     - tablica danych zmiennoprzecinkowych,
lookback - liczba kroków czasu, do których powinny sięgać dane wejsciowe

"""
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None: 
        max_index = len(data) - delay - 1
    i = min_index + lookback 
    while 1: 
        if shuffle: 
            rows = np.random.randint( min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index: 
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index)) 
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),)) 
        
        for j, row in enumerate(rows): 
            indices = range(rows[j] - lookback, rows[j], step) 
            samples[j] = data[indices] 
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
            

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator( float_data, 
                       lookback=lookback, 
                       delay=delay, 
                       min_index=0, 
                       max_index=200000, 
                       shuffle=True, 
                       step=step, 
                       batch_size=batch_size)

val_gen = generator( float_data, 
                     lookback=lookback, 
                     delay=delay, 
                     min_index=200001, 
                     max_index=300000, 
                     step=step, 
                     batch_size=batch_size)

test_gen = generator( float_data, 
                      lookback=lookback, 
                      delay=delay, 
                      min_index=300001, 
                      max_index=None, 
                      step=step, 
                      batch_size=batch_size)

val_steps = ( 300000 - 200001 - lookback ) // batch_size
test_steps = ( len( float_data ) - 300001 - lookback ) // batch_size

### 6.37
# model = Sequential()
# model.add( layers.Flatten( input_shape = ( lookback // step, float_data.shape[ -1 ] ) ) )
# model.add( layers.Dense( 32, activation = 'relu' ) )
# model.add( layers.Dense( 1 ) )

# model.compile( optimizer = RMSprop(), loss = 'mae' )
# history = model.fit_generator( train_gen,
#                                steps_per_epoch = 500,
#                                epochs = 20,
#                                validation_data = val_gen,
#                                validation_steps = val_steps)

# loss = history.history[ 'loss' ]
# val_loss = history.history[ 'val_loss' ]
# epochs = range( len( loss ) )

# plt.figure()

# plt.plot( epochs, loss, 'bo', label = 'Strata trenowania' )
# plt.plot( epochs, val_loss, 'b', label = 'Strata walidacji' )
# plt.title( 'Strata trenowania i walidacji' )
# plt.legend()

# plt.show()


###6.39
# model = Sequential()
# model.add( layers.GRU( 32, input_shape = ( None, float_data.shape[ -1 ] ) ) )
# model.add( layers.Dense( 1 ) )

# model.compile( optimizer = RMSprop(), loss = 'mae' )
# history = model.fit_generator( train_gen, 
#                                steps_per_epoch = 500,
#                                epochs = 20,
#                                validation_data = val_gen,
#                                validation_steps = val_steps)
                                                         

# loss = history.history[ 'loss' ]
# val_loss = history.history[ 'val_loss' ]
# epochs = range( len( loss ) )

# plt.figure()

# plt.plot( epochs, loss, 'bo', label = 'Strata trenowania' )
# plt.plot( epochs, val_loss, 'b', label = 'Strata walidacji' )
# plt.title( 'Strata trenowania i walidacji' )
# plt.legend()

# plt.show()

###6.40
# model = Sequential()
# model.add( layers.GRU( 32,
#                         dropout = 0.2,
#                         recurrent_dropout = 0.2,
#                         input_shape = ( None, float_data.shape[ -1 ] ) ) )

# model.add( layers.Dense( 1 ) )    

# model.compile( optimizer = RMSprop(), loss = 'mae' )
# history = model.fit_generator( train_gen,
#                                 steps_per_epoch = 500,
#                                 epochs = 40,
#                                 validation_data = val_gen,
#                                 validation_steps = val_steps)

# loss = history.history[ 'loss' ]
# val_loss = history.history[ 'val_loss' ]
# epochs = range( len( loss ) )

# plt.figure()

# plt.plot( epochs, loss, 'bo', label = 'Strata trenowania' )
# plt.plot( epochs, val_loss, 'b', label = 'Strata walidacji' )
# plt.title( 'Strata trenowania i walidacji' )
# plt.legend()

plt.show()

###6.41
model = Sequential()
model.add( layers.GRU( 32,
                       dropout = 0.1,
                       recurrent_dropout = 0.5,
                       return_sequences = True,
                       input_shape = ( None, float_data.shape[ -1 ] ) ) ) 

model.add( layers.GRU( 64,
                       activation = 'relu',
                       dropout = 0.1,
                       recurrent_dropout = 0.5 ) )

model.add( layers.Dense( 1 ) )

model.compile( optimizer = RMSprop(), loss = 'mae' )
history = model.fit_generator( train_gen,
                               steps_per_epoch = 500,
                               epochs = 40,
                               validation_data = val_gen,
                               validation_steps = val_steps)


loss = history.history[ 'loss' ]
val_loss = history.history[ 'val_loss' ]
epochs = range( len( loss ) )

plt.figure()

plt.plot( epochs, loss, 'bo', label = 'Strata trenowania' )
plt.plot( epochs, val_loss, 'b', label = 'Strata walidacji' )
plt.title( 'Strata trenowania i walidacji' )
plt.legend()

plt.show()






































