import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Lambda
from keras import backend as K
#from keras.layers import Activation
#from keras.layers import Softmax
#from keras.layers import Multiply 
#from keras.layers import Subtract
#from keras.layers import Concatenate
#from keras.layers import Dense
#from keras.layers import Dot
#from keras.layers import Reshape
#from keras.layers import Dropout
#from keras.activations import tanh, relu, sigmoid
#from keras.initializers import RandomUniform
#from tensorflow import math 





def get_model(num_item, latent_dim):

	index_1 = Input(shape=(1,))
	index_2 = Input(shape=(1,))

	embedding_layer = Embedding(num_item, latent_dim, name="item_embedding")
	embedding_1 = embedding_layer(index_1)
	embedding_2 = embedding_layer(index_2)

	def distance(l):
		diff = l[0] - l[1]
		square = K.square(diff)		
		return K.sum(square, axis=-1)

	# When optimizing, the model will try to minimize the distance between the embeddings of 2 indexes.
	# However, it suffers from the risk that all the embeddings will end up being the same
	output = Lambda(distance)([embedding_1, embedding_2])

	return Model([index_1,index_2],output)



