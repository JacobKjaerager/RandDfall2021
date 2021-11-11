import numpy as np
from keras.models import Sequential
<<<<<<< HEAD
from keras.layers import Dense
from keras.layers import LSTM

=======
from keras.layers import Dense, LSTM, Conv2D, Flatten
import marshal as json
>>>>>>> 69c092ea9fce827ee6ded204d7525d7e60db6005

def get_hyper_opt_conf(train_shape: tuple) -> list:
	return [
	{
<<<<<<< HEAD
		"enabled_for_run": True,
=======
		"enabled_for_run": False,
>>>>>>> 69c092ea9fce827ee6ded204d7525d7e60db6005
		"EPOCHS": 50,
		"model": Sequential(),
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
		"layers": [{
			"layer_type": LSTM,
			"layer_arguments": {
				"units": 50,
				"activation": "relu",
				"input_shape": train_shape,
			}
		},
		{
			"layer_type": Dense,
			"layer_arguments": {
<<<<<<< HEAD
				"units": 1,
=======
				"units": 3,
                "activation": "softmax"
>>>>>>> 69c092ea9fce827ee6ded204d7525d7e60db6005
			}
		}]
	},
	{
		"enabled_for_run": False,
		"EPOCHS": 50,
		"model": Sequential(),
		"optimizer": "adam",
		"loss_function": "mse",
		"layers": [
			{
				"layer_type": LSTM,
				"layer_arguments": {
					"units": 50,
					"activation": "tanh",
					"return_sequences": True,
					"input_shape": train_shape
				}
			},
			{
				"layer_type": LSTM,
				"layer_arguments": {
					"units": 50,
					"activation": "tanh"
				},
			},
			{
				"layer_type": Dense,
				"layer_arguments": {
					"units": 1
				}
			}
		]
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 1,
		"model": Sequential(),
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
		"layers": [
			{
				"layer_type": Conv2D,
				"layer_arguments": {
					"filters": 80,
                    "kernel_size": 3,
					"activation": "tanh",
					"input_shape": train_shape
				}
			},
			{
				"layer_type": Conv2D,
				"layer_arguments": {
					"filters": 40,
                    "kernel_size": 3,
					"activation": "tanh",
				}
			},
            {
				"layer_type": Flatten,
				"layer_arguments": {

				}
			},
			{
				"layer_type": Dense,
				"layer_arguments": {
					"units": 3,
                    "activation": "softmax"
				}
			}
		]
	}
]




