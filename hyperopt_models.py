import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def get_hyper_opt_conf(train_shape: tuple) -> list:
	return [
	{
		"enabled_for_run": True,
		"EPOCHS": 50,
		"model": Sequential(),
		"optimizer": "adam",
		"loss_function": "mse",
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
				"units": 1,
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
	}
]




