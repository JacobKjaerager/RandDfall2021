from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten, \
	ConvLSTM2D, BatchNormalization, Conv3D, Reshape

import numpy as np


def get_hyper_opt_conf() -> list:
	return [
	{
		"enabled_for_run": False,
		"EPOCHS": 1,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": "sequential",
		"optimizer": "adam",
		"loss_function": "mse",
		"layers": [
			{
				"layer_type": "lstm",
				"layer_arguments": {
					"units": 50,
					"activation": "tanh",
					"return_sequences": True,
					"input_shape": (100, 40)
				}
			},
			{
				"layer_type": "lstm",
				"layer_arguments": {
					"units": 50,
					"activation": "tanh"
				},
			},
			{
				"layer_type": "dense",
				"layer_arguments": {
						"units": 3,
						"activation": "softmax"
				}
			}
		]
	},
	{
		"enabled_for_run": True,
		"EPOCHS": 1,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": "sequential",
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
		"layers": [
			{
				"layer_type": "lstm",
				"layer_arguments": {
					"units": 50,
					"activation": "tanh",
					"input_shape": (100, 40)
				}
			},
			{
				"layer_type": "dense",
				"layer_arguments": {
						"units": 3,
						"activation": "softmax"
				}
			}
		]
	},
	{
		"enabled_for_run": False,
		"EPOCHS": 1,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 4,
		"model": "sequential",
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
		"layers": [
			{
				"layer_type": "convlstm2d",
				"layer_arguments": {
					"filters": 64,
					"kernel_size": [5, 5],
					"padding": "same",
					"return_sequences": True,
					"activation": "tanh",
					"input_shape": (100, 40, 1)
				}
			},
			{
				"layer_type": "batchnormalization"
			},
			{
				"layer_type": "convlstm2d",
				"layer_arguments": {
					"filters": 64,
					"kernel_size": [3, 3],
					"padding": "same",
					"return_sequences": True,
					"activation": "tanh"
				}
			},
			{
				"layer_type": "batchnormalization"
			},
			{
				"layer_type": "convlstm2d",
				"layer_arguments": {
					"filters": 64,
					"kernel_size": [1, 1],
					"padding": "same",
					"return_sequences": True,
					"activation": "tanh"
				}
			},
			{
				"layer_type": "conv3d",
				"layer_arguments": {
					"filters": 1,
					"kernel_size": [3, 3, 3],
					"padding": "same",
					"activation": "sigmoid"
				}
			}
		]
	},
	{
		"enabled_for_run": False,
		"EPOCHS": 1,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 4,
		"model": "sequential",
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
		"layers": [
			{
				"layer_type": "conv2d",
				"layer_arguments": {
					"filters": 80,
					"kernel_size": 3,
					"activation": "tanh",
					"input_shape": (100, 40, 1)
				}
			},
			{
				"layer_type": "conv2d",
				"layer_arguments": {
					"filters": 40,
					"kernel_size": 3,
					"activation": "tanh"
				}
			},
			{
				"layer_type": "flatten",
				"layer_arguments": {

				}
			},
			{
				"layer_type": "dense",
				"layer_arguments": {
					"units": 3,
					"activation": "softmax"
				}
			}
		]
	}
]




