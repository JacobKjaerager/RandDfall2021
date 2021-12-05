from keras.models import Sequential
import numpy as np


def get_hyper_opt_conf() -> list:
	return [
    	{
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 60,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 16,
        "inception_num": 32,
        "lstm_num": 64,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 16,
        "inception_num": 32,
        "lstm_num": 64,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 140,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 16,
        "inception_num": 32,
        "lstm_num": 64,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 24,
        "inception_num": 48,
        "lstm_num": 92,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 32,
        "inception_num": 64,
        "lstm_num": 128,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 60,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 24,
        "inception_num": 48,
        "lstm_num": 92,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 60,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 32,
        "inception_num": 64,
        "lstm_num": 128,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 140,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 24,
        "inception_num": 48,
        "lstm_num": 92,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": True,
		"EPOCHS": 100,
		"input_shape_sample": 140,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["deeplob"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "conv_filter_num": 32,
        "inception_num": 64,
        "lstm_num": 128,
        "leaky_relu_alpha": 0.01
	},
    {
		"enabled_for_run": False,
		"EPOCHS": 1,
		"input_shape_sample": 100,
		"input_shape_features": 40,
		"data_dimensions": 3,
		"model": ["tabl"],
		"optimizer": "adam",
		"loss_function": "categorical_crossentropy",
        "dropout": 0.1,
        "hidden_layer_1_shape_1": 60,
        "hidden_layer_1_shape_2": 10,
        "hidden_layer_2_shape_1": 120,
        "hidden_layer_2_shape_2": 5
	},
	{
		"enabled_for_run": False,
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
		"enabled_for_run": False,
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




