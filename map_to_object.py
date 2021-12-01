from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten, \
	ConvLSTM2D, BatchNormalization, Conv3D, Reshape
from Models import DeepLOB, TABL


def map_2_obj(model_name: any):
    if type(model_name) == list:
        model_name = model_name[0]
    model = {
        "lstm": LSTM,
        "dense": Dense,
        "sequential": Sequential(),
        "conv2d": Conv2D,
        "flatten": Flatten,
        "convlstm2d": ConvLSTM2D,
        "batchnormalization": BatchNormalization,
        "conv3d": Conv3D,
        "reshape": Reshape,
        "deeplob": DeepLOB(),
        "tabl": TABL()
    }[model_name]

    return model

if __name__ == '__main__':
    model = map_2_obj("dense")
    print("awef")

