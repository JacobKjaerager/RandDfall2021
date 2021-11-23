import Models
from tensorflow.keras import optimizers


def run_tabl_config():
    template = [[60, 5], [120, 5], [3, 1]]

    # get Bilinear model
    projection_regularizer = None
    projection_constraint = keras.constraints.max_norm(3.0, axis=0)
    attention_regularizer = None
    attention_constraint = keras.constraints.max_norm(5.0, axis=1)
    dropout = 0.1

    model = Models.TABL(template, dropout, projection_regularizer,
                        projection_constraint, attention_regularizer,
                        attention_constraint)

    opt = optimizers.Adam(learning_rate=0.001)  # learning rate and epsilon are the same as paper DeepLOB
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    class_weight = {0: 1, 1: 1, 2: 1}
    return model


