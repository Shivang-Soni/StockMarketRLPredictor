from tensorflow.keras import models, layers, optimizers

def mlp(input_dim, output_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim)
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(0.001))
    return model
