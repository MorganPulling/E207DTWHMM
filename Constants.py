import numpy as np

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_HOP_SIZE_SAMPLES = 512

# Coordinates are steps through X, steps through Y
TrainingDTWSteps = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [0, 2], [0, 3]])