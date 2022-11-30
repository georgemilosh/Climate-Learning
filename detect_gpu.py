import tensorflow as tf

 # check tf version and GPUs
print(f"{tf.__version__ = }")
if int(tf.__version__[0]) < 2:
    print(f"{tf.test.is_gpu_available() = }")
    GPU = tf.test.is_gpu_available()
else:
    print(f"{tf.config.list_physical_devices('GPU') = }")
    GPU = len(tf.config.list_physical_devices('GPU'))
if not GPU:
    print('\nThis machine does not have a GPU: training may be very slow\n')