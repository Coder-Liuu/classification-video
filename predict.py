from tensorflow.keras.models import load_model
from create_image import read_video
import numpy as np

model = load_model("cnn-lstm-10.h5")

path = "videos/train/diving/v_diving_02/v_diving_02_02.mpg"
video = read_video(path)
print(video.shape)
video = np.expand_dims(video, 0)
y = model.predict(video)
y = y[0][0]
print("预测结果为:",round(y),"概率值为:",y)
