from scipy import misc
import numpy as np
import os

def get_image_paths(folder="train"):
  return (os.path.join(folder, f)
      for f in os.listdir(folder)
      if 'png' in f)

face = misc.face()
face = misc.imread("/Users/sanketshinde/Documents/MyPythonPOCs/ObjectClassification/train/1.png")
print(face.reshape(32*32*3,1).shape)

for k, i in enumerate(get_image_paths()):
    print("index: ", k, " path: ", i)
