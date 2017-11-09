from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Process, Pool
import time


def get_image_paths(folder):
  return (os.path.join(folder, f)
      for f in os.listdir(folder)
      if 'png' in f)


def create_training_data(filename):
    face = misc.face()
    face = misc.imread(filename)
    gface = face.reshape(face.shape[0] * face.shape[1] * face.shape[2]) / 255
    return gface


if __name__ == '__main__':
    images = get_image_paths("train")
    procs = []

    for index, file in enumerate(images):
        proc = Process(target=create_training_data, args=(file,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    print(len(proc))

'''if __name__=='__main__':
    start_time = time.time()
    pool = Pool()
    images = get_image_paths("train")
    X = np.zeros(shape=(3072, 50000))
    for line, val in enumerate(images):
        result = pool.apply_async(create_training_data, [val])
        X[:, line] = result.get()
    pool.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    print(X[:, 0:5])'''

'''if __name__ == "__main__":
    #X = []
    X = np.zeros(shape=(3072, 50000))
    #print(X.shape)
    images = get_image_paths("train")
    pool = Pool()
    pool.map(create_training_data,images, X)
    pool.close()
    pool.join()
    print(X.shape)
    print(X[:, 0:5])
    np.savetxt('X.csv.gz', X, delimiter=',')'''