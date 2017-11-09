# ObjectClassification
A playground to experiment with key concepts of Deep Learning.
I am working with the CIFAR-10 - Object Recognition in Images from the kaggle challenge. All the relevant files and related information can be found here: https://www.kaggle.com/c/cifar-10 .

Although a lot of information can be found all over the internet about Deep Learning, there aren't enough places that I could find that give you a good overview of the key concepts of Deep Learning. However, an exception to this is the Deep Learning specialization course from Coursera. From all the things so far I've read in regards to Deep Learning, Andrew Ng's course on Deep Learning specialization provides a good culimination of all the key concepts. The code style is inspired from what Andrew himself professes. And personally I like it too. 

While most of functionality the code pieces written implement could be easily harnessed from some existing and efficient Deep Learning Frameworks such as TensorFlow, I am taking more of a chef's approach to concoct everything with my very own hands. The sole purpose is to get my hands dirty and in the process get some on the field work experience.

To get started, I downloaded the data from the kaggle page mentioned above. The ProcessRawTrainingData.py file will preprocess the images, flatten and normalize them and pack into a zipped file X.csv.gz. Correspondingly, the labels are stored in trainlabels.csv. The NetFramework.py implements a very basic, fully connected, 2 layer neural network. It does not do much and is pretty muuch static in terms of flexibility to create multi-layered architectures. The Main.py file implements a dirty, in progress functionality that gets the thing running. (NOTE: make sure to import appropriate NetFramework to play around.)

NetFramework_Dynamic implements the functionality to generate multi-layered deep neural networks (DNNs). Again there are some things that I plan to implement. The code is not stable and I am still figuring out the problems with the compute_cost.py. I am running into numerical instabilities at the moment. 

# ToDos
- Numerical Stability
- Batch Normalization, This is implemented but I am not sure if this is causing numerical instabilities
- Regularization L2 or Dropout
- Implement Adam optimization 
- Hyperparameter tuning

Implementing batch normalization is straight forward, but if like me you also tried you might have noticed that finding the derivatives of the loss function w.r.t the parameter of the algorithm (gamma, beta) is not straight forward. However, this post here explains this in a much consumable format : https://kevinzakka.github.io/2016/09/14/batch_normalization/ .

# Further in Future ToDos
 - Incorporate multiprocessing capabilities
