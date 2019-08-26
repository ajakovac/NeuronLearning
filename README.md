# NeuronLearning
Verbose AI platform ready to hack and try new algorithms.

**Main goal**: write a tunable network manager where implementation of new algorithms is easy through writing own lambda functions.
Project is writen in *C++*

## Installation:
* clone the github content to some directory
* install the datasets; currently three datasets are used in some ways in the programs: [mnist](http://yann.lecun.com/exdb/mnist/), [cifar-10-binary](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) and [stl-10](http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz)
* create a directory at DATASET_DIR, create mnist, cifar-10-batches-bin and stl10_binary subdirectories. Extract the downloaded file content to these libraries
* in case of mnist, please rename:
  * train-images-idx3-ubyte.gz -> train_images
  * train-labels-idx1-ubyte.gz -> train_labels
  * t10k-images-idx3-ubyte.gz -> test_images
  * t10k-labels-idx1-ubyte.gz -> test_labels
* alternatively, edit the include/cmake_variables.hpp.in file, and provide explicitly the names of the files
* run cmake with a switch: cmake -DDATASET_DIR="<your dataset_dir>" .
* run make or build the executables with cmake --build -DDATASET_DIR="<your dataset_dir>" .




