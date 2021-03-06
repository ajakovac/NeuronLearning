/* Copyright (C) GJ
 * Written by G. and A. Jakovac 2019 */
#include <iostream>
#include <sstream>
#include <string>
#include "disp_img.hpp"
#include "cifar_10.hpp"
#include "stl_10.hpp"
#include "mnist.hpp"
#include "cmake_variables.hpp"

int main() {
  // std::string cifar_data =
  //   DATASET_DIR + "cifar-10-batches-bin/data_batch_1.bin";
  CIFAR_10_dataset *d = new CIFAR_10_dataset(cifar10_data);

  // std::string stl10_img = DATASET_DIR + "stl10_binary/train_X.bin";
  // std::string stl10_lbl = DATASET_DIR + "stl10_binary/train_y.bin";
  STL_10_dataset *s = new STL_10_dataset(stl10_img, stl10_lbl);

  // std::string mnist_img = DATASET_DIR + "mnist/train_images";
  // std::string mnist_lbl = DATASET_DIR + "mnist/train_labels";
  MNIST_dataset *m = new MNIST_dataset(mnist_train_img, mnist_train_lbl);

  auto update1 = [=](Image_View& v) {
    d->next();
    v.fillRGB(d->data(Red), d->data(Green), d->data(Blue));
    v.setTitle(d->labelString());
  };

  auto update2 = [=](Image_View& v) {
    s->next();
    v.fillRGB(s->data(Red), s->data(Green), s->data(Blue), true);
    v.setTitle(s->labelString());
  };

  auto update3 = [=](Image_View& v) {
    m->next();
    v.fillGreyscale(m->data(Red));
    v.setTitle(std::to_string(m->label()) );
  };

  Image_View view(32, 32, update1);
  view.show();
  Image_View view1(96, 96, update2);
  view1.show();
  Image_View MNIST_view(28, 28, update3);
  MNIST_view.show();

  delete m;
  delete s;
  delete d;
}
