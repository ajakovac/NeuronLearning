/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2019 */
#include <iostream>
#include <string>
#include "Structure.hpp"
#include "Backprop.hpp"
#include "mnist.hpp"
#include "Update.hpp"
#include "cmake_variables.hpp"


int main(int argc, char const *argv[]) {
  randiv.set_seed(10);

  // we need the MNIST database for character recognition learning
  MNIST_dataset mnist(mnist_train_img, mnist_train_lbl);
  int nrows = mnist.height;
  int ncols = mnist.width;
  int numdat = 10000;
  int *data = mnist.data(Red);

  // Here we read in the network

  Network ntw;
  ntw.load("learn_mnist.ntw");
  int basely = 0;

  int  ly1 = 1;
  auto ly1update = affine_nonlin_update(ReLU);

  int  lyres = 2;
  auto lyresupdate = affine_nonlin_update(exp_fn(1.0, 0.5));

  // We do not need to treat the loss layer
  // ----------------------------------------------------------------------- //

  auto update = [=](Network *N) {
    N->applytoLayer(ly1, [=](int n){ ly1update(N, n);});
    N->applytoLayer(lyres, [=](int n){ lyresupdate(N, n);});
  };

  // we run through the complete data base
  mnist.restart();
  int printpercent = 5;  // send signal after printpercent percents
  int printnumber = printpercent*numdat/100;
  int ncorr = 0;     // correct answers count
  for (int nimg = 0; nimg < numdat; ++nimg) {
    if (!mnist.next()) {    // load the next image
      mnist.restart();
      mnist.next();
    }
    for (int n = 0; n < nrows*ncols; ++n)
      ntw.axon(basely, n) = data[n]/255.0;

    update(&ntw);
     // find maximal result bin
    int maxn = -1;
    double maxv = 0;
    ntw.applytoLayer(lyres, [&](int n){
      if (ntw.axon(n) > maxv) {
        maxn = ntw.sitelayerIndex(n);
        maxv = ntw.axon(n);
      }
    });
    if (maxn == mnist.label()) ++ncorr;

    if (nimg%printnumber == 0) {
      std::cout << "                                \r";
      std::cout << "update=" << (100.0*nimg)/numdat;
      std::cout << "%\r" << std::flush;
    }
  }
  std::cout << "                                \r";
  std::cout << "update=100.0%";
  std::cout << "\n------------------" << std::endl;
  std::cout << "correct = " << (100.0*ncorr)/numdat << "%" << std::endl;
  return 0;
}
