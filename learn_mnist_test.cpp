/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <string>
#include <SFML/Graphics.hpp>
#include "Structure.h"
#include "ImageLayer.h"
#include "Backprop.h"
#include "Mnist_db.h"
#include "Update.h"

int main(int argc, char const *argv[]) {
  randiv.set_seed(10);

  // we need the MNIST database for character recognition learning
  sf::Image img;
  // MNIST_Database mnist("train-images.idx3-ubyte",
  //                      "train-labels-idx1-ubyte", img);
  MNIST_Database mnist("testimage.dat",
                        "testlabels.dat", img);

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
  int printnumber = printpercent*mnist.numdat/100;
  int ncorr = 0;     // correct answers count
  for (int nimg = 0; nimg < mnist.numdat; ++nimg) {
    if (!mnist.next_pic()) {    // load the next image
      mnist.restart();
      mnist.next_pic();
    }
    Image_to_Layer(&ntw, basely, &img);

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
    if (maxn == mnist.label) ++ncorr;

    if (nimg%printnumber == 0) {
      std::cout << "                                \r";
      std::cout << "update=" << (100.0*nimg)/mnist.numdat;
      std::cout << "%\r" << std::flush;
    }
  }
  std::cout << "\n------------------" << std::endl;
  std::cout << "correct = " << (100.0*ncorr)/mnist.numdat << "%" << std::endl;
  return 0;
}
