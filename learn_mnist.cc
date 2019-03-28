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
#include "Learning.h"

int main(int argc, char const *argv[]) {
  randiv.set_seed(10);

  // we need the MNIST database for character recognition learning
  sf::Image img;
  MNIST_Database mnist("train-images.idx3-ubyte",
                       "train-labels-idx1-ubyte", img);

  // Here we set up the network

  Network ntw;
  int basely = ntw.AddLayer({mnist.nrows, mnist.ncols});

  int  ly1 = ntw.AddLayer({mnist.nrows, mnist.ncols});
  ntw.ConnectLastLayer(alllayer(basely), normal_cf(0.0, 0.01));
  auto ly1update = affine_nonlin_update(ReLU);
  auto ly1bp = affine_nonlin_bp(dReLU);


  int  lyres = ntw.AddLayer({10});
  ntw.ConnectLastLayer(alllayer(ly1), normal_cf(0.0, 0.01));
  auto lyresupdate = affine_nonlin_update(exp_fn(1.0, 0.5));
  auto lyresbp = affine_nonlin_bp(dexp(1.0, 0.5));


  int lossly = ntw.AddLayer({1});
  ntw.ConnectLastLayer(alllayer(lyres), const_cf(1.0));
  std::vector<double> expected_output(10, 0);
  // auto lossupdate =  pnorm_loss(2, &expected_output);
  // auto lossbp = d_pnorm_loss(2, &expected_output);
  auto lossupdate =  KL_loss(&expected_output);
  auto lossbp = d_KL_loss(&expected_output);

  // ----------------------------------------------------------------------- //

  DNetwork bpntw(&ntw);
  DNetwork bpntwadd(&ntw);

  auto update = [=](Network *N) {
    N->applytoLayer(ly1, [=](int n){ ly1update(N, n);});
    N->applytoLayer(lyres, [=](int n){ lyresupdate(N, n);});
    N->applytoLayer(lossly, [=](int n){ lossupdate(N, n);});
  };

  auto backpropagate = [=](DNetwork *BPN) {
    Network *N = BPN->associatedNetwork();
    BPN->Dsite(N->nSites()-1) = 1.0;  // start with unit derivative
    N->applytoLayer(lossly, [=](int n){ lossbp(BPN, n);});
    N->applytoLayer(lyres, [=](int n){ lyresbp(BPN, n);});
    N->applytoLayer(ly1, [=](int n){ ly1bp(BPN, n);});
  };

  auto learn = steepest_descent;

  int epochnumber = 5;  // train in epochs
  int batchsize = 50;    // gradient collectionin batches, each batchsize long
  int batchnumber = mnist.numdat/batchsize;  // number of batches
  double learningrate = 0.02;  // average learning rate

  int printnumber = 10;  // print after printnumber batches
  int nerrprint = 0;  // count the batches after last print
  double err = 0;    // error calculation
  int ncorr = 0;     // correct answers count

  std::cout << "#epoch\tnbtch\tavrerr\tavrcorr(%)" << std::endl;
  for (int epoch = 0; epoch < epochnumber; ++epoch) {
    mnist.restart();  // in each epoch we start the training set from beginning
    std::cout << std::endl << std::endl;

    for (int nbtch = 0; nbtch < batchnumber; ++nbtch) {
      bpntwadd.reset();  // in each new batch restart derivative collection

      for (int btch = 0; btch < batchsize; ++btch) {
        if (!mnist.next_pic()) {    // load the next image
          mnist.restart();
          mnist.next_pic();
        }
        Image_to_Layer(&ntw, basely, &img);

        for (double& x : expected_output) x = 0.01;
        expected_output[mnist.label] = 1.0;
        normalize(&expected_output);

        update(&ntw);
        err += ntw[ntw.nSites()-1];

        bpntw.reset();
        backpropagate(&bpntw);
        bpntwadd += bpntw;

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
      }

      if (nerrprint == printnumber) {
        std::cout << epoch << " " << nbtch;
        std::cout << " " << err/(nerrprint*batchsize);
        std::cout << " " << (ncorr*100.0)/(nerrprint*batchsize);
        std::cout << std::endl;
        nerrprint = 1;
        err = 0;
        ncorr = 0;
      } else {
        ++nerrprint;
      }

      // learning: steepest descent
      double lrate = learningrate/batchsize;
      learn(&bpntwadd, ly1, lrate);
      learn(&bpntwadd, lyres, 10*lrate);
    }
  }
  // save the optimized network
  // ntw.save("learn_mnist.ntw");

  return 0;
}
