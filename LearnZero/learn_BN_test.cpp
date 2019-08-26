/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <string>
#include <chrono>
#include "Structure.hpp"
#include "Backprop.hpp"
#include "mnist.hpp"
#include "Update.hpp"
#include "Learning.hpp"
#include "cmake_variables.hpp"

int main(int argc, char const *argv[]) {
  randiv.set_seed(10);

  // we need the MNIST database for character recognition learning
  MNIST_dataset mnist(mnist_test_img, mnist_test_lbl);
  int nrows = mnist.height;
  int ncols = mnist.width;
  int numdat = 60000;
  int *data = mnist.data(Red);

  // Here we set up the network
  Network ntw;
  int basely = ntw.AddLayer({nrows, ncols});

  int  ly1 = ntw.AddLayer({nrows-7, ncols-7});
  ntw.ConnectLastLayer(masquedlist(basely, {8, 8}, {1, 1}, {0, 0}),
                       normal_dist(0.0, 0.01));
  auto ly1update = affine_nonlin_update(tanh_fn(1.0, 1.0));

  int  ly2 = ntw.AddLayer({nrows-11, ncols-11});
  ntw.ConnectLastLayer(masquedlist(ly1, {5, 5}, {1, 1}, {0, 0}),
                       normal_dist(0.0, 0.01));
  auto ly2update = affine_nonlin_update(tanh_fn(1.0, 1.0));

  int  lyres = ntw.AddLayer({10});
  ntw.ConnectLastLayer(alllayer(ly2), normal_dist(0.0, 0.01));
  auto lyresupdate = affine_nonlin_update(exp_fn(1.0, 0.5));

  int lossly = ntw.AddLayer({1});
  ntw.ConnectLastLayer(alllayer(lyres), 1.0);
  std::vector<double> expected_output(10, 0);
  // auto lossupdate =  pnorm_loss(2, &expected_output);
  // auto lossbp = d_pnorm_loss(2, &expected_output);
  auto lossupdate =  KL_loss(&expected_output);

  ntw.load("learn_mnist.ntw");
  // ----------------------------------------------------------------------- //

  auto update = [=](Network *N) {
    N->forallSitesinLayer(ly1, [=](int n){ ly1update(N, n);});
    N->forallSitesinLayer(ly2, [=](int n){ ly2update(N, n);});
    N->forallSitesinLayer(lyres, [=](int n){ lyresupdate(N, n);});
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
      ntw.site(basely, n) = data[n]/255.0;

    update(&ntw);
     // find maximal result bin
    int maxn = -1;
    double maxv = 0;
    ntw.forallSitesinLayer(lyres, [&](int n){
      if (ntw.site(n) > maxv) {
        maxn = ntw.sitelayerIndex(lyres, n);
        maxv = ntw.site(n);
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