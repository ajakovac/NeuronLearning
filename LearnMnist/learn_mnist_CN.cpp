/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include "Structure.hpp"
#include "Backprop.hpp"
#include "mnist.hpp"
#include "Update.hpp"
#include "Learning.hpp"
#include "cmake_variables.hpp"

int main(int argc, char const *argv[]) try {
  randiv.set_seed(10);

  // we need the MNIST database for character recognition learning
  MNIST_dataset mnist(mnist_train_img, mnist_train_lbl);
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
  auto ly1update = affine_nonlin_update(ReLU);
  auto ly1bp = affine_nonlin_bp(dReLU);

  int  ly2 = ntw.AddLayer({nrows-11, ncols-11});
  ntw.ConnectLastLayer(masquedlist(ly1, {5, 5}, {1, 1}, {0, 0}),
                       normal_dist(0.0, 0.01));
  auto ly2update = affine_nonlin_update(ReLU);
  auto ly2bp = affine_nonlin_bp(dReLU);

  int  lyres = ntw.AddLayer({10});
  ntw.ConnectLastLayer(alllayer(ly2), normal_dist(0.0, 0.01));
  auto lyresupdate = affine_nonlin_update(exp_fn(1.0, 0.5));
  auto lyresbp = affine_nonlin_bp(dexp(1.0, 0.5));

  int lossly = ntw.AddLayer({1});
  ntw.ConnectLastLayer(alllayer(lyres), 1.0);
  std::vector<double> expected_output(10, 0);
  // auto lossupdate =  pnorm_loss(2, &expected_output);
  // auto lossbp = d_pnorm_loss(2, &expected_output);
  auto lossupdate =  KL_loss(&expected_output);
  auto lossbp = d_KL_loss(&expected_output);

  // ----------------------------------------------------------------------- //

  DNetwork bpntw(&ntw);
  DNetwork bpntwadd(&ntw);

  auto update = [=](Network *N) {
    N->forallSitesinLayer(ly1, [=](int n){ ly1update(N, n);});
    N->forallSitesinLayer(ly2, [=](int n){ ly2update(N, n);});
    N->forallSitesinLayer(lyres, [=](int n){ lyresupdate(N, n);});
    N->forallSitesinLayer(lossly, [=](int n){ lossupdate(N, n);});
  };

  auto backpropagate = [=](DNetwork *DN) {
    Network *N = DN->associatedNetwork();
    DN->Dsite(N->nSites()-1) = 1.0;  // start with unit derivative
    N->forallSitesinLayer(lossly, [=](int n){ lossbp(DN, n);});
    N->forallSitesinLayer(lyres, [=](int n){ lyresbp(DN, n);});
    N->forallSitesinLayer(ly2, [=](int n){ ly2bp(DN, n);});
    N->forallSitesinLayer(ly1, [=](int n){ ly1bp(DN, n);});
  };

  // use this setting for a complete learning
  int epochnumber = 5;  // train in epochs
  int batchsize = 50;  // gradient collection in batches, each batchsize long
  int batchnumber = numdat/batchsize;  // number of batches
  double learningrate = 0.02;  // average learning rate
  int printnumber = 10;  // print after printnumber batches

  double releaserate = 0.999;  // reducing learning rate after each learning

  // use the following setting for learn the first picture
  // int epochnumber = 20;  // train in epochs
  // int batchsize = 1;  // gradient collection in batches, each batchsize long
  // int batchnumber = 1;  // number of batches
  // double learningrate = 0.02;  // average learning rate
  // int printnumber = 1;  // print after printnumber batches

  int nerrprint = 1;  // count the batches after last print
  double err = 0;    // error calculation
  int ncorr = 0;     // correct answers count


  ADAM learnmethod(&bpntwadd);
  std::cout << "#ADAM learning";

  // ConjugateGadient learnmethod(&bpntwadd);
  // std::cout << "#conjugate gradient learning";

  // SteepestDescent learnmethod(&bpntwadd);
  // std::cout << "#steepest descent learning";

  // StochGrad learnmethod(&bpntwadd);
  // std::cout << "#stochastic gradient learning";
  // batchsize = 1;
  // batchnumber = mnist.numdat;
  // printnumber = 499;

  std::cout << "\n#---------\n";

  std::cout << "#cnt  epoch  nbtch  avrerr  avrcorr(%)" << std::endl;
  int globalcount = 0;

  for (int epoch = 0; epoch < epochnumber; ++epoch) {
    mnist.restart();  // in each epoch we start the training set from beginning
    // std::cout << std::endl << std::endl;

    for (int nbtch = 0; nbtch < batchnumber; ++nbtch) {
      bpntwadd.reset();  // in each new batch restart derivative collection

      // auto t1 = std::chrono::high_resolution_clock::now();

      for (int btch = 0; btch < batchsize; ++btch) {
        if (!mnist.next()) {    // load the next image
          mnist.restart();
          mnist.next();
        }
        for (int n = 0; n < nrows*ncols; ++n)
          ntw.site(basely, n) = data[n]/255.0;
        for (double& x : expected_output) x = 0.01;
        expected_output[mnist.label()] = 1.0;
        normalize(&expected_output);

        update(&ntw);
        err += ntw[ntw.nSites()-1];

        bpntw.reset();
        backpropagate(&bpntw);
        bpntwadd += bpntw;

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
      }

      // auto t2 = std::chrono::high_resolution_clock::now();
      // auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
      // std::cout << "dt = " << dt << " ms\n";


      if (nerrprint > printnumber) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "                                 \r";
        std::cout << std::flush;
        std::cout << std::right
                  << std::setw(4) << globalcount++
                  << std::setw(7) << epoch
                  << std::setw(7) << nbtch
                  << std::setw(8) << err/(nerrprint*batchsize)
                  << std::setw(9) << (ncorr*100.0)/(nerrprint*batchsize);
        std::cout << " %" << std::flush;
        nerrprint = 1;
        err = 0;
        ncorr = 0;
      } else {
        ++nerrprint;
      }

      // learning with the actual learn method
      double lrate = learningrate/batchsize;
      learnmethod.startlearncycle();
      learnmethod.learn(ly1, lrate);
      learnmethod.learn(ly2, 10*lrate);
      learnmethod.learn(lyres, 20*lrate);
    }
  }
  // save the optimized network
  ntw.save("learn_mnist.ntw");
  std::cout << "\n__END__\n";

  return 0;
} catch(Error & u) {
  std::cerr << u.error_message << std::endl;
}
