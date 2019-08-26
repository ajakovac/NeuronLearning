/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <chrono>
#include "Structure.hpp"
#include "Backprop.hpp"
#include "mnist.hpp"
#include "Update.hpp"
#include "Learning.hpp"
#include "cmake_variables.hpp"

void normalize_connections(Network *N, int sn) {
  int cnmax = N->nConnections(sn);
  double nrm = 0;
  for (uint cn = 0; cn < cnmax; ++cn) {
    double w = N->connection(sn, cn);
    nrm += w*w;
  }
  nrm = sqrt(nrm);
  for (uint cn = 0; cn < cnmax; ++cn)
    N->connection(sn, cn) *= 3.0/nrm;
}


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
  auto ly1update = affine_nonlin_update(tanh_fn(1.0, 1.0));
  auto ly1bp = affine_nonlin_bp(dtanh(1.0, 1.0));

  int  ly2 = ntw.AddLayer({nrows-11, ncols-11});
  ntw.ConnectLastLayer(masquedlist(ly1, {5, 5}, {1, 1}, {0, 0}),
                       normal_dist(0.0, 0.01));
  auto ly2update = affine_nonlin_update(tanh_fn(1.0, 1.0));
  auto ly2bp = affine_nonlin_bp(dtanh(1.0, 1.0));

  int  lyres = ntw.AddLayer({10});
  // ntw.ConnectLastLayer_advanced(
  //   alllayer(ly2),
  //   [&](Network* N, int sn0, int sn1) {
  //     if ((sn0+sn1)%2) return true;
  //     else
  //       return false;
  //   },
  //   normal_dist(0.0, 0.01));
  // auto lyresupdate = affine_nonlin_update(tanh_fn(1.0, 1.0));
  // auto lyresupdate = affine_nonlin_update(id_fn);
  // auto lyresbp = affine_nonlin_bp(d_id);
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

  // the new staff: normalize connections site by site
  for (int sn = 0; sn < ntw.nSites(); ++sn) normalize_connections(&ntw, sn);


  // In this version the loss is implicit, it will restrict the
  // result to be on a shell with radius R.

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
  int epochnumber = 10;  // train in epochs
  int batchsize = 50;  // gradient collection in batches, each batchsize long
  int batchnumber = numdat/batchsize;  // number of batches
  int nallbatches = 0;  // number of all batches
  double learningrate = 0.02;  // average learning rate
  int printnumber = 10;  // print info after printnumber batches
  int savenumber = 100;  // save after savenumber batches
  int nerrprint = 1;  // count the batches after last print
  double err = 0;    // error calculation
  int ncorr = 0;     // correct answers count

  ADAM learnmethod(&bpntwadd);
  std::cerr << "#ADAM learning";
  std::cerr << "\n#---------\n";
  std::cout << "#cnt  epoch  nbtch  avrerr" << std::endl;
  int globalcount = 0;

  for (int epoch = 0; epoch < epochnumber; ++epoch) {
    // int he beginning of each epoch we save the output of NN pics
    // std::cerr << "\nSaving...";
    // int NN = 10000;
    // mnist.restart();
    // std::ofstream fres[10];
    // for (int i = 0; i < 10; i++) {
    //   std::stringstream ss;
    //   ss << "twoDpoints" << i << ".dat";
    //   std::cout << ss.str() << std::endl;
    //   fres[i].open(ss.str());
    //   if (!fres[i].is_open()) throw(Error("File can not be opened!"));
    // }
    // for (int ncount = 0; ncount < NN && mnist.next(); ++ncount) {
    //   for (int n = 0; n < nrows*ncols; ++n)
    //     ntw.site(basely, n) = data[n]/255.0;
    //   update(&ntw);
    //   std::ofstream & ff = fres[mnist.label()];
    //   ntw.forallSitesinLayer(lyres, [&](int n){
    //     ff << ntw[n] << "  ";});
    //   ff << std::endl;
    // }
    // std::cerr << "\ndone" << std:: endl;
    // for (int i = 0; i< 10; i++) fres[i].close();

    mnist.restart();  // in each epoch we start the training set from beginning
    // std::cerr << "#" << epoch << std::endl;

    for (int nbtch = 0; nbtch < batchnumber; ++nbtch, ++nallbatches) {
      bpntwadd.reset();  // in each new batch restart derivative collection

      // auto t1 = std::chrono::high_resolution_clock::now();

      for (int btch = 0; btch < batchsize; ++btch) {
        if (!mnist.next()) {    // load the next image
          mnist.restart();
          mnist.next();
        }
        // if (mnist.label() != 0) continue;
        for (int n = 0; n < nrows*ncols; ++n)
          ntw.site(basely, n) = data[n]/255.0;
        for (auto & x : expected_output) x = 0.01;
        expected_output[mnist.label()] = 1.0;
        normalize(&expected_output);

        update(&ntw);
        err += ntw[ntw.nSites()-1];

        bpntw.reset();
        backpropagate(&bpntw);
        bpntwadd.add(bpntw, 1.0);

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

      // the new staff: normalize connections site by site
      for (int sn = 0; sn < ntw.nSites(); ++sn) normalize_connections(&ntw, sn);
    }
  }

  // // final save
  // std::cerr << "\nSaving...";
  // int NN = 10000;
  // mnist.restart();
  // std::ofstream fres[10];
  // for (int i = 0; i < 10; i++) {
  //   std::stringstream ss;
  //   ss << "twoDpoints" << i << ".dat";
  //   std::cout << ss.str() << std::endl;
  //   fres[i].open(ss.str());
  //   if (!fres[i].is_open()) throw(Error("File can not be opened!"));
  // }
  // for (int ncount = 0; ncount < NN && mnist.next(); ++ncount) {
  //   for (int n = 0; n < nrows*ncols; ++n)
  //     ntw.site(basely, n) = data[n]/255.0;
  //   update(&ntw);
  //   std::ofstream & ff = fres[mnist.label()];
  //   ntw.forallSitesinLayer(lyres, [&](int n){
  //     ff << ntw[n] << "  ";});
  //   ff << std::endl;
  // }
  // std::cerr << "done" << std:: endl;
  // for (int i = 0; i< 10; i++) fres[i].close();

  // save the optimized network
  ntw.save("learn_mnist.ntw");
  std::cout << "\n__END__\n";

  return 0;
} catch(Error & u) {
  std::cerr << u.error_message << std::endl;
}
