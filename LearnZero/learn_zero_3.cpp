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

void normalize_connections(Network *N, int sn, double nn) {
  int cnmax = N->nConnections(sn);
  double nrm = 0;
  for (uint cn = 0; cn < cnmax; ++cn) {
    double w = N->connection(sn, cn);
    nrm += w*w;
  }
  nrm = sqrt(nrm);
  for (uint cn = 0; cn < cnmax; ++cn)
    N->connection(sn, cn) *= nn/nrm;
}

void save_layer(Network *NW,
                auto update,
                int lyres,
                MNIST_dataset *DS,
                int NN,
                std::string filename) {
  int nrows = DS->height;
  int ncols = DS->width;
  int *data = DS->data(Red);

  std::ofstream fres[10];
  for (int i = 0; i < 10; i++) {
    std::stringstream ss;
    ss << filename << i << ".dat";
    std::cout << ss.str() << std::endl;
    fres[i].open(ss.str());
    if (!fres[i].is_open()) throw(Error("File can not be opened!"));
  }
  for (int ncount = 0; ncount < NN && DS->next(); ++ncount) {
    for (int n = 0; n < nrows*ncols; ++n)
      NW->site(0, n) = data[n]/255.0;
    update(NW);
    std::ofstream & ff = fres[DS->label()];
    NW->forallSitesinLayer(lyres, [&](int n){
      ff << (*NW)[n] << "  ";});
    ff << std::endl;
  }
  for (int i = 0; i< 10; i++) fres[i].close();
}


int main(int argc, char const *argv[]) try {
  randiv.set_seed(10);

  enum task { train = 0, test = 1, gen = 2 };
  task actualtask = train;
  // task actualtask = test;

  // we need the MNIST database for character recognition learning
  MNIST_dataset mnist(mnist_train_img, mnist_train_lbl);
  MNIST_dataset mnist_test(mnist_test_img, mnist_test_lbl);
  int nrows = mnist.height;
  int ncols = mnist.width;
  int numdat = 60000;
  int numdat_test = 10000;
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
  // auto ly2update = affine_nonlin_update(ReLU);
  // auto ly2bp = affine_nonlin_bp(dReLU);

  int dres = 2;
  int lyres = ntw.AddLayer({dres});
  ntw.ConnectLastLayer_advanced(
    alllayer(ly2),
    [&](Network* N, int sn0, int sn1) {
      if ((sn0+sn1)%dres) return true;
      else
        return false;
    },
    normal_dist(0.0, 0.01));
  // ntw.ConnectLastLayer(alllayer(ly2), normal_dist(0.0, 0.1));
  // auto lyresupdate = affine_nonlin_update(tanh_fn(1.0, 1.0));
  auto lyresupdate = affine_nonlin_update(id_fn);
  auto lyresbp = affine_nonlin_bp(d_id);

  // the new staff: normalize connections site by site
  for (int sn = 0; sn < ntw.nSites(); ++sn)
    normalize_connections(&ntw, sn, 3.0);

  // use pretrained network
  std::ifstream dum("pretrain3.ntw");
  // check if file exists;if so, use its data to load connections
  if (dum.is_open()) {
    std::cout << "Load from pretrained network...";
    ntw.load("pretrain3.ntw");
    std::cout << "done\n";
  }
  dum.close();

  // In this version the loss is implicit, it will restrict the
  // result to be on a shell with radius R.

  // ----------------------------------------------------------------------- //

  DNetwork bpntw(&ntw);
  DNetwork bpntwadd(&ntw);

  auto update = [=](Network *N) {
    N->forallSitesinLayer(ly1, [=](int n){ ly1update(N, n);});
    N->forallSitesinLayer(ly2, [=](int n){ ly2update(N, n);});
    N->forallSitesinLayer(lyres, [=](int n){ lyresupdate(N, n);});
  };

  auto backpropagate = [=](DNetwork *DN, double R2) {
    Network *N = DN->associatedNetwork();
    double y2 = 0.0;
    // we are tending to a circle with radius R
    N->forallSitesinLayer(lyres, [&](int n){ y2 += N->site(n)*N->site(n);});
    N->forallSitesinLayer(lyres, [&](int n){
      DN->Dsite(n) = N->site(n)*(y2-R2);
    });
    N->forallSitesinLayer(lyres, [=](int n){ lyresbp(DN, n);});
    N->forallSitesinLayer(ly2, [=](int n){ ly2bp(DN, n);});
    N->forallSitesinLayer(ly1, [=](int n){ ly1bp(DN, n);});
  };

  // use this setting for a complete learning
  int epochnumber = 20;  // train in epochs
  int batchsize = 50;  // gradient collection in batches, each batchsize long
  int batchnumber = numdat/batchsize;  // number of batches
  int nallbatches = 0;  // number of all batches
  double learningrate = 0.01;  // average learning rate
  int printnumber = 10;  // print info after printnumber batches
  int savenumber = 100;  // save after savenumber batches
  int globalcount = 0;

  ADAM learnmethod(&bpntwadd);
  std::cerr << "#ADAM learning";
  std::cerr << "\n#---------\n";
  std::cout << "#cnt  epoch  nbtch  avrerr" << std::endl;

  if (actualtask == train) {
    for (int epoch = 0; epoch < epochnumber; ++epoch) {

      // in the beginning of each epoch we save the output of NN pics
      std::cerr << "\nSaving...";
      mnist.restart();
      save_layer(&ntw, update, lyres, &mnist, 10000, "twoDpoints");
      std::cerr << "\ndone" << std:: endl;
      // std::cerr << "Enter to continue ...";
      // std::string dum; std::cin >> dum;
      // std::cerr << "... OK\n";

      // mnist.restart();  // in each epoch we start the training set from beginning
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

          update(&ntw);

          bpntw.reset();
          if (mnist.label() == 0)
            backpropagate(&bpntw, 0);
          else
            backpropagate(&bpntw, 1.0);
          bpntwadd.add(bpntw, 1.0);
        }

        // auto t2 = std::chrono::high_resolution_clock::now();
        // auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        // std::cout << "dt = " << dt << " ms\n";


        if (globalcount++ % printnumber == 0) {
          std::cout << std::fixed << std::setprecision(2);
          std::cout << "                                 \r";
          std::cout << std::flush;
          std::cout << std::right
                    << std::setw(4) << globalcount++
                    << std::setw(7) << epoch
                    << std::setw(7) << nbtch;
          std::cout << std::flush;
        }

        // learning with the actual learn method
        double lrate = learningrate/batchsize;
        learnmethod.startlearncycle();
        learnmethod.learn(ly1, lrate);
        learnmethod.learn(ly2, lrate);
        learnmethod.learn(lyres, lrate);

        // the new staff: normalize connections site by site
        for (int sn = 0; sn < ntw.nSites(); ++sn)
          normalize_connections(&ntw, sn, 3.0);
      }
    }

    // save the optimized network
    ntw.save("learn_mnist.ntw");
    std::cout << "\n__END__\n";
  } else {
    ntw.load("learn_mnist.ntw");
  }
  // final save
  std::cerr << "\nSaving...";
  mnist.restart();
  save_layer(&ntw, update, lyres, &mnist, 60000, "twoDpoints");
  std::cerr << "\ndone" << std:: endl;

  //  test set
  std::cerr << "\nSaving test set...";
  mnist_test.restart();
  save_layer(&ntw, update, lyres, &mnist_test, 60000,
             "twoDpoints_test");
  std::cerr << "\ndone" << std:: endl;

  return 0;
} catch(Error & u) {
  std::cerr << u.error_message << std::endl;
}
