/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <cmath>
#include "Structure.hpp"
#include "Update.hpp"
#include "Backprop.hpp"
#include "Numder.hpp"

// the best test to check whether we caclculate the derivative correctly,
// if we compute them numerically, too.

int main(int argc, char const *argv[]) {
  // to be reproducable, we set the
  randiv.set_seed(10);

  // ----------------------------------------------------------------------- //
  // First we define the network of layers, set up connections
  Network ntw;

  // base layer, where external input comes in
  int   basely = ntw.AddLayer({5});
  // we fill the base layer with random numbers
  ntw.applytoLayer(basely, [&](int n){ ntw[n] = 1.0; });
  // ntw.applytoLayer(basely, [&](int n){ ntw[n] = normal_dist(0.0, 1.0)(); });

  // Here come the internal (hidden) layers with arbitrary number
  int      ly1 = ntw.AddLayer({3});
  // ly1 connected with a masque (windowed) to the base layer
  ntw.ConnectLastLayer(masquedlist(basely, {3}, {1}), const_cf(1.0));

  int ly2 = ntw.AddLayer({3});
  ntw.ConnectLastLayer(directlist(ly1), const_cf(1.0));

  // // We have a result layer
  // int resultly = ntw.AddLayer({3});
  // // resultly is fully connected to the last hidden layer
  // ntw.ConnectLastLayer(alllayer(ly2), normal_cf(0.0, 0.5) );

  // Finally we have to compute a single number, the loss:
  int lossly = ntw.AddLayer({1});
  // loss layer is unifromly connected to the result layer (have unit weights)
  ntw.ConnectLastLayer(alllayer(ly2), const_cf(1.0));

  // ----------------------------------------------------------------------- //
  // We have to define the update functions; we do it together with the
  // corresponding backpropagation partner

  // The base layer does not need update or backpropagation

  // first hidden layer:
  auto ly1update = affine_nonlin_update(id_fn);
  auto     ly1bp = affine_nonlin_bp(d_id);

  // second hidden layer:
  double p = 0.4;
  auto ly2update = dropout_update(&p);
  auto     ly2bp = affine_nonlin_bp(d_id);
  // auto ly2update = affine_nonlin_update(id_fn);
  // auto     ly2bp = affine_nonlin_bp(d_id);
  // In case of the result layer it is recommended to map only to positive
  // values
  // auto resupdate = affine_nonlin_update(exp_fn(1.0, 1.0));
  // auto  resbp = affine_nonlin_bp(dexp(1.0, 1.0));

  // For the loss we need to have expected output to compare with
  std::vector<double> expected_output(10, 1);
  // and apply some loss functions
  auto lossupdate = pnorm_loss(1, &expected_output);
  auto lossbp = d_pnorm_loss(1, &expected_output);

  // it is worth to build up an update function for the complete network
  auto update = [=](Network* L) {
    // basely is not updated
    randiv.set_seed(10);  // to be able to reproduce the same runs
    L->applytoLayer(ly1, [=](int n){      ly1update(L, n); });
    L->applytoLayer(ly2, [=](int n){      ly2update(L, n); });
    // L->applytoLayer(resultly, [=](int n){ resupdate(L, n); });
    L->applytoLayer(lossly, [=](int n){   lossupdate(L, n); });
  };

  // we collect the important functions in the beginning
  // the update and backpropagation functions:

  // auto ly1update = affine_nonlin_update(ReLU);
  // auto  ly1bp = affine_nonlin_bp(dReLU);

  // auto ly1update = maxpool_update;
  // auto  ly1bp = affine_nonlin_bp(d_id);


  // auto lossupdate = KL_loss(&expected_output);
  // auto lossbp = d_KL_loss(&expected_output);

  // auto lossupdate = pnorm_loss(2, &expected_output);
  // auto lossbp = d_pnorm_loss(2, &expected_output);

  // choose the value of the expected output
  // for (double& x : expected_output) x = 0.0001;
  // expected_output[5] = 1.0;
  // double nrm = 0;
  // for (double& x : expected_output) nrm+= x;
  // for (double& x : expected_output) x/= nrm;

  // perform the first update
  update(&ntw);

  // We can save the network geometry and default connections
  ntw.save("mynet2.ntw");

  // ----------------------------------------------------------------------- //
  // now the basic network is defined, next we should apply the desired skins
  // we will need a backpropagation skin
  DNetwork bpntw(&ntw);

  // it is worth to collect all backpropagations into a common bp function
  auto backpropagate = [=](DNetwork *BPL) {
    Network *L = BPL->associatedNetwork();
    BPL->Dsite(L->nSites()-1) = 1.0;  // start with unit derivative
    L->applytoLayer(lossly, [=](int n){ lossbp(BPL, n); });
    // L->applytoLayer(resultly, [=](int n){ resbp(BPL, n); });
    L->applytoLayer(ly2, [=](int n){ ly2bp(BPL, n); });
    L->applytoLayer(ly1, [=](int n){ ly1bp(BPL, n); });
    // the base layer does not need backpropagation
  };

  // we can now compute all derivatives at once
  backpropagate(&bpntw);

  // ----------------------------------------------------------------------- //
  // this program tests the backpropagation results against
  // numerical derivation: to this latter we need partial_update
  auto partial_update = [=](Network* L, int ly) {
    randiv.set_seed(10);  // the same seed as before
    // ly0 is not updated
    if (ly<= ly1) L->applytoLayer(ly1, [=](int n){ ly1update(L, n); });
    if (ly<= ly2) L->applytoLayer(ly2, [=](int n){ ly2update(L, n); });
    if (ly<= lossly) L->applytoLayer(lossly, [=](int n){ lossupdate(L, n); });
  };
  Numder ndntw(&ntw, partial_update);
  ndntw.getD();

  // -------------------------------------------------------------- //
  // we can compare the results of the numeric and analytic derivatives
  double error2 = 0;
  int nerr = 0;
  auto sq = [](double x){return x*x;};

  // here we can determine which elements should be taken into account:
  // print site derivatives? (layerwise)
  // basely, ly1, ly2, resultly, lossly
  bool pr_Dsite[] = {true, true, true,  true};
  // print connections?
  bool pr_Dconn[] = {true, true, false, false};

  for (int n = ntw.nSites()-1; n>= 0; --n) {
    int lyn = ntw.sitelayerID(n);
    std::cout << n << "(" << lyn << ")";
    std::cout << "\taxon=" << ntw.axon(n);
    std::cout << std::endl;
    if (pr_Dsite[lyn]) {
      std::cout << "\tz=" << bpntw.Dsite(n) << " -- "
                          << ndntw.z[n];
      error2+= sq(bpntw.Dsite(n)-ndntw.z[n]);
      nerr++;
      std::cout << std::endl;
    }
    if (pr_Dconn[lyn]) {
      for (int cid = 0; cid < ntw.nConnections(n); cid++) {
        std::cout << "\t\t[" << cid << "]";
        std::cout << "\tconn=" << ntw.siteConnection(n, cid);
        std::cout << "\tdconn=" << bpntw.Dconn(n, cid);
        std::cout << " -- " << ndntw.dconn[ ntw.getconnID(n, cid)] << std::endl;
        error2+= sq(bpntw.Dconn(n, cid)-ndntw.dconn[ ntw.getconnID(n, cid)]);
        nerr++;
      }
    }
  }
  std::cout << "Average error = " << sqrt(error2/nerr) << '\n';

  return 0;
}
