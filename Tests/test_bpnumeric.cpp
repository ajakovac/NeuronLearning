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

  // these bool vectors are used to decide whether the different derivatives
  // are to be compared with the numeric derivation
  std::vector<bool>  pr_Dsite;    // site derivative
  std::vector<bool>  pr_Dconn;    // connection derivatives

  Network ntw;

  int basely = ntw.AddLayer({10});
  ntw.forallSitesinLayer(basely, [&](int n){
    ntw[n] = normal_dist(0.0, 1.0)();
  });
  pr_Dsite.push_back(true);
  pr_Dconn.push_back(false);

  int ly1 = ntw.AddLayer({8});
  ntw.ConnectLastLayer(masquedlist(basely, {3}, {1}, {0}),
                       normal_dist(0.0, 1.0));
  auto ly1update = affine_nonlin_update(tanh_fn(1.0, 1.0));
  auto ly1bp = affine_nonlin_bp(dtanh(1.0, 1.0));
  pr_Dsite.push_back(true);
  pr_Dconn.push_back(true);

  int lypool = ntw.AddLayer({6});
  ntw.ConnectLastLayer(masquedlist(ly1, {3}, {1}, {0}), 1.0);
  auto lypoolupdate = maxpool_update;
  auto lypoolbp = affine_nonlin_bp(d_id);
  pr_Dsite.push_back(true);
  pr_Dconn.push_back(false);

  int ly3 = ntw.AddLayer({12});
  ntw.ConnectLastLayer(alllayer(lypool), 0.1);
  auto ly3update = affine_nonlin_update(ReLU);
  auto ly3bp = affine_nonlin_bp(dReLU);
  pr_Dsite.push_back(true);
  pr_Dconn.push_back(true);

  int lydrop = ntw.AddLayer({12});
  ntw.ConnectLastLayer(directlist(ly3), 1.0);
  double p = 0.2;
  auto lydropupdate = dropout_update(&p);
  auto lydropbp = affine_nonlin_bp(d_id);
  pr_Dsite.push_back(true);
  pr_Dconn.push_back(false);  // connection does not play role

  int resly = ntw.AddLayer({8});
  ntw.ConnectLastLayer(alllayer(lydrop), 0.1);
  auto reslyupdate = affine_nonlin_update(exp_fn(1.0, 0.5));
  auto reslybp = affine_nonlin_bp(dexp(1.0, 0.5));
  pr_Dsite.push_back(true);
  pr_Dconn.push_back(true);

  int lossly = ntw.AddLayer({1});
  ntw.ConnectLastLayer(alllayer(resly), 1.0);
  std::vector<double> expected_output(8, 1.0);
  auto lossupdate = pnorm_loss(1, &expected_output);
  // auto lossupdate = KL_loss(&expected_output);  // pnorm_loss(1, &expected_output);
  auto lossbp = d_pnorm_loss(1, &expected_output);
  // auto lossbp = d_KL_loss(&expected_output);  // d_pnorm_loss(1, &expected_output);
  pr_Dsite.push_back(true);
  pr_Dconn.push_back(false);  // connection does not play role

  // ----------------------------------------------------------------------- //
  auto update = [=](Network* L) {
    randiv.set_seed(10);  // to be able to reproduce the same runs
    L->forallSitesinLayer(ly1, [=](int n){   ly1update(L, n); });
    L->forallSitesinLayer(lypool, [=](int n){   lypoolupdate(L, n); });
    L->forallSitesinLayer(ly3, [=](int n){   ly3update(L, n); });
    L->forallSitesinLayer(lydrop, [=](int n){   lydropupdate(L, n); });
    L->forallSitesinLayer(resly, [=](int n){    reslyupdate(L, n); });
    L->forallSitesinLayer(lossly, [=](int n){   lossupdate(L, n); });
  };

  update(&ntw);
  ntw.save("test_bp_numeric_all_1.ntw");

  // ----------------------------------------------------------------------- //
  DNetwork bpntw(&ntw);

  auto backpropagate = [=](DNetwork *BPL) {
    Network *L = BPL->associatedNetwork();
    BPL->Dsite(L->nSites()-1) = 1.0;  // start with unit derivative
    L->forallSitesinLayer(lossly, [=](int n){ lossbp(BPL, n); });
    L->forallSitesinLayer(resly, [=](int n){ reslybp(BPL, n); });
    L->forallSitesinLayer(lydrop, [=](int n){ lydropbp(BPL, n); });
    L->forallSitesinLayer(ly3, [=](int n){ ly3bp(BPL, n); });
    L->forallSitesinLayer(lypool, [=](int n){ lypoolbp(BPL, n); });
    L->forallSitesinLayer(ly1, [=](int n){ ly1bp(BPL, n); });
  };

  backpropagate(&bpntw);

  // ----------------------------------------------------------------------- //
  // this program tests the backpropagation results against
  // numerical derivation: to this latter we need partial_update
  auto partial_update = [=](Network* L, int ly) {
    randiv.set_seed(10);  // the same seed as before
    if (ly<= ly1) L->forallSitesinLayer(ly1, [=](int n){ ly1update(L, n); });
    if (ly<= lypool) L->forallSitesinLayer(lypool, [=](int n){ lypoolupdate(L, n); });
    if (ly<= ly3) L->forallSitesinLayer(ly3, [=](int n){ ly3update(L, n); });
    if (ly<= lydrop) L->forallSitesinLayer(lydrop, [=](int n){ lydropupdate(L, n); });
    if (ly<= resly) L->forallSitesinLayer(resly, [=](int n){ reslyupdate(L, n); });
    if (ly<= lossly) L->forallSitesinLayer(lossly, [=](int n){ lossupdate(L, n); });
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
  // basely, ly1, ly2, lossly

  int lynprev = 0;
  for (int n = ntw.nSites()-1; n>= 0; --n) {
    int lyn = ntw.sitelayerID(n);
    if (lyn != lynprev) std::cout << " ----------------------------\n";
    lynprev = lyn;
    std::cout << n << "(" << lyn << ")";
    std::cout << ": axon=" << ntw.site(n);
    std::cout << std::endl;
    if (pr_Dsite[lyn]) {
      std::cout << "\tDaxon=" << bpntw.Dsite(n) << " -- "
                          << ndntw.z[n];
      error2+= sq(bpntw.Dsite(n)-ndntw.z[n]);
      nerr++;
      std::cout << std::endl;
    }
    if (pr_Dconn[lyn]) {
      for (int cid = 0; cid < ntw.nConnections(n); cid++) {
        std::cout << "\t\t[" << cid << "]";
        std::cout << "\tconn=" << ntw.connection(n, cid);
        std::cout << "\tDconn=" << bpntw.Dconn(n, cid);
        std::cout << " -- " << ndntw.dconn[ ntw.getconnID(n, cid)] << std::endl;
        error2+= sq(bpntw.Dconn(n, cid)-ndntw.dconn[ ntw.getconnID(n, cid)]);
        nerr++;
      }
    }
  }
  std::cout << "Average error = " << sqrt(error2/nerr) << '\n';

  return 0;
}
