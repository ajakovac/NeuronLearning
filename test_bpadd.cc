/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <string>
#include "Structure.h"
#include "Backprop.h"
#include "Update.h"

int main(int argc, char const *argv[]) {
  randiv.set_seed(10);

  Network ntw;
  int basely = ntw.AddLayer({4, 4});

  int  lyres = ntw.AddLayer({5});
  ntw.ConnectLastLayer(alllayer(basely), normal_cf(0.0, 0.01));
  auto lyresupdate = affine_nonlin_update(exp_fn(1.0, 0.5));
  auto lyresbp = affine_nonlin_bp(dexp(1.0, 0.5));

  int lossly = ntw.AddLayer({1});
  ntw.ConnectLastLayer(alllayer(lyres), const_cf(1.0));
  std::vector<double> expected_output(10, 0);
  auto lossupdate =  KL_loss(&expected_output);
  auto lossbp = d_KL_loss(&expected_output);

  // ----------------------------------------------------------------------- //

  auto update = [=](Network *N) {
    N->applytoLayer(lyres, [=](int n){ lyresupdate(N, n);});
    N->applytoLayer(lossly, [=](int n){ lossupdate(N, n);});
  };

  BPNetwork bpntw(&ntw);
  BPNetwork bpntw_collect(&ntw);

  auto backpropagate = [=](BPNetwork *BPN) {
    Network *N = BPN->associatedNetwork();
    BPN->Dsite(N->nSites()-1) = 1.0;  // start with unit derivative
    N->applytoLayer(lossly, [=](int n){ lossbp(BPN, n);});
    N->applytoLayer(lyres, [=](int n){ lyresbp(BPN, n);});
  };

  auto learn = [=](BPNetwork *BPN, int lynum, double lrate) {
    Network *N = BPN->associatedNetwork();
    N->applytoLayer(lynum, [&](int n) {
      for (int cnn = 0; cnn < N->nConnections(n); cnn++) {
        N->siteConnection(n, cnn) -= lrate* BPN->Dconn(n, cnn);
      }
    });
  };

  std::vector<double> diff(ntw.nallConnections(), 0);
  std::cout << "Network details: " << std::endl;
  std::cout << "nLayers = " << ntw.nLayers()  << std::endl;
  std::cout << "nSites = " << ntw.nSites()  << std::endl;
  std::cout << "nConnections = " << ntw.nallConnections()  << std::endl;

  //////////////////////////////////////////////////////////////////////

  ntw.applytoLayer(basely, [&](int n){ ntw[n] = normal_dist(0, 1.0)();});
  for (double& x : expected_output) x = 0.01;
  expected_output[2] = 1.0;
  normalize(&expected_output);

  update(&ntw);
  bpntw.reset();
  backpropagate(&bpntw);

  bpntw_collect.reset();
  bpntw_collect += bpntw;

  for (int cid = 0; cid < ntw.nallConnections(); cid++)
    if (bpntw.Dconn(cid) != bpntw_collect.Dconn(cid)) {
      std::cout << "PROBLEM!!" << std::endl;
      return -1;
    }

  for (int cid = 0; cid < ntw.nallConnections(); cid++) {
    diff[cid] = bpntw.Dconn(cid);
  }

  //////////////////////////////////////////////////////////////////////

  ntw.applytoLayer(basely, [&](int n){ ntw[n] = normal_dist(0, 1.0)();});
  for (double& x : expected_output) x = 0.01;
  expected_output[3] = 1.0;
  normalize(&expected_output);

  update(&ntw);
  bpntw.reset();
  backpropagate(&bpntw);
  bpntw_collect += bpntw;

  for (int cid = 0; cid < ntw.nallConnections(); cid++) {
    diff[cid] += bpntw.Dconn(cid);
  }

  for (int cid = 0; cid < ntw.nallConnections(); cid++)
    if (diff[cid] != bpntw_collect.Dconn(cid)) {
      std::cout << cid << " --" << diff[cid] << " vs "
                << bpntw_collect.Dconn(cid)
                << std::endl;
    }


  return 0;
}
