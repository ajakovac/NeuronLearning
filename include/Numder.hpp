/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#ifndef INCLUDE_NUMDER_HPP_
#define INCLUDE_NUMDER_HPP_

#include <vector>
#include "Structure.hpp"

// In this file we numerically determine all the derivatives

class Numder {
 public:
  Network *L;  // the network we associate backpropagation
  // vectors of number_of_connections size
  std::vector<double> dconn;  // to store the derivative wrt. the connections

  // vectors of number_of_axons size
  std::vector<double> z;  // to store the derivative wrt. the axons
  // numeric derivation needs a partial updat function that updates the network
  // starting from a given layer
  std::function< void(Network*, int) >  partial_update;

  // in the constructor we set no bp propagation
  Numder(Network *Lin, std::function< void(Network*, int) > pupd) :
    L(Lin), dconn(Lin->nallConnections()),
    z(Lin->nSites(), 0), partial_update(pupd) {}

  void getD() {
    static const double dx = 1e-5;
    partial_update(L, 0);  // update the complete network
    int lossID = L->nSites()-1;
    z[lossID] = 1;
    double y = L->site(lossID);
    for (int n = lossID-1; n>= 0; --n) {
      int ly = L->sitelayerID(n);
      double x = L->site(n);
      L->site(n) +=dx;
      partial_update(L, ly+1);
      z[n] = (L->site(lossID)-y)/dx;
      L->site(n)-= dx;
      partial_update(L, ly+1);
      for (int cid = 0; cid < L->nConnections(n); cid++) {
        x = L->connection(n, cid);
        L->connection(n, cid) += dx;
        partial_update(L, ly);
        dconn[ L->getconnID(n, cid) ] = (L->site(lossID)-y)/dx;
        L->connection(n, cid)-= dx;
        partial_update(L, ly);
      }
    }
  }
};

#endif  // INCLUDE_NUMDER_HPP_
