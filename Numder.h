/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#ifndef NUMDER_H_
#define NUMDER_H_

#include "Structure.h"
#include <vector>

// In this file we numerically determine all the derivatives

class Numder {
 public:
  Network *L;  // the network we associate backpropagation
  // vectors of number_of_connections size
  std::vector<double> dconn;  // to store the derivative wrt. the connections

  // vectors of number_of_axons size
  std::vector<double> z;  // to store the derivative wrt. the axons
  std::vector<double> doffset;  // to store the derivative wrt. the offsets
  // numeric derivation needs a partial updat function that updates the network
  // starting from a given layer
  std::function< void(Network*, int) >  partial_update;

  // in the constructor we set no bp propagation
  Numder(Network *Lin, std::function< void(Network*, int) > pupd) :
    L(Lin), dconn(Lin->nallConnections()),
    z(Lin->nSites(), 0), doffset(Lin->nSites(), 0),
    partial_update(pupd) {}

  void getD() {
    static const double dx = 1e-5;
    partial_update(L, 0);  // update the complete network
    int lossID = L->nSites()-1;
    z[lossID] = 1;
    double y = L->axon(lossID);
    for (int n = lossID-1; n>= 0; --n) {
        int ly = L->sitelayerID(n);
        double x = L->axon(n);
        L->axon(n) +=dx;
        partial_update(L, ly+1);
        z[n] = (L->axon(lossID)-y)/dx;
        L->axon(n)-= dx;
        partial_update(L, ly+1);
        for (int cid = 0; cid < L->nConnections(n); cid++) {
          x = L->siteConnection(n, cid);
          L->siteConnection(n, cid) += dx;
          partial_update(L, ly);
          dconn[ L->getconnID(n, cid) ] = (L->axon(lossID)-y)/dx;
          L->siteConnection(n, cid)-= dx;
          partial_update(L, ly);
        }
        x = L->siteOffset(n);
        L->siteOffset(n) +=dx;
        partial_update(L, ly);
        doffset[n] = (L->axon(lossID)-y)/dx;
        L->siteOffset(n)-= dx;
        partial_update(L, ly);
    }
  }
};

#endif  // NUMDER_H_
