/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#ifndef BACKPROP_H_
#define BACKPROP_H_

#include "Structure.h"
#include <vector>

// The backpropagation is a skin to the Network, it is not necessary for the
// basic functionalities.
// It must be created giving the Network it will work on.
// As new data it contains z and dconn which are the derivatives with respect to
// the axons and connections, respectively.
// A new function data should provide f' as a function of the present axon.
// BPNetwork then have a single functionality, which performs the
// backpropagation.

class BPNetwork {
  // here are the data: all hidden
  Network *L;  // the network we associate backpropagation
  // vectors of number_of_connections size
  std::vector<double> dconn;  // to store the derivative wrt. the connections

  // vectors of number_of_axons size
  std::vector<double> z;  // to store the derivative wrt. the axons
  std::vector<double> doffset;  // to store the derivative wrt. the offsets

  // the core of the backpropagation method: how to propagate the derivative
  // at a given site to earlier sites. Each layer should have such a function

 public:
  // first the functions to reach data
  Network* associatedNetwork() { return L;}
  double& Dsite(int sn) {return z[sn];}
  double& Doffset(int sn) { return doffset[sn]; }
  double& Dconn(int sn, int cn) { return dconn[ L->getconnID(sn, cn)]; }
  double& Dconn(int cid) { return dconn[cid]; }

  // in the constructor we set no bp propagation
  explicit BPNetwork(Network *Lin) :
    L(Lin), dconn(Lin->nallConnections(), 0),
    z(Lin->nSites(), 0), doffset(Lin->nSites(), 0) {}

  // the gradient computing function accumlates the derivatives, so for
  // new learning process we have to reset all the derivatives.
  void reset() {
    for (auto& y : dconn) y = 0;
    for (auto& x : z) x = 0;
    for (auto& x : doffset) x = 0;
  }

  void operator+=(const BPNetwork &bpadd) {
    for (int n = 0; n < dconn.size(); ++n) dconn[n] += bpadd.dconn[n];
    for (int n = 0; n < z.size(); ++n) z[n] += bpadd.z[n];
    for (int n = 0; n < doffset.size(); ++n) doffset[n] += bpadd.doffset[n];
  }
};

// the most usual layer type that has a linear transformation followed by a
// nonlinear function. Its backpropagation can be described by the following
// function, that uses the derivative of the nonlinear function with an argument
// of the function value
auto affine_nonlin_bp = [](std::function< double(double) > df) {
  return [=](BPNetwork* BL, int an) {
    Network* L = BL->associatedNetwork();
    double dfn = df(L->axon(an));
    BL->Doffset(an) += BL->Dsite(an)*dfn;  // refresh the offset derivative
    for (int cn = 0; cn< L->nConnections(an); cn++) {
      // this is the site we are connected to
      int sn = L->siteConnectedSite(an, cn);
      // refresh the connection derivative
      BL->Dconn(an, cn) += BL->Dsite(an)*dfn* L->axon(sn);
      // backpropagate information to the precedeing sites
      BL->Dsite(sn) += BL->Dsite(an)*dfn*L->siteConnection(an, cn);
    }
  };
};

//////////////////////////////////////////////////////////////////////////
// Here are some examples for the df functions

auto dReLU = [](double x) { return x > 0? 1:0; };

auto dtanh = [](double hght, double wdth) {
  return [=](double x) {return wdth/hght*(hght*hght-x*x); };
};

auto dexp = [](double hght, double wdth) {
  return [=](double x) {return wdth*x;};
};

// derivatives of the maxpool and dropout
// can also be used as a normal affine_nonlin_bp
// with the identity function derivative:
auto d_id = [](double x){return 1;};


//////////////////////////////////////////////////////////////////////////
// the loss function derivatives require comparison with an expected output

// derivative of pnorm_loss
auto d_pnorm_loss = [](int p, std::vector<double>* wanted_output) {
  return [=](BPNetwork* BL, int sn) {
    Network* L = BL->associatedNetwork();
    for (int i = 0; i < L->nConnections(sn); i++) {
      double xi = L->siteConnectedValue(sn, i) - (*wanted_output)[i];
      if (p > 1) {
        BL->Dsite(L->siteConnectedSite(sn, i)) +=
          BL->Dsite(sn)*xi* std::pow(std::fabs(xi), p-2);
      } else {
        BL->Dsite(L->siteConnectedSite(sn, i)) +=
          BL->Dsite(sn)*(xi > 0 ? 1: -1);
      }
    }
  };
};

// derivative of normalized_pnorm_loss
auto d_normalized_pnorm_loss = [](int p, std::vector<double>* wanted_output) {
  return [=](BPNetwork* BL, int sn) {
    Network* L = BL->associatedNetwork();
    double nrm = 0;
    int nconns = L->nConnections(sn);
    for (int i = 0; i < nconns; i++)
      nrm += L->siteConnectedValue(sn, i);
    double CC = 0;
    for (int i = 0; i < nconns; i++) {
      double xi = L->siteConnectedValue(sn, i);
      double dxi = xi/nrm - (*wanted_output)[i];
      CC+= xi * dxi* std::pow(std::fabs(dxi), p-2);
    }
    for (int i = 0; i < nconns; i++) {
      double dxi = L->siteConnectedValue(sn, i)/nrm - (*wanted_output)[i];
      BL->Dsite(L->siteConnectedSite(sn, i)) +=
        BL->Dsite(sn)*(dxi* std::pow(std::fabs(dxi), p-2) - CC/nrm) /nrm;
    }
  };
};

// derivative of KL_loss
auto d_KL_loss = [](std::vector<double>* wanted_output) {
  return [=](BPNetwork* BL, int sn) {
    Network* L = BL->associatedNetwork();
    double nrm = 0;
    int nconns = L->nConnections(sn);
    for (int i = 0; i < nconns; i++) {
      nrm += L->siteConnectedValue(sn, i);
    }
    for (int i = 0; i < nconns; i++) {
      double xi = L->siteConnectedValue(sn, i);
      double s = xi/(nrm* (*wanted_output)[i]);
      BL->Dsite(L->siteConnectedSite(sn, i)) +=
         BL->Dsite(sn)*(std::log(s*s)/2 - L->axon(sn)) /nrm;
    }
  };
};


#endif  // BACKPROP_H_