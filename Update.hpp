/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#ifndef UPDATE_H_
#define UPDATE_H_

#include <vector>
#include "Structure.h"


////////////////////////////////////////////////////////////////////////
// The logics of update is that we build up a network update function
// Here is an example:
// auto update = [&](Network *L){
//   L->applytoLayer( ly0, [=](int n){ ly0update(L,n); });
//   L->applytoLayer( ly1, [=](int n){ ly1update(L,n); });
//       ...
//   L->applytoLayer( lyn, [=](int n){ lynupdate(L,n); });
// }
// Then we simply run update(&ntw) for the actual network.
//
// What is worth to specialize are the update functions.
// Here are some examples for them.
//

// a big class of update functions are the combination of an affine and a
// nonlinear function. to define it we have to provide the nonlin function
auto affine_nonlin_update = [](auto f) {
  return [=](Network* N, int sn) {
    double res = N->bias(sn);
    auto cnfn = N->readConnection(sn);
    auto cnvfn = N->readConnectedValue(sn);
    int cnmax = N->nConnections(sn);
    for (int cn = 0; cn < cnmax; ++cn)
      res+= cnfn(cn)* cnvfn(cn);
    N->axon(sn) = f(res);
  };
};

// the f above can be something like these:
auto id_fn = [](double x){return x;};
auto ReLU = [](double x){return x > 0? x : 0;};

auto tanh_fn = [](double hght, double wdth) {
  return [=](double x){ return hght*std::tanh(wdth*x);};
};

auto exp_fn = [](double hght, double wdth) {
  return [=](double x){ return hght*std::exp(wdth*x);};
};

// we also can implement a pooling update function: choose the maximum of
// the input axons; here the connections play no role, they are used to
// sign the actual maximal value
auto maxpool_update = [](Network* N, int sn) {
  double res = N->siteConnectedValue(sn, 0);
  int si = 0;
  for (int i = 1; i < N->nConnections(sn); ++i) {
    double r1 = N->siteConnectedValue(sn, i);
    if (r1 > res) {
      res = r1;
      si = i;
    }
  }
  // we flag the maximum index connection by 1
  for (int i = 0; i < N->nConnections(sn); ++i)
    N->siteConnection(sn, i) = 0;
  N->siteConnection(sn, si) = 1;

  // and finally update the axon.
  N->axon(sn) = res;
};

// dropout is used to make the learning more robust
// the dropout probability can be changed externally (it is nonzero only
// for learning!)
auto dropout_update = [](double* p) {
  return [=](Network* N, int sn) {
    // there must be only one connection!
    double res = N->siteConnectedValue(sn, 0);
    double rnd = uniform_dist(0.0, 1.0)();
    if (rnd > *p) {  // no dropout; true for prob=0!
      N->siteConnection(sn, 0) = 1/(1-*p);  // compensate missing information
      N->axon(sn) = res/(1-*p);
    } else {  // dropout
      N->siteConnection(sn, 0) = 0;
      N->axon(sn) = 0;
    }
  };
};

//////////////////////////////////////////////////////////////////////////
// another big family of update functions are the loss functions: we compare
// the result of the last layer by the desired output. In these functions
// compatibility of sizes is not examined!

// simplest loss, using p-norm
auto pnorm_loss = [](int p, std::vector<double>* wanted_output) {
  return [=](Network* N, int sn) {
    double diff = 0.0;
    for (int i = 0; i < N->nConnections(sn); ++i) {
      double dx = std::fabs(N->siteConnectedValue(sn, i) - (*wanted_output)[i]);
      diff += std::pow(dx, p);
    }
    N->axon(sn) = diff / p;
  };
};

// to normalize the output and compare with a normalized sample vector
auto normalized_pnorm_loss = [](int p, std::vector<double>* wanted_output) {
  return [=](Network* N, int sn) {
    double nrm = 0;
    for (int i = 0; i < N->nConnections(sn); ++i)
      nrm += N->siteConnectedValue(sn, i);
    double diff = 0.0;
    for (int i = 0; i < N->nConnections(sn); ++i) {
      double dxi = N->siteConnectedValue(sn, i)/nrm - (*wanted_output)[i];
      diff += std::pow(std::fabs(dxi), p);
    }
    N->axon(sn) = diff / p;
  };
};

// The Kullback-Leibler divergence: always normalized
auto KL_loss = [](std::vector<double>* wanted_output) {
  return [=](Network* N, int sn) {
    double nrm = 0;
    for (int i = 0; i < N->nConnections(sn); ++i)
      nrm += N->siteConnectedValue(sn, i);
    double KLdiv = 0;
    for (int i = 0; i < N->nConnections(sn); ++i) {
      double x = N->siteConnectedValue(sn, i);
      double s = x/(nrm* (*wanted_output)[i]);
      KLdiv += x*std::log(s*s);
    }
    N->axon(sn) = KLdiv/(2*nrm);
  };
};

void normalize(std::vector<double> *v) {
  double nrm = 0;
  for (auto& x : *v) nrm+= x;
  for (auto& x : *v) x/= nrm;
}


#endif  // UPDATE_H_
