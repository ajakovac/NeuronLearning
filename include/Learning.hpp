/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2019 */
#ifndef INCLUDE_LEARNING_HPP_
#define INCLUDE_LEARNING_HPP_

#include <vector>
#include "Structure.hpp"
#include "Backprop.hpp"

// In this file we collect all the basic information needed to perform
// learning.

// performs scalar product of the connections of A and B at lyn layer level
inline double scalar_product_conn(int lyn, DNetwork *A, DNetwork* B) {
    double rn2 = 0.0;
    Network *N = A->associatedNetwork();
    N->forallSitesinLayer(lyn, [&](int sn) {
        int cnmax = N->nConnections(sn);
        for (int cn = 0; cn < cnmax; ++cn)
            rn2 += A->Dconn(sn, cn)*B->Dconn(sn, cn);
    });
    return rn2;
}

class SteepestDescent {
 public:
    Network *N;
    DNetwork *DN;
    explicit SteepestDescent(DNetwork *DNin) : DN(DNin) {
        N = DN->associatedNetwork();
    }
    void learn(int lyn, double lrate) {
        N->forallSitesinLayer(lyn, [&](int n) {
            int cnnmax = N->nConnections(n);
            for (int cnn = 0; cnn < cnnmax; cnn++) {
                N->connection(n, cnn) -= lrate* DN->Dconn(n, cnn);
            }
        });
    }
    void startlearncycle() {}
};

class StochGrad {
 public:
    Network *N;
    DNetwork *DN;
    double lrfactor;   // the learning rate reduction factor
    double lrdecrease;  // the lrfactor reduction factor
    double invmaxgrad;   // inverse of the maximal modulus of the gradient
    explicit StochGrad(DNetwork *DNin) : DN(DNin) {
        N = DN->associatedNetwork();
        invmaxgrad = 0.005;
        lrfactor = 1.0;
        lrdecrease = 1.0;  // 0.9999;
    }

    void learn(int lyn, double lrate) {
        double x = 1.0;
        if (invmaxgrad > 0) {
            double rn2 = scalar_product_conn(lyn, DN, DN);
            x = 1.0/(1.0 + invmaxgrad*sqrt(rn2));
        }
        N->forallConnectionsinLayer(lyn, [&](int sn, int cn) {
            N->connection(sn, cn) -= lrate*lrfactor*x*DN->Dconn(sn, cn);
        });
    }
    void startlearncycle() {
        lrfactor *= lrdecrease;
    }
};


class ADAM {
 public:
    Network *N;
    DNetwork *DN;
    double beta1;
    double beta2;
    DNetwork *G;
    DNetwork *V;
    double beta1n;  // nth power of beta1
    double beta2n;  // nth power of beta2
    explicit ADAM(DNetwork* DNin)
    : DN(DNin) {
        N = DN->associatedNetwork();
        G = new DNetwork(N);
        G->reset();
        V = new DNetwork(N);
        V->reset();
        beta1 = 0.9;
        beta1n = 1.0;
        beta2 = 0.999;
        beta2n = 1.0;
    }
    void learn(int lyn, double lrate) {
        double A1 = 1.0/(1-beta1n);
        double A2 = 1.0/(1-beta2n);
        N->forallConnectionsinLayer(lyn, [&](int sn, int cn) {
            double x = DN->Dconn(sn, cn);
            G->Dconn(sn, cn) = beta1*G->Dconn(sn, cn) + (1-beta1)*x;
            V->Dconn(sn, cn) = beta2*V->Dconn(sn, cn) + (1-beta2)*x*x;
            N->connection(sn, cn) -=
                lrate*A1*G->Dconn(sn, cn)/(ep+A2*sqrt(V->Dconn(sn, cn)));
        });
    }
    void startlearncycle() {
        beta1n *= beta1;
        beta2n *= beta2;
    }

 private:
    double ep = 1e-8;
};



class ConjugateGadient {
 public:
    explicit ConjugateGadient(DNetwork* DNin)
    : DN(DNin), z(0), updcnt(0) {
        N = DN->associatedNetwork();
        P = new DNetwork(N);
        P->reset();
        z.resize(N->nLayers(), 1.0);
        updcnt.resize(N->nLayers(), 0);
    }
    ~ConjugateGadient() {
        delete P;
    }
    Network *N;
    DNetwork *DN;
    DNetwork *P;
    std::vector<double> z;  // lyer number sized vector to store the old P.R
    int restorenm = 20;  // after restornnm update fall back to P=R
    std::vector<int> updcnt;  // current status in layers

    void learn(int lyn, double lrate) {
        // teach connections in layer lyn;
        double rn2 = scalar_product_conn(lyn, DN, DN);
        double rnpn1 = scalar_product_conn(lyn, DN, P);
        // double beta = rn2/z[lyn];
        double beta = rn2/(z[lyn] - rt*rnpn1);
        if (updcnt[lyn]-- == 0) {
            updcnt[lyn] = restorenm;
            beta = 0;  // fall back to P=R;
        }
        N->forallSitesinLayer(lyn, [&](int sn) {
            int cnmax = N->nConnections(sn);
            for (int cn = 0; cn < cnmax; ++cn)
                P->Dconn(sn, cn) = DN->Dconn(sn, cn)+beta*P->Dconn(sn, cn);
        });
        double pnorm2 = scalar_product_conn(lyn, P, P);
        // z[lyn] = rn2;
        z[lyn] = scalar_product_conn(lyn, DN, P);
        double alpha = scalar_product_conn(lyn, DN, P)/(ep+sqrt(pnorm2*rn2));
        N->forallSitesinLayer(lyn, [&](int sn) {
            int cnmax = N->nConnections(sn);
            for (int cn = 0; cn < cnmax; ++cn)
                N->connection(sn, cn) -=
                    lrate*alpha*P->Dconn(sn, cn);
        });
    }
    void startlearncycle() {}

 private:
    double pnorm2;
    double ep = 1e-6;  // regulator
    double rt = 0.1;   // parameter
};

class ConjugateGadient_xtd {
 public:
    explicit ConjugateGadient_xtd(DNetwork* DNin)
    : DN(DNin), z(0), updcnt(0) {
        N = DN->associatedNetwork();
        P = new DNetwork(N);
        P->reset();
        z.resize(N->nLayers(), 1.0);
        updcnt.resize(N->nLayers(), 0);
    }
    ~ConjugateGadient_xtd() {
        delete P;
    }
    Network *N;
    DNetwork *DN;
    DNetwork *P;
    std::vector<double> z;  // lyer number sized vector to store the old P.R
    int updatemax = 30;  // so many times update in one direction
    std::vector<int> updcnt;  // current status in layers
    void learn(int lyn, double lrate) {
        // teach connections in layer lyn;
        double rn2 = scalar_product_conn(lyn, DN, DN);
        if (updcnt[lyn]-- == 0) {
            updcnt[lyn] = updatemax;
            double rnpn1 = scalar_product_conn(lyn, DN, P);
            double beta = rn2/(z[lyn] - rnpn1);
            N->forallSitesinLayer(lyn, [&](int sn) {
                int cnmax = N->nConnections(sn);
                for (int cn = 0; cn < cnmax; ++cn)
                    P->Dconn(sn, cn) = DN->Dconn(sn, cn)+beta*P->Dconn(sn, cn);
            });
        }
        double pnorm2 = scalar_product_conn(lyn, P, P);
        z[lyn] = scalar_product_conn(lyn, DN, P);
        double alpha = z[lyn]/(ep+sqrt(pnorm2*rn2));
        N->forallSitesinLayer(lyn, [&](int sn) {
            int cnmax = N->nConnections(sn);
            for (int cn = 0; cn < cnmax; ++cn)
                N->connection(sn, cn) -=
                    lrate*alpha*P->Dconn(sn, cn);
        });
    }
    void startlearncycle() {}

 private:
    double pnorm2;
    double ep = 1e-8;  // regulator
};

#endif  // INCLUDE_LEARNING_HPP_
