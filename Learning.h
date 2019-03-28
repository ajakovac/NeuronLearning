/* Copyright (C) AJ
 * Written by A. Jakovac 2019 */
#ifndef LEARNING_H_
#define LEARNING_H_

#include "Structure.h"
#include "Backprop.h"

// In this file we collect all the basic information needed to perform
// learning.

auto steepest_descent = [](DNetwork *BPN, int lynum, double lrate) {
    Network *N = BPN->associatedNetwork();
    N->applytoLayer(lynum, [&](int n) {
        N->bias(n) -= lrate* BPN->Dbias(n);
        int cnnmax = N->nConnections(n);
        for (int cnn = 0; cnn < cnnmax; cnn++) {
            N->siteConnection(n, cnn) -= lrate* BPN->Dconn(n, cnn);
        }
    });
};

class ConjugateGadient {
 public:
    ConjugateGadient(DNetwork* DNin) : DN(DNin) {
        N = DN->associatedNetwork();
        P = new DNetwork(N);
    }
    ~ConjugateGadient() {
        delete P;
    }
    Network *N;
    DNetwork *DN;
    DNetwork *P;
};


#endif  // LEARNING_H_
