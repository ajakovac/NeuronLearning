/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#ifndef LEARNING_H_
#define LEARNING_H_

#include "Structure.h"
#include "Backprop.h"

// This file is intented to describe the learning methods. In this version
// learning is applied to each site of the network!
// Thus all learning function will require for input
// a Network, a site, some of them DNetwork.

// The most simple learning process: sttepest descent method
class congrad {
 public:
    explicit congrad(int lr) : learningrate(lr) {}
    double learningrate;
    void operator()(Network *L, int sn, DNetwork *BPL) {
        L->bias(sn) -= learningrate* BPL->Dbias(sn);
        for (int cid = 0; cid < L->nConnections(sn); ++cid)
            L->siteConnection(sn, cid) -= learningrate* BPL->Dconn(sn, cid);
    }
};

// Th conjugate gradient method

// We define a function that equate the weights and derivatives
// of a layer: this is needed for a convolutional learning
void equate(Network *L, int lyid, DNetwork *BPL) {
    double avr = 0.0;
    L->applytoLayer(lyid, [&](int n) { avr+= L->bias(n); });
    avr /= L->nSitesinLayer(lyid);
    L->applytoLayer(lyid, [&](int n) { L->bias(n) = avr; });
    //
    avr = 0.0;
    L->applytoLayer(lyid, [&](int n) { avr+= BPL->Dbias(n); });
    avr /= L->nSitesinLayer(lyid);
    L->applytoLayer(lyid, [&](int n) { BPL->Dbias(n) = avr; });
    //
    avr = 0.0;
    L->applytoLayer(lyid, [&](int n) {
        for (int cid = 0; cid < L->nConnections(n); cid++)
            avr+= L->siteConnection(n, cid);
    });
    avr /= L->nSitesinLayer(lyid);
    L->applytoLayer(lyid, [&](int n) {
        for (int cid = 0; cid < L->nConnections(n); cid++)
            L->siteConnection(n, cid) = avr;
    });
    //
    avr = 0.0;
    L->applytoLayer(lyid, [&](int n) {
        for (int cid = 0; cid < L->nConnections(n); cid++)
            avr+= BPL->Dconn(n, cid);
    });
    avr /= L->nSitesinLayer(lyid);
    L->applytoLayer(lyid, [&](int n) {
        for (int cid = 0; cid < L->nConnections(n); cid++)
            BPL->Dconn(n, cid) = avr;
    });
}



#endif  // LEARNING_H_
