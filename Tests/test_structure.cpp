/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include<iostream>
#include "Error.hpp"
#include "Structure.hpp"
#include "Update.hpp"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  randiv.set_seed(10);
  cout << "Random number testing:\n";
  auto nd = normal_dist(0.0, 1.0);
  cout<< "Random number with Gaussian distribution: " << nd() << endl;
  auto c2 = cnst(2.0);
  cout<< "This is always 2 :) : " << c2() << endl;

  cout << "Structure testing:\n";
  Network ntw;
  // the base layer
  int ly0 = ntw.AddLayer({4, 4});
  // it has no connections, we set its values by hand
  ntw.applytoLayer(ly0, [&](int n){ ntw.axon(n)= normal_dist(0.0, 1.0)(); });

  // first layer
  int ly1 = ntw.AddLayer({3, 3});
  // connect to the first layer fully
  ntw.ConnectLastLayer(alllayer(ly0), normal_cf(0.0, 1.0));

  int ly2 = ntw.AddLayer({2, 2});
  ntw.ConnectLastLayer(masquedlist(ly1, {2, 3}, {1, 0}),
                       normal_cf(0.0, 1.0));



  auto ly1update = affine_nonlin_update(ReLU);
  auto ly2update = affine_nonlin_update(tanh_fn(1.0, 1.0));
  auto update = [=](Network* L) {
    // ly0 is not updated
    L->applytoLayer(ly1, [=](int n){ ly1update(L, n); });
    L->applytoLayer(ly2, [=](int n){ ly2update(L, n); });
  };
  update(&ntw);
  ntw.save("mynet.ntw");

  // test also a larger layer (but do not save it)
  int lyl1 = ntw.AddLayer({1000});
  int lyl2 = ntw.AddLayer({1000});
  ntw.ConnectLastLayer(alllayer(lyl1), normal_cf(0.0, 1.0));

  Network nt1;
  std::cout << "Read file: mynet.ntw" << std::endl;

  nt1.load("mynet.ntw");
  nt1.save("mynet_copy.ntw");
  return 0;
}
