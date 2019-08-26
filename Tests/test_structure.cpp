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

  cout << "Structure testing:\n";
  Network ntw;
  // the base layer
  int ly0 = ntw.AddLayer({4, 4});
  // it has no connections, we set its values by hand
  ntw.forallSitesinLayer(ly0, [&](int n){
    ntw.site(n)= 1.0+ 0.0*normal_dist(0.0, 1.0)();
  });

  // first layer
  int ly1 = ntw.AddLayer({3, 3});
  // connect to the first layer fully
  ntw.ConnectLastLayer(alllayer(ly0), 1.0);
  // ntw.ConnectLastLayer(alllayer(ly0), normal_dist(0.0, 1.0));

  int ly2 = ntw.AddLayer({2, 2});
  ntw.ConnectLastLayer(
    masquedlist(ly1, {2, 3}, {1, 0}, {0, 0}),
    1.0);
  // ntw.ConnectLastLayer(
  //   masquedlist(ly1, {2, 3}, {1, 0}, {0, 0}),
  //   normal_dist(0.0, 1.0));

  std::cout << "nSites()=" << ntw.nSites() << std::endl;
  std::cout << "nLayers()=" << ntw.nLayers() << std::endl;
  std::cout << "nallConnections()=" << ntw.nallConnections() << std::endl;
  for (uint lyn = 0; lyn < ntw.nLayers(); ++lyn)
    std::cout << "nSitesinLayer(" << lyn << ")="
              << ntw.nSitesinLayer(lyn) << std::endl;
  for (uint sn = 0; sn < ntw.nSites(); ++sn)
    std::cout << "nConnections(" << sn << ")="
              << ntw.nConnections(sn) << std::endl;
  std::cout << "--------------------------\n";
  std::cout << ntw.sitelayerID(8) << std::endl;
  std::cout << ntw.sitelayerID(18) << std::endl;
  std::cout << ntw.sitelayerID(27) << std::endl;

  std::cout << ntw.getconnSite(8) << std::endl;
  std::cout << ntw.getconnSite(145) << std::endl;
  std::cout << ntw.getconnSite(165) << std::endl;

  auto ly1update = affine_nonlin_update(ReLU);
  auto ly2update = affine_nonlin_update(tanh_fn(1.0, 1.0));
  auto update = [=](Network* L) {
    // ly0 is not updated
    L->forallSitesinLayer(ly1, [=](int n){ ly1update(L, n); });
    L->forallSitesinLayer(ly2, [=](int n){ ly2update(L, n); });
  };
  update(&ntw);

  ntw.save("mynet.ntw", true);  // save verbatim

  Network ntw1;
  ntw1.load("mynet.ntw", true);  // load and create
  ntw1.save("mynet1.ntw", true);

  Network ntw2;
  int ly20 = ntw2.AddLayer({4, 4});
  int ly21 = ntw2.AddLayer({3, 3});
  ntw2.ConnectLastLayer(alllayer(ly20), normal_dist(0.0, 1.0));
  int ly22 = ntw2.AddLayer({2, 2});
  ntw2.ConnectLastLayer(
    masquedlist(ly1, {2, 3}, {1, 0}, {0, 0}),
    normal_dist(0.0, 1.0));
  ntw2.load("mynet.ntw", false);  // load only
  ntw1.save("mynet2.ntw", true);

  // test also a larger layer (but do not save it)
  int lyl1 = ntw.AddLayer({1000});
  int lyl2 = ntw.AddLayer({1000});
  ntw.ConnectLastLayer(alllayer(lyl1), normal_dist(0.0, 1.0));

  // test convolutional averaging
  Network ntw3;
  int ly30 = ntw3.AddLayer({3, 3, 3});  // three channels
  int ly31 = ntw3.AddLayer({2, 2, 3});  // three channels
  Shape shp = ntw3.layerShape(ly31);
  ntw3.ConnectLastLayer(
    masquedlist(ly30, {2, 2, 3}, {1, 1, 0}, {0, 0, 0}),
    normal_dist(0.0, 1.0));
  // for (uint rsn = 0; rsn < ntw3.nSitesinLayer(ly31); ++rsn) {
  //     std::vector<int> p = shp.components(rsn);
  //     int chn = p.back();
  //     int sn = ntw3.getsiteID(ly31, rsn);
  //     int ncmax = ntw3.nConnections(sn);
  //     for (uint cn = 0; cn < ncmax; ++cn)
  //       ntw3.connection(sn, cn) = chn;
  // }
  ntw3.average_weights(ly31);
  ntw3.save("mynet3.ntw", true);

  return 0;
}
