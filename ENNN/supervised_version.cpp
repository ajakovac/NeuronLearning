/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <string>
#include <chrono>
#include "../include/Structure.hpp"
#include "../include/Backprop.hpp"
#include "../include/mnist.hpp"
#include "../include/Update.hpp"
#include "../include/Learning.hpp"
//#include "cmake_variables.hpp"

#include<map>

//map< label, < vector of layernum, sitenum, n >  >
std::map < int, std::vector<std::vector<double> > > ideal_memory;
std::map < int, int> label_occurence_num;

std::string mnist_train_img = "../../Datasets/mnist/train_images";
std::string mnist_train_lbl = "../../Datasets/mnist/train_labels";;

// DOES NOT SAVE THE 0TH LAYER !! (there are no connections to save)
void SaveToIdealMemory(int label, const Network& netw) {
  if( ideal_memory.count(label) == 0) {
    for(uint i = 0; i < netw.nLayers(); ++i) {
      ideal_memory[label].push_back( std::vector<double> (netw.nSitesinLayer(i), 0) );
      for(uint j = 0; j < netw.nSitesinLayer(i); ++j)
	ideal_memory[label][i][j] = netw.axon(i, j); 
      //*0.1; // if we want a maximal value for Ideal  
    }
    label_occurence_num[label] = 1;
    return;
  }
  std::vector<std::vector<double> >& v = ideal_memory[label];
  int &N = label_occurence_num[label];
  for(uint i = 0; i < netw.nLayers(); ++i)
    for(uint j = 0; j < netw.nSitesinLayer(i); ++j)
      v[i][j] = (N*v[i][j] + netw.axon(i, j))/ (N + 1); // i' = (N*i + x)/(N+1)
  ++N;
}

void NormalizeConnections(Network& netw) {
  for(uint i = 0; i < netw.nSites(); ++i) {
    double norm = 0;
    for(uint cn = 0; cn < netw.nConnections(i); ++cn)
      norm += netw.siteConnection(i, cn)*netw.siteConnection(i, cn);
    norm = sqrt(norm);
    for(uint cn = 0; cn < netw.nConnections(i); ++cn)
      netw.siteConnection(i, cn) /= norm;
  }
}

double GetIdealOfSite(int label, int n, const Network& netw) {
  return ideal_memory[label][netw.sitelayerID(n)][netw.sitelayerIndex(n)];  
}

int main(int, char const *[]) try {
  randiv.set_seed(10);
  
  //---------------------------------------------------------------------------
  // Init dataset:
  
  MNIST_dataset mnist(mnist_train_img, mnist_train_lbl);
  int nrows = mnist.height;
  int ncols = mnist.width;
  // int numdat = 60000;
  int *data = mnist.data(Red); // the pictures are greyscale – using the r channel

  //---------------------------------------------------------------------------
  // Creating the network:
  
  Network ntw;
  int basely = ntw.AddLayer({nrows, ncols});
  
  int  ly1 = ntw.AddLayer({nrows-4, ncols-4});
  ntw.ConnectLastLayer( masquedlist(basely, {5,5}, {1,1}), normal_cf(0.0, 0.01));
  auto ly1update = affine_nonlin_update( tanh_fn(1, 1) );
  
  int ly2 = ntw.AddLayer( {ntw.layerShape(ly1)[0]-4, ntw.layerShape(ly1)[0] -4} );
  ntw.ConnectLastLayer( masquedlist(ly1, {5,5}, {1,1}), normal_cf(0.0, 0.01));
  auto ly2update = affine_nonlin_update( tanh_fn(1,1)  );

  NormalizeConnections(ntw);

  auto update = [=](Network *N) 
    {
      N->applytoLayer(ly1, [=](int n) {ly1update(N, n);});
      N->applytoLayer(ly2, [=](int n) {ly2update(N, n);});
    };

  //---------------------------------------------------------------------------
  // The iteration:
  
  int epochnum = 1;
  double curr_im = 0; // current image is 0
  double learning_rate = 0.01;
  double VAR = 0.1;
  
  // batchsize is not needed: we update similarly to SGD
  for (int epoch = 0; epoch < epochnum; ++epoch) {
    std::cerr << "epoch " << epoch << "\n";
    mnist.restart();
    while( mnist.next() ) {
      for (int n = 0; n < nrows*ncols; ++n)
	ntw.axon(basely, n) = data[n]/255.0;
      update(&ntw);
      SaveToIdealMemory(mnist.label(), ntw);
      // change the network tructure:
      
      auto lambda = [&](int n) {
      	  double sum = 0;
      	  for(uint cn = 0; cn < ntw.nConnections(n); ++cn)
      	    sum += ntw.siteConnectedValue(n, cn)*ntw.siteConnectedValue(n, cn);
      	  double alpha = learning_rate*(GetIdealOfSite(mnist.label(), n, ntw)
      					-ntw[n])/(sum+0.001);
      	  for(uint cn = 0; cn < ntw.nConnections(n); ++cn)
      	    ntw.siteConnection(n, cn) += alpha*ntw.siteConnectedValue(n, cn)*
      	      normal_dist(1, VAR)();
      };
      ntw.applytoLayer(ly1, lambda);
      ntw.applytoLayer(ly2, lambda);
      NormalizeConnections(ntw);
      if(int(curr_im) % 600 == 0)
	std::cerr << "                     \r" << int(curr_im/600.0) << "%";
      ++curr_im;
    }
  }

  for(auto& v : ideal_memory) {
    std::cout << "\n" << v.first << " distances:\n";
    for(auto& v2 : ideal_memory) {
      std::cout << "\t from " << v2.first << ": ";
      double D = 0;
      double d;
      for(uint i = 0; i < v.second.size(); ++i)
	for(uint j = 0; j < v.second[i].size(); ++j) {
	  d = v.second[i][j]-v2.second[i][j];
	  D += d*d;
	}
      std::cout << sqrt(D) << "\n";
    }
    
  ntw.save("netw01.ntw");

  }
    
  

  // // print the ideals:
  // for(auto& v : ideal_memory) {
  //   std::cout << "\n" << v.first << ": ";
  //   for(auto& v2 : v.second) {
  //     for(auto& d : v2)
  // 	std::cout << d << " ";
  //     std::cout << "\n----------------------------------------------------------\n";
  //   }
  // }
  
 }
 catch(Error & u) {
   std::cerr << u.error_message << std::endl;
 }
 catch(...) {
   std::cerr << "unknown error\n";
 }
