/* Copyright (C)
 * Written by A. Jakovac 2019 */
#ifndef INCLUDE_NTWSCRIPT_HPP_
#define INCLUDE_NTWSCRIPT_HPP_
#include<iostream>
#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include "Error.hpp"
#include "Structure.hpp"
#include "Update.hpp"
#include "Backprop.hpp"

class Ntwscript {
 public:
  Network &ntw;
  Ntwscript(Network &ntw_in) : ntw(ntw_in), maxlynum(-1) {}

  // container for the expected output
  std::vector<double> expected_output;

  // containers for layer update and backpropagation
  std::vector< std::function< void(Network*, int)> > updatelist;
  std::vector< std::function< void(DNetwork*, int)>> backprlist;
  int maxlynum;  // maximal layer number
  int result() {return maxlynum-1;}
  int loss() {return maxlynum;}
  std::vector<bool> is_conv;  // if the layer is convolution

  void update(Network *N) {
    // going forward, the first layer is not updated
    for (int lyn = 1; lyn <= maxlynum; ++lyn)
      N->forallSitesinLayer(lyn, [=](int n){ updatelist[lyn](N, n);});
  }
  void backpropagate(DNetwork *DN) {
    Network *N = DN->associatedNetwork();
    DN->Dsite(N->nSites()-1) = 1.0;  // start with unit derivative
    // going back, the first layer is not affected
    for (int lyn = maxlynum; lyn > 0; --lyn) {
      N->forallSitesinLayer(lyn, [=](int n){ backprlist[lyn](DN, n);});
    }
  }


  int read_scriptfile(std::string filename) {
    auto relu_update = affine_nonlin_update(ReLU);
    auto relu_backpr = affine_nonlin_bp(dReLU);
    auto tanh_update = affine_nonlin_update(tanh_fn(1.0, 1.0));
    auto tanh_backpr = affine_nonlin_bp(dtanh(1.0, 1.0));
    auto exp_update = affine_nonlin_update(exp_fn(1.0, 0.5));
    auto exp_backpr = affine_nonlin_bp(dexp(1.0, 0.5));
 
    std::ifstream scriptfile(filename);
    if (!scriptfile.is_open()) throw(Error("File is unreadable\n"));

    while (!scriptfile.eof()) {
      inp = read_name(scriptfile);
      if (inp.compare("layer") == 0) {
        std::cout << "layer:\n";

        // read layer geometry
        std::vector<int> v = read_intvector(scriptfile);
        if (v.size() == 0) throw("bad shape for a layer\n");
        int lyn = ntw.AddLayer(Shape(v));
        std::cout << " - layer number = " << lyn << std::endl;
        maxlynum = lyn;
        std::cout << " - shape = " << ntw.layerShape(lyn) << std::endl;

        // read connection type
        std::cout << " - connection type: ";
        inp = read_name(scriptfile);
        if (inp.compare("none") == 0) {
          std::cout << "none\n";
          is_conv.push_back(false);
        } else if (inp.compare("full") == 0) {
          std::cout << "fully connected\n";
          is_conv.push_back(false);
          if (lyn == 0) throw("first layer cannot be connected\n");
          ntw.ConnectLastLayer(alllayer(lyn-1), normal_dist(0.0, 0.01));
        } else if (inp.compare("masq") == 0 || inp.compare("conv") == 0) {
          if (inp.compare("masq") == 0) {
            std::cout << "masqued\n";
            is_conv.push_back(false);
          } else {
            std::cout << "convolution\n";
            is_conv.push_back(true);
          }
          if (lyn == 0) throw("first layer cannot be connected\n");
          v = read_intvector(scriptfile);
          v.push_back(ntw.layerShape(lyn-1).back());  // last dimension: channel
          std::cout << " - window geometry:" << v << std::endl;
          if (v.size() != ntw.layerShape(lyn-1).size())
            throw("size mismatch in convolution\n");
          std::vector<int> stride(v.size(), 1);
          stride.back() = 0;
          std::cout << " - window stride:" << stride << std::endl;
          std::vector<int> offset(v.size(), 0);
          std::cout << " - window offset:" << offset << std::endl;
          ntw.ConnectLastLayer(
            masquedlist(lyn-1, Shape(v), Shape(stride), Shape(offset)),
            normal_dist(0.0, 0.01));
        } else {
          throw("unknown connection type\n");
        }

        // read activation function
        std::cout << " - activation function: ";
        inp = read_name(scriptfile);
        if (inp.compare("ReLU") == 0) {
          std::cout << "ReLU\n";
          updatelist.push_back(relu_update);
          backprlist.push_back(relu_backpr);
        } else if (inp.compare("tanh") == 0) {
          std::cout << "tanh\n";
          updatelist.push_back(tanh_update);
          backprlist.push_back(tanh_backpr);
        } else if (inp.compare("exp") == 0) {
          std::cout << "exp\n";
          updatelist.push_back(exp_update);
          backprlist.push_back(exp_backpr);
        } else {
          throw("unknown activation function\n");
        }

      } else {
        if (!scriptfile.eof()) throw("unknown command\n");
      }
    }

    // finally add a loss layer with KL divergence
    std::cout << "layer: KL loss\n";
    ntw.AddLayer({1});
    ntw.ConnectLastLayer(alllayer(maxlynum), 1.0);
    expected_output.resize(ntw.layerShape(maxlynum).vol(), 0);
    auto lossupdate =  KL_loss(&expected_output);
    auto lossbp = d_KL_loss(&expected_output);
    updatelist.push_back(lossupdate);
    backprlist.push_back(lossbp);
    is_conv.push_back(false);
    maxlynum++;
    std::cout << "Max layer number = " << maxlynum << std::endl;
    return maxlynum;
  }

 private:
  std::string inp;
  std::vector<int> read_intvector(std::ifstream &ifs) {
    ifs.clear();
    std::vector<int> out;
    char c;
    ifs >> c;
    if (c != '(') {
      ifs.unget();
      ifs.clear();
      return out;
    }
    int nx;
    while (ifs >> nx) {
      out.push_back(nx);
      ifs >> c;
      if (ifs.eof()) throw("bad vector format");
      if (c != ',' && c != ')') throw("bad delimiter");
    }
    if (c != ')') throw("bad vector format");
    // ifs.clear();
    return out;
  }
  std::string read_name(std::ifstream &ifs) {
    ifs.clear();
    std::string st;
    char c = 0;
    ifs >> c;
    while (std::isalpha(c) && !ifs.eof()) {
      st.append(&c, 1);
      c = ifs.get();
    }
    if (!ifs.eof()) ifs.unget();
    // ifs.clear();
    return st;
  }


};

#endif  // INCLUDE_NTWSCRIPT_HPP_
