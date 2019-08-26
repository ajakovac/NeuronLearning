/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#ifndef INCLUDE_STRUCTURE_HPP_
#define INCLUDE_STRUCTURE_HPP_

#include<vector>
#include<cmath>
#include<functional>
#include<algorithm>
#include<fstream>
#include<string>
#include "Shape.hpp"
#include "Rnd.hpp"

// This file defines the basic sructural building blocks. These are collected
// in the class Network.
// The Network class is designed to be effectively placed into a GPU.
// Exception is the geometry manager, but it can be separated easily

class Network {
  // the data members: all are private:
  // vector of number_of_connections size:
  std::vector<double> conn;   // all connection values
  std::vector<int> connsite;  // corresponding connected sites

  // vectors of number_of_axons size:
  std::vector<double> axons;
  std::vector<int> layers;  // all axons have a layer number
  std::vector<int> conn_offset;  // here starts the connections of a given site

  // vector of number_of layers size:
  // the axons belonging to a layer are placed continuously, starting from
  // an index value: that is stored in layer_offset vector:
  std::vector<int> layer_offset;
  // the shapes of the layers is a vector of vectors which is not well suited
  // for GPU. Therefore all applications that use shape must stay on CPU.
  // Nevertheless geometry is a very important aspect of layers.
  std::vector<Shape>  shapes;

 public:
  //////////////////////////////////////////////////////////////
  // functions for extracting data //
  //////////////////////////////////////////////////////////////

  // Network acts like a vector of axons
  double& operator[](int sn) { return axons[sn];}
  // get the axon value from site ID sn
  double& site(int sn) {return axons[sn];}
  // get the axon value from layer ID and relative index
  double site(int sn) const {return axons[sn];}
  // get the axon value from layer ID and relative index
  double& site(int lyn, int lyindex) {
    return axons[layer_offset[lyn] + lyindex];}
  double site(int lyn, int lyindex) const {
    return axons[layer_offset[lyn] + lyindex];}
  // site ID from layer number and relative index
  uint getsiteID(int lyn, int sli) { return layer_offset[lyn] + sli; }
  // number of all sites
  uint nSites() {return axons.size();}
  // number of layers
  uint nLayers() const {return layer_offset.size();}
  // number of sites in a layer lyn
  uint nSitesinLayer(int lyn) const {
    if (lyn== static_cast<int>(layer_offset.size())-1)
      return axons.size()-layer_offset[lyn];
    return layer_offset[lyn+1]-layer_offset[lyn];
  }
  // gives layer ID to that site sn belongs to
  int sitelayerID(int sn) const { return layers[sn]; }
  // index of the site within its layer
  int sitelayerIndex(int sn) const {
    return sn-layer_offset[ layers[sn]]; }

  //////////////////////////////////////////////////////////////
  // functions for handle connections //
  //////////////////////////////////////////////////////////////

  // number of all connections
  uint nallConnections() {return conn.size();}
  // number of connections of site sn
  uint nConnections(int sn) {  // number of connections for a site
    if (sn== static_cast<int>(axons.size())-1)
      return conn.size()-conn_offset[sn];
    return conn_offset[sn+1]-conn_offset[sn];
  }
  // connection ID of the cn-th connection of site sn
  int getconnID(int sn, int cn) { return conn_offset[sn]+cn; }
  // get the site the connection belongs to
  int getconnSite(int cid) {
    for (int nn = 1; nn < static_cast<int>(axons.size()); ++nn)
      if (conn_offset[nn] < cid) return(nn-1);
    return(axons.size()-1);
  }
  // the connection value from conn ID
  double& connection(int connID) {return conn[connID];}
  // value of the cn-th connection of site sn
  double& connection(int sn, int cn) {
    return conn[ conn_offset[sn]+cn ]; }
  // the connected site of the cn-th connection of site sn
  int& connectedSite(int sn, int cn) {
      return connsite[ conn_offset[sn]+cn ]; }
  // axon of the connected site of the cn-th connection of site sn
  double& connectedValue(int sn, int cn) {
    return axons[ connsite[ conn_offset[sn]+cn ] ]; }

  //////////////////////////////////////////////////////////////
  // functions for handle connections //
  //////////////////////////////////////////////////////////////

  // shape of a layer lyn
  Shape& layerShape(int lyn) { return shapes[lyn]; }
  // shape of a layer of site sn
  Shape& siteShape(int sn) { return shapes[layers[sn]]; }
  // position of the site sn within its layer
  std::vector<int> sitePos(int sn) {
    int lind = layers[sn];
    return shapes[lind].components(sn - layer_offset[lind]);
  }
  // site ID of a site in layer lyn at position pos
  int getsiteID(int lyn, Position pos) {
    Shape& sh = shapes[lyn];
    return layer_offset[lyn] + sh.index(pos);
  }
  // change the layer shape
  void setlayerShape(int lyn, Shape s) {
    if (s.vol()!= nSitesinLayer(lyn))
      throw(Error("Incompatible shapes in setlayerShape\n"));
    shapes[lyn] = s;
  }

  //////////////////////////////////////////////////////////////
  // data access through lambda expressions //
  //////////////////////////////////////////////////////////////

  // provides a lambda accessing the connection
  auto readConnection(int sn) {
    double *conndat = &conn[ conn_offset[sn]];
    return [=](int cn) {
      return *(conndat+cn);
    };
  }

  // provides a lambda accessing the connected value
  auto readConnectedSite(int sn) {
    int *connst = &connsite[ conn_offset[sn]];
    return [=](int cn) {
      return *(connst+cn);
    };
  }

  // provides a lambda accessing the connected value
  auto readConnectedValue(int sn) {
    int *connst = &connsite[ conn_offset[sn]];
    return [=](int cn) {
      return axons[*(connst+cn)];
    };
  }

  //////////////////////////////////////////////////////////////
  // functions to build up a network //
  //////////////////////////////////////////////////////////////

  // apply a function to all axons of a layer:
  // F : int -> void, int means site ID.
  template < typename T >
  void forallSitesinLayer(int lyn, T F) {
    int nmax;
    if (lyn == static_cast<int>(layer_offset.size())-1) nmax = axons.size();
    else
      nmax = layer_offset[lyn+1];
    for (int n = layer_offset[lyn]; n < nmax; n++) F(n);
  }

  // apply a function to all connections in a layer
  template< typename T>
  void forallConnectionsinLayer(int lyn, T F) {
    int nmax;
    if (lyn == static_cast<int>(layer_offset.size()-1)) nmax = axons.size();
    else
      nmax = layer_offset[lyn+1];
    for (int n = layer_offset[lyn]; n < nmax; n++) {
      int cnmax = nConnections(n);
      for (int cn = 0; cn < cnmax; ++cn) F(n, cn);
    }
  }

  // create a layer by specifying its shape
  // IMPORTANT! Layer creation order means layer hierarchy as well!!
  int AddLayer(const Shape s) {
    int lysize = s.vol();
    int axind = axons.size();  // the actual position in axons
    layer_offset.push_back(axind);  // is the current layer offset
    int lind = 0;
    if (axind > 0) lind=layers[axind-1]+1;  // get the next layer number
    for (int n = 0; n < lysize; n++) {
      axons.push_back(0.0);  // default axon value is zero
      layers.push_back(lind);  // all of these axons have the same layer number
      conn_offset.push_back(0);  // no connections for this site
    }
    shapes.push_back(s);
    return lind;
  }


  // Connection manager: connection of the last layer can be easily added,
  // the structure supports this type of connection building.
  // We create connections with the help of an fn and a cf function
  // fn(Network *, int, vector<int> *): gives site list to where we connect
  // cf(Network*, int, int) -> double: provides the strength of the connection
  template < typename FNT, typename CFT >
  void ConnectLastLayer(FNT fn, CFT cf) {
    std::vector<int> vs;  // placeholder for the connection site list
    int ls = axons.size()-1;
    int lyn = layers[ls];  // the last layer ID
    for (unsigned int sn = layer_offset[lyn]; sn < axons.size(); sn++) {
      conn_offset[sn] = conn.size();  // the next connection offset
      vs.clear();
      fn(this, sn, &vs);  // get the corresponding connection sites
      for (const auto& csn : vs) {
        connsite.push_back(csn);
        double cn = cf(this, sn, csn);  // compute the corresponding weights
        conn.push_back(cn);
      }
    }
  }


  // save the network
  void save_verbatim(const char* filename) {
    std::ofstream file;
    file.open(filename);
    file << "#Saving network\n";

    file << "\n#laynum\toffset\tshape\n";
    for (unsigned int lyn = 0; lyn < layer_offset.size(); ++lyn)
      file << lyn << "\t" << layer_offset[lyn]<< "\t"
           << shapes[lyn] << std::endl;


    file << "\n#num\taxons\toffset\tlayer\tposition\tconn_offset\n";
    for (unsigned int sn = 0; sn < axons.size(); ++sn) {
      file << sn << "\t" << axons[sn] <<"\t";
      file << layers[sn] << "\t";
      file << sitePos(sn) << "\t\t" << conn_offset[sn];
      file  << std::endl;
    }

    file << "\n#nconn\tconnto\tpos\tfrom\tpos\tconnstrength\n";
    for (int sn = 0; sn < static_cast<int>(axons.size()); sn++) {
      int cnmax = conn.size();
      if (sn < static_cast<int>(axons.size())-1) cnmax = conn_offset[sn+1];
      for (int cn = conn_offset[sn]; cn < cnmax; cn++) {
        file << cn << "\t" << sn << "\t" << sitePos(sn) <<"\t";
        file << connsite[cn] << "\t" << sitePos(connsite[cn]);
        file << "\t" << conn[cn];
        file << std::endl;
      }
    }
    file.close();
  }

  // save the network
  void save(const char* filename) {
    std::ofstream file;
    file.open(filename);
    file << "#Saving network\n";

    file << "\n#layer_offset shape\n";
    file << layer_offset.size() << std::endl;
    for (unsigned int lyn = 0; lyn < layer_offset.size(); lyn++)
      file << layer_offset[lyn] << " " << shapes[lyn] << std::endl;


    file << "\n#axons layer conn_offset\n";
    file << layers.size() << std::endl;
    for (unsigned int sn = 0; sn < axons.size(); sn++) {
      file << axons[sn] << " ";
      file << layers[sn] << " ";
      file << conn_offset[sn];
      file  << std::endl;
    }

    file << "\n#connectedsite connstrength\n";
    file << conn.size() << std::endl;
    for (unsigned int cn = 0; cn < conn.size(); ++cn) {
      file << connsite[cn]  << " ";
      file << conn[cn];
      file << std::endl;
    }
    file.close();
  }

  // load the network from scratch
  void load(const char* filename) {
    std::ifstream file;
    std::string line;
    file.open(filename);
    int c;
    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    int lynum;
    file >> lynum;

    for (int ly = 0; ly < lynum; ++ly) {
      int intread;
      file >> intread;
      layer_offset.push_back(intread);
      std::vector<int> shaperead;
      char chr;
      file >> chr;
      while (chr != ')') {
        file >> intread;
        shaperead.push_back(intread);
        file >> std::ws;
        file >> chr;
      }
      shapes.push_back(Shape(shaperead));
    }

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    int axnum;
    file >> axnum;

    for (int an = 0; an < axnum; ++an) {
      double xread;
      int intread;
      file >> xread;
      axons.push_back(xread);
      file >> intread;
      layers.push_back(intread);
      file >> intread;
      conn_offset.push_back(intread);
    }

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    int connum;
    file >> connum;
    for (int cn = 0; cn < connum; ++cn) {
      double xread;
      int intread;
      file >> intread;
      connsite.push_back(intread);
      file >> xread;
      conn.push_back(xread);
    }

    file.close();
  }


  // fill the axon and connection values assuming the same structure
  void fill(const char* filename) {
    std::ifstream file;
    std::string line;
    file.open(filename);
    int c;
    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    unsigned int lynum;
    file >> lynum;
    if (lynum != layer_offset.size())
      throw(Error("Bad network data file!"));

    for (unsigned int ly = 0; ly < lynum; ++ly) {
      int intread;
      file >> intread;
      // layer_offset.push_back(intread);
      std::vector<int> shaperead;
      char chr;
      file >> chr;
      while (chr != ')') {
        file >> intread;
        shaperead.push_back(intread);
        file >> std::ws;
        file >> chr;
      }
      // shapes.push_back(Shape(shaperead));
    }

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    unsigned int axnum;
    file >> axnum;
    if (axnum != axons.size())
      throw(Error("Bad network data file!"));

    for (unsigned int an = 0; an < axnum; ++an) {
      double xread;
      int intread;
      file >> xread;
      axons[an] = xread;
      file >> intread;
      // layers.push_back(intread);
      file >> intread;
      // conn_offset.push_back(intread);
    }

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    unsigned int connum;
    file >> connum;
    if (connum != conn.size())
      throw(Error("Bad network data file!"));

    for (unsigned int cn = 0; cn < connum; ++cn) {
      double xread;
      int intread;
      file >> intread;
      // connsite.push_back(intread);
      file >> xread;
      conn[cn] = xread;
    }

    file.close();
  }
};

//////////////////////////////////////////////////////////////////
// Here we propose some functions for connections:

// site list functions: this lists all the sites of a layer
auto alllayer = [](int lyn) {
  return [=](Network* N, int, std::vector<int> *vs){
    N->forallSitesinLayer(lyn, [&](int n){ vs->push_back(n); });
  };
};

// masqued local neighbourhood with shift
auto masquedlist = [](int lyn, const Shape &masque, const Shape &shift) {
  return [=](Network* N, int an, std::vector<int> *v) {
    auto p1 = cdot(N->sitePos(an), shift);
    multi_for(masque, [&](auto x){
      Position y = p1+x;
      if ((y>= 0) && (y < N->layerShape(lyn)) )
        v->push_back(N->getsiteID(lyn, y));
    });
  };
};

// the most simple connection: only the same position is connected
auto directlist = [](int lyn){
  return [=](Network* N, int an, std::vector<int> *v) {
    v->push_back(N->getsiteID(lyn, N->sitelayerIndex(an)));
  };
};

///////////////////////////////////////////////////////////////////////////
// some examples for the connection functions

auto normal_cf = [](double mean, double var) {
  auto rr = normal_dist(mean, var);
  return [=](Network*, int, int) { return rr();};
};

auto uniform_cf = [](double mean, double var) {
  auto rr = uniform_dist(mean, var);
  return [=](Network*, int, int) { return rr();};
};

auto const_cf = [](double xcnst) {
  return [=](Network*, int, int) { return xcnst;};
};


#endif  // INCLUDE_STRUCTURE_HPP_
