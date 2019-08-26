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
#include<iomanip>
#include "Shape.hpp"
#include "Rnd.hpp"

// This file defines the basic sructural building blocks. These are collected
// in the class Network.
// The Network class is designed to be effectively placed into a GPU.
// Exception is the geometry manager, but it can be separated easily

class Network {
  // the data members: all are private:

  // the basic units are the sites, their values are stored here
  std::vector<double> sites;

  // sites are organized in layers; layers have shape
  std::vector<Shape> shapes;  // vector of number_of_layers size

  // site values are stored continuousy, so we must remember,
  // where the sites of a given layer start (last element is sites.size())
  std::vector<uint> layer_offset;  // vector of number_of_layers+1 size:

  // sites are connected:
  std::vector<double> conn;   // connection values
  std::vector<uint> connsite;  // corresponding connected sites

  // connections are also stored continuously, so we must remember,
  // where the connections of a given site starts
  // the last element is always equal to conn.size()
  std::vector<uint> conn_offset;  // vector of number_of_sites+1 size

 public:
  // constructor: only offsets need to be adjusted
  Network() {
    layer_offset.push_back(0);
    conn_offset.push_back(0);
  }

  //////////////////////////////////////////////////////////////
  // functions for reaching data one by one //
  //////////////////////////////////////////////////////////////

  // Network acts like a vector of sites
  double& operator[](uint sn) { return sites[sn];}
  // get the site value from site ID sn
  double& site(uint sn) {return sites[sn];}
  // get the site value from site ID sn, const version
  double site(uint sn) const {return sites[sn];}
  // get the site value from layer ID and relative index
  double& site(uint lyn, uint lyindex) {
    return sites[layer_offset[lyn] + lyindex];}
  // get the site value from layer ID and relative index, const version
  double site(uint lyn, uint lyindex) const {
    return sites[layer_offset[lyn] + lyindex];}
  // site ID from layer number and relative index
  uint getsiteID(uint lyn, uint sli) { return layer_offset[lyn] + sli; }
  // site ID of a site in layer lyn at position pos
  uint getsiteID(uint lyn, Position pos) {
    Shape& sh = shapes[lyn];
    return layer_offset[lyn] + sh.index(pos);
  }
  // connection ID of the cn-th connection of site sn
  uint getconnID(uint sn, uint cn) { return conn_offset[sn]+cn; }
  // the connection value from conn ID
  double& connection(uint connID) {return conn[connID];}
  // value of the cn-th connection of site sn
  double& connection(uint sn, uint cn) {
    return conn[ conn_offset[sn]+cn ]; }
  // the connected site of the cn-th connection of site sn
  uint& connectedSite(uint sn, uint cn) {
      return connsite[ conn_offset[sn]+cn ]; }
  // site of the connected site of the cn-th connection of site sn
  double& connectedValue(uint sn, uint cn) {
    return sites[ connsite[ conn_offset[sn]+cn ] ]; }
  // shape of a layer lyn
  Shape& layerShape(uint lyn) { return shapes[lyn]; }
  // change the layer shape
  void setlayerShape(uint lyn, Shape s) {
    if (s.vol()!= nSitesinLayer(lyn))
      throw(Error("Incompatible shapes in setlayerShape\n"));
    shapes[lyn] = s;
  }


  //////////////////////////////////////////////////////////////
  // functions for having size information //
  //////////////////////////////////////////////////////////////

  // number of all sites
  uint nSites() {return sites.size();}
  // number of layers
  uint nLayers() const {return shapes.size();}
  // number of sites in a layer lyn
  uint nSitesinLayer(uint lyn) const {
    return layer_offset[lyn+1]-layer_offset[lyn];
  }
  // number of all connections
  uint nallConnections() {return conn.size();}
  // number of connections of site sn
  uint nConnections(int sn) {  // number of connections for a site
    return conn_offset[sn+1]-conn_offset[sn];
  }

  //////////////////////////////////////////////////////////////
  // data access through lambda expressions //
  //////////////////////////////////////////////////////////////

  // access to layer ly
  auto readLayer(uint ly) {
    double* data = &sites[layer_offset[ly]];
    return [=](uint snr) -> double& {
      return *(data+snr);
    };
  }
  // access to connections of sn
  auto readConnection(uint sn) {
    double* data = &conn[conn_offset[sn]];
    return [=](uint cnr) -> double& {
      return *(data +cnr);
    };
  }
  // access to connected sites to site sn
  auto readConnectedSite(uint sn) {
    uint* data = &connsite[conn_offset[sn]];
    return [=](uint cnr) -> uint {
      return *(data +cnr);
    };
  }
  // provides a lambda accessing the connected value
  auto readConnectedValue(uint sn) {
    uint* connst = &connsite[conn_offset[sn]];
    return [=](uint cn) {
      return sites[*(connst+cn)];
    };
  }

  /////////////////////////
  // search in hierarchy //
  /////////////////////////

  // get the site the connection belongs to; does not check range
  uint getconnSite(uint cid) {
    for (uint nn = 1; nn < conn_offset.size(); ++nn)
      if (conn_offset[nn] > cid) return(nn-1);
    return(sites.size());
  }
  // get the layer ID to that site sn belongs to; no range check
  uint sitelayerID(uint sn) const {
    for (uint nn = 1; nn < layer_offset.size(); ++nn)
      if (layer_offset[nn] > sn) return(nn-1);
    return(shapes.size());
  }
  // index of the site within its layer
  uint sitelayerIndex(uint lyn, uint sn) const {
    return sn-layer_offset[lyn]; }

  // position of the relative site sn within its layer
  std::vector<int> sitePos(uint lind, uint sn) {
    return shapes[lind].components(sn);
  }
  // position of the site sn
  std::vector<int> sitePos(uint sn) {
    uint lind = sitelayerID(sn);
    return shapes[lind].components(sn - layer_offset[lind]);
  }

  //////////////////////////////////////////////////////////////
  // functions to build up a network //
  //////////////////////////////////////////////////////////////

  // apply a function to all sites of a layer:
  // F : int -> void, int means site ID.
  template < typename T >
  void forallSitesinLayer(uint lyn, T F) {
    uint nmax = layer_offset[lyn+1];
    for (uint n = layer_offset[lyn]; n < nmax; n++) F(n);
  }

  // apply a function to all connections in a layer
  template< typename T>
  void forallConnectionsinLayer(uint lyn, T F) {
    uint nmax = layer_offset[lyn+1];
    for (uint n = layer_offset[lyn]; n < nmax; n++) {
      uint cnmax = nConnections(n);
      for (uint cn = 0; cn < cnmax; ++cn) F(n, cn);
    }
  }

  // create a layer by specifying its shape
  // IMPORTANT! Layer creation order means layer hierarchy as well!!
  uint AddLayer(const Shape s) {
    uint lysize = s.vol();
    // the last connection; also the last element of conn_offset
    uint lastconn = conn.size();
    conn_offset.pop_back();  // delete last element of conn_offset
    for (uint n = 0; n < lysize; n++) {
      sites.push_back(0.0);  // default site value is zero
      conn_offset.push_back(lastconn);  // no new connections for this site
    }
    conn_offset.push_back(lastconn);  // closing element
    layer_offset.push_back(sites.size());  // end of current layer
    shapes.push_back(s);
    return shapes.size()-1;
  }

  // Connection managers: we always add connections for the last layer.
  // We define several methods to do that.

  // The basic tool to create connections uses an fn and a cf function
  // fn(Network *, int ly, int rsn, vector<int> *): gives site list to connect
  // cf() -> double: provides the strength of the connection
  template < typename FNT, typename CFT >
  void ConnectLastLayer(FNT fn, CFT cf) {
    std::vector<int> vs;  // placeholder for the connection site list
    uint lyn = shapes.size()-1;  // the last layer ID
    uint nst = layer_offset[lyn+1]-layer_offset[lyn];  // number of sites
    for (uint snr = 0; snr < nst; ++snr) {  // relative site indexing
      uint sntot = layer_offset[lyn]+snr;  // this is the absolute site id
      conn_offset[sntot] = conn.size();  // the next connection offset
      vs.clear();
      fn(this, lyn, snr, &vs);  // get the corresponding connection sites
      for (const auto& csn : vs) {
        connsite.push_back(csn);  // connect to this site
        conn.push_back(cf());  // set the weight using the cf function
      }
    }
    conn_offset[sites.size()] = conn.size();  // the last element
  }

  // a shorthand notation for constant strength connections
  template < typename FNT >
  void ConnectLastLayer(FNT fn, double cnst) {
    ConnectLastLayer(fn, [=](){return cnst;});
  }

  // The advanced tool to create connections uses three functions
  // fn(Network *, int ly, int snr, vector<int> *): gives site list to connect
  // rf(Network *, int sn0, int sn1) -> bool: should it be connected?
  // cf() -> double: provides the strength of the connection
  template < typename FNT, typename RFT, typename CFT >
  void ConnectLastLayer_advanced(FNT fn, RFT rf, CFT cf) {
    std::vector<int> vs;  // placeholder for the connection site list
    uint lyn = shapes.size()-1;  // the last layer ID
    uint nst = layer_offset[lyn+1]-layer_offset[lyn];  // number of sites
    for (uint snr = 0; snr < nst; ++snr) {  // relative site indexing
      uint sntot = layer_offset[lyn]+snr;  // this is the absolute site id
      conn_offset[sntot] = conn.size();  // the next connection offset
      vs.clear();
      fn(this, lyn, snr, &vs);  // get the corresponding connection sites
      for (const auto& csn : vs) {
        if (rf(this, sntot, csn)) {  // connect if rf is true
          connsite.push_back(csn);  // connect to this site
          conn.push_back(cf());  // set the weight using the cf function
        }
      }
    }
    conn_offset[sites.size()] = conn.size();  // the last element
  }

  // for convolution: average spatial weights in a layer
  // we assume here that the last dimension is the channel
  void average_weights(int lyn) {
    int n0 = getsiteID(lyn, 0);  // take the first site in layer
    int ncinsite = nConnections(n0);  // find its number of connections
    Shape shp = shapes[lyn];
    int nofch = shp.back();  // last dimension: number of channels
    std::vector<double> avrw(ncinsite*nofch);  // for averaging the weights
    int nsinl = shp.vol();  // number of sites in layer
    for (uint rsn = 0; rsn < nsinl; ++rsn) {
      std::vector<int> p = shp.components(rsn);
      int chn = p.back();
      int sn = getsiteID(lyn, rsn);
      for (uint cn = 0; cn < ncinsite; ++cn)
        avrw[chn*ncinsite+cn] += connection(sn, cn);  // add up connections
    }
    int nspatialsites = nsinl/nofch;  // number of sites per channel
    for (auto &sw : avrw) sw /= nspatialsites;  // normalize
    for (uint rsn = 0; rsn < nsinl; ++rsn) {
      std::vector<int> p = shp.components(rsn);
      int chn = p.back();
      int sn = getsiteID(lyn, rsn);
      for (uint cn = 0; cn < ncinsite; ++cn)
        connection(sn, cn) = avrw[chn*ncinsite+cn];  // rewrite connections
    }
  }

  // save the network
  void save(const char* filename, bool verbatim = false) {
    std::ofstream file;
    file.open(filename);
    file << "#Saving network\n\n";

    file << std::fixed << std::setprecision(6);

    file << "#number of layers:\n";
    file << shapes.size() << std::endl;
    file << "#l#    l_offset       shape\n";
    for (uint lyn = 0; lyn < shapes.size(); ++lyn)
      file << std::right
           << std::setw(3) << lyn
           << std::setw(9) << layer_offset[lyn]
           << std::setw(10) << shapes[lyn]
           << std::endl;

    file << "\n#number of sites:\n";
    file << sites.size();
    file << "\n#s#          value   conn_offset";
    if (verbatim) file << "   position";
    file  << std::endl;
    for (uint lyn = 0; lyn < shapes.size(); ++lyn) {
      uint nst = nSitesinLayer(lyn);
      for (uint nl = 0; nl < nst; ++nl) {
        uint sn = getsiteID(lyn, nl);
        file << std::setw(3) << sn
             << std::setw(15) << sites[sn]
             << std::setw(10) << conn_offset[sn];
        if (verbatim)
          file << std::setw(10) << sitePos(lyn, nl);
        file  << std::endl;
      }
    }

    file << "\n#number of connections:\n";
    file << conn.size() << std::endl;
    file << "#c#  connsite     conn";
    if (verbatim) file << "       layer:pos --> layer:pos";
    file  << std::endl;
    for (uint lyn = 0; lyn < shapes.size(); ++lyn) {
      uint nst = nSitesinLayer(lyn);
      for (uint nl = 0; nl < nst; ++nl) {
        uint sn = getsiteID(lyn, nl);
        uint cnmin = conn_offset[sn];
        uint cnmax = conn_offset[sn+1];
        for (uint cn = cnmin; cn < cnmax; ++cn) {
          uint ncs = connsite[cn];
          file << std::setw(3) << cn
               << std::setw(6) << ncs
               << std::setw(15) << conn[cn];
          if (verbatim) {
            int ncsl = sitelayerID(ncs);
            int ncsr = ncs - layer_offset[ncsl];
            file << std::setw(10) << lyn << ":" << sitePos(lyn, nl)
                 << " --> "
                 << std::setw(8) << ncsl << ":" << sitePos(ncsl, ncsr);
          }
          file << std::endl;
        }
      }
    }
    file.close();
  }

  // load the network from scratch
  void load(const char* filename, bool create = false) {
    std::ifstream file;
    std::string line;
    file.open(filename);
    int c;
    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    int lynum;  // read number of layers
    file >> lynum;
    if (!create && lynum != shapes.size())
      throw(Error("bad network data file!"));

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    for (int ly = 0; ly < lynum; ++ly) {
      int lnum;
      file >> lnum;  // read line number
      int lyoffs;
      file >> lyoffs;  // read layer offset
      if (create) {
        layer_offset[lnum] = lyoffs;
        layer_offset.push_back(lyoffs);
      }
      std::vector<int> shaperead;
      char chr;
      file >> chr;
      while (chr != ')') {
        int intread;
        file >> intread;
        shaperead.push_back(intread);
        file >> std::ws;
        file >> chr;
      }
      if (create) shapes.push_back(Shape(shaperead));
      // if not create, we assume that the shapes are already given
    }

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    int axnum;
    file >> axnum;
    if (create) layer_offset[shapes.size()]= axnum;
    if (!create && axnum != sites.size())
      throw(Error("bad network data file!"));

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    for (int an = 0; an < axnum; ++an) {
      int lnum;
      file >> lnum;  // read line number
      double xread;
      int intread;
      file >> xread;  // read site value
      if (create) sites.push_back(xread);
      else
        sites[an] = xread;
      file >> intread;  // read conn offset
      if (create) {
        conn_offset[lnum] = intread;
        conn_offset.push_back(intread);
      } else {
        conn_offset[an] = intread;
      }
      // read the remainder of the line
      if (file.peek() != '\n') std::getline(file, line);
    }

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    int connum;
    file >> connum;
    if (create) conn_offset[sites.size()] = connum;
    if (!create && connum != conn.size())
      throw(Error("bad network data file!"));

    // skip remark and empty lines
    while ((c =file.peek()) == '#' || c == '\n') {
      std::getline(file, line);
    }
    for (int cn = 0; cn < connum; ++cn) {
      int lnum;
      file >> lnum;  // read line number
      double xread;
      int intread;
      file >> intread;  // read connected site
      if (create) connsite.push_back(intread);
      else
        connsite[cn] = intread;
      file >> xread;  // read connection strength
      if (create) conn.push_back(xread);
      else
        conn[cn] = xread;
      // read the remainder of the line
      if (file.peek() != '\n') std::getline(file, line);
    }

    // finally the last elements of offsets are set
    if (create) {
      layer_offset.push_back(sites.size());
      conn_offset.push_back(conn.size());
    } else {
      layer_offset[shapes.size()] = sites.size();
      conn_offset[sites.size()] = conn.size();
    }

    file.close();
  }
};

//////////////////////////////////////////////////////////////////
// Here we propose some functions for connections:

// site list functions: this lists all the sites of a layer
auto alllayer = [](uint lyn) {
  return [=](Network* N, uint, uint, std::vector<int> *vs){
    N->forallSitesinLayer(lyn, [&](uint n){ vs->push_back(n); });
  };
};

// masqued local neighbourhood with shift
auto masquedlist = [](uint lyn,  // this is the layer from where we choose sites
                      const Shape &masque,  // the window of collection
                      const Shape &shift,  // the shift of neighboring windows
                      const Shape &offset  // the starting offset
                      ) {
  // lf is the layer from we connect, anr the relative site index
  return [=](Network* N, uint lf, uint anr, std::vector<int> *v) {
    auto p1 = cdot(N->sitePos(lf, anr), shift) + offset;
    multi_for(masque, [&](auto x){
      Position y = p1+x;
      if ((y>= 0) && (y < N->layerShape(lyn)) )
        v->push_back(N->getsiteID(lyn, y));
    });
  };
};

// the most simple connection: only the same position is connected
auto directlist = [](int lyn){
  return [=](Network* N, uint lf, uint anr, std::vector<int> *v) {
    v->push_back(N->getsiteID(lyn, anr));
  };
};

///////////////////////////////////////////////////////////////////////////
// some examples for the connection functions
// normal_dist(mean, var);
// uniform_dist(mean, var);
// [](){return 1.0;};  for a constant value

#endif  // INCLUDE_STRUCTURE_HPP_
