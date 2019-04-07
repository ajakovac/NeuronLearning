/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#ifndef INCLUDE_SHAPE_HPP_
#define INCLUDE_SHAPE_HPP_
// In this file we collect all routines that help to define a
// multidimensional vector object. Shape is meant to describe the
// geometry of a Tensor. We define multi_for that cycles through all
// the directions. We define addition and multiplication by real
// numbers of a vector.

#include <vector>
#include <iostream>
#include <ostream>
#include <cmath>
#include "Error.hpp"
// ----------------------------------------------------------//
// Here starts the definition of the shape class
// ----------------------------------------------------------//

template<typename T> bool operator<(std::vector<T> x, std::vector<T> y);
template<typename T> bool operator>(std::vector<T> x, std::vector<T> y);

template<typename T>
std::ostream& operator<< (std::ostream& stream, const std::vector<T> &v) {
  stream << "(";
  auto x = v;
  auto vl = x.back();
  x.pop_back();
  for (auto vi : x) stream << vi<< ", ";
  stream << vl << ")";
  return stream;
}

class Position : public std::vector<int> {
 public:
  using std::vector<int>::vector;
  Position(std::vector<int> v) : std::vector<int>(v) {}
  Position(std::initializer_list<Position> poslist) : std::vector<int>() {
    for (auto p : poslist) for (int n : p) this->push_back(n);
  }
};

class Shape : public Position {
 public:
  Shape() : Position() {_make_e();}
  explicit Shape(int n) : Position(1, n) { _make_e();}
  Shape(std::initializer_list<int> l) : Position(l) {_make_e();}
  explicit Shape(std::vector<int> v) : Position(v) {_make_e(); }
  // capacity of the shape
  int vol() const {
    int n = 1;
    for (auto i : *this) n*=i;
    return n;
  }
  // flatten the position vector
  int index(const std::vector<int> L) const {
    if (L.size() != size()) throw("size mismatch!");
    int ind = *L.begin();
    int i = 1;
    for (auto it = L.begin()+1; it != L.end(); it++, i++)
      ind = *it+ ind* at(i);
    return ind;
  }
  // components of the index
  std::vector<int> components(int index) const {
    std::vector<int> L(size());
    for (unsigned int i = 0; i< size(); i++) {
      L[i] = index/e[i];
      index-= e[i]*L[i];
    }
    return L;
  }

  enum FIT_METHOD { CLOSEST, PERIODIC };
  // if v does not fint into a box described by the shape, it projects back:
  // CLOSEST: find the coolosest point or PERIODIC: periodically extend shape
  void fit(vector<int> *v, Shape::FIT_METHOD fm = CLOSEST) const {
    if (v->size()!= size()) throw(Error("Shape::fit: size mismatch!"));
    if ((*v) < (*this)) return;  // if it already fits to the volume, OK
    for (unsigned int i = 0; i < size(); i++) if ((*v)[i]>= at(i)) {
      switch (fm) {
        case FIT_METHOD::CLOSEST  : (*v)[i] = at(i)-1; break;
        case FIT_METHOD::PERIODIC : (*v)[i]%= at(i); break;
        }
      }
  }

 private:
  std::vector<int> e;
  // create the basis vectors
  void _make_e() {
    if (size() == 0) return;
    e.resize(size());
    e[size()-1] = 1;
    for (unsigned int i = size()-1; i > 0; i--) e[i-1] = e[i]*at(i);
  }
};

// -------------------------------------------------------------------//
// A shaped vector is called tensor, with all necessary reshaping
// properties.

template<class T>
class Tensor : public std::vector< T > {
 public:
  explicit Tensor(int N) : std::vector< T >(N), _shape(N) {}
  explicit Tensor(const Shape &s) : std::vector< T >(s.vol()), _shape(s) {}
  const Shape & shape() const {return _shape;}
  void reshape(Shape s) {
    if (s.vol()!= _shape.vol()) throw(Error("Tensor::reshape: invalid shape!"));
    _shape = s;
  }

 private:
  Shape _shape;
};

// help function for multi_for
template < typename T >
void _multi_for(std::vector< int > args, std::vector< int > limits, T f);

// multi_for performs enbedded for cycles, where each cycle runs through
// 0 to limits[i]. The body of the for cycle is a typename that can take
// the indexes (the actual position) as argument: f : vector<int> -> void.
template < typename T >
void multi_for(std::vector< int > limits, T f) {
  _multi_for({}, limits, f);
}

template < typename T >
void _multi_for(std::vector< int > args, std::vector< int > limits, T f) {
  if (limits.size() == 0) {
    f(args);
  } else {
    int n = limits[0];
    limits.erase(limits.begin());
    args.push_back(n);
    for (int i = 0; i < n; i++) {
      args.back() = i;
      _multi_for(args, limits, f);
    }
  }
}

// Here we define the basic algebraic functions on vectors:

template<typename T>
bool operator==(std::vector<T> x, std::vector<T> y) {
  if (x.size() != y.size()) return false;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] != y[i]) return false;
  return true;
}


template<typename Tx, typename Ty>
auto operator+(std::vector<Tx> x, std::vector<Ty> y) {
  decltype(x) z;
  if (x.size() != y.size()) throw(Error("operator+ (Shape): size mismatch!"));
  for (unsigned int i = 0; i < x.size(); i++) z.push_back(x[i]+y[i]);
  return z;
}

template<typename Tx, typename Ty>
auto operator-(std::vector<Tx> x, std::vector<Ty> y) {
  decltype(x) z;
  if (x.size() != y.size()) throw(Error("operator- (Shape): size mismatch!"));
  for (unsigned int i = 0; i < x.size(); i++) z.push_back(x[i]-y[i]);
  return z;
}

template<typename Tx, typename Ty>
auto cdot(std::vector<Tx> x, std::vector<Ty> y) {
  decltype(x) z;
  if (x.size() != y.size()) throw(Error("operator- (Shape): size mismatch!"));
  for (unsigned int i = 0; i < x.size(); i++) z.push_back(x[i]*y[i]);
  return z;
}


template<class T, typename P>
T operator*(P x, T y) {
  T z;
  for (unsigned int i = 0; i < y.size(); i++) z.push_back(x*y[i]);
  return z;
}

template<typename T>
bool operator<(std::vector<T> x, std::vector<T>  y) {
  if (x.size() != y.size()) throw(Error("operator< (Shape): size mismatch!"));
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] >= y[i]) r = false;
  return r;
}

template<typename T>
bool operator<=(std::vector<T> x, std::vector<T>  y) {
  if (x.size() != y.size()) throw(Error("operator< (Shape): size mismatch!"));
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] > y[i]) r = false;
  return r;
}

template<typename T>
bool operator>(std::vector<T> x, std::vector<T>  y) {
  if (x.size() != y.size()) throw(Error("operator< (Shape): size mismatch!"));
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] <= y[i]) r = false;
  return r;
}

template<typename T>
bool operator>=(std::vector<T> x, std::vector<T>  y) {
  if (x.size() != y.size()) throw(Error("operator< (Shape): size mismatch!"));
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] < y[i]) r = false;
  return r;
}


template<typename T>
bool operator<(std::vector<T> x, int y) {
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] >= y) r = false;
  return r;
}

template<typename T>
bool operator<=(std::vector<T> x, int y) {
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] > y) r = false;
  return r;
}

template<typename T>
bool operator>(std::vector<T> x, int  y) {
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] <= y) r = false;
  return r;
}

template<typename T>
bool operator>=(std::vector<T> x, int  y) {
  bool r = true;
  for (unsigned int i = 0; i < x.size(); i++)
    if (x[i] < y) r = false;
  return r;
}

// Here are the distance functions. dist(n1, n2) measures the distance
// n2 from n1, ie. the possibility to connect FROM n2 TO n1.

// the geometrical distance
auto sq = [](double x){return x*x;};

auto L2_dist = [](Shape sh1, Shape sh2) {
  if (sh1.size() != sh2.size())
    throw(Error("Incompatible geometries in geom_dist!"));
  double res2 = 0;
  for (unsigned int j = 0; j < sh1.size(); j++) res2 += sq(sh1[j]-sh2[j]);
  return std::sqrt(res2);
};



#endif  // INCLUDE_SHAPE_HPP_
