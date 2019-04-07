/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#ifndef INCLUDE_RND_HPP_
#define INCLUDE_RND_HPP_

#include <random>
#include "Error.hpp"

// The random class with normal and uniform distributions

class RandomDevice {
 public:
  RandomDevice()  : e2(r()), dis(0, 1), nrm(0, 1), seed(-1) {}
  explicit RandomDevice(int seedin)  :
    e2(seedin), dis(0, 1), nrm(0, 1), seed(seedin) {}
  double normal_dist() {return nrm(e2);}
  double normal_dist(double mean, double var) {return mean+var*nrm(e2);}
  double uniform_dist() {return dis(e2);}
  double uniform_dist(double mean, double var) {return mean+var*dis(e2);}
  void set_seed(int seed_in) {
    seed = seed_in;
    e2 = std::mt19937(seed);
  }
 private:
  std::random_device r;
  std::mt19937 e2;
  std::uniform_real_distribution<> dis;
  std::normal_distribution<> nrm;
  int seed;
} randiv;

// lambda expressions to use the above functionalities

auto normal_dist =  [](double mean, double var) {
  return [=](){
    return randiv.normal_dist(mean, var);
  };
};

auto uniform_dist = [](double mean, double var) {
  return [=](){
    return randiv.uniform_dist(mean, var);
  };
};

// for completeness here is a constant function:
// this is the same as normal_dist(val, 0) or uniform_dist(val ,0)

auto cnst = [](double val) {
  return [=](){return val;};
};

#endif  // INCLUDE_RND_HPP_
