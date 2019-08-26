/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include<iostream>
#include "Rnd.hpp"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  randiv.set_seed(10);
  cout << "Random number testing:\n";
  auto nd = normal_dist(0.0, 1.0);
  cout<< "Random number with Gaussian distribution: " << nd() << endl;
  auto c2 = cnst(2.0);
  cout<< "This is always 2 :) : " << c2() << endl;

  return 0;
}
