/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <vector>
#include "Shape.hpp"
#include "Functional.hpp"

auto f = [](double x){return x*x-1;};

int main(int argc, char const *argv[]) {
  std::vector<double> x = {1, 2, 3};
  std::vector<double> y = map(f, x);
  std::vector<double> z = zip([](double a, double b){return a*b;}, x, y);
  double r = foldl([](double a, double b){return a + b;}, 0, z);
  std::cout << x << "->" << y << '\n';
  std::cout << x<< "." << y << "->" << z << '\n';
  std::cout << "sum of" << z << "=" << r << '\n';
  return 0;
}
