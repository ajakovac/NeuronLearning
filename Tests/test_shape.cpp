/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include "Shape.hpp"

int main(int argc, char const *argv[]) {
  std::cout << "\nShape testing!\n";
  Position p0(3, 0);
  Position p1({3, 4, 5});
  Position p2 = {4, 5, 6};
  Position p3 = {p1, p2, {3, 4}};
  Position p4 = p3;
  Shape sp({5, 5});

  std::cout << p0 << '\n';
  std::cout << p1 << '\n';
  std::cout << p2 << '\n';
  std::cout << p3 << '\n';
  std::cout << "The shape = " << sp << std::endl;
  std::cout << "dimension of the shape d=" << sp.size() << std::endl;
  std::cout << "total volume vol=" << sp.vol()<< std::endl;
  int ndx = sp.index({3, 2});
  std::cout << "index of "<< Shape({3, 2}) << "= " << ndx << std::endl;
  std::cout << "components of " << ndx << " = " << sp.components(ndx)
            << std::endl;
  std::cout << "{2, 3} + {4, 5} =" << Shape({2, 3})+Shape({4, 5}) << std::endl;
  Shape vv = Shape({0, 6});
  std::cout << "project " << vv;
  sp.fit(&vv);
  std::cout << " back to the original volume (CLOSEST) "  << vv << std::endl;
  vv = Shape({0, 6});
  sp.fit(&vv, Shape::FIT_METHOD::PERIODIC);
  std::cout << "project this vector back to the original volume (PERIODIC) "
            << vv << std::endl;
  Tensor<double> M(9);
  std::cout << "shape of the tensor= " << M.shape() << std::endl;
  M.reshape({3, 3});
  std::cout << "shape of the tensor= " << M.shape() << std::endl;
  multi_for(Shape({3, 3}), [=](auto x){ std::cout << x << std::endl;});
  std::cout << "distance of {1, 2} and {4, 6} = "
            << L2_dist(Shape({1, 2}), Shape({4, 6})) << std::endl;

  return 0;
}
