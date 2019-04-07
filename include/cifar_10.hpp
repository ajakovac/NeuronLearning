/* Copyright (C) NeuronLearning_project
 * Written by G and A. Jakovac 2019 */
#ifndef INCLUDE_CIFAR_10_HPP_
#define INCLUDE_CIFAR_10_HPP_

#include <string>
#include <fstream>
#include <iostream>
#include "dataset.hpp"


class CIFAR_10_dataset : public dataset {
 public:
  inline bool next() final;
  inline explicit CIFAR_10_dataset(const std::string& filename_in);
  std::string labelString() const;
};

CIFAR_10_dataset::CIFAR_10_dataset(const std::string& fname)
  : dataset(fname, 32, 32)
{ }


bool CIFAR_10_dataset::next() {
  int pixel_num = height * width;
  char c;
  fs.get(c);
  _label = static_cast<int>(c);
  if (fs.eof()) {
    _label = -1;
    return false;
  }
  for (int i = 0; i < pixel_num; ++i) {
    fs.get(c);
    dat_red[i] = reinterpret_cast<unsigned char&>(c);
  }
  for (int i = 0; i < pixel_num; ++i) {
    fs.get(c);
    dat_green[i] = reinterpret_cast<unsigned char&>(c);
  }
  for (int i = 0; i < pixel_num; ++i) {
    fs.get(c);
    dat_blue[i] = reinterpret_cast<unsigned char&>(c);
  }
  return true;
}
std::string CIFAR_10_dataset::labelString() const {
  switch (_label) {
    case 0: return "airplane";
    case 1: return "automobile";
    case 2: return "bird";
    case 3: return "cat";
    case 4: return "deer";
    case 5: return "dog";
    case 6: return "frog";
    case 7: return "horse";
    case 8: return "ship";
    case 9: return "truck";
    default: return "Invalid_Label";
  }
}
#endif  // INCLUDE_CIFAR_10_HPP_
