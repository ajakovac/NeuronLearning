/* Copyright (C) AGJ
 * Written by G. and A. Jakovac 2019 */
#ifndef INCLUDE_MNIST_HPP_
#define INCLUDE_MNIST_HPP_

#include <string>
#include "dataset.hpp"

class MNIST_dataset : public dataset {
 public:
  void restart() override;
  bool next() override;
  MNIST_dataset(const std::string& fin, const std::string& Lfin);
};

MNIST_dataset::MNIST_dataset(const std::string& fin, const std::string& Lfin)
  : dataset(fin, Lfin, 28, 28) {
  restart();
}

void MNIST_dataset::restart() {
  fs.clear();
  label_fs.clear();
  fs.seekg(16);
  label_fs.seekg(8);
}

bool MNIST_dataset::next() {
  int pixel_num = height * width;
  char c;
  label_fs.get(c);
  _label = static_cast<int>(c);
  if (fs.eof()) {
    _label = -1;
    return false;
  }
  auto a = fs.tellg();
  fs.get(c);
  if (fs.eof())
    return false;
  fs.seekg(a);
  for (int i = 0; i < pixel_num; ++i) {
    fs.get(c);
    dat_red[i] = reinterpret_cast<unsigned char&>(c);
    dat_green[i] = reinterpret_cast<unsigned char&>(c);
    dat_blue[i] = reinterpret_cast<unsigned char&>(c);
  }
  return true;
}

#endif  // INCLUDE_MNIST_HPP_
