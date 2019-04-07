/* Copyright (C) NeuronLearning_project
 * Written by G. and A. Jakovac 2019 */
#ifndef INCLUDE_STL_10_HPP_
#define INCLUDE_STL_10_HPP_

#include <string>
#include "dataset.hpp"

class STL_10_dataset : public dataset {
 public:
  inline STL_10_dataset(const std::string& fname, const std::string& Lfname);
  inline bool next() override;
  inline std::string labelString() const;
};

STL_10_dataset::STL_10_dataset(const std::string& fname,
                               const std::string& Lfname)
  : dataset(fname, Lfname, 96, 96)
{ }

bool STL_10_dataset::next() {
  int pixel_num = height * width;
  char c;
  label_fs.get(c);
  _label = static_cast<int>(c);
  if (label_fs.eof()) {
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

std::string STL_10_dataset::labelString() const {
  switch (_label) {
    case 1: return "airplane";
    case 2: return "bird";
    case 3: return "car";
    case 4: return "cat";
    case 5: return "deer";
    case 6: return "dog";
    case 7: return "horse";
    case 8: return "monkey";
    case 9: return "ship";
    case 10: return "truck";
    default: return "Invalid_Label";
  }
}

#endif  // INCLUDE_STL_10_HPP_
