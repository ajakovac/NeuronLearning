/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#ifndef INCLUDE_ERROR_HPP_
#define INCLUDE_ERROR_HPP_

#include <string>

class Error {
 public:
  explicit Error(char * u) : error_message(u) {}
  explicit Error(const char * u) : error_message(u) {}
  explicit Error(std::string &s) : error_message(s.c_str()) {}
  explicit Error(const std::string &s) : error_message(s.c_str()) {}
  const char * error_message;
};

#endif  // INCLUDE_ERROR_HPP_
