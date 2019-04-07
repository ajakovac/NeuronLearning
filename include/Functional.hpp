/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#ifndef INCLUDE_FUNCTIONAL_HPP_
#define INCLUDE_FUNCTIONAL_HPP_

#include<algorithm>
#include<vector>
#include<type_traits>

template< typename F, typename A >
std::vector< typename std::result_of<F(A)>::type > map(F f, std::vector<A> &a) {
  std::vector< typename std::result_of<F(A)>::type > b(a.size());
  std::transform(a.begin(), a.end(), b.begin(), f);
  return b;
}

template< typename F, typename A, typename B >
std::vector< typename std::result_of<F(A, B)>::type >
zip(F f, std::vector<A> &a, std::vector<B> &b) {
  std::vector< typename std::result_of<F(A, B)>::type > c(a.size());
  std::transform(a.begin(), a.end(), b.begin(), c.begin(), f);
  return c;
}

template< typename F, typename A, typename B >
A foldl(F f, A a0, B &b) {
  A res = a0;
  for (auto bi : b) res = f(res, bi);
  return res;
}

#endif  // INCLUDE_FUNCTIONAL_HPP_
