/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2019 */
#ifndef INCLUDE_IMAGELAYER_HPP_
#define INCLUDE_IMAGELAYER_HPP_

#include <iostream>
#include <vector>
#include <SFML/System.hpp>
#include <SFML/Graphics.hpp>
#include "Structure.hpp"

constexpr int NCOLOR = 255;

void Image_to_Layer(Network *N, int lyn, sf::Image *img) {
  Shape sh = N->layerShape(lyn);
  sf::Vector2u size = img->getSize();
  if (sh[1] != size.x || sh[1] != size.y)
    throw(Error("Image_to_Layer: size mismatch"));

  // there is no color channel
  if (sh.size() == 2) {
    for (int i = 0; i < sh[0]; i++) {
      for (int j = 0; j < sh[1]; j++) {
        sf::Color clr = img->getPixel(i, j);
        int sn = N->getsiteID(lyn, {i, j});
        N->site(sn) =  static_cast<double>((clr.r + clr.g + clr.b)/(3*NCOLOR));
      }
    }
  // there is only one color
  } else if (sh.size() ==3 && sh[2] == 1) {
    for (int i = 0; i < sh[0]; i++) {
      for (int j = 0; j < sh[1]; j++) {
        sf::Color clr = img->getPixel(i, j);
        int sn = N->getsiteID(lyn, {i, j, 0});
        N->site(sn) =  static_cast<double>((clr.r + clr.g + clr.b)/(3*NCOLOR));
      }
    }
  // there are three colors
  } else if (sh.size() == 3 && sh[2] == 3) {
    for (int i = 0; i < sh[0]; i++) {
      for (int j = 0; j < sh[1]; j++) {
        sf::Color clr = img->getPixel(i, j);
        int sn = N->getsiteID(lyn, {i, j, 0});
        N->site(sn) =  static_cast<double>(clr.r)/NCOLOR;
        sn = N->getsiteID(lyn, {i, j, 1});
        N->site(sn) =  static_cast<double>(clr.g)/NCOLOR;
        sn = N->getsiteID(lyn, {i, j, 2});
        N->site(sn) =  static_cast<double>(clr.b)/NCOLOR;
      }
    }
  }
}

void Layer_to_Image(Network *N, int lyn, sf::Image *img) {
  Shape sh = N->layerShape(lyn);
  sf::Vector2u size = img->getSize();
  if (sh[1] != size.x || sh[1] != size.y)
    throw(Error("Layer_to_Image: size mismatch"));

  // there is no color channel
  if (sh.size() == 2) {
    for (int i = 0; i < sh[0]; i++) {
      for (int j = 0; j < sh[1]; j++) {
        int sn = N->getsiteID(lyn, {i, j});
        int clrbw = static_cast<int>(NCOLOR*N->site(sn));
        sf::Color clr;
        clr.r = clrbw;
        clr.g = clrbw;
        clr.b = clrbw;
        img->setPixel(i, j, clr);
      }
    }
  // there is only one color
  } else if (sh.size() ==3 && sh[2] == 1) {
    for (int i = 0; i < sh[0]; i++) {
      for (int j = 0; j < sh[1]; j++) {
        int sn = N->getsiteID(lyn, {i, j, 0});
        int clrbw = static_cast<int>(NCOLOR*N->site(sn));
        sf::Color clr;
        clr.r = clrbw;
        clr.g = clrbw;
        clr.b = clrbw;
        img->setPixel(i, j, clr);
      }
    }
  // there are three colors
  } else if (sh.size() == 3 && sh[2] == 3) {
    for (int i = 0; i < sh[0]; i++) {
      for (int j = 0; j < sh[1]; j++) {
        sf::Color clr;
        int sn = N->getsiteID(lyn, {i, j, 0});
        clr.r = static_cast<int>(NCOLOR*N->site(sn));
        sn = N->getsiteID(lyn, {i, j, 1});
        clr.g = static_cast<int>(NCOLOR*N->site(sn));
        sn = N->getsiteID(lyn, {i, j, 2});
        clr.b = static_cast<int>(NCOLOR*N->site(sn));
        img->setPixel(i, j, clr);
      }
    }
  }
}

#endif  // INCLUDE_IMAGELAYER_HPP_
