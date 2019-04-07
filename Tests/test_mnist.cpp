/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <string>
#include <SFML/Graphics.hpp>
#include "Mnist_db.h"

int main(int argc, char const *argv[]) {
  std::string database_pix("train-images.idx3-ubyte");
  std::string database_labels("train-labels-idx1-ubyte");

  sf::Image img;
  std::cout << "--------------------------------------" << std::endl;
  std::cout << "Read the database header" << std::endl;
  std::cout << "--------------------------------------" << std::endl;

  MNIST_Database mnist("train-images.idx3-ubyte",
                       "train-labels-idx1-ubyte", img);

  std::cout << "Magic number (pix data type) = " << mnist.magic1_data << std::endl;
  std::cout << "Magic number (pix dimension) = " << mnist.magic2_data << std::endl;
  std::cout << "Magic number (lab data type) = " << mnist.magic1_label << std::endl;
  std::cout << "Magic number (lab dimension) = " << mnist.magic2_label << std::endl;
  std::cout << "Number of data points = " << mnist.numdat << std::endl;
  std::cout << "Number of rows = " << mnist.nrows << std::endl;
  std::cout << "Number of cols = " << mnist.ncols << std::endl;

  mnist.next_pic();
  std::cout << "label= " << mnist.label << std::endl;
  sf::RenderWindow window;
  sf::Texture texture;
  texture.loadFromImage(img);
  sf::Sprite sprite(texture);
  window.create(sf::VideoMode(mnist.nrows, mnist.ncols), "Digit");

  window.clear();

  window.draw(sprite);

  window.display();

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) window.close();
      if (event.type == sf::Event::KeyPressed) {
        if (event.key.code == sf::Keyboard::Escape) window.close();
        if (event.key.code == sf::Keyboard::Return) {
          if (!mnist.next_pic()) window.close();
          std::cout << "label= " << mnist.label << std::endl;
          texture.loadFromImage(img);
        }
      }
    }
    window.clear();
    window.draw(sprite);
    window.display();
  }


  std::cout << "\nEND\n";
  return 0;
}
