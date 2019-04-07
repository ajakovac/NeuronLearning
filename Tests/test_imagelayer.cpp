/* Copyright (C) NeuronLearning_project
 * Written by A. Jakovac 2018 */
#include <iostream>
#include <cmath>
#include <SFML/Graphics.hpp>
#include "ImageLayer.hpp"
#include "Error.hpp"
#include "Update.hpp"

const double pi = 3.1415926535897932384626433832795028841971693993751;

int main()
try {
  sf::Image img;
  if (!img.loadFromFile("indexa.jpeg")) return EXIT_FAILURE;
  sf::Vector2u size = img.getSize();
  int szx = static_cast<int>(size.x);
  int szy = static_cast<int>(size.y);
  std::cout << "Image size=("<< size.x<< "," << size.y << ")\n";
  sf::Color clr = img.getPixel(0, 0);
  std::cout << "color of (100,100) point = ("
       << static_cast<int>(clr.r) << ", "
       << static_cast<int>(clr.g) << ", "
       << static_cast<int>(clr.b) << ", "
       << static_cast<int>(clr.a) << ")\n";
  Network ntw;
  int basely = ntw.AddLayer({szx, szy, 3});
  Image_to_Layer(&ntw, basely, &img);

  int resly = ntw.AddLayer({szx, szy, 3});
  ntw.ConnectLastLayer([=](Network *N, int an, std::vector<int> *v) {
    Position off = {szx/2, szy/2, 0};
    Position p0 = N->sitePos(an);
    Position p = p0 -off;
    double cf = std::cos(pi/6);
    double sf = std::sin(pi/6);
    Position prot = { static_cast<int>(cf*p[0] - sf*p[1]),
                      static_cast<int>(sf*p[0] + cf*p[1]), p[2]};
    prot = prot+off;
    if ( (prot >= 0) && (prot < N->layerShape(1)) )
      v->push_back(N->getsiteID(0, prot));
  }, const_cf(1.0));

  auto upd = affine_nonlin_update(id_fn);
  ntw.applytoLayer(resly, [&](int n){ upd(&ntw, n); });

  sf::Image img1;
  img1.create(size.x, size.y);
  Layer_to_Image(&ntw, resly, &img1);

  sf::RenderWindow window;
  sf::Texture texture;
  texture.loadFromImage(img1);
  sf::Sprite sprite(texture);
  window.create(sf::VideoMode(size.x, size.y), "My hedgehog!");

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed) window.close();
      if (event.type == sf::Event::KeyPressed)
      if (event.key.code == sf::Keyboard::Escape) window.close();
    }

    window.clear();

    window.draw(sprite);

    window.display();
  }
  return 0;
} catch(Error & u) {
  std::cerr << u.error_message << std::endl;
}
