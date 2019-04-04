#ifndef DISP_IMG_HPP_
#define DISP_IMG_HPP_

#include <SFML/Graphics.hpp>
#include <functional>

class Image_View
{
public: 
  inline Image_View(int hin, int win);
  inline Image_View(int hin, int win, std::function<void(Image_View&)> fin);
  inline void fillRGB(int* red, int* green, int* blue, bool reverse = false);
  inline void fillGreyscale(int* vals, bool reverse = false);
  inline void show();
  inline void close();
  inline void setTitle(const std::string& ti);
  
private:
  int _width;
  int _height;
  bool opened;
  sf::Image img;
  sf::RenderWindow window;
  sf::Texture texture;
  sf::Sprite sprite;
  std::function< void(Image_View&) > f;
  inline void update_image();
  std::string _title;
};

Image_View::Image_View(int hin, int win) :  _width(win), _height(hin), opened(false),
					    window(), sprite(), _title("")
{
  f = [](Image_View&) -> void {};
  img.create(win, hin);  
}

Image_View::Image_View(int hin, int win, std::function<void(Image_View&)> fin) 
  :  Image_View(hin, win)
{
  f = fin;
  img.create(win, hin);  
}

void Image_View::update_image()
{
  //f(*this);
  texture.loadFromImage(img);
  sprite.setTexture(texture);
  window.clear();
  window.draw(sprite);
  window.display();
}

void Image_View::show()
{
  if(opened) return;
  opened = true;
  // for(int i = 0; i < 32; ++i)
  //   for(int j = 0; j < 32; ++j) {
  //     img.setPixel(j, i, sf::Color(i % 256, j %256, (i+j) % 256));
  //   }
  if(_title == "")
    window.create(sf::VideoMode(_width, _height), "Image View");
  else window.create(sf::VideoMode(_width, _height), _title.c_str());
  f(*this);
  update_image();
  while(window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) 
      {
  	if (event.type == sf::Event::Closed) close();
  	if (event.type == sf::Event::KeyPressed) {
  	  if (event.key.code == sf::Keyboard::Escape) close();
  	  else if (event.key.code == sf::Keyboard::Return) f(*this);
  	}
      }
    window.clear();
    window.draw(sprite);
    window.display();
  }
}

void Image_View::close()
{
  window.close();
  opened = false;
}

void Image_View::fillRGB(int* r, int* g, int* b, bool reversed)
{
  for(int i = 0; i < _height; ++i)
    for(int j = 0; j < _width; ++j) {
      int n = i*_width + j;
      if(reversed) n = j*_height + i;
      img.setPixel(j, i, sf::Color(r[n], g[n], b[n]));
    }
  if(opened)
    update_image();
}

void Image_View::fillGreyscale(int* v, bool reversed)
{
  for(int i = 0; i < _height; ++i)
    for(int j = 0; j < _width; ++j) {
      int n = i*_width + j;
      if(reversed) n = j*_height + i;
      img.setPixel(j, i, sf::Color(v[n], v[n], v[n]));
    }
  if(opened) 
    update_image();
}

void Image_View::setTitle(const std::string& ti)
{
  _title = ti;
  if(opened)
    window.setTitle(_title.c_str());
}

#endif  // DISP_IMG_HPP_
