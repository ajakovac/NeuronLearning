#include "disp_img.hpp"
#include "cifar_10.hpp"
#include "stl_10.hpp"
#include "mnist.hpp"

#include <iostream>
#include <sstream>
#include <string> 

int main()
{

  using namespace std;  

  CIFAR_10_dataset *d = new CIFAR_10_dataset("./cifar-10-batches-bin/data_batch_1.bin");

  STL_10_dataset *s = new STL_10_dataset("./stl10_binary/train_X.bin",
					 "./stl10_binary/train_y.bin");
 
  MNIST_dataset *m = new MNIST_dataset("./mnist/train_images",
				       "./mnist/train_labels");

  auto update1 = [=](Image_View& v)
    {
      d->next();
      v.fillRGB(d->data(Red), d->data(Green), d->data(Blue));
      v.setTitle(d->labelString());
    };

  auto update2 = [=](Image_View& v)
    {
      s->next();
      v.fillRGB(s->data(Red), s->data(Green), s->data(Blue), true);
      v.setTitle(s->labelString());
    };

  auto update3 = [=](Image_View& v)
    {
      m->next();
      v.fillGreyscale(m->data(Red));
      v.setTitle( std::to_string(m->label()) );
    };
  
  Image_View view(32, 32, update1);
  view.show();
  Image_View view1(96, 96, update2);
  view1.show();
  Image_View MNIST_view(28, 28, update3);
  MNIST_view.show();  

  delete m;
  delete s;
  delete d;
}
