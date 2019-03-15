/* Copyright (C) AJ
 * Written by A. Jakovac 2018 */
#ifndef MNIST_DB_H_
#define MNIST_DB_H_

#include "Error.h"
#include <stdio.h>
#include <SFML/Graphics.hpp>

void inttohex(int in, int* out) {
  for (int i = 0; i < 8; i++) {
    out[i] = in%16;
    in = in/16;
  }
}

void inttobytes(int in, int* out) {
  for (int i = 0; i < 4; i++) {
    out[i] = in%256;
    in = in/256;
  }
}

int bytestoint(int* out) {
  int dd = out[0];
  for (int i = 1; i < 4; i++) dd = 256*dd+out[i];
  return dd;
}

// int readnextdigit() {
//   static int gg=0;
//   cout << "gg=" << gg << endl;
//   gg++;
//   if(gg==4) gg=0;
//   return 0;
// }


class MNIST_Database {
 public:
  MNIST_Database(const char * datafile, const char *labelfile, sf::Image& img);
  ~MNIST_Database();
  sf::Image& image() {return _image;}
  bool next_pic();
  void restart();
  int magic1_data, magic2_data;
  int magic1_label, magic2_label;
  int numdat;
  int nrows, ncols;
  int label;
 private:
  sf::Image& _image;
  FILE *PIX;
  FILE *LABL;
  int header[4];
  int lablheader[2];
  int bytes[4];
  int labelbytes[4];
  int labelbuf;
  int labelread;
  int pxsize;
  int * pixbuf;
};

MNIST_Database::MNIST_Database(
  const char * datafile,
  const char *labelfile,
  sf::Image &img
)
  : labelread(3), _image(img) {
  PIX = fopen(datafile, "r");
  int rr = fread(&header[0], sizeof(int), 4, PIX);
  if (rr != 4) throw(Error("MNIST_Database: wrong header"));

  inttobytes(header[0], bytes);
  magic1_data = bytes[2];
  magic2_data = bytes[3];

  inttobytes(header[1], bytes);
  numdat = bytestoint(bytes);

  inttobytes(header[2], bytes);
  nrows = bytestoint(bytes);

  inttobytes(header[3], bytes);
  ncols = bytestoint(bytes);

  LABL = fopen(labelfile, "r");
  rr = fread(&lablheader[0], sizeof(int), 2, LABL);
  if (rr != 2) throw(Error("MNIST_Database: wrong header"));

  inttobytes(lablheader[0], bytes);
  magic1_label = bytes[2];
  magic2_label = bytes[3];

  inttobytes(lablheader[1], bytes);
  if (numdat != bytestoint(bytes))
    throw(Error("Corrupt MNIST database: no matching files!"));

  _image.create(ncols, nrows);
  pxsize = nrows*ncols;
  pixbuf = new int[pxsize/4];
}

MNIST_Database::~MNIST_Database() {
  delete[] pixbuf;
  fclose(PIX);
  fclose(LABL);
}

bool MNIST_Database::next_pic() {
  if (labelread == 3) {
    int rr = fread(&labelbuf, sizeof(int), 1, LABL);
    if (rr !=1 ) return rr;
    // throw(Error("MNIST_Database::next_pic(): wrong data"));

    inttobytes(labelbuf, labelbytes);
    labelread = 0;
  } else {
    labelread++;
  }
  label = labelbytes[labelread];

  int rr = fread(pixbuf, sizeof(int), pxsize/4, PIX);
  for (int i = 0, ptr = 0; i < pxsize/4; i++, ptr+=4) {
    inttobytes(pixbuf[i], bytes);
    int row = (int)(ptr/ncols);
    int col = ptr - row*ncols;
    for (int j = 0; j < 4; j++) {
      int clr = 255-bytes[j];
      _image.setPixel(col+j, row, sf::Color(clr, clr, clr) );
    }
  }
  return rr;
}


void MNIST_Database::restart() {
  fseek(LABL, 8, SEEK_SET);
  labelread = 3;
  fseek(PIX, 16, SEEK_SET);
}



#endif  // MNIST_DB_H_
