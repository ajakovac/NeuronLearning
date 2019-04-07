#ifndef DATASET_HPP_
#define DATASET_HPP_

enum color
  {
    Red = 0, Green = 1, Blue = 2
  };

class dataset
{
public:
  virtual void restart();
  virtual bool next() = 0;
  
  inline dataset(const std::string& filename_in, int w, int h);
  inline dataset(const std::string& fin, const std::string& Lfin,int w, int h);
  inline virtual ~dataset() {fs.close(); if(label_fs.is_open()) label_fs.close();}
  inline int label() const {return _label;}
  inline int value(int i, int j , color c) const;
  inline int* data(color);
  inline const std::string& filename() const {return _filename;}

  //  const int pixel_num = 1024;
  const int width;
  const int height;

protected:
  std::string _filename;
  std::ifstream fs;
  std::vector< int > dat_red;
  std::vector< int > dat_green;
  std::vector< int > dat_blue;
  int _label;
  std::string _label_file_name;
  std::ifstream label_fs;
};

dataset::dataset(const std::string& fname, int w, int h)
  : width(w), height(h),  _filename(fname), _label(-1), _label_file_name("") 
{
  int pixel_num = height*width;
  fs.open(fname, std::ios_base::binary);
  dat_red.resize(pixel_num);
  dat_green.resize(pixel_num);
  dat_blue.resize(pixel_num);
}

dataset::dataset(const std::string& fname, const std::string& Lfname, int w, int h)
  : dataset(fname, w, h)
{
  _label_file_name = Lfname;
  label_fs.open(Lfname, std::ios_base::binary);
}

void dataset::restart()
{
  fs.clear();
  fs.seekg(0);
  if(label_fs.is_open())
    {
      label_fs.clear();
      label_fs.seekg(0);
    }
}

int dataset::value(int i, int j , color c) const
{
  if(c == Red)
    return dat_red[i*width + j];
  if(c == Green)
    return dat_green[i*width + j];
  if(c == Blue)
    return dat_blue[i*width + j];
  return 0;
}

int* dataset::data(color c)
{
   if(c == Red)
     return dat_red.data();
   if(c == Green)
     return dat_green.data();
   if(c == Blue)
     return dat_blue.data();
   return nullptr;
}


#endif  // DATASET_HPP_
