//
//  File:	error.h
//  Id:		$Id$
//  Author:	Antal Jakovac
//
//  Description: the basic class of eror handling
//
#ifndef ERROR_H
#define ERROR_H

#include <string>

class Error {
public:
  Error(char * u) : error_message(u) {};
  Error(const char * u) : error_message(u) {};
  Error(std::string &s) : error_message(s.c_str()) {};
  Error(const std::string &s) : error_message(s.c_str()) {};
  const char * error_message;
};

#endif //ERROR_H
