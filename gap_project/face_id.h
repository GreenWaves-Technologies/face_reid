#ifndef __shufflenet_H__
#define __shufflenet_H__

#define __PREFIX(x) shufflenet ## x
// Include basic GAP builtins defined in the Autotiler
#include "at_api.h"

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 

extern AT_DEFAULTFLASH_EXT_ADDR_TYPE face_id_L3_Flash;

#define FACE_ID_W 112
#define FACE_ID_H 112
#define FACE_ID_C 3

#define FACE_ID_SIZE (FACE_ID_W*FACE_ID_H*FACE_ID_C)

#endif