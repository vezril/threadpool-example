/*
 * ImageUtil.h
 *
 *  Created on: 8 janv. 2012
 *      Author: bruno
 */

#ifndef IMAGEUTIL_H_
#define IMAGEUTIL_H_

typedef unsigned char Byte;

typedef struct {
	int		x;
	int		y;
}PosXY;

typedef struct {
	char	*Name;
	Byte	*Image;
	float	*ImageFloat;
	int		 Width;
	int		 Height;
	float	 Response;
} ImageStruct;


void  SaveByteImagePgm(char FileName[], Byte *Image, int Width, int Height);
Byte *LoadByteImagePgm(char FileName[], int *Width, int *Height);
Byte *ReduceImageBilinear(Byte *Image, int *Width, int *Height, float scale);
Byte *ReduceImageBicubic(Byte *Image, int *Width, int *Height, float scale);
int   GetImageFileNames (const char *Directory, char ***FileNames);
int   GetImageFiles (char *RepName, char ***FileNames, ImageStruct **Images);
void  FreeImageFiles (char ***FileNames, ImageStruct **Images, int numFiles);

#endif /* IMAGEUTIL_H_ */
