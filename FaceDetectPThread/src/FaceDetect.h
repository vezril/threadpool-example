/*
 * FaceDetect.h
 *
 *  Created on: 3 janv. 2012
 *      Author: bruno
 */

#ifndef FACEDETECT_H_
#define FACEDETECT_H_

#define REDUCTION_SCALE (1.2)
#define NUM_SCALE		(10)


#define REDUCTION_BILINEAR  0
#define REDUCTION_BICUBIC	1
#if ((REDUCTION_BILINEAR == 1) && (REDUCTION_BICUBIC == 1))
#error "Erreur : Les reductions BILINEAR et BICUBIC ont été choisies ensembles !"
#endif

void CorrigeIllumination(float *ImageWindow, Byte *MaskImage, int MaskSize, int threadID);
void EgaliseHistogramme(float *ImageWindow, Byte *MaskImage, int MaskSize, int threadID);

Byte *ReduceImageBilinear(Byte *Image, int *Width, int *Height, float scale);
Byte *ReduceImageBicubic(Byte *Image, int *Width, int *Height, float scale);

float CalculeNet(Network *NetLocal);

#endif /* FACEDETECT_H_ */
