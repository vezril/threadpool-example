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


void CorrigeIllumination(float *ImageWindow, Byte *Mask, int MaskSize);
void EgaliseHistogramme(float *ImageWindow, Byte *Mask, int MaskSize);

#if REDUCTION_BILINEAR == 1
Byte *ReduceImageBilinear(Byte *Image, int *Width, int *Height, float scale);
#endif
#if REDUCTION_BICUBIC == 1
Byte *ReduceImageBicubic(Byte *Image, int *Width, int *Height, float scale);
#endif

float CalculeNet(Network *Net);


#endif /* FACEDETECT_H_ */
