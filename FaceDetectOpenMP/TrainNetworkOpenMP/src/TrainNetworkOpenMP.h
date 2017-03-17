/*
 * TrainNetworkOpenMP.h
 *
 *  Created on: 8 janv. 2012
 *      Author: bruno
 */

#ifndef TRAINNETWORKOPENMP_H_
#define TRAINNETWORKOPENMP_H_

#define LEARN_RATE (0.2)

void  CorrigeIllumination(float *ImageWindow, Byte *Mask, int MaskSize);
void  EgaliseHistogramme(float *ImageWindow, Byte *Mask, int MaskSize);
float CalculeNet(Network *Net);
float TrainNet(Network *Net, float LearnRate, float Momentum, float Error);

#endif /* TRAINNETWORKOPENMP_H_ */
