/*
 * TrainNetwork.h
 *
 *  Created on: 8 janv. 2012
 *      Author: bruno
 */

#ifndef TRAINNETWORK_H_
#define TRAINNETWORK_H_

#define LEARN_RATE (0.2)

void  CorrigeIllumination(float *ImageWindow, Byte *Mask, int MaskSize);
void  EgaliseHistogramme(float *ImageWindow, Byte *Mask, int MaskSize);
float CalculeNet(Network *Net);
float TrainNet(Network *Net, float LearnRate, float Momentum, float Error);

#endif /* TRAINNETWORK_H_ */
