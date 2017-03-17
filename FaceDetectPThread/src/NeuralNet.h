/*
 * NeuralNet.h
 *
 *  Created on: 8 janv. 2012
 *      Author: bruno
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#define NUM_HIDDEN_TYPES	3

#define NUM_UNITS_HIDDEN_1	4
#define NUM_UNITS_HIDDEN_2	16
#define NUM_UNITS_HIDDEN_3	6
#define NUM_UNITS_OUTPUT	1

#define HIDDEN_1_SIZE	(10*10)
#define HIDDEN_2_SIZE	(5*5)
#define HIDDEN_3_SIZE	(20*5)

#define BIAS_VALUE	1.0

#define NUM_UNITS_HIDDEN	(NUM_UNITS_HIDDEN_1 + NUM_UNITS_HIDDEN_2 + NUM_UNITS_HIDDEN_3)

enum { HIDDEN_1, HIDDEN_2, HIDDEN_3, OUTPUT };

enum { LOAD_WEIGHT, INIT_WEIGHT };

typedef struct {
	int		Type;
	float	*WeightVal;
	float	*PrevWeightVal;
	float	**InputValp;
	float	*OutputValp;
	float	*LocErrorValp;
	int		numConnections;
} Units;

typedef struct {
	float	*WeightVal;
	float	*PrevWeightVal;
	float	*InputVal;
	float	**InputAdr;
	float	*OutputVal;
	float	*ErrorVal;
	Units	*NetNeurones;
	int		numInputs;
	int		numUnits;
	int		numGroups;
	int		numConnections;
} Network;

void GetNetInfo (FILE *netFile, Network *Net);
void CreateNetStructure (Network *Net, int MaskSize);
long InitializeNet (FILE *netFile, FILE *weightFile, Network *Net);
long LoadNet (char FileName[], Network *Net, int WindowSize, int Load_Init);
int  SaveNet (char FileName[], Network *Net, long int numEpoch);


#endif /* NEURALNET_H_ */
