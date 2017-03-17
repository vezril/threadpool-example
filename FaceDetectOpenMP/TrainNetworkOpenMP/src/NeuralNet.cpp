/*
 * NeuralNet.cpp
 *
 *  Created on: 8 janv. 2012
 *      Author: bruno
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "NeuralNet.h"



void GetNetInfo (FILE *netFile, Network *Net) {
	int	numUnits, numInputs;

    fscanf(netFile,"%d units\n", &numUnits);
    fscanf(netFile,"%d inputs\n", &numInputs);
    Net->numUnits = numUnits;
	Net->numGroups = (Net->numUnits-NUM_UNITS_OUTPUT)/NUM_UNITS_HIDDEN;
	Net->numInputs = numInputs;

	fscanf(netFile,"%d connections\n",&(Net->numConnections));
}


void CreateNetStructure (Network *Net, int MaskSize) {
	float	*tpW, *tpPW, **tpI, *tpO, *tpE;
	int		H1Size, H2Size, H3Size;
	int     type, inc, num;
	int		i, j, n;

	// Réserve les espaces de mémoire du réseau au complet
	Net->WeightVal     = (float *)  calloc(Net->numConnections, sizeof(float));
	Net->PrevWeightVal = (float *)  calloc(Net->numConnections, sizeof(float));
	Net->InputVal      = (float *)  calloc(Net->numInputs+Net->numUnits, sizeof(float));
	Net->InputAdr      = (float **) calloc(Net->numConnections, sizeof(float *));
	Net->ErrorVal      = (float *)  calloc(Net->numUnits, sizeof(float));
	Net->NetNeurones   = (Units *)  calloc(Net->numUnits, sizeof(Units));

	// Le vecteur des sorties d'une couche de neurones
	// est le vecteur des entrées de la couche de neurones suivante
	Net->OutputVal	= &(Net->InputVal[Net->numInputs]);

	Net->InputVal[Net->numInputs-1] = BIAS_VALUE;

	H1Size = (MaskSize*MaskSize)/4;
	H2Size = (MaskSize*MaskSize)/16;
	H3Size = (MaskSize*MaskSize)/4;
	// Chaque neurone "reçoit" un morceau des vecteurs réservés pour le réseau
	tpW  = &(Net->WeightVal[0]);
	tpPW = &(Net->PrevWeightVal[0]);
	tpI  = &(Net->InputAdr[0]);
	tpO  = &(Net->OutputVal[0]);
	tpE  = &(Net->ErrorVal[0]);
	n = 0;
	for (type = HIDDEN_1; type <= HIDDEN_3; type++) {
		switch (type) {
			case HIDDEN_1 :	inc = H1Size + 1;
							num = NUM_UNITS_HIDDEN_1;
							break;
			case HIDDEN_2 :	inc = H2Size + 1;
							num = NUM_UNITS_HIDDEN_2;
							break;
			case HIDDEN_3 :	inc = H3Size + 1;
							num = NUM_UNITS_HIDDEN_3;
							break;
		}
		for (i = 0; i < Net->numGroups; i++)
			for (j = 0; j < num; j++) {
				Net->NetNeurones[n].Type = type;
				Net->NetNeurones[n].WeightVal     = tpW;
				Net->NetNeurones[n].PrevWeightVal = tpPW;
				Net->NetNeurones[n].InputValp     = tpI;
				Net->NetNeurones[n].OutputValp    = tpO;
				Net->NetNeurones[n].LocErrorValp  = tpE;
				tpW  += inc;
				tpPW += inc;
				tpI  += inc;
				tpO  += 1;
				tpE  += 1;
				n++;
			}
	}
	// Output Neurone
	Net->NetNeurones[n].Type		  = OUTPUT;
	Net->NetNeurones[n].WeightVal     = tpW;
	Net->NetNeurones[n].PrevWeightVal = tpPW;
	Net->NetNeurones[n].InputValp     = tpI;
	Net->NetNeurones[n].OutputValp    = tpO;
	Net->NetNeurones[n].LocErrorValp  = tpE;
}


long InitializeNet (FILE *netFile, FILE *weightFile, Network *Net) {
	int		from, to, num;
	float	weight;
	long	numEpoch = 0;
	int		i;

	if (weightFile != NULL) {		// Si un fichier de "poids" est fournit, ignore ces infos
		fscanf(weightFile, "%ld epochs\n", &numEpoch);
		fscanf(weightFile, "%*d weights\n");
	}
	for (i = 0; i < Net->numConnections; i++) {
		fscanf(netFile, "%d %d\n", &from, &to);
		if (weightFile != NULL)						// Si un fichier de "poids" est fournit
			fscanf(weightFile, "%e\n", &weight);
		else										// sinon choisit un poids au hazard
			weight = 2.0*(((float)random())/RAND_MAX) - 1.0;
		// Établit les connections des neurones
		to -= Net->numInputs;
		num = Net->NetNeurones[to].numConnections;
		Net->NetNeurones[to].InputValp[num] = &(Net->InputVal[from]);
		Net->NetNeurones[to].WeightVal[num] = weight;
		Net->NetNeurones[to].numConnections++;
	}
	memcpy(Net->PrevWeightVal, Net->WeightVal, Net->numConnections*sizeof(float));
	return numEpoch;
}

long LoadNet (char FileName[], Network *Net, int WindowSize, int Load_Init) {
	FILE	*netFile, *weightFile;
	char	name[1024];
	long	numEpoch = 0;

    sprintf(name, "%s.net", FileName);
    netFile	= fopen(name, "rb");
	if (netFile == NULL)
		return -1;

	GetNetInfo(netFile, Net);
	CreateNetStructure(Net, WindowSize);

	if (Load_Init == LOAD_WEIGHT) {			// Lecture des poids existants
		sprintf(name, "%s.wet", FileName);
		weightFile = fopen(name, "rb");
		if (weightFile == NULL)
			return -2;
		numEpoch = InitializeNet (netFile, weightFile, Net);
		fclose(weightFile);
	}

	if (Load_Init == INIT_WEIGHT)			// Initialisation au hazard des poids
		numEpoch = InitializeNet (netFile, NULL, Net);

	fclose(netFile);

	return numEpoch;
}


int SaveNet (char FileName[], Network *Net, long int numEpoch) {
	FILE	*weightFile, *netFile;
	char	name[1024];
	long	n;
	int		*num, from, to;

	sprintf(name, "%s.net", FileName);
	netFile = fopen(name, "rb");
	if (netFile == NULL)
		return 2;
	GetNetInfo(netFile, Net);

	sprintf(name, "%s.wet", FileName);
	weightFile = fopen(name, "wb");
	if (weightFile == NULL)
		return 2;

	// Ajout d'info inutile
	fprintf(weightFile, "%lu epochs\n", numEpoch);
	fprintf(weightFile, "%u weights\n", Net->numConnections);
	num = (int *) calloc(Net->numInputs, sizeof(int));
	for (n = 0; n < Net->numConnections; n++) {
		fscanf(netFile, "%d %d\n", &from, &to);
		to -= Net->numInputs;
		fprintf(weightFile, "%e\n", Net->NetNeurones[to].WeightVal[num[to]]);
		num[to]++;
	}
	free(num);
	fclose(weightFile);

	return 0;
}
