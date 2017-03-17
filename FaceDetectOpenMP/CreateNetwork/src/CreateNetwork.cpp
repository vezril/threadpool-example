//============================================================================
// Name        : CreateNetwork.cpp
// Author      : Bruno De Kelper
// Version     :
// Copyright   : Your copyright notice
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "NeuralNet.h"
#include "CreateNetwork.h"

int main(int argc, char *argv[]) {
	FILE		*netFile;
	char		*NetName, nom[1024];
	Network		Net;
	int			WindowSize;
	int			numUnits, numInputs, numGroups, numConnections;
	int			numH1UnitConn, numH2UnitConn, numH3UnitConn;
	int			numH1Conn, numH2Conn, numH3Conn;
	int			from, to, gr, offset, H3offset;

	if (argc < 4) {
		printf("Erreur : La commande devrait être :\n");
		printf("            CreateNetwork <nom du réseau> <Taille de la fenêtre> <nombre de groupe>\n");
		exit(1);
	}

	NetName    = argv[1];
	WindowSize = atoi(argv[2]);
	numGroups  = atol(argv[3]);

	numUnits  = numGroups*(NUM_UNITS_HIDDEN) + NUM_UNITS_OUTPUT;
	numInputs = (WindowSize*WindowSize)+1;

	numH1UnitConn = 1 + (WindowSize*WindowSize)/4;
	numH2UnitConn = 1 + (WindowSize*WindowSize)/16;
	numH3UnitConn = 1 + (WindowSize*WindowSize)/4;

	numH1Conn = NUM_UNITS_HIDDEN_1*numH1UnitConn;
	numH2Conn = NUM_UNITS_HIDDEN_2*numH2UnitConn;
	numH3Conn = NUM_UNITS_HIDDEN_3*numH3UnitConn;

	numConnections = numGroups*(numH1Conn + numH2Conn + numH3Conn + NUM_UNITS_HIDDEN) + 1;

	sprintf(nom, "%s.net", NetName);
	netFile = fopen(nom,"w");

    fprintf(netFile,"%d units\n", numUnits);
    fprintf(netFile,"%d inputs\n", numInputs);
    fprintf(netFile,"%d connections\n", numConnections);

	offset = numInputs;
	// Pour H1
    for (gr = 0; gr < numGroups; gr++) {
        for (from = 0; from < numInputs-1; from++) {
        	to = ((2*from/WindowSize) % 2) + 2*(2*from/(numInputs-1));
            fprintf(netFile,"%d %d\n", from, to + offset);
        }
        for (to = 0; to < NUM_UNITS_HIDDEN_1; to++) {
            fprintf(netFile,"%d %d\n", numInputs-1, to + offset);
        }
    	offset += NUM_UNITS_HIDDEN_1;
    }
    // Pour H2
    for (gr = 0; gr < numGroups; gr++) {
        for (from = 0; from < numInputs-1; from++) {
        	to = ((4*from/WindowSize) % 4) + 4*(4*from/(numInputs-1));
            fprintf(netFile,"%d %d\n", from, to + offset);
        }
        for (to = 0; to < NUM_UNITS_HIDDEN_2; to++) {
            fprintf(netFile,"%d %d\n", numInputs-1, to + offset);
        }
    	offset += NUM_UNITS_HIDDEN_2;
    }
    // Pour H3
    H3offset = (numInputs-1)*(NUM_UNITS_HIDDEN_3-4)/(4*(NUM_UNITS_HIDDEN_3-1));
    for (gr = 0; gr < numGroups; gr++) {
        for (from = 0; from < 3*(numInputs-1)/2; from++) {
        	to = (4*from/(numInputs-1));
            fprintf(netFile,"%d %d\n", from - to*H3offset, to + offset);
        }
        for (to = 0; to < NUM_UNITS_HIDDEN_3; to++) {
            fprintf(netFile,"%d %d\n", numInputs-1, to + offset);
        }
    	offset += NUM_UNITS_HIDDEN_3;
    }
    // Pour Output
	offset -= numGroups*NUM_UNITS_HIDDEN;
    for (gr = 0; gr < numGroups; gr++) {
        for (from = 0; from < NUM_UNITS_HIDDEN_1; from++) {
            fprintf(netFile,"%d %d\n", from + offset, numInputs + numUnits - 1);
        }
    	offset += NUM_UNITS_HIDDEN_1;
    }
    for (gr = 0; gr < numGroups; gr++) {
        for (from = 0; from < NUM_UNITS_HIDDEN_2; from++) {
            fprintf(netFile,"%d %d\n", from + offset, numInputs + numUnits - 1);
        }
    	offset += NUM_UNITS_HIDDEN_2;
    }
    for (gr = 0; gr < numGroups; gr++) {
        for (from = 0; from < NUM_UNITS_HIDDEN_3; from++) {
            fprintf(netFile,"%d %d\n", from + offset, numInputs + numUnits - 1);
        }
    	offset += NUM_UNITS_HIDDEN_3;
    }
    fprintf(netFile,"%d %d\n", numInputs-1, offset);

    fclose(netFile);

    LoadNet(NetName, &Net, WindowSize, INIT_WEIGHT);
	SaveNet(NetName, &Net, 0);

	free(Net.WeightVal);
	free(Net.InputVal);
	free(Net.InputAdr);
	free(Net.ErrorVal);
	free(Net.NetNeurones);

	return 0;
}
