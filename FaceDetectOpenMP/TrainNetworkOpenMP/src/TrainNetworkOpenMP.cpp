//============================================================================
// Name        : TrainNetworkOpenMP.cpp
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
#include "ImageUtil.h"
#include "TrainNetworkOpenMP.h"

#include "signal.h"

#include "omp.h"

#define GET_TSC(T) { __asm__ volatile ("push %eax"); \
					 __asm__ volatile ("push %ebx"); \
					 __asm__ volatile ("push %ecx"); \
					 __asm__ volatile ("push %edx"); \
					 __asm__ volatile ("cpuid"); \
					 __asm__ volatile ("rdtsc":"=A"(T)); \
					 __asm__ volatile ("pop %edx"); \
					 __asm__ volatile ("pop %ecx"); \
					 __asm__ volatile ("pop %ebx"); \
					 __asm__ volatile ("pop %eax"); \
				   }

volatile unsigned long long t1, t2;

// Pour la détection du CTRL-C
volatile int Terminate = 0;
void trap(int signal) {
	Terminate = 1;
}



void CorrigeIllumination(float *ImageWindow, Byte *Mask, int MaskSize) {
	static  int		LCMinit = 1;
	static	float	LCM[3][3];
	float	vec1_0, vec1_1, vec1_2;   // Part of affine fitting
	float	vec2[3];   // Part of affine fitting
	int		halfX = (MaskSize/2);
	int		halfY = (MaskSize/2);
	float	xi, yi;
	int		i;

	if (LCMinit) {	// Si la matrice de correction n'est pas initialisée
		float	A = 0.0, B = 0.0, C = 0.0;

		LCMinit = 0;	// Calcule les termes essentiels de la matrice
		for (i = 0; i < MaskSize*MaskSize; i++) {
			xi = (float)((i % MaskSize) - halfX);
			yi = (float)((i / MaskSize) - halfY);
			if (Mask[i] > 0) {
				A += xi*xi;
				B += xi*yi;
				C += yi*yi;
			}
		}				// Calcule l'inverse de la matrice (formule prédéterminée)
		LCM[0][0] = 1.0/(A-B);
		LCM[0][1] = 0.0;
		LCM[0][2] = 1.0/(2.0*(A-B));
		LCM[1][0] = 0.0;
		LCM[1][1] = 1.0/(C-B);
		LCM[1][2] = 1.0/(2.0*(C-B));
		LCM[2][0] = 1.0/(2.0*(A-B));
		LCM[2][1] = 1.0/(2.0*(C-B));
		LCM[2][2] = (A*C-B*B)/(4.0*B*(A-B)*(C-B));
	}

  #pragma omp parallel
  {
	// Calcul la fonction affine - Étape 1
	#pragma omp for private(xi, yi) reduction(+ : vec1_0, vec1_1, vec1_2)
	for (i = 0; i < MaskSize*MaskSize; i++) {
		if (Mask[i] > 0) {
			xi = (float)((i % MaskSize) - halfX);
			yi = (float)((i / MaskSize) - halfY);
			vec1_0 += (xi*ImageWindow[i]);
			vec1_1 += (yi*ImageWindow[i]);
			vec1_2 += (ImageWindow[i]);
		}
	}

	// Calcul la fonction affine - Étape2
	#pragma omp for
	for (i = 0; i < 3; i++)
		vec2[i] = LCM[i][0]*vec1_0 + LCM[i][1]*vec1_1 + LCM[i][2]*vec1_2;

	// Applique la correction affine
	#pragma omp for private(xi, yi)
	for (i = 0; i < MaskSize*MaskSize; i++) {
		xi = (float)((i % MaskSize) - halfX);
		yi = (float)((i / MaskSize) - halfY);
		ImageWindow[i] = (float) floor(ImageWindow[i] - (xi*vec2[0] + yi*vec2[1] + vec2[2] - 128.5f));
		ImageWindow[i] = (ImageWindow[i] < 0.0f) ? 0.0f : ((ImageWindow[i] > 255.0f) ? 255.0f : ImageWindow[i]);
	}
  }
}


void EgaliseHistogramme(float *ImageWindow, Byte *Mask, int MaskSize) {
	long	hist[256];
	float	scaleFactor;
	long	min;
	int		i, j;

  #pragma omp parallel private(j)
  { long	hh[256];
    int		id, num;

    id = omp_get_thread_num();
    num = 256/omp_get_num_threads();
	memset(&hist[id*num], 0, num*sizeof(long));

	memset(hh, 0, 256*sizeof(long));
	#pragma omp for
	for (i = 0; i < MaskSize*MaskSize; i++)
		if (Mask[i] > 0)
			hh[(Byte)(ImageWindow[i])]++;

	// Calcul l'histogramme-cumulatif
	hist[0] += hh[0];
	for (j = 1; j < 256; j++) {
		hh[j] += hh[j-1];
		hist[j] += hh[j];
	}
	#pragma omp master
	{
	min = 0;
	for (i = 0; i < 256; i++)
		if (hist[i] > 0) {
			min = hist[i];
			break;
		}
	}
	#pragma omp barrier

	// Applique l'égalisation d'histogramme
    if (hist[255] > min) {
        scaleFactor = 255.0/(hist[255]-min);
		#pragma omp for
        for (i = 0; i < MaskSize*MaskSize; i++)
            ImageWindow[i] = (float) floor((float) ((hist[(Byte)(ImageWindow[i])]-min)*scaleFactor));
    }
  }
}


float CalculeNet(Network *Net) {
	float	tp;
	Units  *Neurone;
	int		i, j;

  #pragma omp parallel
  {
    #pragma omp for private(Neurone, tp, j)	// Calcule la somme des entrées, pondérée par les poids
	for (i = 0; i < Net->numUnits-1; i++) {
		Neurone = &(Net->NetNeurones[i]);
		tp = 0.0;							// Calcule la somme des entrées, pondérée par les poids
		for (j = 0; j < Neurone->numConnections; j++)
			tp += (float) ((Neurone->WeightVal[j])*(*(Neurone->InputValp[j])));
		*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide
	}
	Neurone = &(Net->NetNeurones[Net->numUnits-1]);
    #pragma omp for reduction(+ : tp)	// Calcule la somme des entrées, pondérée par les poids
	for (j = 0; j < Neurone->numConnections; j++)
		tp += (float) ((Neurone->WeightVal[j])*(*(Neurone->InputValp[j])));
  }
	*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide
	return Net->OutputVal[Net->numUnits-1];
}


float TrainNet(Network *Net, float LearnRate, float Momentum, float Error) {
	float  *BackwardError, WeightDiff;
	Units  *Neurone;
	int		i, j;

  #pragma omp parallel
  { int  id, num;

    id = omp_get_thread_num();
    num = Net->numUnits/omp_get_num_threads();
	memset(&(Net->ErrorVal[id*num]), 0, num*sizeof(float));		// Mettre toutes les erreurs à zéro

    #pragma omp master
    {
	Neurone = &(Net->NetNeurones[Net->numUnits-1]);			// Choisit le neurone de sortie et
	*(Neurone->LocErrorValp) = Error*(1.0-(*(Neurone->OutputValp))*(*(Neurone->OutputValp)));	// assigne l'erreur du réseau à cet neurone
	BackwardError = Neurone->LocErrorValp - (Neurone->numConnections - 1);
    }
    #pragma omp barrier
    #pragma omp for
	for (j = 0; j < Neurone->numConnections-1; j++)	// Rétropropage l'erreur aux neurones précédents
		BackwardError[j] += ((Neurone->WeightVal[j])*(*(Neurone->LocErrorValp)));
    #pragma omp for private(Neurone)
	for (i = Net->numUnits-2; i >= 0; i--) {	// Calcule le gradient d'erreur local
		Neurone = &(Net->NetNeurones[i]);
		*(Neurone->LocErrorValp) *= (1.0-(*(Neurone->OutputValp))*(*(Neurone->OutputValp)));
	}
    #pragma omp for private(Neurone, WeightDiff, j)
	for (i = 0; i < Net->numUnits; i++) {		// Applique la correction des poids
		Neurone = &(Net->NetNeurones[i]);
		for (j = 0; j < Neurone->numConnections; j++) {
			WeightDiff = Neurone->WeightVal[j] - Neurone->PrevWeightVal[j];
			Neurone->PrevWeightVal[j] = Neurone->WeightVal[j];
			Neurone->WeightVal[j] += LearnRate*(*(Neurone->LocErrorValp))*(*(Neurone->InputValp[j])) + Momentum*WeightDiff;
		}
	}
  }
	return Error;
}


int main(int argc, char *argv[]) {
	char		*LearnRepName;
	char		*MaskName;
	char		*NetName;
	char 		**LearnFileNames = NULL;
	ImageStruct	*LearnImages;
	ImageStruct	Mask;
	int			numLearnFiles = 0;
	int			numTestFiles = 0;

	Network		Net;
	float		*Window;
	float		Reponse, LearnRate, Momentum;
	float		NetErreur, TotalErreur, TErreur, FErreur;
	long		epoch, numEpoch;
	int			m, n, Mode = INIT_WEIGHT;

	if (argc < 8) {
		printf("Erreur : La commande devrait être :\n");
		printf("            TrainNetworkFast <répertoire d'apprentissage> <nom du masque> <nom du réseau> <Taux d'apprentissage> <Momentum> <numEpoch> <Mode=(LOAD,INIT)>\n");
		exit(1);
	}

	LearnRepName = argv[1];
	MaskName     = argv[2];
	NetName      = argv[3];
	LearnRate    = atof(argv[4]);
	Momentum     = atof(argv[5]);
	numEpoch     = atol(argv[6]);
	if ((strcmp(argv[7],"INIT") == 0)||(strcmp(argv[7],"init") == 0))
		Mode = INIT_WEIGHT;
	if ((strcmp(argv[7],"LOAD") == 0)||(strcmp(argv[7],"load") == 0))
		Mode = LOAD_WEIGHT;

	numLearnFiles = GetImageFiles(LearnRepName, &LearnFileNames, &LearnImages);
	printf("numLearnFiles = %u    numTestFiles = %u  numEpoch = %lu\n", numLearnFiles, numTestFiles, numEpoch);
	for (m = 0; m < numLearnFiles; m++) {
		LearnImages[m].Response = (LearnImages[m].Name[strlen(LearnImages[m].Name)-1] == 'T') ? 1.0 : -1.0;
	}

	Mask.Image = LoadByteImagePgm(MaskName, &Mask.Width, &Mask.Height);
	if (Mask.Image == NULL) {
		printf("Erreur : Ne peut pas charger le fichier de masque : %s.pgm\n", MaskName);
		exit(1);
	}
	printf("Mask (%s) : (Width, Height) = (%dx%d)\n", MaskName, Mask.Width, Mask.Height);

	epoch = LoadNet(NetName, &Net, Mask.Width, Mode);

	if (epoch < 0) {
		printf("Erreur : Ne peut pas charger le fichier du réseau : %s.net\n", NetName);
		exit(1);
	}
	numEpoch += epoch;
	printf("Net.numUnits = %d  Net.numGroups = %d  Net.numConnections = %d   numEpoch = %lu\n", Net.numUnits, Net.numGroups, Net.numConnections, numEpoch);

	for (m = 0; m < numLearnFiles; m++) {
		LearnImages[m].ImageFloat = (float *) calloc(Mask.Height*Mask.Width, sizeof(float));
		for (n = 0; n < Mask.Height*Mask.Width; n++)			// Transfère la portion d'image aux entrées du réseau
			LearnImages[m].ImageFloat[n] = (float) LearnImages[m].Image[n];
		free(LearnImages[m].Image);
		CorrigeIllumination(LearnImages[m].ImageFloat, Mask.Image, Mask.Width);
		EgaliseHistogramme(LearnImages[m].ImageFloat, Mask.Image, Mask.Width);
		for (n = 0; n < Mask.Height*Mask.Width; n++)   // Convertit la fenêtre en valeurs 0.0 à 1.0
			LearnImages[m].ImageFloat[n] = LearnImages[m].ImageFloat[n]/255.0;
	}

    signal(SIGINT, &trap);
    Terminate = 0;

	printf("\nDépart !  (numProcs = %u)\n", omp_get_num_procs());
	Window = Net.InputVal;
	for (; epoch < numEpoch; epoch++) {
GET_TSC(t1);
		TErreur = 0.0;
		FErreur = 0.0;
		TotalErreur = 0.0;
		for (m = 0; m < numLearnFiles; m++) {
			memcpy(Window, LearnImages[m].ImageFloat, Mask.Height*Mask.Width*sizeof(float));			// Transfère la portion d'image aux entrées du réseau
			Reponse = CalculeNet(&Net);
			NetErreur = (LearnImages[m].Response - Reponse);
			TrainNet(&Net, LearnRate, Momentum, NetErreur);
			if (LearnImages[m].Response > 0.0)
				TErreur += NetErreur;
			else
				FErreur += NetErreur;
			TotalErreur += (NetErreur > 0) ? NetErreur : -NetErreur;
		}
GET_TSC(t2);
		printf("(%f) : (%lu/%lu) = (%e, %e, %e)  (%e)\n", LearnRate, epoch+1, numEpoch, TErreur, FErreur, TotalErreur, ((float)t2-t1)/2.2e9);

		// CTRL-C détecté ?
		if (Terminate)
			break;
	}
	printf("\nFini !\n");

	SaveNet(NetName, &Net, epoch+1);

	free(Net.WeightVal);
	free(Net.PrevWeightVal);
	free(Net.InputVal);
	free(Net.InputAdr);
	free(Net.ErrorVal);
	free(Net.NetNeurones);

	free(Mask.Image);

	FreeImageFiles (&LearnFileNames, &LearnImages, numLearnFiles);

	// Retransmet le CTRL-C au système
	if (Terminate)
		signal(SIGINT, SIG_DFL);

	return 0;
}
