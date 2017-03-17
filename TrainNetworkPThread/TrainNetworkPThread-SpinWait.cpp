//============================================================================
// Name        : TrainNetworkPThread.cpp
// Author      : Bruno De Kelper
// Version     :
// Copyright   : Your copyright notice
//============================================================================

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "NeuralNet.h"
#include "ImageUtil.h"
#include "TrainNetworkPThread.h"

#include <time.h>
#include <signal.h>

#include <pthread.h>
#include <semaphore.h>


// Pour la détection du CTRL-C
volatile int Terminate = 0;
void trap(int signal) {
	Terminate = 1;
}



void CorrigeIllumination(float *ImageWindow, Byte *Mask, int MaskSize) {
	static  int		LCMinit = 1;
	static	float	LCM[3][3];
	float	vec1_0, vec1_1, vec1_2;   // Part of affine fitting
	float			vec2[3];   // Part of affine fitting
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

	// Calcul la fonction affine - Étape 1
	vec1_0 = vec1_1 = vec1_2 = 0.0;
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
	for (i = 0; i < 3; i++)
		vec2[i] = LCM[i][0]*vec1_0 + LCM[i][1]*vec1_1 + LCM[i][2]*vec1_2;

	// Applique la correction affine
	for (i = 0; i < MaskSize*MaskSize; i++) {
		xi = (float)((i % MaskSize) - halfX);
		yi = (float)((i / MaskSize) - halfY);
		ImageWindow[i] = (float) floor(ImageWindow[i] - (xi*vec2[0] + yi*vec2[1] + vec2[2] - 128.5f));
		ImageWindow[i] = (ImageWindow[i] < 0.0f) ? 0.0f : ((ImageWindow[i] > 255.0f) ? 255.0f : ImageWindow[i]);
	}
}


void EgaliseHistogramme(float *ImageWindow, Byte *Mask, int MaskSize) {
	long	hist[256];
	float	scaleFactor;
	long	min;
	int		i, j;

	memset(hist, 0, 256*sizeof(long));

	for (i = 0; i < MaskSize*MaskSize; i++)
		if (Mask[i] > 0)
			hist[(Byte)(ImageWindow[i])]++;

	// Calcul l'histogramme-cumulatif
	for (j = 1; j < 256; j++)
		hist[j] += hist[j-1];

	min = 0;
	for (i = 0; i < 256; i++)
		if (hist[i] > 0) {
			min = hist[i];
			break;
		}

	// Applique l'égalisation d'histogramme
    if (hist[255] > min) {
        scaleFactor = 255.0/(hist[255]-min);
        for (i = 0; i < MaskSize*MaskSize; i++)
            ImageWindow[i] = (float) floor((float) ((hist[(Byte)(ImageWindow[i])]-min)*scaleFactor));
    }
}

void *HiddenNeuroneTask ( void *ptr ) {
	TaskWork	*Work = (TaskWork *) ptr;
	Units  		*Neurone;
	float  		WeightDiff;
	float		tp;
	int			i, j;
	struct sched_param param;
	int			minprio, maxprio, retval;
	cpu_set_t   cpuset;

	minprio = sched_get_priority_min(POLICY);
	maxprio = sched_get_priority_max(POLICY);

    param.sched_priority = minprio + (maxprio - minprio)/2;
    retval = pthread_setschedparam(pthread_self(), POLICY, &param);

    if (retval)
		printf("Tâche (%u) : setschedparam - (retval = %d) opération non-permise\n", Work->TaskID, retval);

    CPU_ZERO(&cpuset);
    CPU_SET(Work->TaskID, &cpuset);
	retval = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

	if (retval) {
		printf("Tâche (%u) : setaffinity - (retval = %d) opération non-permise\n", Work->TaskID, retval);
	}

	pthread_barrier_wait(&(barrier_Start));
	while (1) {

		while ((!CalculeNet)&&(!Fini))
			__asm__ volatile ("pause");

	    if (Fini)
	    	break;

		for (i = 0; i < Work->numUnits; i++) {
			Neurone = Work->LocalNeurones[i];
			if (Neurone->Type == OUTPUT)
				continue;
			tp = 0.0;							// Calcule la somme des entrées, pondérée par les poids
			for (j = 0; j < Neurone->numConnections; j++)
				tp += (float) ((Neurone->WeightVal[j])*(*(Neurone->InputValp[j])));
			*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide
		}

		Work->CalculeFini = 1;

		while (!TrainNet)
			__asm__ volatile ("pause");

		for (i = 0; i < Work->numUnits; i++) {		// Applique la correction des poids
			Neurone = Work->LocalNeurones[i];
			if (Neurone->Type != OUTPUT)
				*(Neurone->LocErrorValp) *= (1.0-(*(Neurone->OutputValp))*(*(Neurone->OutputValp)));
			tp = LearnRate*(*(Neurone->LocErrorValp));
			for (j = 0; j < Neurone->numConnections; j++) {
				WeightDiff = Neurone->WeightVal[j] - Neurone->PrevWeightVal[j];
				Neurone->PrevWeightVal[j] = Neurone->WeightVal[j];
				Neurone->WeightVal[j] += tp*(*(Neurone->InputValp[j])) + Momentum*WeightDiff;
			}
		}

		Work->TrainFini = 1;
	}

  pthread_exit(0); /* exit thread */
}


int CreateWorkLoad (Network  *Net, TaskWork **TaskWorkLoad) {
	cpu_set_t   cpuset;
	int			i, j, numCPU, TaskPerCPU;

	pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	numCPU = 0;
	for (i = 0; i < CPU_SETSIZE; i++)
		if (CPU_ISSET(i, &cpuset))
			numCPU++;
		else
			break;
	numCPU = numCPU-1;
	*TaskWorkLoad = (TaskWork *) calloc(numCPU, sizeof(TaskWork));

	TaskPerCPU = (((Net->numUnits) % numCPU) > 0) ? ((Net->numUnits)/numCPU) + 1 : ((Net->numUnits)/numCPU);
	for (i = 0; i < numCPU; i++) {
		(*TaskWorkLoad)[i].LocalNeurones = (Units **) calloc(TaskPerCPU, sizeof(Units *));
		(*TaskWorkLoad)[i].TaskID = i;
		(*TaskWorkLoad)[i].CalculeFini = 0;
		(*TaskWorkLoad)[i].TrainFini = 0;
	}

	for (i = 0; i < (Net->numUnits); i++) {
		j = i % numCPU;
		(*TaskWorkLoad)[j].LocalNeurones[(*TaskWorkLoad)[j].numUnits] = &(Net->NetNeurones[i]);
		(*TaskWorkLoad)[j].numUnits++;
	}

	return numCPU; /* exit thread */
}

pthread_t *CreateAllTasks (TaskWork *TaskWorkLoad, int numTask) {
	pthread_t			*thread;
	pthread_attr_t		attr;
	struct sched_param	param;
	int					minprio, maxprio;
	cpu_set_t   		cpuset;
	int					i;

	CPU_ZERO(&cpuset);

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	minprio = sched_get_priority_min(POLICY);
	maxprio = sched_get_priority_max(POLICY);
	pthread_attr_setschedpolicy(&attr, POLICY);
	param.sched_priority = minprio + (maxprio - minprio)/2;
	pthread_attr_setschedparam(&attr, &param);

	thread = (pthread_t *) calloc(numTask, sizeof(pthread_t));

	for(i = 0; i < numTask; i++) {
		printf("Creating thread %d\n", i);
		CPU_SET(TaskWorkLoad[i].TaskID, &cpuset);
		pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
		pthread_create(&thread[i], &attr, HiddenNeuroneTask, &(TaskWorkLoad[i]));
		CPU_ZERO(&cpuset);
	}
	pthread_attr_destroy(&attr);
	return thread; /* exit thread */
}

int main(int argc, char *argv[]) {
	char		*LearnRepName;
	char		*MaskName;
	char		*NetName;
	char 		**LearnFileNames = NULL;
	int			numLearnFiles = 0;
	int			numTestFiles = 0;

	Units  		*Neurone;
	float  		*BackwardError, tp;

	float		*Window;
	long		epoch, numEpoch;
	int			m, n, Mode = INIT_WEIGHT;
	int			numTaskArrived;


	if (argc < 8) {
		printf("Erreur : La commande devrait être :\n");
		printf("            TrainNetworkPThread <répertoire d'apprentissage> <nom du masque> <nom du réseau> <Taux d'apprentissage> <Momentum> <numEpoch> <Mode=(LOAD,INIT)>\n");
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

	numTask = CreateWorkLoad (&Net, &TaskWorkLoad);

	{	struct sched_param param;
		cpu_set_t   cpuset;
		int			retval;

	    param.sched_priority = sched_get_priority_max(POLICY);
	    retval = pthread_setschedparam(pthread_self(), POLICY, &param);

		if (retval)
			printf("Maitre : setschedparam - (retval = %d) opération non-permise\n", retval);

		CPU_ZERO(&cpuset);
	    CPU_SET(numTask, &cpuset);
		retval = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

		if (retval)
			printf("Maitre : setaffinity - (retval = %d) opération non-permise\n", retval);

	}

    pthread_barrier_init(&barrier_Start, NULL, numTask+1);

	Thread = CreateAllTasks (TaskWorkLoad, numTask);

	Neurone = &(Net.NetNeurones[Net.numUnits-1]);

	Fini = 0;
    printf("\nDépart !\n");
	pthread_barrier_wait(&(barrier_Start));
	Window = Net.InputVal;
	for (; epoch < numEpoch; epoch++) {
GET_TSC(t1);
		TErreur = 0.0;
		FErreur = 0.0;
		TotalErreur = 0.0;
		for (m = 0; m < numLearnFiles; m++) {
			memcpy(Window, LearnImages[m].ImageFloat, Mask.Height*Mask.Width*sizeof(float));			// Transfère la portion d'image aux entrées du réseau

			//////////////////////////////////////////////////////////////////////////////////////
			CalculeNet = 1;

			numTaskArrived = 0;
			while (numTaskArrived != numTask) {
				for (n = 0; n < numTask; n++)
					if (TaskWorkLoad[n].CalculeFini) {
						numTaskArrived++;
						TaskWorkLoad[n].CalculeFini = 0;
					}
				__asm__ volatile ("pause");
			}
			CalculeNet = 0;

    		tp = 0.0;	// Calcule la somme des entrées, pondérée par les poids
			for (n = 0; n < Neurone->numConnections; n++)
				tp += (float) ((Neurone->WeightVal[n])*(*(Neurone->InputValp[n])));
			*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide

			Reponse = *(Neurone->OutputValp);
			NetErreur = (LearnImages[m].Response - Reponse);

			*(Neurone->LocErrorValp) = NetErreur*(1.0-Reponse*Reponse);	// assigne l'erreur du réseau à cet neurone
			BackwardError = Neurone->LocErrorValp - (Neurone->numConnections - 1);
			for (n = 0; n < Neurone->numConnections-1; n++)	// Rétropropage l'erreur aux neurones précédents
				BackwardError[n] = ((Neurone->WeightVal[n])*(*(Neurone->LocErrorValp)));

			TrainNet = 1;

			numTaskArrived = 0;
			while (numTaskArrived != numTask) {
				for (n = 0; n < numTask; n++)
					if (TaskWorkLoad[n].TrainFini) {
						numTaskArrived++;
						TaskWorkLoad[n].TrainFini = 0;
					}
				__asm__ volatile ("pause");
			}
			TrainNet = 0;
			/////////////////////////////////////////////////////////////////////////////////////

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
	Fini = 1;

	SaveNet(NetName, &Net, epoch+1);


	for (n = 0; n < numTask; n++) {
		pthread_join(Thread[n], NULL);
		free(TaskWorkLoad[n].LocalNeurones);
	}
	free(TaskWorkLoad);
	free(Thread);

	pthread_barrier_destroy(&barrier_Start);

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
