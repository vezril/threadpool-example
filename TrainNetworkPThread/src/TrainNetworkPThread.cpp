//============================================================================
// Name        : TrainNetworkPThread.cpp
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
#include "TrainNetworkPThread.h"

#include <signal.h>

#include <pthread.h>
#include <semaphore.h>
#include <mpi.h>

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
		pthread_mutex_lock(&mutex_CalculeNet);
		while ((!CalculeNet)&&(!Fini))
			pthread_cond_wait(&cond_CalculeNet, &mutex_CalculeNet);
	    pthread_mutex_unlock(&mutex_CalculeNet);

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

	    sem_post(&sem_CalculFini);

		pthread_mutex_lock(&mutex_TrainNet);
		while (!TrainNet)
			pthread_cond_wait(&cond_TrainNet, &mutex_TrainNet);
	    pthread_mutex_unlock(&mutex_TrainNet);

		for (i = 0; i < Work->numUnits; i++) {		// Applique la correction des poids
			Neurone = Work->LocalNeurones[i];
			if (Neurone->Type != OUTPUT)
				*(Neurone->LocErrorValp) *= (1.0-(*(Neurone->OutputValp))*(*(Neurone->OutputValp)));
			for (j = 0; j < Neurone->numConnections; j++) {
				WeightDiff = Neurone->WeightVal[j] - Neurone->PrevWeightVal[j];
				Neurone->PrevWeightVal[j] = Neurone->WeightVal[j];
				Neurone->WeightVal[j] += LearnRate*(*(Neurone->LocErrorValp))*(*(Neurone->InputValp[j])) + Momentum*WeightDiff;
			}
		}

		sem_post(&sem_TrainFini);

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
	numCPU = 1;
	*TaskWorkLoad = (TaskWork *) calloc(numCPU, sizeof(TaskWork));

	TaskPerCPU = (((Net->numUnits) % numCPU) > 0) ? ((Net->numUnits)/numCPU) + 1 : ((Net->numUnits)/numCPU);
	for (i = 0; i < numCPU; i++) {
		(*TaskWorkLoad)[i].LocalNeurones = (Units **) calloc(TaskPerCPU, sizeof(Units *));
		(*TaskWorkLoad)[i].TaskID = i;
	}

	for (i = 0; i < (Net->numUnits); i++) {
		j = i % numCPU;
		(*TaskWorkLoad)[j].LocalNeurones[(*TaskWorkLoad)[j].numUnits] = &(Net->NetNeurones[i]);
		(*TaskWorkLoad)[j].numUnits++;
	}

	return numCPU; /* exit thread */
}

pthread_t *CreateAllTasks (TaskWork *TaskWorkLoad, int numTask) {
	pthread_t		*thread;
	pthread_attr_t	attr;
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
    int			fileToProcess = 0;
    ImageStruct	*ImageBuffer;
    
	Units  		*Neurone;
	float  		*BackwardError, tp, tpRcv;

	float		*Window;
	long		epoch, numEpoch;
	int			m, n, Mode = INIT_WEIGHT;
    int         myid, numprocs, err;
    int			i,j, preData[2];
    MPI_Status  rcvStatus;

	if (argc < 8) {
		printf("Erreur : La commande devrait être :\n");
		printf("            TrainNetworkPThread <répertoire d'apprentissage> <nom du masque> <nom du réseau> <Taux d'apprentissage> <Momentum> <numEpoch> <Mode=(LOAD,INIT)>\n");
		exit(1);
	}
    
    // MPI Initialization
    err = MPI_Init(&argc, &argv);
    
    if(err != MPI_SUCCESS){
		fprintf(stderr, "Failed to start MPI\n");
		exit(1);
	}
    
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  
	// Arguments
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

	// Init of arguments and network
	numLearnFiles = GetImageFiles(LearnRepName, &LearnFileNames, &LearnImages);
	printf("numLearnFiles = %u    numTestFiles = %u  numEpoch = %lu\n", numLearnFiles, numTestFiles, numEpoch);
	if(myid == 0){
		for (m = 0; m < numLearnFiles; m++) {
			LearnImages[m].Response = (LearnImages[m].Name[strlen(LearnImages[m].Name)-1] == 'T') ? 1.0 : -1.0;
		}
	}
	
	ImageBuffer = (ImageStruct	*) malloc(sizeof(ImageStruct) * numLearnFiles);

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

	if(myid==0){
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
	}

    signal(SIGINT, &trap);
    Terminate = 0;

    sem_init(&sem_CalculFini, 0, 0);
    sem_init(&sem_TrainFini, 0, 0);
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
	// INIT Finished
	Fini = 0;
    
	pthread_barrier_wait(&(barrier_Start));
	Window = Net.InputVal;
	
	fileToProcess = (int)(numLearnFiles / numprocs);
	
	
	// Can't do MPI_Scatter due to the nature of the data. http://stackoverflow.com/questions/9864510/struct-serialization-in-c-and-transfer-over-mpi
	// So, doing this extremely ugly thing.
	// If root, then send every (TotalPictures / nSystems) to every node
	// if worker, receive pictures
	if(myid==0){
		for(i = 1; i<numprocs;i++){
			preData[0] = LearnImages[0].Width * LearnImages[0].Height;
			preData[1] = fileToProcess;
			MPI_Send(&preData, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
			for(j=0;j<fileToProcess;j++){
				MPI_Send(LearnImages[j].ImageFloat,preData[0], MPI_FLOAT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&LearnImages[j].Response,1,MPI_FLOAT,i,0,MPI_COMM_WORLD);
			}
		}
		// Screw good practices, that's why.
		// This is lazyness so that the algorithm will continue with the root
		for(i=0;i<fileToProcess;i++){
			ImageBuffer[i].ImageFloat = LearnImages[i].ImageFloat;
			ImageBuffer[i].Response = LearnImages[i].Response;
		}
		
	} else {
		MPI_Recv(&preData, 2, MPI_INT, 0, 0, MPI_COMM_WORLD,&rcvStatus);
		fileToProcess = preData[1];
		for(i=0;i<fileToProcess;j++){
			MPI_Recv(ImageBuffer[i].ImageFloat, preData[0], MPI_FLOAT, 0, 0, MPI_COMM_WORLD,&rcvStatus);
			MPI_Recv(&ImageBuffer[i].Response, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &rcvStatus);
		}
	}
	
	
	printf("\nDépart !\n");
	
	for (; epoch < numEpoch; epoch++) {
		TErreur = 0.0;
		FErreur = 0.0;
		TotalErreur = 0.0;
		
		for (m = 0; m < fileToProcess; m++) {
			memcpy(Window, ImageBuffer[m].ImageFloat, Mask.Height*Mask.Width*sizeof(float));			// Transfère la portion d'image aux entrées du réseau

			pthread_mutex_lock(&mutex_CalculeNet);
			CalculeNet = 1;
            pthread_cond_broadcast(&cond_CalculeNet);
            pthread_mutex_unlock(&mutex_CalculeNet);
            for (n = 0; n < numTask; n++)
            	sem_wait(&sem_CalculFini);
			CalculeNet = 0;

    		tp = 0.0;	// Calcule la somme des entrées, pondérée par les poids
			for (n = 0; n < Neurone->numConnections; n++)
				tp += (float) ((Neurone->WeightVal[n])*(*(Neurone->InputValp[n])));
		
		
			*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide

			Reponse = *(Neurone->OutputValp);
			NetErreur = (ImageBuffer[m].Response - Reponse);

			*(Neurone->LocErrorValp) = NetErreur*(1.0-Reponse*Reponse);	// assigne l'erreur du réseau à cet neurone
			BackwardError = Neurone->LocErrorValp - (Neurone->numConnections - 1);
			for (n = 0; n < Neurone->numConnections-1; n++)	// Rétropropage l'erreur aux neurones précédents
				BackwardError[n] = ((Neurone->WeightVal[n])*(*(Neurone->LocErrorValp)));

			pthread_mutex_lock(&mutex_TrainNet);
			TrainNet = 1;
			pthread_cond_broadcast(&cond_TrainNet);
			pthread_mutex_unlock(&mutex_TrainNet);

			for (n = 0; n < numTask; n++)
				sem_wait(&sem_TrainFini);
			TrainNet = 0;
			
			if (ImageBuffer[m].Response > 0.0)
				TErreur += NetErreur;
			else
				FErreur += NetErreur;
			TotalErreur += (NetErreur > 0) ? NetErreur : -NetErreur;
	
		}
		
		// CTRL-C détecté ?
		if (Terminate)
			break;
		
		// Get errors from other nodes
		MPI_Reduce(&tp, &tpRcv, sizeof(int), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);	
		MPI_Reduce(&TErreur, &TErreurRcv, sizeof(int), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);	
		MPI_Reduce(&FErreur, &FErreurRcv, sizeof(int), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);	
		MPI_Reduce(&TotalErreur, &TotalErreurRcv, sizeof(int), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);	
		

		
		if(myid == 0)
			printf("(%f) : (%lu/%lu) = (%e, %e, %e)\n", LearnRate, epoch+1, numEpoch, TErreur, FErreur, TotalErreur);
	}
	printf("\nFini !\n");
	pthread_mutex_lock(&mutex_CalculeNet);
	Fini = 1;
    pthread_cond_broadcast(&cond_CalculeNet);
    pthread_mutex_unlock(&mutex_CalculeNet);

	SaveNet(NetName, &Net, epoch+1);
	
	for (n = 0; n < numTask; n++) {
		pthread_join(Thread[n], NULL);
		free(TaskWorkLoad[n].LocalNeurones);
	}
	free(TaskWorkLoad);
	free(Thread);
	
	pthread_cond_destroy(&cond_CalculeNet);
	pthread_mutex_destroy(&mutex_CalculeNet);
	pthread_cond_destroy(&cond_TrainNet);
	pthread_mutex_destroy(&mutex_TrainNet);
	
	sem_destroy(&sem_CalculFini);
	sem_destroy(&sem_TrainFini);
	
	free(Net.WeightVal);
	free(Net.PrevWeightVal);
	free(Net.InputVal);
	free(Net.InputAdr);
	free(Net.ErrorVal);
	free(Net.NetNeurones);

	free(Mask.Image);
	
	// Causes MPI to crash. Go figure.
	//FreeImageFiles(&LearnFileNames, &LearnImages, numLearnFiles);

	// Retransmet le CTRL-C au système
	if (Terminate)
		signal(SIGINT, SIG_DFL);
    
    // MPI Cleanup
    MPI_Finalize();
	return 0;
}
