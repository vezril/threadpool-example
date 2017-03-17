// FaceDetect.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <unistd.h>

#include "NeuralNet.h"
#include "ImageUtil.h"
#include "FaceDetect.h"
#include "threadpool.h"


typedef struct {
    ImageStruct  Image;
    Byte        *ImageHits;
    Byte        *ImageReponse;
    char        *MaskName;
    float       *Window;
    float       *Response;
    long        x;
    long        y;
    long        numDetected;
    float       answers[9];
    float       Average;
    int         structID;
    pthread_spinlock_t	lock;
    int 		synchronization;
    int			synch_val;
} image_t;

// Global vars
image_t      	*images;
float 			**Window = NULL;
float 			Seuil = 1.0;
ImageStruct  	Mask;
Network      	Net;
Network			*Networks;
char    		name[256];
char        	*ImageName = NULL;
static	float	LCM[3][3];

void CorrigeIllumination(float *ImageWindow, Byte *MaskImage, int MaskSize, int threadID) {
	float	vec1_0, vec1_1, vec1_2;   // Part of affine fitting [CALVIN] ???
	float	vec2[3];                  // Part of affine fitting
	int		halfX = (MaskSize/2);
	int		halfY = (MaskSize/2);
	float	xi, yi;
	int		i;

	// Calcul la fonction affine - Étape 1
	vec1_0 = vec1_1 = vec1_2 = 0.0;
	
	for (i = 0; i < MaskSize*MaskSize; i++) {
		if (MaskImage[i] > 0) {
			xi = (float)((i % MaskSize) - halfX);
			yi = (float)((i / MaskSize) - halfY);
			vec1_0 += (xi*ImageWindow[i]);
			vec1_1 += (yi*ImageWindow[i]);
			vec1_2 += (ImageWindow[i]);
		}
	}

	// Calcul la fonction affine - Étape2
    vec2[i] = LCM[0][0]*vec1_0 + LCM[0][1]*vec1_1 + LCM[0][2]*vec1_2;
    vec2[i] = LCM[1][0]*vec1_0 + LCM[1][1]*vec1_1 + LCM[1][2]*vec1_2;
    vec2[i] = LCM[2][0]*vec1_0 + LCM[2][1]*vec1_1 + LCM[2][2]*vec1_2;

	// Applique la correction affine
	for (i = 0; i < MaskSize*MaskSize; i++) {
		xi = (float)((i % MaskSize) - halfX);
		yi = (float)((i / MaskSize) - halfY);
		ImageWindow[i] = (float) floor(ImageWindow[i] - (xi*vec2[0] + yi*vec2[1] + vec2[2] - 128.5f));
		ImageWindow[i] = (ImageWindow[i] < 0.0f) ? 0.0f : ((ImageWindow[i] > 255.0f) ? 255.0f : ImageWindow[i]);
	}
}


void EgaliseHistogramme(float *ImageWindow, Byte *MaskImage, int MaskSize, int threadID) {
	long	hist[256];
	float	scaleFactor;
	long	min;
	int		i;
    
	// Calcul l'histogramme
	memset(hist, 0, 256*sizeof(long));
    for(i = 0; i < MaskSize*MaskSize; i++)
		if (MaskImage[i] > 0)
			hist[(Byte)(ImageWindow[i])]++;

	// Calcul l'histogramme-cumulatif
	for(i = 1; i < 256; i++)
		hist[i] += hist[i-1];
	min = 0;
	for(i = 0; i < 256; i++)
		if(hist[i] > 0) {
			min = hist[i];
			break;
		}

	// Applique l'égalisation d'histogramme
    if(hist[255] > min) {
        scaleFactor = 255.0/(hist[255]-min);
        for(i = 0; i < MaskSize*MaskSize; i++)
            ImageWindow[i] = (float) floor((float) ((hist[(Byte)(ImageWindow[i])]-min)*scaleFactor));
	}
}


Byte *ReduceImageBilinear(Byte *Image, int *Width, int *Height, float scale) {
    Byte   *NewImage;
    int     NewWidth, NewHeight;
    float   Ux[2], Uy[2];
    float   kx, ky, kkx, kky;
    float   tp1, tp2;
    int     x, y, x1, y1;
    long    xy, x1y1;
    int     i, j;

    if (scale <= 1.0f)
        return NULL;

    NewWidth   = (int) ((*Width)/scale);
    NewWidth  += (((*Width)-((int)(NewWidth*scale))) == 1) ? -1 : 0;
    NewHeight  = (int) ((*Height)/scale);
    NewHeight += (((*Height)-((int)(NewHeight*scale))) == 1) ? -1 : 0;
    NewImage   = (Byte *) malloc(NewWidth*NewHeight);

    for (y = 0; y < NewHeight; y++) {
        ky = scale*(y+1);
        y1 = (int) ky;
        if (y1+1 >= (*Height))
            break;
        kky = ky - y1;
        Uy[0] = 1.0 - kky;
        Uy[1] = kky;
        for (x = 0; x < NewWidth; x++) {
            kx = scale*(x+1);
            x1 = (int) kx;
            if (x1+1 >= (*Width))
                break;
            kkx = kx - x1;
            Ux[0] = 1.0 - kkx;
            Ux[1] = kkx;
            x1y1 = ((long)y1)*(*Width)+((long)x1);
            tp1 = 0.0f;
            for (i = 0; i <= 1; i++) {
                tp2 = 0.0;
                for (j = 0; j <= 1; j++) {
                    x1y1 = ((long)y1+i)*(*Width)+((long)x1+j);
                    tp2 +=    Image[x1y1]*Ux[j];
                }
                tp1 += Uy[i]*tp2;
            }
            xy     = ((long)y)*NewWidth+((long)x);
            NewImage[xy] = (tp1 < 0.0) ? 0 : ((tp1 > 255.0) ? 255 : ((Byte) tp1));
        }
    }
    free(Image);
    Image = NULL;
    *Width = NewWidth;
    *Height = NewHeight;

    return NewImage;
}

Byte *ReduceImageBicubic(Byte *Image, int *Width, int *Height, float scale) {
    Byte   *NewImage;
    int     NewWidth, NewHeight;
    float   Q[4][4] = {{-0.5, 1.0, -0.5, 0.0},{1.5,-2.5,0.0,1.0},{-1.5,2.0,0.5,0.0},{0.5,-0.5,0.0,0.0}};
    float   Ax[4], Ay[4], Ux[4], Uy[4];
    float   kx, ky, ax, ay;
    float   tp1, tp2;
    int     x, y, x1, y1;
    long    xy, x1y1;
    int     i, j;

    if (scale <= 1.0)
        return NULL;

    NewWidth   = (int) ((*Width)/scale);
    NewWidth  += (((*Width)-((int)(NewWidth*scale))) == 1) ? -2 : -1;
    NewHeight  = (int) ((*Height)/scale);
    NewHeight += (((*Height)-((int)(NewHeight*scale))) == 1) ? -2 : -1;
    NewImage   = (Byte *) malloc(NewWidth*NewHeight);

    for (y = 0; y < NewHeight; y++) {
        ky = scale*(y+1);
        y1 = (int) ky;
        if (y1+2 >= (*Height))
            continue;
        ay = ky-y1;
        Ay[3] = 1.0;
        for (i = 2; i >= 0; i--)
            Ay[i] = Ay[i+1]*ay;
        for (i = 0; i < 4; i++) {
            tp1 = 0.0;
            for (j = 0; j < 4; j++)
                tp1 += Q[i][j]*Ay[j];
            Uy[i] = tp1;
        }
        for (x = 0; x < NewWidth; x++) {
            kx = scale*(x+1);
            x1 = (int) kx;
            if (x1+2 >= (*Width))
                continue;
            ax = kx-x1;
            Ax[3] = 1.0;
            for (i = 2; i >= 0; i--)
                Ax[i] = Ax[i+1]*ax;
            for (i = 0; i < 4; i++) {
                tp1 = 0.0;
                for (j = 0; j < 4; j++)
                    tp1 += Q[i][j]*Ax[j];
                Ux[i] = tp1;
            }
            tp1 = 0.0;
            for (i = -1; i <= 2; i++) {
                tp2 = 0.0;
                for (j = -1; j <= 2; j++) {
                    x1y1 = ((long)y1+i)*(*Width)+((long)x1+j);
                    tp2 +=    Image[x1y1]*Ux[j+1];
                }
                tp1 += Uy[i+1]*tp2;
            }
            xy     = ((long)y)*NewWidth+((long)x);
            NewImage[xy] = (tp1 < 0.0) ? 0 : ((tp1 > 255.0) ? 255 : ((Byte) tp1));
        }
    }
    free(Image);
    Image = NULL;
    *Width = NewWidth;
    *Height = NewHeight;

    return NewImage;
}


float CalculeNet(Network *NetLocal) {
	Units  *Neurone;
    double   tp;
    int      i, j;

	for (i = 0; i < NetLocal->numUnits-1; i++) {
		Neurone = &(NetLocal->NetNeurones[i]);
		tp = 0.0; // Calcule la somme des entrées, pondérée par les poids
			
		
		for (j = 0; j < Neurone->numConnections; j++)
			tp += (float) ((Neurone->WeightVal[j]) * (*(Neurone->InputValp[j])));
		
		*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide
	}
	Neurone = &(NetLocal->NetNeurones[NetLocal->numUnits-1]);
	tp = 0.0;							// Calcule la somme des entrées, pondérée par les poids
	for (j = 0; j < Neurone->numConnections; j++)
		tp += (float) ((Neurone->WeightVal[j])*(*(Neurone->InputValp[j])));
	*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide
    return NetLocal->OutputVal[NetLocal->numUnits-1];
}

/*
 * Simple function to print a fatal message and exit
 */
void fatal(const char *string, ...) {
    va_list list;
    va_start(list, string);
    vfprintf(stderr, string, list);
    va_end(list);
    exit(1);
}

/*
 * Main Work function, arguments required are the the image to work on (s)
 * and the row to work on (Y), argument zero MUST be used for the threadID
 */
void work(void *arg){
	unsigned int X, pos1, pos2, y, x, cX, cY, Y, s,i;
	int * arguments;
	float Average;
	int threadID;
	
	
	arguments = (int *) arg;
	
	threadID = arguments[0];
	Y = arguments[1];
	s = arguments[2];
	free(arg);
	arg = NULL;
	
    // Setup the new input for the neural network
	for(X = 0; (int) X < images[s].Image.Width-Mask.Width; X++) {
        cX = X + Mask.Width/2;
        cY = Y + Mask.Height/2;

		for(y = Y; y < Y+Mask.Height; y++) {   // Transfère la portion d'image aux entrées du réseau
			pos1 = (y-Y)*Mask.Width;
			pos2 = y*images[s].Image.Width;
			for(x = X; x < X+Mask.Width; x++) {
				Window[threadID][pos1+(x-X)] = (float) images[s].Image.Image[pos2 + x];
			}
		}
		
        // Run the two functions
		CorrigeIllumination(Window[threadID], Mask.Image, Mask.Width, threadID);
		EgaliseHistogramme(Window[threadID], Mask.Image, Mask.Width, threadID);

        // Do the normal stuff here for calculating the neural network output
		for(pos1 = 0; (int) pos1 < Mask.Height*Mask.Width; pos1++)   // Convertit la fenêtre en valeurs 0.0 à 1.0
			Window[threadID][pos1] = Window[threadID][pos1]/255.0;


		images[s].Response[cY*images[s].Image.Width+cX] = CalculeNet(&Networks[threadID]);

		Average = 
			(1*images[s].Response[(cY-1)*images[s].Image.Width+(cX-1)]+
			 2*images[s].Response[(cY-1)*images[s].Image.Width+(cX+0)]+
			 1*images[s].Response[(cY-1)*images[s].Image.Width+(cX+1)]+
			 2*images[s].Response[(cY+0)*images[s].Image.Width+(cX-1)]+
			 3*images[s].Response[(cY+0)*images[s].Image.Width+(cX+0)])/9;
				   
		Average = (Average > 1.0) ? 1.0 : ((Average < -1.0) ? -1.0 : Average);
		
        // Lock the image and put in the result
		pthread_spin_lock(&images[s].lock);
		images[s].ImageReponse[(cY-1)*images[s].Image.Width+(cX-1)] = (Byte)((0.5+Average/2.0)*255.0);
		pthread_spin_unlock(&images[s].lock);
		
        // Check if there was a hit, if so lock the image again and increase the numDetected
		if (Average >= Seuil) {                // S'il y a eu une détection
			pthread_spin_lock(&images[s].lock);
			images[s].numDetected++;
			pthread_spin_unlock(&images[s].lock);
			for (y = Y; y < Y+Mask.Height; y++) {    // Dessine le masque sur l'image
				pos1 = (y-Y)*Mask.Width;
				for (x = X; x < X+Mask.Width; x++) {
					if (Mask.Image[pos1+(x-X)] == 0)
						images[s].ImageHits[(y*images[s].Image.Width) + x] = 0;
				}
			}
		}
	}
    
    // Ugly but it works. This is to synchronize all work() functions to the corresponding
    // cleanup function.
	pthread_spin_lock(&images[s].lock);
	images[s].synchronization++;
	pthread_spin_unlock(&images[s].lock);
}

/* 
 * Cleanup function. This is scheduled after all the work() functions for a given image.
 * It also has to synchronize and wait for all scheduled work() functions to finish (which shouldn't be long)
 */
void cleanup(void *arg){
	int s;
	int * arguments;
	int cnt=0;
	int threadID;
	
	arguments = (int *) arg;
	threadID = arguments[0];
	s = arguments[1];
	free(arg);
	arg = NULL;
	
	// TODO Refactor this. It's ugly as hell.
    // This was added to create a certain delay in the spin lock acquisition...
    // for some reason it was working weirdly without the cnt check
	while(1){
		if(cnt > 150000){
			pthread_spin_lock(&images[s].lock);
			if(images[s].synchronization < images[s].synch_val){
				pthread_spin_unlock(&images[s].lock);
			} else {
				pthread_spin_unlock(&images[s].lock);
				break;
			}
			cnt=0;
		}
		cnt++;
	}
	
    // Alright, good to save the image and crap out the results.
	printf("[INFO] (%s) numDetected = %lu\n", __func__, images[s].numDetected);
	sprintf(name,"%s_Out%d",ImageName, s);
	SaveByteImagePgm(name, images[s].ImageHits, images[s].Image.Width, images[s].Image.Height);
	sprintf(name,"%s_Res%d",ImageName, s);
	SaveByteImagePgm(name, images[s].ImageReponse, images[s].Image.Width, images[s].Image.Height);
		
	free(images[s].ImageReponse);
	free(images[s].ImageHits);
	free(images[s].Response);
	free(images[s].Image.Image);
	//images[s].ImageReponse = NULL;
	//images[s].ImageHits = NULL;
	//images[s].Response = NULL;
	//images[s].Image.Image = NULL;
}

 
int main(int argc, char *argv[]) {
    double timespent;
    char *MaskName = NULL, *NetName = NULL;
    int          s=0, Y, i, YY;
    threadpool_t *pool;
    int *args;
    long nCPU = sysconf(_SC_NPROCESSORS_ONLN);
    int Y_total;
    float	A = 0.0, B = 0.0, C = 0.0;
    float	xi, yi;
    int     Erreur;
    
    if (argc < 5) {
        printf("Erreur : La commande devrait être :\n");
        printf("            %s <nom du réseau> <nom du masque> <nom de l'image> <Seuil de détection>\n", argv[0]);
        exit(1);
    }

    NetName   = argv[1];
    MaskName  = argv[2];
    ImageName = argv[3];
    Seuil     = atof(argv[4]);
    
    // INIT
    // The *2 is to fix a strange bug. This is a temporary fix. 
    // TODO: If time permits, find the source of this bug. (Double Free or memory corruption)
    images = (image_t *) malloc(sizeof(image_t)*NUM_SCALE*2);
    if(images == NULL)
        fatal("[ERROR] (%s) Failed to allocate memory for images\n", __func__);
    
    // Loading the inital image and mask
    images[0].Image.Image = LoadByteImagePgm(ImageName,&images[0].Image.Width, &images[0].Image.Height);
    if(images[0].Image.Image == NULL)
        fatal("[ERROR] (%s) Unable to load image file: %s.pgm\n", __func__, ImageName);
    
    Mask.Image = LoadByteImagePgm(MaskName, &Mask.Width, &Mask.Height);
    if(Mask.Image == NULL)
        fatal("[ERROR] (%s) Unable to load Mask file: %s.pgm\n", __func__, MaskName); 
    
    Networks = (Network *) malloc(sizeof(Network) * nCPU);
    for(i=0;i<nCPU;i++){
		Erreur = LoadNet(NetName, &Networks[i], Mask.Width, LOAD_WEIGHT);
		if (Erreur < 0) {
			switch (Erreur) {
				case -1 :    
				case -2 :    fatal("[ERROR] (%s) Unable to load neural network: %s.wet\n", __func__, NetName);
							 break;
			}
		}
	}
    
    // Start the threadpool
    pool = threadpool_create(nCPU-1, 35000, 0);
    if(pool == NULL)
		fatal("[ERROR] (%s) Unable to create threadpool, exiting.\n", __func__);
    
    Window = (float **) malloc(sizeof(float *)*nCPU);
    for(i=0;i<nCPU;i++){
		Window[i] = Networks[i].InputVal;
	}
    
    // INIT LCM Matrix
    // This is directly stripped from the CorrigeIllumination function
    // To avoid having to do synchronization on that (I'm lazy), I'll just
    // init it before hand and stick it global.
    for (i = 0; i <  Mask.Width*Mask.Height; i++) {
        xi = (float)((i % Mask.Width) - Mask.Width/2);
        yi = (float)((i / Mask.Height) - Mask.Height/2);
        if (Mask.Image[i] > 0) {
            A += xi*xi;
            B += xi*yi;
            C += yi*yi;
        }
    }	
    
    LCM[0][0] = 1.0/(A-B);
    LCM[0][1] = 0.0;
    LCM[0][2] = 1.0/(2.0*(A-B));
    LCM[1][0] = 0.0;
    LCM[1][1] = 1.0/(C-B);
    LCM[1][2] = 1.0/(2.0*(C-B));
    LCM[2][0] = 1.0/(2.0*(A-B));
    LCM[2][1] = 1.0/(2.0*(C-B));
    LCM[2][2] = (A*C-B*B)/(4.0*B*(A-B)*(C-B));

    // Alright! PUROGURAMU E IKIMASU!!!
    fprintf(stdout, "[INFO] (%s) Start!\n", __func__);
	fprintf(stdout, "[INFO] (%s) Mask (%s) : Width = %d  Height = %d\n", __func__, MaskName, Mask.Width, Mask.Height);
    
    fprintf(stdout, "[INFO] (%s) Threadpool queue size: %i\n", __func__, threadpool_get_queue_size(pool));
    fprintf(stdout, "[INFO] (%s) Threadpool thread count: %i\n", __func__,threadpool_get_thread_count(pool));
    
    // Scheduling and Reducing loop
    for(s=0;s<NUM_SCALE;s++){
        // Init required stuff for a given image
		pthread_spin_init(&images[s].lock, 0);
		
		images[s].ImageHits    = (Byte *) malloc(images[s].Image.Width*images[s].Image.Height*sizeof(Byte));
        images[s].ImageReponse = (Byte *) malloc(images[s].Image.Width*images[s].Image.Height*sizeof(Byte));
        images[s].Response     = (float *) malloc(images[s].Image.Width*images[s].Image.Height*sizeof(float));
        
        // TODO Refactor this, not necessairy anymore
        Y_total = images[s].Image.Height;
        memcpy(images[s].ImageHits, images[s].Image.Image, images[s].Image.Width*images[s].Image.Height);
        images[s].numDetected = 0;
		images[s].synch_val = Y_total-Mask.Height;
        
        // TODO Do we really need this?
        fprintf(stdout, "[INFO] (%s) Image (%s) : Width = %d  Height = %d Scale = %d\n", __func__, ImageName, images[s].Image.Width, images[s].Image.Height, s);
        
        // Scheduling section.
        // To avoid masks that go over one another, I'm scheduling things
        // one mask apart. 
        for(YY=0;YY<Mask.Height;YY++) {
            for(Y = 0; YY+Y < images[s].Image.Height-Mask.Height; Y=Y+Mask.Height) {
                // Alocate memory for arguments and pass that to the threadpool
                args = (int *) malloc(sizeof(int)*3);
                args[0] = 0; //ThreadID, gets passed by the actual thread, this is just a placeholder
                args[1] = Y+YY;
                args[2] = s;
                threadpool_add(pool, &work, (void *) args,0);
            }
        }
        
        // Previous args will get freed in their work process. These are for cleanup()
        args = (int *) malloc(sizeof(int)*2);
        args[0] = 0; //ThreadID, gets passed by the actual thread
        args[1] = s;
        threadpool_add(pool, &cleanup, (void *) args, 0);
		
		// Reduction
		// This has to be here because who ever wrote the reduction algorithms... well, let's say he free's memory quite freely.
		images[s+1].Image.Width =  images[s].Image.Width;
		images[s+1].Image.Height = images[s].Image.Height;
		images[s+1].Image.Image = (Byte *) malloc(sizeof(Byte) * images[s].Image.Height * images[s].Image.Width);
		
        // Some checks and then reduce the image if necessairy.
		if(images[s+1].Image.Image == NULL){
			fprintf(stderr, "[ERROR] (%s) Unable to allocate memory\n", __func__);
		} else {
			memcpy(images[s+1].Image.Image, images[s].Image.Image, images[s].Image.Height * images[s].Image.Width);
		
			if (s+1 < NUM_SCALE) {
                // Bloody hell this is inelegant.
				#ifdef REDUCTION_BILINEAR
					images[s+1].Image.Image = ReduceImageBilinear(images[s+1].Image.Image, &images[s+1].Image.Width, &images[s+1].Image.Height, REDUCTION_SCALE);
				#else
					images[s+1].Image.Image = ReduceImageBicubic(images[s+1].Image.Image, &images[s+1].Image.Width, &images[s+1].Image.Height, REDUCTION_SCALE);
				#endif
			}
		}
    }
    
    // Start a new worker since scheduling is done and gracefully shutdown the threadpool
    fprintf(stdout, "[INFO] (%s) Scheduling done creating new worker from freed thread\n", __func__);
    threadpool_new_worker(pool);
    
    threadpool_destroy(pool, THREADPOOL_GRACEFUL);
    
    
    // Freeing memory cause that's how I roll
    for(i=0;i<nCPU;i++){		
		Window[i] = NULL;
		free(Networks[i].WeightVal);
		free(Networks[i].PrevWeightVal);
		free(Networks[i].InputVal);
		free(Networks[i].InputAdr);
		free(Networks[i].NetNeurones);
	}
	
	free(Window);
	free(Networks);
    free(images);
    free(Mask.Image);
    exit(0);
}
