// FaceDetect.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "NeuralNet.h"
#include "ImageUtil.h"
#include "FaceDetect.h"

#define GET_TSC(T) { __asm__ volatile ("cpuid\n\t":::"%eax", "%ebx", "%ecx", "%edx"); \
                     __asm__ volatile ("rdtsc\n\t":"=A"(T)::"%ebx","%ecx"); \
				   }



volatile unsigned long long T0, T1, T2, T3, T4, T5, T6;
volatile unsigned long long t1, t2;

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
	int		i;

	// Calcul l'histogramme
	memset(hist, 0, 256*sizeof(long));
 	for (i = 0; i < MaskSize*MaskSize; i++)
		if (Mask[i] > 0)
			hist[(Byte)(ImageWindow[i])]++;

	// Calcul l'histogramme-cumulatif
	for (i = 1; i < 256; i++)
		hist[i] += hist[i-1];
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


#if REDUCTION_BILINEAR == 1
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
#endif


#if REDUCTION_BICUBIC == 1
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
#endif




float CalculeNet(Network *Net) {
	Units  *Neurone;
    double   tp;
    int      i, j;

	for (i = 0; i < Net->numUnits-1; i++) {
		Neurone = &(Net->NetNeurones[i]);
		tp = 0.0;							// Calcule la somme des entrées, pondérée par les poids
		for (j = 0; j < Neurone->numConnections; j++)
			tp += (float) ((Neurone->WeightVal[j])*(*(Neurone->InputValp[j])));
		*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide
	}

	Neurone = &(Net->NetNeurones[Net->numUnits-1]);
	tp = 0.0;							// Calcule la somme des entrées, pondérée par les poids
	for (j = 0; j < Neurone->numConnections; j++)
		tp += (float) ((Neurone->WeightVal[j])*(*(Neurone->InputValp[j])));
	*(Neurone->OutputValp) = (float) tanh(tp);	// Fonction d'activation Sigmoide

    return Net->OutputVal[Net->numUnits-1];
}


int main(int argc, char *argv[]) {
    Network      Net;
    ImageStruct  Image, Mask;
    Byte        *ImageHits = NULL, *ImageReponse = NULL;
    char        *ImageName = NULL, *MaskName = NULL, *NetName = NULL;
    float       *Window = NULL, *Response = NULL, Average, Seuil = 1.0;
    int          X, Y, cX, cY, pos1, pos2;
    long         x, y, numDetected;
    int          s;

    char    name[256];
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

    Mask.Image = LoadByteImagePgm(MaskName, &Mask.Width, &Mask.Height);
    if (Mask.Image == NULL) {
        printf("Erreur : Ne peut pas charger le fichier de masque : %s.pgm\n", MaskName);
        exit(1);
    }
    printf("Mask (%s) : Width = %d  Height = %d\n", MaskName, Mask.Width, Mask.Height);

    Image.Image = LoadByteImagePgm(ImageName,&Image.Width, &Image.Height);
    if (Image.Image == NULL) {
        printf("Erreur : Ne peut pas charger le fichier d'image : %s.pgm\n", ImageName);
        exit(1);
    }

    Erreur = LoadNet(NetName, &Net, Mask.Width, LOAD_WEIGHT);

    if (Erreur < 0) {
        switch (Erreur) {
            case -1 :    printf("Erreur : Ne peut pas charger le fichier du réseau : %s.net\n", NetName);
                         break;
            case -2 :    printf("Erreur : Ne peut pas charger le fichier du réseau : %s.wet\n", NetName);
                         break;
        }
        exit(1);
    }
    printf("numUnits = %d numGroups = %d numConnections = %d\n",Net.numUnits,Net.numGroups,Net.numConnections);

printf("\nDépart!\n\n");
    Window = Net.InputVal;
    s = 0;
    do {
        ImageHits    = (Byte *) realloc(ImageHits, Image.Width*Image.Height*sizeof(Byte));
    	ImageReponse = (Byte *) calloc(Image.Width*Image.Height, sizeof(Byte));
    	Response     = (float *) realloc(Response, Image.Width*Image.Height*sizeof(float));
        memcpy(ImageHits, Image.Image, Image.Width*Image.Height);
        printf("Image (%s) : Width = %d  Height = %d Scale = %d\n", ImageName, Image.Width, Image.Height, s);
        numDetected = 0;
//T0=T1=T2=T3=T4=T5=T6=0;
GET_TSC(t1);
        for (Y = 0; Y < Image.Height-Mask.Height; Y++)     // Balaye l'image avec le masque
            for (X = 0; X < Image.Width-Mask.Width; X++) {
                cX = X + Mask.Width/2;                // Coordonnées du centre du masque sur l'image
                cY = Y + Mask.Height/2;
//T0++;
//GET_TSC(t1);
                for (y = Y; y < Y+Mask.Height; y++) {   // Transfère la portion d'image aux entrées du réseau
                	pos1 = (y-Y)*Mask.Width;
                	pos2 = (y*Image.Width);
                    for (x = X; x < X+Mask.Width; x++)
                        Window[pos1+(x-X)] = (float) Image.Image[pos2 + x];
                }
//GET_TSC(t2);
//T1 += t2-t1;

//GET_TSC(t1);
                CorrigeIllumination(Window, Mask.Image, Mask.Width);
//GET_TSC(t2);
//T2 += t2-t1;
//GET_TSC(t1);
                EgaliseHistogramme(Window, Mask.Image, Mask.Width);
//GET_TSC(t2);
//T3 += t2-t1;

//GET_TSC(t1);
                for (pos1 = 0; pos1 < Mask.Height*Mask.Width; pos1++)   // Convertit la fenêtre en valeurs 0.0 à 1.0
                	Window[pos1] = Window[pos1]/255.0;
//GET_TSC(t2);
//T4 += t2-t1;

//GET_TSC(t1);
                Response[cY*Image.Width+cX] = CalculeNet(&Net);
//GET_TSC(t2);
//T5 += t2-t1;
            	Average = (1*Response[(cY-1)*Image.Width+(cX-1)]+
               			   2*Response[(cY-1)*Image.Width+(cX+0)]+
               			   1*Response[(cY-1)*Image.Width+(cX+1)]+
               			   2*Response[(cY+0)*Image.Width+(cX-1)]+
               			   3*Response[(cY+0)*Image.Width+(cX+0)])/9;
               	Average = (Average > 1.0) ? 1.0 : ((Average < -1.0) ? -1.0 : Average);
               	ImageReponse[(cY-1)*Image.Width+(cX-1)] = (Byte)((0.5+Average/2.0)*255.0);
//GET_TSC(t1);
                if (Average >= Seuil) {                // S'il y a eu une détection
                	numDetected++;
                    printf("(%d,%d) -> %f\n", cX-1, cY-1, Average);
                    for (y = Y; y < Y+Mask.Height; y++) {    // Dessine le masque sur l'image
                    	pos1 = (y-Y)*Mask.Width;
                        for (x = X; x < X+Mask.Width; x++)
                            if (Mask.Image[pos1+(x-X)] == 0)
                                ImageHits[(y*Image.Width) + x] = 0;
                    }
                }
//GET_TSC(t2);
//T6 += t2-t1;
            }
GET_TSC(t2);
//printf("\nT1 = %llu  T2 = %llu  T3 = %llu  T4 = %llu  T5 = %llu  T6 = %llu\n\n", T1/T0, T2/T0, T3/T0, T4/T0, T5/T0, T6/T0);
printf("\nnumDetected = %lu  Time = %e\n\n", numDetected, ((float)t2-t1)/2.2e9);
//printf("\nnumDetected = %lu  \n", numDetected);
        sprintf(name,"%s_Out%d",ImageName, s);
        SaveByteImagePgm(name, ImageHits, Image.Width, Image.Height);
        sprintf(name,"%s_Res%d",ImageName, s);
        SaveByteImagePgm(name, ImageReponse, Image.Width, Image.Height);
        free(ImageReponse);
        if (s+1 < NUM_SCALE) {
#if REDUCTION_BILINEAR == 1
            Image.Image  = ReduceImageBilinear(Image.Image, &Image.Width, &Image.Height, REDUCTION_SCALE);
#endif
#if REDUCTION_BICUBIC == 1
			Image.Image    = ReduceImageBicubic(Image.Image, &Image.Width, &Image.Height, REDUCTION_SCALE);
#endif
        }
    } while (++s < NUM_SCALE);
    printf("\nFini!\n");
    free(ImageHits);
    free(Response);

    free(Image.Image);
    free(Mask.Image);

    free(Net.WeightVal);
    free(Net.PrevWeightVal);
    free(Net.InputVal);
    free(Net.InputAdr);
    free(Net.NetNeurones);

    return 0;
}