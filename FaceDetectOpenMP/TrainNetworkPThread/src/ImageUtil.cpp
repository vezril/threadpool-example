/*
 * ImageUtil.cpp
 *
 *  Created on: 8 janv. 2012
 *      Author: bruno
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

#include "ImageUtil.h"


void SaveByteImagePgm(char FileName[], Byte *Image, int Width, int Height) {
 	FILE    *imageFile = NULL;
	char	name[1024];

    sprintf(name,"%s.pgm",FileName);
    imageFile = fopen(name,"wb");
	fprintf(imageFile,"P5\n%d %d\n255\n", Width, Height);
	fwrite((void *) Image, 1, (unsigned long) Width*Height, imageFile);

	fclose(imageFile);
}


Byte *LoadByteImagePgm(char FileName[], int *Width, int *Height) {
 	FILE    *imageFile = NULL;
    char	line[80];
	Byte	*Image = NULL;
	char	*comment, *token, *input;
	int		tokens = 0;
	char	name[1024];

	sprintf(name,(strrchr(FileName, '.') == NULL) ? "%s.pgm" : "%s",FileName);
    imageFile = fopen(name,"rb");
	if (imageFile == NULL)
		return NULL;

	do {
		fgets(line,80,imageFile);
		comment = strchr(line,'#');
		if (comment != NULL)
			*comment = 0;
		token = NULL;
		input = &line[0];
		while ((tokens < 4) && ((token = strtok(input," \t\n\r")) != NULL)) {
			switch (tokens) {
				case 0:	if (strcmp(token,"P5") != 0) {
							fprintf(stderr,"(%s) Bad PGM file.\n", name);
							exit(1);
						}
						break;
				case 1:	*Width = atoi(token);
						break;
				case 2:	*Height = atoi(token);
						break;
				default:
						break;
			}
			tokens++;
			input=NULL;
		}
	} while (tokens < 4);

	// Read the image data itself
	Image = (Byte *) malloc((*Width)*(*Height));
	fread((void *)Image, 1, (*Width)*(*Height), imageFile);

	fclose(imageFile);

	return Image;
}

#define MAXLENGTH 100
int GetImageFileNames (const char *Directory, char ***FileNames) {
	    FILE* fp;
	    char CmdString[MAXLENGTH];
	    char name[MAXLENGTH];
	    char convert[MAXLENGTH];
	    char *tp;
		int	 numRead = 0, nameLength = 0, numFiles = 0;
		int  Erreur;

		// Récupère la liste des fichiers du répertoire
		snprintf(CmdString, MAXLENGTH, "ls -B1 %s", Directory);
	    fp = popen(CmdString,"r");

	    while (numRead != EOF) {		// Analyse la liste
	    	memset(name, 0, MAXLENGTH);
	    	numRead = fscanf(fp, "%s", name);
	    	if (numRead == EOF)					// Fin de la liste
	    		break;
    		if (!(tp = strrchr(name, '.')))		// S'il n'y a pas d'extension, ignore ce fichier
    			continue;
    		*tp = 0;
    		Erreur = 0;
    		// Si le fichier est d'un autre format que PGM, convertit le fichier en PGM
    		if (strcasecmp(tp+1, "gif") == 0) {
    			sprintf(convert, "giftopnm %s//%s.gif > %s//%s.pgm", Directory, name, Directory, name);
    			strcpy(tp+1,"pgm");
    			Erreur = system(convert);
    		}
    		if (strcasecmp(tp+1, "jpeg") == 0) {
    			sprintf(convert, "jpegtopnm %s//%s.jpeg > %s//%s.pgm", Directory, name, Directory, name);
				Erreur = system(convert);
    		}
    		if (strcasecmp(tp+1, "tiff") == 0) {
    			sprintf(convert, "tifftopnm %s//%s.tiff > %s//%s.pgm", Directory, name, Directory, name);
				Erreur = system(convert);
    		}
    		if (strcasecmp(tp+1, "bmp") == 0) {
    			sprintf(convert, "bmptopnm %s//%s.bmp > %s//%s.pgm", Directory, name, Directory, name);
    			Erreur = system(convert);
    		}
    		if (strcasecmp(tp+1, "raw") == 0) {
    			sprintf(convert, "rawtopgm %s//%s.raw > %s//%s.pgm", Directory, name, Directory, name);
    			Erreur = system(convert);
    		}
    		// S'il y a eu une erreur de convertion ou si ce n'est pas une image PGM, ignore ce fichier
    		if ((Erreur < 0) || (strcasecmp(tp+1, "pgm") != 0))
    			continue;
    		nameLength = strlen(name) + 1;
    		// Ajoute le nom du fichier à la liste
    		(*FileNames) = (char **) realloc((void *) (*FileNames), (numFiles+1)*sizeof(char *));
    		(*FileNames)[numFiles] = (char *) malloc(nameLength*sizeof(char));
    		memcpy((*FileNames)[numFiles], name, nameLength);
    		numFiles++;
	    }

	    fclose (fp);

	    return numFiles;
}

int GetImageFiles (char *RepName, char ***FileNames, ImageStruct **Images) {
	int		n, numFiles = 0;
    char	name[MAXLENGTH];

	numFiles = GetImageFileNames (RepName, FileNames);
//printf("numFiles = %u\n", numFiles);
	*Images = (ImageStruct *) malloc(numFiles*sizeof(ImageStruct));
	for (n = 0; n < numFiles; n++) {
		(*Images)[n].Name = (char *) malloc(strlen(RepName)+strlen((*FileNames)[n])+2);
		sprintf((*Images)[n].Name, "%s/%s", RepName, (*FileNames)[n]);
		sprintf(name, "%s.pgm", (*Images)[n].Name);
		(*Images)[n].Image = LoadByteImagePgm(name, &((*Images)[n]).Width, &((*Images)[n]).Height);
		if ((*Images)[n].Image == NULL) {
			printf("Erreur : Ne peut pas charger le fichier d'image : %s.pgm\n", (*FileNames)[n]);
			exit(1);
		}
//printf("Image (%s) : Width = %d  Height = %d\n", (*FileNames)[n], (*Images)[n].Width, (*Images)[n].Height);
	}

	return numFiles;
}


void FreeImageFiles (char ***FileNames, ImageStruct **Images, int numFiles) {
	int n;

	for (n = 0; n < numFiles; n++) {
		free((*Images)[n].Name);
		free((*Images)[n].ImageFloat);
		free((*FileNames)[n]);
	}
	free(*Images);
	free(*FileNames);
	*Images    = NULL;
	*FileNames = NULL;
}

