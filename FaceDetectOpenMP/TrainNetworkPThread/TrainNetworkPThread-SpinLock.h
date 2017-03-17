/*
 * TrainNetworkPThread.h
 *
 *  Created on: 8 janv. 2012
 *      Author: bruno
 */

#ifndef TRAINNETWORKPTHREAD_H_
#define TRAINNETWORKPTHREAD_H_

#define POLICY SCHED_RR

#define LEARN_RATE (0.2)

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

typedef pthread_spinlock_t pthread_spinlock_t_aligned __attribute__((aligned(128)));

typedef struct {
	Units	**LocalNeurones;
	pthread_spinlock_t_aligned spinlock_CalculeNet;
	pthread_spinlock_t_aligned spinlock_TrainNet;
	pthread_spinlock_t_aligned spinlock_CalculeFini;
	pthread_spinlock_t_aligned spinlock_TrainFini;
	int		numUnits;
	int		TaskID;
} TaskWork;

volatile unsigned long long t1, t2;

ImageStruct	*LearnImages;
ImageStruct	Mask;
Network		Net;
pthread_t	*Thread;
TaskWork	*TaskWorkLoad;
int			numTask;


pthread_barrier_t  barrier_Start;
int		Fini = 0;

float		LearnRate, Momentum;
float		Reponse, NetErreur, TotalErreur, TErreur, FErreur;

void  CorrigeIllumination(float *ImageWindow, Byte *Mask, int MaskSize);
void  EgaliseHistogramme(float *ImageWindow, Byte *Mask, int MaskSize);
void *HiddenNeuroneTask ( void *ptr );
int   CreateWorkLoad (Network  *Net, TaskWork *TaskWorkLoad);

#endif /* TRAINNETWORKPTHREAD_H_ */
