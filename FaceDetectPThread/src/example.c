#include <stdio.h>
#include <stdlib.h>
#include "threadpool.h"

#define MAXTHREADS 4
#define QUEUESIZE 32

threadpool_t * pool;

void function1(void){
    printf("Worker took function1\n");
}

void function2(void){
    printf("Worker took function2\n");
}

void main(void){
    
    int ret, i;
    
    pool = threadpool_create(MAXTHREADS, QUEUESIZE, 0);
    if(pool == NULL){
        printf("Threadpool creation: failed\n");
    } else {
        printf("Threadpool creation: success\n");
    }
    
    
    printf("Threadpool queue size: %i\n", threadpool_get_queue_size(pool));
    printf("Threadpool thread count: %i\n", threadpool_get_thread_count(pool));
    printf("Threadpool is started: %i\n", threadpool_is_started(pool));
    printf("Threadpool is shutdown: %i\n", threadpool_is_shutdown(pool));
    
    for(i=0;i<16;i++){
        threadpool_add(pool, &function1, NULL, 0);
        if(i%2 == 0)
            threadpool_add(pool, &function2, NULL, 0);
    }
    
    ret = threadpool_destroy(pool, THREADPOOL_GRACEFUL);
    if(ret != 0){
        printf("Threadpool destroy: failed\n");
    } else {
        printf("Threadpool destroy: success\n");
    }
    
    printf("Threadpool is shutdown: %i\n", threadpool_is_shutdown(pool));
}
