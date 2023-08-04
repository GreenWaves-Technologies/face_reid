
/*
 * Copyright (C) 2023 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */


/* Autotiler includes. */
#include <stdlib.h>
#include <math.h>
#include "face_id.h"
#include "face_idKernels.h"
#include "gaplib/fs_switch.h"
#include "gaplib/ImgIO.h"

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif
#define CI
#ifdef CI
#include "golden.h"
#endif

typedef struct{
    unsigned char* input;
    F16* output;
}face_id_clusterArg;


AT_DEFAULTFLASH_EXT_ADDR_TYPE face_id_L3_Flash = 0;

static void cluster(void*Arg)
{
    face_id_clusterArg* fi_cluster_arg= (face_id_clusterArg*)Arg;

    #ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    #endif

    face_idCNN(fi_cluster_arg->input,fi_cluster_arg->output);
    printf("Runner completed\n");

}

int face_id(void)
{
    printf("Entering main controller\n");
    
    face_id_clusterArg fi_cluster_arg;
    unsigned char* Input = (unsigned char*)pi_l2_malloc(FACE_ID_SIZE*sizeof(unsigned char));
    if(Input==NULL){
        printf("Error allocating input buffer...\n");
        return -1;
    }
    F16* Output = (F16*)pi_l2_malloc(128*sizeof(F16));
    if(Output==NULL){
        printf("Error allocating output buffer...\n");
        return -1;
    }

    char* ImageName = __XSTR(INPUT_IMAGE);
    printf("Reading image %s\n", ImageName);
	//Reading Image from Bridge
	if (ReadImageFromFile(ImageName, FACE_ID_W, FACE_ID_H, FACE_ID_C, Input, FACE_ID_SIZE*sizeof(char), IMGIO_OUTPUT_CHAR, 0)) {
        printf("Failed to load image %s\n", ImageName);
        return 1;
	}
    
     for(int i = 0;i<FACE_ID_W*FACE_ID_H;i++){
        unsigned char tmp = Input[i*3];
        Input[i*3]=Input[i*3+2];
        Input[i*3+2]=tmp;
    }

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = STACK_SIZE;

    pi_open_from_conf(&cluster_dev, (void *) &cl_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-4);
    }

    printf("FC Frequency = %d Hz CL Frequency = %d Hz PERIPH Frequency = %d Hz\n", 
            pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));

	#ifdef VOLTAGE
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
	#endif
	printf("Voltage: %dmV\n", pi_pmu_voltage_get(PI_PMU_VOLTAGE_DOMAIN_CHIP));
    

    printf("Constructor\n");
    int ConstructorErr = face_idCNN_Construct();
    if (ConstructorErr)
    {
        printf("Graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
        pmsis_exit(-6);
    }
    
    fi_cluster_arg.input=Input;
    fi_cluster_arg.output=Output;

    {
        pi_perf_conf(1 << PI_PERF_CYCLES | 1 << PI_PERF_ACTIVE_CYCLES);
        gap_fc_starttimer();
        gap_fc_resethwtimer();
        int start = gap_fc_readhwtimer();
        struct pi_cluster_task task_ctor;
        pi_cluster_task(&task_ctor, (void (*)(void *)) face_idCNN_ConstructCluster, NULL);
        pi_cluster_send_task_to_cl(&cluster_dev, &task_ctor);
        int elapsed = gap_fc_readhwtimer() - start;
        printf("L1 Promotion copy took %d FC Cycles\n", elapsed);
    }

    printf("Call cluster\n");
    struct pi_cluster_task task;
    pi_cluster_task(&task, (void (*)(void *))cluster, &fi_cluster_arg);
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

    pi_cluster_send_task_to_cl(&cluster_dev, &task);

    face_idCNN_Destruct();

#ifdef PERF
	{
		unsigned long long int TotalCycles = 0, TotalOper = 0;
		printf("\n");
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
		}
		for (unsigned int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100*((float) (AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100*((float) (AT_GraphOperInfosNames[i]) / TotalOper), ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
		}
		printf("\n");
		printf("%45s: Cycles: %12llu, Cyc%%: 100.0%%, Operations: %12llu, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
		printf("\n");
	}
#endif

    #ifdef CI
    for(int i=0;i<128;i++){
        if(fabs(Output[i]-golden[i])>0.001){
            printf("CI check error...\n");
            printf("%d index - value output %f : value golden %f\n",i,Output[i],golden[i]);
            return -1;
        }
    }
    printf("CI Passed successfully!\n");
    #endif

    printf("Ended\n");
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL shufflenet Example ***\n\n");
    int ret = face_id();
    return ret;
}
