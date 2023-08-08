
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
#include "he.h"

#define IMG_TEST_N 4

#ifndef STACK_SIZE
#define STACK_SIZE      1024
#endif

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

char*image_list[128]={
    "../cropped_faces/francesco_1.png_face_crop.ppm",
    "../cropped_faces/francesco_2.png_face_crop.ppm",
    "../cropped_faces/manuele_1.png_face_crop.ppm",
    "../cropped_faces/manuele_2.png_face_crop.ppm"
};

// def cos_sim(a,b):
//     return 100*round(1 - (np.dot(a, b)/(norm(a)*norm(b))),4)

float cosine_similarity(F16*a, F16*b){
    F16 norm_a=0,norm_b=0;
    for(int i=0;i<128;i++){
        norm_a += a[i]*a[i];
        norm_b += b[i]*b[i];
    }
    norm_a=sqrt(norm_a);
    norm_b=sqrt(norm_b);
    
    F16 dot_p=0;
    for(int i=0;i<128;i++){
        dot_p+=(a[i]*b[i]);
    }

    return dot_p/(norm_a*norm_b);
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
    // Create space for 3 images to test face reid
    F16** Output = (F16**)pi_l2_malloc(IMG_TEST_N*sizeof(F16*));

    for (int i=0;i<IMG_TEST_N;i++){
        Output[i] = (F16*)pi_l2_malloc(128*sizeof(F16));
        if(Output[i]==NULL){
            printf("Error allocating output buffer...\n");
            return -1;
        }
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
    

    for (int i=0;i<IMG_TEST_N;i++){
        printf("Reading image %s\n", image_list[i]);
        if (ReadImageFromFile(image_list[i], FACE_ID_W, FACE_ID_H, FACE_ID_C, Input, FACE_ID_SIZE*sizeof(char), IMGIO_OUTPUT_CHAR, 0)) {
            printf("Failed to load image %s\n", image_list[i]);
            return 1;
        }


        #if EQ_HIST
        histogram_eq_HWC_fc(Input,FACE_ID_W, FACE_ID_H);
        #endif
        fi_cluster_arg.input=Input;
        fi_cluster_arg.output=Output[i];

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
    }
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


    printf("Cosine similarity results:\n");
    printf("Cosine similarity francesco1 - francesco2: %f\n",cosine_similarity(Output[0],Output[1]));
    printf("Cosine similarity manuele1 - manuele2: %f\n",cosine_similarity(Output[2],Output[3]));
    printf("Cosine similarity francesco1 - manuele1: %f\n",cosine_similarity(Output[0],Output[2]));
    printf("Cosine similarity francesco2 - manuele2: %f\n",cosine_similarity(Output[1],Output[3]));
    
    // Decomment to print output tensor
    // printf("Output:\n");
    // for(int i=0;i<128;i++)printf("%f ",Output[i]);
    // printf("\n\n");

    #ifdef CI
    for(int i=0;i<128;i++){
        if(fabs(Output[0][i]-golden[i])>0.001){
            printf("CI check error...\n");
            printf("%d index - value output %f : value golden %f\n",i,Output[0][i],golden[i]);
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
