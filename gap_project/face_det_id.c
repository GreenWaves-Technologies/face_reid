
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
#include "face_det.h"
#include "DeMosaicKernels.h"
#include "face_idKernels.h"
#include "face_detKernels.h"
#include "gaplib/fs_switch.h"
#include "gaplib/ImgIO.h"
#include "he.h"
#include "post_process.h"

#define IMG_TEST_N 1

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s

#ifndef STACK_SIZE
#define STACK_SIZE 1024
#endif

#define IMG_IN_W 480
#define IMG_IN_H 480

typedef struct
{
    unsigned char *input;
    F16 *output;
} face_id_clusterArg;

AT_DEFAULTRAM_T DefaultRam;

AT_DEFAULTFLASH_EXT_ADDR_TYPE face_id_L3_Flash = 0;
AT_DEFAULTFLASH_EXT_ADDR_TYPE face_det_L3_Flash = 0;

typedef struct ArgISPCluster
{
    uint32_t Win;
    uint32_t Hin;
    uint32_t Wout;
    uint32_t Hout;
    uint8_t *ImageIn;
    uint8_t *ImageOut;
} ArgISPCluster_T;

static void ISP_cluster_main(ArgISPCluster_T *ArgC)
{
    printf ("cluster master start\n");
    int32_t perf_count;
    pi_perf_conf(1 << PI_PERF_CYCLES);
    pi_perf_start();

    demosaic_image(ArgC->ImageIn, ArgC->ImageOut);
    white_balance_HWC_L3Histogram(ArgC->ImageOut,95);
    
    pi_perf_stop();
    perf_count = pi_perf_read(PI_PERF_CYCLES);

    printf("\nCycles on cluster: %d Cycles\n", (perf_count));
    
}

void ISP_Filtering(pi_device_t* cluster_dev,uint8_t*in, uint8_t*out){
   /* Allocating L1 memory for cluster */
    DeMosaic_L1_Memory = (char *) pi_l1_malloc(cluster_dev, _DeMosaic_L1_Memory_SIZE);
    if (DeMosaic_L1_Memory == 0)
    {
        printf("Failed to allocate %d bytes for L1_memory\n", _DeMosaic_L1_Memory_SIZE);
        pmsis_exit(-5);
    }
    DeMosaic_L2_Memory = (char *) pi_l2_malloc( _DeMosaic_L2_Memory_SIZE);
    if (DeMosaic_L2_Memory == 0)
    {
        printf("Failed to allocate %d bytes for L1_memory\n", _DeMosaic_L2_Memory_SIZE);
        pmsis_exit(-5);
    }

    ArgISPCluster_T cluster_call;

    //Assinging all input variables to Cluster structure
    cluster_call.ImageIn     = in;
    cluster_call.Win         = IMG_IN_W;
    cluster_call.Hin         = IMG_IN_H;
    cluster_call.Wout        = IMG_IN_W;
    cluster_call.Hout        = IMG_IN_H;
    cluster_call.ImageOut    = out;

    /* Prepare task to be offload to Cluster. */
    pi_cluster_task_t task;
    pi_cluster_task(&task, (void *) ISP_cluster_main, &cluster_call);

    // /* Execute the function "cluster_main" on the Core 0 of cluster. */
    pi_cluster_send_task(cluster_dev, &task);


    pi_l1_free(cluster_dev, DeMosaic_L1_Memory, _DeMosaic_L1_Memory_SIZE);
    pi_l2_free(DeMosaic_L2_Memory, _DeMosaic_L2_Memory_SIZE);
}

static void cluster(void *Arg)
{
    face_id_clusterArg *fi_cluster_arg = (face_id_clusterArg *)Arg;

#ifdef PERF
    printf("Start timer\n");
    gap_cl_starttimer();
    gap_cl_resethwtimer();
#endif

    face_idCNN(fi_cluster_arg->input, fi_cluster_arg->output, NULL);
    printf("Runner completed\n");
}

char *image_list[128] = {
    __XSTR(INPUT_IMAGE_1),
    __XSTR(INPUT_IMAGE_2),
    __XSTR(INPUT_IMAGE_3),
    __XSTR(INPUT_IMAGE_4)};

float cosine_similarity(F16 *a, F16 *b)
{
    F16 norm_a = 0, norm_b = 0;
    for (int i = 0; i < 128; i++)
    {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = sqrt(norm_a);
    norm_b = sqrt(norm_b);

    F16 dot_p = 0;
    for (int i = 0; i < 128; i++)
    {
        dot_p += (a[i] * b[i]);
    }

    return dot_p / (norm_a * norm_b);
}

void bilinear_resize_hwc(KerResize_ArgT* Arg)
{
        uint8_t * __restrict__ In  = (uint8_t * __restrict__) Arg->In;
        unsigned int Win               = Arg->Win;
        unsigned int Hin               = Arg->Hin;
        uint8_t * __restrict__ Out = (uint8_t * __restrict__) Arg->Out;
        unsigned int Wout              = Arg->Wout;
        unsigned int HTileIn           = Arg->HTileIn;
        unsigned int HTileOut          = Arg->HTileOut;
        unsigned int HTileInIndex      = Arg->HTileInIndex;
        unsigned int HTileOutIndex     = Arg->HTileOutIndex;
        unsigned int Channels          = 3;

        unsigned int CoreId = 0;//gap_coreid();                                       
        unsigned int ChunkCell = Wout;//ChunkSize(Wout);                                 
        //unsigned int First = CoreId*ChunkCell, Last  = Min(Wout, First+ChunkCell);
        unsigned int First = 0, Last = Wout;
                                                                                  
        unsigned int WStep = Arg->WStep;                                          
        unsigned int HStep = Arg->HStep;                                          
                                                                                  
        unsigned int x, y, c;                                                     
        unsigned int hCoeff = HStep*HTileOutIndex;                                
        for (y = 0 ; y < HTileOut && First < Last ; y++) {                        
                unsigned int offsetYfloor = (hCoeff >> 16) - HTileInIndex;        
                unsigned int offsetYceil  = gap_min(offsetYfloor+1, HTileIn);     
                unsigned int hc2 = (hCoeff >> 9) & 127;                           
                unsigned int hc1 = 128 - hc2;                                     
                unsigned int wCoeff = First*WStep;                                
                                                                                  
                for (x = First ; x < Last ; x++) {                                
                        unsigned int offsetXfloor = (wCoeff >> 16);               
                        unsigned int offsetXceil =  gap_min(offsetXfloor+1, Win); 
                        unsigned int wc2 = (wCoeff >> 9) & 127;                   
                        unsigned int wc1 = 128 - wc2;                             
                        for (c = 0 ; c < Channels ; c++) {                        
                                unsigned int P1 = In[(offsetYfloor * Win + offsetXfloor) * Channels + c];
                                unsigned int P2 = In[(offsetYceil  * Win + offsetXfloor) * Channels + c];
                                unsigned int P3 = In[(offsetYfloor * Win + offsetXceil)  * Channels + c];
                                unsigned int P4 = In[(offsetYceil  * Win + offsetXceil)  * Channels + c];

                                Out[(y*Wout + x) * Channels + c] =
                                        ((P1*hc1 + P2*hc2)*wc1 + (P3*hc1 + P4*hc2)*wc2) >> 14;
                        }
                        wCoeff += WStep;
                }
                hCoeff += HStep;
        }
}

typedef struct ArgFACE_DETCluster
{
    uint8_t* in;
    F16* scores_out;
    F16* boxes_out;
} ArgFACE_DETCluster_T;


static void RunFaceDetection(ArgFACE_DETCluster_T*Arg)
{
#ifdef PERF
	//printf("Start timer\n");
	gap_cl_starttimer();
	gap_cl_resethwtimer();
#endif
  printf("Running on cluster\n");

  face_detCNN(Arg->in,Arg->boxes_out,Arg->scores_out,&(Arg->boxes_out[512*16]),&(Arg->scores_out[512]),NULL);
  printf("Runner completed\n");

}

F16 francesco[]={
    -0.064453,-0.315918,-0.023193,-0.056152,-0.159424,0.157349,-0.430908,-0.434082,-0.623047,-0.024414,0.027344,-0.427734,0.366943,0.417969,0.389404,0.343506,0.059082,0.034180,-0.032227,-0.013428,0.442871,-0.411133,0.275146,0.109375,-0.414551,0.680664,0.174072,0.284424,0.015625,-0.014648,-0.089478,0.068604,0.036133,0.314453,0.335449,0.105957,0.094727,-0.596680,0.027466,-0.253906,0.452148,-0.212158,0.260742,0.045898,0.128418,-0.570312,0.295410,0.145508,-0.385742,-0.376465,0.429199,-0.443359,0.225952,-0.014252,0.474609,0.188965,-0.264160,0.042664,-0.229736,-0.275391,-0.041016,0.483154,-0.451172,0.286133,0.420410,-0.520020,-0.191040,-0.508789,0.019531,-0.068176,0.150391,-0.303711,0.161133,-0.009277,-0.069336,0.115234,-0.344238,0.852539,0.252197,0.057251,-0.225342,0.505859,0.058838,0.107910,0.048828,0.219482,0.172852,0.414551,0.302002,-0.085083,0.371826,0.066528,-0.038330,-0.900879,-0.708008,-0.300537,-0.127930,-0.169067,-0.331787,0.059448,-0.235718,0.709961,0.254150,0.104492,-0.007080,0.341309,0.305664,0.433350,0.481445,-0.131836,-0.647461,-0.175781,-0.007568,-0.079346,0.139160,-0.532227,-0.546875,-0.235840,-0.083008,0.242065,-0.026611,-0.083740,-0.153809,0.432861,-0.742676,-0.739746,-0.083984,-0.342041
};

F16 manuele[]={
    0.223633,0.190430,0.102295,-0.967773,0.054565,-0.282227,0.463623,0.074097,-0.275391,-0.165527,0.610352,-0.642090,0.223877,-0.642090,0.327393,0.182861,-0.477539,-0.783203,-0.030029,0.044189,0.359619,-0.120361,-0.122620,0.071289,-0.959961,-0.506836,0.227295,0.289307,-0.396484,-0.292969,-0.476807,0.478516,-1.039062,1.046875,0.233765,0.377197,0.381348,-0.026611,-0.537598,0.073242,0.334473,0.044189,-0.565430,0.046387,-0.401855,0.168213,-0.061035,-0.453125,0.306641,-0.235962,-0.292969,-0.233765,-0.197388,0.054169,-0.013580,0.046875,0.591309,-0.353027,0.129883,-0.285645,-0.604004,0.372070,-0.381836,0.703613,0.044922,-0.456299,0.223755,-0.123535,0.863770,0.344238,-0.201904,-0.536621,0.565918,-0.539551,-0.174316,-0.095093,-0.812988,1.018555,-0.229858,-0.033569,-0.374268,0.348633,0.215576,0.427246,0.320068,-0.028564,-0.671387,0.258301,-0.122925,-0.496094,-0.181152,-0.128296,0.156494,-0.312256,0.367188,-0.616211,0.147705,0.169434,-0.547363,-0.278320,-0.232300,0.078613,0.534668,0.311035,-0.160889,-0.186523,-0.230957,-0.441650,0.577148,0.406494,-0.742188,0.086914,0.167480,0.262451,-0.063110,0.537109,0.532227,0.061707,-0.401123,-0.469971,-0.411621,0.012207,0.055664,-1.171875,-0.366699,0.018555,0.289062,-0.769531
};
int face_id(void)
{
    printf("Entering main controller\n");
    uint8_t *ImageIn;
    uint8_t *ImageOut_ram;

    face_id_clusterArg fi_cluster_arg;

    // Create space for IMG_TEST_N images to test face reid
    F16** Output = (F16**)pi_l2_malloc(IMG_TEST_N*sizeof(F16*));

    for (int i=0;i<IMG_TEST_N;i++){
        Output[i] = (F16*)pi_l2_malloc(128*sizeof(F16));
        if(Output[i]==NULL){
            printf("Error allocating output buffer...\n");
            return -1;
        }
    }

    F16* boxes_out=pi_l2_malloc(sizeof(F16)*(16*896));
	F16* scores_out=pi_l2_malloc(sizeof(F16)*(1*896));

    pi_device_t *ram = &DefaultRam;
    struct pi_default_ram_conf conf_ram;
    pi_default_ram_conf_init(&conf_ram);

    pi_open_from_conf(ram, &conf_ram);

    if (pi_ram_open(ram))
        return -3;

    // if(pi_open(PI_RAM_DEFAULT, &ram)){
    //     printf("Error opening ram!\n");
    //     return -1;
    // }

    if (pi_ram_alloc(ram, (uint32_t *)&ImageOut_ram, IMG_IN_W * IMG_IN_H * 3 * sizeof(F16)))
    {
        printf("Error allocating output buffer into ram");
        return -1;
    }

    /* Configure And open cluster. */
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.cc_stack_size = 4096;
    struct pi_cluster_task task;
    pi_open_from_conf(&cluster_dev, (void *)&cl_conf);
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

    printf("Constructor Face ID\n");
    ////face_idCNN_Construct(int DoL1Alloc, int DoL2Alloc, int DoL2DynAlloc, int DoL3Init, int DoL3Alloc, int DoPromotion)
    {
        int ConstructorErr = face_idCNN_Construct(0, 1, 0, 1, 1, 1);
        if (ConstructorErr)
        {
            printf("Face ID graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
            pmsis_exit(-6);
        }
    }
    printf("Constructor Face Detection\n");
    //// face_detCNN_Construct(int DoL1Alloc, int DoL2Alloc, int DoL2DynAlloc, int DoL3Init, int DoL3Alloc, int DoL3DynAlloc, int DoPromotion)
    {
        int ConstructorErr = face_detCNN_Construct(0, 1, 0, 1, 1, 1, 1);
        if (ConstructorErr)
        {
            printf("Face Detection graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
            pmsis_exit(-6);
        }
    }

    for (int iter = 0; iter < IMG_TEST_N; iter++)
    //for (int i = 0; i < 1; i++)
    {
        ImageIn = (uint8_t* )pi_l2_malloc(480*480);
        if (ReadImageFromFile(image_list[iter], IMG_IN_W, IMG_IN_H, 1, ImageIn, IMG_IN_W * IMG_IN_H * sizeof(unsigned char), IMGIO_OUTPUT_CHAR, 0))
        {
            printf("Failed to load image %s or dimension mismatch \n");
            return -1;
        }

        //////// Calling ISP
        ISP_Filtering(&cluster_dev,ImageIn, ImageOut_ram);
        pi_l2_free(ImageIn,480*480);
        
        WriteImageToFileL3(ram,"../input_rgb.ppm", 480,480,3, (uint32_t)ImageOut_ram, RGB888_IO);

        //////// Calling Face Detection
        {
            int ConstructorErr = face_detCNN_Construct(1, 0, 1, 0, 0, 0, 0);
            if (ConstructorErr)
            {
                printf("Face Detection graph warm constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
                pmsis_exit(-6);
            }
        }

        //printf("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
        
        ArgFACE_DETCluster_T facedet_arg;
        facedet_arg.in=ImageOut_ram;
        facedet_arg.boxes_out=boxes_out;
        facedet_arg.scores_out=scores_out;

        pi_cluster_task(&task, (void (*)(void *))&RunFaceDetection, &facedet_arg);
        pi_cluster_task_stacks(&task, NULL, 1024);
        pi_cluster_send_task_to_cl(&cluster_dev,&task);
        face_detCNN_Destruct(1, 0, 1, 0, 0, 0);

        float *scores = pi_l2_malloc(896*sizeof(float));
	    float *boxes  = pi_l2_malloc(16*896*sizeof(float));
	    bbox_float_t* bboxes = pi_l2_malloc(MAX_BB_OUT*sizeof(bbox_float_t));

        if(scores==NULL || boxes==NULL || bboxes==NULL){
            printf("Alloc error\n");
            return (-1);
        }
        printf("\n");
        for(int i=0;i<896;i++){
            //printf("Scores[%d] %f\n", i, scores_out[i]);
            scores[i] = 1/(1+expf(-(((float)scores_out[i]))));
            

            for(int j=0;j<16;j++){
                //printf("boxes[%d] %f\n", i, boxes_out[i]);
                boxes[(i*16)+j] = ((float)boxes_out[(i*16)+j]);
            }
        }

        post_process(scores,boxes,bboxes,480,480, 0.50f);

        non_max_suppress(bboxes);
        //printBboxes_forPython(bboxes);

        pi_l2_free(scores,896*sizeof(float));
        pi_l2_free(boxes,16*896*sizeof(float));


        // This is for debugging
        for(int i=0;i<MAX_BB_OUT;i++){
            if (bboxes[i].alive)
                printf("Detected Face %d score: %f\n",i,bboxes[i].score);
                //printf("score: %f xmin: %f ymin: %f w:%f h:%f\n",i,bboxes[i].score, bboxes[i].xmin,bboxes[i].ymin,bboxes[i].w,bboxes[i].h);
        }

        // For each found face run face ID on it
        for(int i=0;i<MAX_BB_OUT;i++){
            if (bboxes[i].alive){
                // Allocate space to load Face Bounding Box
                uint8_t * face_out = pi_l2_malloc(112*112*3);
                uint8_t * face_in = pi_l2_malloc((int)bboxes[i].w*(int)bboxes[i].h*3);
                if(face_in==NULL || face_out==NULL){
                    printf("Error allocating faces inpu! \n");
                    return -1;
                }
                
                pi_ram_read_2d(ram, (uint32_t) ImageOut_ram + (((int)bboxes[i].ymin*480 + (int)bboxes[i].xmin)*3), (void*)face_in, 
                (int)bboxes[i].w*(int)bboxes[i].h*3, 480*3,(int)bboxes[i].w*3);

                //WriteImageToFile("../face_id_resize_rgb.ppm", (int)bboxes[i].w,(int)bboxes[i].h,3, face_in, RGB888_IO);

                //Resize for face ID (112*112*3)
                KerResize_ArgT ResizeArg;
                ResizeArg.In             = face_in;
                ResizeArg.Win            = bboxes[i].w;
                ResizeArg.HTileIn        = bboxes[i].h;
                ResizeArg.Out            = face_out;
                ResizeArg.Wout           = 112;
                ResizeArg.HTileOut       = 112;
                ResizeArg.WStep          = (((int)bboxes[i].w-1)<<16)/(112-1);
                ResizeArg.HStep          = (((int)bboxes[i].h-1)<<16)/(112-1);
                ResizeArg.HTileInIndex   = 0;
                ResizeArg.HTileOutIndex  = 0;
                ResizeArg.Channels       = 3;
                bilinear_resize_hwc(&ResizeArg);
                
                pi_l2_free(face_in,(int)bboxes[i].w*(int)bboxes[i].h*3);

                WriteImageToFile("../face_id_input_rgb.ppm", 112,112,3, face_out, RGB888_IO);
                
                histogram_eq_HWC_fc(face_out, FACE_ID_W, FACE_ID_H);

                WriteImageToFile("../face_id_input_rgb_after_hist.ppm", 112,112,3, face_out, RGB888_IO);

                fi_cluster_arg.input = face_out;
                fi_cluster_arg.output = Output[iter];

                face_idCNN_Construct(1, 0, 1, 0, 0, 0);
                printf("Call cluster\n");
                pi_cluster_task(&task, (void (*)(void *))cluster, &fi_cluster_arg);
                //pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

                pi_cluster_send_task_to_cl(&cluster_dev, &task);
                face_idCNN_Destruct(1, 0, 1, 0, 0);
                pi_l2_free(face_out,112*112*3);
            }
        }

        float fra1_fra2,fra1_manu1;
        fra1_fra2  = cosine_similarity(Output[0],francesco);
        fra1_manu1 = cosine_similarity(Output[0],manuele);
        printf("Cosine similarity results:\n");
        printf("Cosine similarity francesco1 - francesco2: %f\n",fra1_fra2);
        printf("Cosine similarity francesco1 - manuele: %f\n"   ,fra1_manu1);
        if(fra1_fra2<0.58 && fra1_manu1 > 0.13){
            printf("CI ERROR, Cosine Similarity degradation\n");
            printf("Cosine similarity francesco1 - francesco2 should be >=0.58 and actually is %f\n",fra1_fra2);
            printf("Cosine similarity francesco1 - manuele should be <0.13 and actually is %f\n",fra1_manu1);
            return -1;
        }
    }
    
    //Nothing is left to do thus deallocate L2 static and L3
    face_idCNN_Destruct(0, 1, 0, 1, 1);
    face_detCNN_Destruct(0, 1, 0, 1, 1, 1);

#ifdef PERF
    {
        unsigned long long int TotalCycles = 0, TotalOper = 0;
        printf("\n");
        for (unsigned int i = 0; i < (sizeof(AT_GraphPerf) / sizeof(unsigned int)); i++)
        {
            TotalCycles += AT_GraphPerf[i];
            TotalOper += AT_GraphOperInfosNames[i];
        }
        for (unsigned int i = 0; i < (sizeof(AT_GraphPerf) / sizeof(unsigned int)); i++)
        {
            printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_GraphNodeNames[i], AT_GraphPerf[i], 100 * ((float)(AT_GraphPerf[i]) / TotalCycles), AT_GraphOperInfosNames[i], 100 * ((float)(AT_GraphOperInfosNames[i]) / TotalOper), ((float)AT_GraphOperInfosNames[i]) / AT_GraphPerf[i]);
        }
        printf("\n");
        printf("%45s: Cycles: %12llu, Cyc%%: 100.0%%, Operations: %12llu, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float)TotalOper) / TotalCycles);
        printf("\n");
    }
#endif

#ifdef FACE_DET_PERF
    {
        unsigned long long int TotalCycles = 0, TotalOper = 0;
        printf("\n");
        for (unsigned int i = 0; i < (sizeof(AT_FaceDet_GraphPerf) / sizeof(unsigned int)); i++)
        {
            TotalCycles += AT_FaceDet_GraphPerf[i];
            TotalOper += AT_FaceDet_GraphOperInfosNames[i];
        }
        for (unsigned int i = 0; i < (sizeof(AT_FaceDet_GraphPerf) / sizeof(unsigned int)); i++)
        {
            printf("%45s: Cycles: %12u, Cyc%%: %5.1f%%, Operations: %12u, Op%%: %5.1f%%, Operations/Cycle: %f\n", AT_FaceDet_GraphNodeNames[i], AT_FaceDet_GraphPerf[i], 100 * ((float)(AT_FaceDet_GraphPerf[i]) / TotalCycles), AT_FaceDet_GraphOperInfosNames[i], 100 * ((float)(AT_FaceDet_GraphOperInfosNames[i]) / TotalOper), ((float)AT_FaceDet_GraphOperInfosNames[i]) / AT_FaceDet_GraphPerf[i]);
        }
        printf("\n");
        printf("%45s: Cycles: %12llu, Cyc%%: 100.0%%, Operations: %12llu, Op%%: 100.0%%, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float)TotalOper) / TotalCycles);
        printf("\n");
    }
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
