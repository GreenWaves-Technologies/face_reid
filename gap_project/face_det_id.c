
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

#define IMG_TEST_N 4

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

    //#if IMG_VGA
    demosaic_image(ArgC->ImageIn, ArgC->ImageOut);
    //#else //IMG_HD
    //demosaic_image_HWC_HD(ArgC->ImageIn, ArgC->ImageOut);
    //#endif

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

int face_id(void)
{
    printf("Entering main controller\n");
    uint8_t *ImageIn_ram;
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

    if (pi_ram_alloc(ram, (uint32_t *)&ImageIn_ram, IMG_IN_W * IMG_IN_H * sizeof(uint8_t)))
    {
        printf("Error allocating input buffer into ram");
        return -1;
    }
    if (pi_ram_alloc(ram, (uint32_t *)&ImageOut_ram, IMG_IN_W * IMG_IN_H * 3 * sizeof(F16)))
    {
        printf("Error allocating output buffer into ram");
        return -1;
    }

    // unsigned char *Input = (unsigned char *)pi_l2_malloc(FACE_ID_SIZE * sizeof(unsigned char));
    // if (Input == NULL)
    // {
    //     printf("Error allocating input buffer...\n");
    //     return -1;
    // }
    // // Create space for 3 images to test face reid
    // F16 **Output = (F16 **)pi_l2_malloc(IMG_TEST_N * sizeof(F16 *));

    // for (int i = 0; i < IMG_TEST_N; i++)
    // {
    //     Output[i] = (F16 *)pi_l2_malloc(128 * sizeof(F16));
    //     if (Output[i] == NULL)
    //     {
    //         printf("Error allocating output buffer...\n");
    //         return -1;
    //     }
    // }

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

    printf("Constructor\n");
    ////face_idCNN_Construct(int DoL1Alloc, int DoL2Alloc, int DoL2DynAlloc, int DoL3Init, int DoL3Alloc, int DoPromotion)
    {
        int ConstructorErr = face_idCNN_Construct(0, 1, 0, 1, 1, 1);
        if (ConstructorErr)
        {
            printf("Face ID graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
            pmsis_exit(-6);
        }
    }
    printf("Constructor\n");
    //// face_detCNN_Construct(int DoL1Alloc, int DoL2Alloc, int DoL2DynAlloc, int DoL3Init, int DoL3Alloc, int DoL3DynAlloc, int DoPromotion)
    {
        int ConstructorErr = face_detCNN_Construct(0, 1, 0, 1, 1, 1, 1);
        if (ConstructorErr)
        {
            printf("Face Detection graph constructor exited with error: (%s)\n", GetAtErrorName(ConstructorErr));
            pmsis_exit(-6);
        }
    }

    // for (int i = 0; i < IMG_TEST_N; i++)
    for (int i = 0; i < 1; i++)
    {
        if (ReadImageFromFileL3(ram, image_list[i], IMG_IN_W, IMG_IN_H, 1, ImageIn_ram, IMG_IN_W * IMG_IN_H * sizeof(unsigned char), IMGIO_OUTPUT_CHAR, 0))
        {
            printf("Failed to load image %s or dimension mismatch \n");
            return -1;
        }
        // uint8_t * ImageIn=pi_l2_malloc(128*128*3);
        // F16 * ImageIn_f=pi_l2_malloc(128*128*3*2);
        // if (ReadImageFromFile("../input_rgb.ppm", 128, 128, 3, ImageIn, 128 * 128 * 3 *sizeof(unsigned char), IMGIO_OUTPUT_CHAR, 0))
        // {
        //     printf("Failed to load image %s or dimension mismatch \n");
        //     return -1;   
        // }
        
        // #define AT_INPUT_HEIGHT 128
        // #define AT_INPUT_WIDTH 128
        // #define AT_INPUT_COLORS 3

        // //HWC to CHW + casting and normalizing to F16
        // for (int h=0; h<AT_INPUT_HEIGHT; h++) {
        //     for (int w=0; w<AT_INPUT_WIDTH; w++) {
        //         for (int c=0; c<AT_INPUT_COLORS; c++) {
        //             //ImageIn_f[c*AT_INPUT_WIDTH*AT_INPUT_HEIGHT+h*AT_INPUT_WIDTH+w] = (((F16) ImageIn[h*AT_INPUT_WIDTH*AT_INPUT_COLORS+w*AT_INPUT_COLORS+c]) / 128) - 1.0f;
                    
        //             ImageIn_f[h*AT_INPUT_WIDTH*AT_INPUT_COLORS+w*AT_INPUT_COLORS+c] = (((F16) ImageIn[h*AT_INPUT_WIDTH*AT_INPUT_COLORS+w*AT_INPUT_COLORS+c]) );
        //         }
        //     }
        // }

        //////// Calling ISP
        ISP_Filtering(&cluster_dev,ImageIn_ram, ImageOut_ram);
        //pi_ram_write(ram, ImageOut_ram, ImageIn_f, 128*128*3*2);
        // pi_ram_copy(ram, ImageOut_ram, (void *) ImageIn, 128*128*3, 0);
        // pi_l2_free(ImageIn_f,128*128*3*2);
        // pi_l2_free(ImageIn,128*128*3);
        
        //WriteImageToFileL3(ram,"../input_rgb.ppm", 480,480,3, ImageOut_ram, RGB888_IO);

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
        printBboxes_forPython(bboxes);

        pi_l2_free(scores,896*sizeof(float));
        pi_l2_free(boxes,16*896*sizeof(float));


        // This is for debugging
        for(int i=0;i<MAX_BB_OUT;i++){
            if (bboxes[i].alive)
                printf("%f %f %f %f %f\n",bboxes[i].score, bboxes[i].xmin,bboxes[i].ymin,bboxes[i].w,bboxes[i].h);
        }


        for(int i=0;i<MAX_BB_OUT;i++){
            if (bboxes[i].alive){
                // Allocate space to load Face Bounding Box
                uint8_t * face_in = pi_l2_malloc((int)bboxes[i].w*(int)bboxes[i].h*3);
                uint8_t * face_out = pi_l2_malloc(112*112*3);
                if(face_in==NULL || face_out==NULL){
                    printf("Error allocating faces inpu! \n");
                    return -1;
                }
                
                // Load from L3
                // pi_ram_read_2d(ram, ImageOut_ram + (((int)bboxes[i].ymin*(int)bboxes[i].w + (int)bboxes[i].xmin)*3), face_in, 
                // (int)bboxes[i].w*(int)bboxes[i].h*3, (480-(int)bboxes[i].w)*3, (int)bboxes[i].w*3);

                // WriteImageToFile("../face_id_resize_rgb.ppm", (int)bboxes[i].w,(int)bboxes[i].h,3, face_in, RGB888_IO);
                
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
                //WriteImageToFile("../face_id_input_rgb.ppm", 112,112,3, face_out, RGB888_IO);
                
                fi_cluster_arg.input = face_out;
                fi_cluster_arg.output = Output[i];

                face_idCNN_Construct(1, 0, 1, 0, 0, 0);
                printf("Call cluster\n");
                pi_cluster_task(&task, (void (*)(void *))cluster, &fi_cluster_arg);
                //pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);

                pi_cluster_send_task_to_cl(&cluster_dev, &task);
                face_idCNN_Destruct(1, 0, 1, 0, 0);
            }
        }

        printf("Cosine similarity results:\n");
        printf("Cosine similarity francesco1 - francesco2: %f\n",cosine_similarity(Output[0],Output[0]));
    

        

//#ifdef EQ_HIST
//        histogram_eq_HWC_fc(Input, FACE_ID_W, FACE_ID_H);
//#endif


    }




    face_idCNN_Destruct(0, 1, 0, 1, 1);
    face_detCNN_Destruct(0, 1, 0, 1, 1,1);

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

    // printf("Cosine similarity results:\n");
    // printf("Cosine similarity francesco1 - francesco2: %f\n", cosine_similarity(Output[0], Output[1]));
    // printf("Cosine similarity manuele1 - manuele2: %f\n", cosine_similarity(Output[2], Output[3]));
    // printf("Cosine similarity francesco1 - manuele1: %f\n", cosine_similarity(Output[0], Output[2]));
    // printf("Cosine similarity francesco2 - manuele2: %f\n", cosine_similarity(Output[1], Output[3]));

    // Decomment to print output tensor
    // printf("Output:\n");
    // for(int i=0;i<128;i++)printf("%f ",Output[0][i]);
    // printf("\n\n");

    printf("Ended\n");
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n\n\t *** NNTOOL shufflenet Example ***\n\n");
    int ret = face_id();
    return ret;
}
