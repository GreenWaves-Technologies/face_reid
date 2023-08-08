/*
 * Copyright (C) 2023 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

#include "he.h"
#include <math.h>

#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define max(a,b)            (((a) > (b)) ? (a) : (b))

static float threeway_max(float a, float b, float c) {
    return max(a, max(b, c));
}

static float threeway_min(float a, float b, float c) {
    return min(a, min(b, c));
}

static void rgbToHsv(uint8_t r, uint8_t g, uint8_t b, float* hsv) {

    float rd = (float) r/255;
    float gd = (float) g/255;
    float bd = (float) b/255;
    float max = threeway_max(rd, gd, bd), min = threeway_min(rd, gd, bd);
    float h=0, s, v = max;

    float d = max - min;
    s = max == 0 ? 0 : d / max;

    if (max == min) { 
        h = 0; // achromatic
    } else {
        if (max == rd) {
            h = (gd - bd) / d + (gd < bd ? 6 : 0);
        } else if (max == gd) {
            h = (bd - rd) / d + 2;
        } else if (max == bd) {
            h = (rd - gd) / d + 4;
        }
        h /= 6;
    }

    hsv[0] = h;
    hsv[1] = s;
    hsv[2] = v;
}

static void hsvToRgb(float h, float s, float v, uint8_t* rgb) {
    float r=0, g=0, b=0;

    int i   =  (int)(h * 6);
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    switch(i % 6){
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    rgb[0] = r * 255;
    rgb[1] = g * 255;
    rgb[2] = b * 255;
}
 
void histogram_eq_HWC_fc(uint8_t* img,uint32_t W, uint32_t H){

    //1KB
    uint32_t* v_hist = pi_l2_malloc(256*sizeof(uint32_t));
    //150KB 
    float* hsv_img = pi_l2_malloc(W*H*3*sizeof(float));
    //50KB
    float* new_v_chan = pi_l2_malloc(W*H*sizeof(float));


    //Convert to HSV
    for(uint32_t j=0;j<H;j++){
        for(uint32_t i=0;i<W;i++){
            rgbToHsv(img[3*(j*W+i)],img[1+3*(j*W+i)],img[2+3*(j*W+i)],&(hsv_img[3*(j*W+i)]));
        }
    }
 
    //Calculate Histogram on Value channel
    for(int i=0;i<256;i++){
        v_hist[i]=0;
    }

    for(uint32_t j=0;j<H;j++){
        for(uint32_t i=0;i<W;i++){
            float v_i = hsv_img[2+3*(j*W+i)];
            v_hist[(int)(v_i*255)]++;
        }
    }
    //1KB
    float* transfer_function = (float*)pi_l2_malloc(sizeof(float)*256);
    for(int i = 0; i < 256; i++) transfer_function[i]=0;

    // finding the normalised values using cumulative mass function
    // different scheduling clauses can be used here for comparative analysis
    //#pragma omp parallel for num_threads(n_threads) schedule(static,1) 
    
    for(int i = 0; i < 256; i++){
        float sum = 0.0;
        for(int j = 0; j < i+1; j++){
            sum += (float)v_hist[j];
        }
        //transfer_function[i] += (256)*((float)sum)/(W*H);
        transfer_function[i] += (((float)sum)/(W*H));
    }

    //#pragma omp parallel for num_threads(n_threads)
    for(unsigned int i = 0; i < W*H; i++){
        new_v_chan[i] = transfer_function[(int)(hsv_img[2+3*(i)]*255)];
    }

    pi_l2_free(transfer_function,sizeof(float)*256);
 
    //Convert back to RGB
    //Convert to HSV
    for(uint32_t j=0;j<H;j++){
        for(uint32_t i=0;i<W;i++){
            hsvToRgb(hsv_img[3*(j*W+i)],hsv_img[1+3*(j*W+i)],new_v_chan[(j*W+i)],&(img[3*(j*W+i)]));
        }
    }

    // Free used memory
    pi_l2_free(new_v_chan,W*H*sizeof(float));
    pi_l2_free(hsv_img,W*H*3*sizeof(float));
    pi_l2_free(v_hist,256*sizeof(uint32_t));
}