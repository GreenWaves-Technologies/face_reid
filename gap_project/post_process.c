
#include "post_process.h"


void printBboxes_forPython(bbox_float_t *boundbxs){
	printf("\n\n======================================================");
	printf("\nThis can be copy-pasted to python to draw BoudingBoxs   ");
	printf("\n\n");

	for (int counter=0;counter< MAX_BB_OUT;counter++){
		if(boundbxs[counter].alive){
			printf("rect = patches.Rectangle((%f,%f),%f,%f,linewidth=1,edgecolor='r',facecolor='none')\nax.add_patch(rect)\n",
				boundbxs[counter].xmin,
				boundbxs[counter].ymin,
				boundbxs[counter].w,
				boundbxs[counter].h
				);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k1_x,
				boundbxs[counter].k1_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k2_x,
				boundbxs[counter].k2_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k3_x,
				boundbxs[counter].k3_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k4_x,
				boundbxs[counter].k4_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k5_x,
				boundbxs[counter].k5_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k6_x,
				boundbxs[counter].k6_y);
		}
	}//
}

float rect_intersect_area(float a_x, float a_y, float a_w, float a_h,
                         float b_x, float b_y, float b_w, float b_h ){

    #define MIN(a,b) ((a) < (b) ? (a) : (b))
    #define MAX(a,b) ((a) > (b) ? (a) : (b))

    float x = MAX(a_x,b_x);
    float y = MAX(a_y,b_y);

    float size_x = MIN(a_x+a_w,b_x+b_w) - x;
    float size_y = MIN(a_y+a_h,b_y+b_h) - y;

    if(size_x <=0 || size_x <=0)
        return 0;
    else
        return size_x*size_y;

    #undef MAX
    #undef MIN
}

void non_max_suppress(bbox_float_t * boundbxs){

    int idx,idx_int;

    //Non-max supression
     for(idx=0;idx<MAX_BB_OUT;idx++){
        //check if rect has been removed (-1)
        if(boundbxs[idx].alive==0)
            continue;

        for(idx_int=0;idx_int<MAX_BB_OUT;idx_int++){

            if(boundbxs[idx_int].alive==0 || idx_int==idx)
                continue;

            //check the intersection between rects
            float intersection = rect_intersect_area(boundbxs[idx].xmin,boundbxs[idx].ymin,boundbxs[idx].w,boundbxs[idx].h,
                                                   boundbxs[idx_int].xmin,boundbxs[idx_int].ymin,boundbxs[idx_int].w,boundbxs[idx_int].h);

            if(intersection >= NON_MAX_THRES){ //is non-max
                //supress the one that has lower score that is alway the internal index, since the input is sorted
                boundbxs[idx_int].alive =0;
            }
        }
    }
}

#ifdef FLOAT_POST_PROCESS
#include "anchors.h"

void post_process(float* scores,float * boxes,bbox_float_t* bboxes,int img_w,int img_h, float thres){
    float center_y;
    float center_x;
    float bbh;
    float bbw;
    float min_x;
    float min_y;
    int det_bb=0;
	
	//This peace of code was used to generate anchors for fixed point post processing
	/*for(int i=0;i<896;i++){
    	for(int a=0;a<4;a++){
    		printf("%d, ",((int) (anchors[i*4+a]*128))<<8);
    	}
    	printf("\n");
    }*/

    for(int i=0;i<MAX_BB_OUT;i++){
		bboxes[i].alive=0;
	}
	for(int i=0;i<896;i++){
		if(scores[i]>thres){
			if(det_bb>MAX_BB_OUT){
				printf("MAX BB REACHED!!!\n");
				break;
			}

			center_x = boxes[(i*16)+box_offset_x] /  (float)X_SCALE * anchors[(i*4)+2] + anchors[(i*4)+0];
	    	center_y = boxes[(i*16)+box_offset_y] /  (float)Y_SCALE * anchors[(i*4)+3] + anchors[(i*4)+1];
			//printf("FP Scores: %f %f %f\n",scores[i],center_x,center_y);
	    	bbh = boxes[(i*16)+box_offset_height] /  (float) H_SCALE * anchors[(i*4)+3];
	    	bbw = boxes[(i*16)+box_offset_width]  /  (float) W_SCALE * anchors[(i*4)+2];
	    	//printf("bbh bbw: %f %f\n",bbh,bbw);
	    	bboxes[det_bb].h =((boxes[(i*16)+box_offset_height] /  (float) H_SCALE * anchors[(i*4)+3])*img_h);
	    	bboxes[det_bb].w =((boxes[(i*16)+box_offset_width]  /  (float) W_SCALE * anchors[(i*4)+2])*img_w);
	    	bboxes[det_bb].xmin = ((center_x - 0.5 * bbw)*img_w);
	    	bboxes[det_bb].ymin = ((center_y - 0.5 * bbh)*img_h);
	    	//printf("%f %f %f %f\n",bboxes[det_bb].xmin,bboxes[det_bb].ymin,bboxes[det_bb].w,bboxes[det_bb].h);
	    	//Decoding Keypoints
	    	bboxes[det_bb].k1_x=(boxes[(i*16)+keypoints_coord_offset+0+keypoint_offset_x] /  X_SCALE * anchors[(i*4)+2] + anchors[(i*4)+0])*img_w;
	    	bboxes[det_bb].k1_y=(boxes[(i*16)+keypoints_coord_offset+0+keypoint_offset_y] /  Y_SCALE * anchors[(i*4)+3] + anchors[(i*4)+1])*img_h;
	    	bboxes[det_bb].k2_x=(boxes[(i*16)+keypoints_coord_offset+2+keypoint_offset_x] /  X_SCALE * anchors[(i*4)+2] + anchors[(i*4)+0])*img_w;
	    	bboxes[det_bb].k2_y=(boxes[(i*16)+keypoints_coord_offset+2+keypoint_offset_y] /  Y_SCALE * anchors[(i*4)+3] + anchors[(i*4)+1])*img_h;
	    	bboxes[det_bb].k3_x=(boxes[(i*16)+keypoints_coord_offset+4+keypoint_offset_x] /  X_SCALE * anchors[(i*4)+2] + anchors[(i*4)+0])*img_w;
	    	bboxes[det_bb].k3_y=(boxes[(i*16)+keypoints_coord_offset+4+keypoint_offset_y] /  Y_SCALE * anchors[(i*4)+3] + anchors[(i*4)+1])*img_h;
	    	bboxes[det_bb].k4_x=(boxes[(i*16)+keypoints_coord_offset+6+keypoint_offset_x] /  X_SCALE * anchors[(i*4)+2] + anchors[(i*4)+0])*img_w;
	    	bboxes[det_bb].k4_y=(boxes[(i*16)+keypoints_coord_offset+6+keypoint_offset_y] /  Y_SCALE * anchors[(i*4)+3] + anchors[(i*4)+1])*img_h;
	    	bboxes[det_bb].k5_x=(boxes[(i*16)+keypoints_coord_offset+8+keypoint_offset_x] /  X_SCALE * anchors[(i*4)+2] + anchors[(i*4)+0])*img_w;
	    	bboxes[det_bb].k5_y=(boxes[(i*16)+keypoints_coord_offset+8+keypoint_offset_y] /  Y_SCALE * anchors[(i*4)+3] + anchors[(i*4)+1])*img_h;
	    	bboxes[det_bb].k6_x=(boxes[(i*16)+keypoints_coord_offset+10+keypoint_offset_x] /  X_SCALE * anchors[(i*4)+2] + anchors[(i*4)+0])*img_w;
	    	bboxes[det_bb].k6_y=(boxes[(i*16)+keypoints_coord_offset+10+keypoint_offset_y] /  Y_SCALE * anchors[(i*4)+3] + anchors[(i*4)+1])*img_h;
	
			bboxes[det_bb].score=scores[i];
	    	bboxes[det_bb++].alive=1;
		}
	}
}



#else

#include "anchors_fix.h"

void post_process_fix(int16_t* scores,int16_t * boxes,bbox_t* bboxes,int img_w,int img_h, int16_t thres){
	int center_y;
    int center_x;
    int bbh;
    int bbw;
    int min_x;
    int min_y;
    int det_bb=0;
    for(int i=0;i<MAX_BB_OUT;i++){
		bboxes[i].alive=0;
	}
	for(int i=0;i<896;i++){
		if(scores[i]>thres){
			if(det_bb>MAX_BB_OUT){
				printf("MAX BB REACHED!!!\n");
				break;
			}
			// boxes are Q8, anchors scale is removed since it is always 1 in float version
			// Anchors are already scaled to Q8 to avoid further processing
			// center_x and y is Q15
			center_x = (int) boxes[(i*16)+box_offset_x] + anchors_fix[(i*4)+0];
	    	center_y = (int) boxes[(i*16)+box_offset_y] + anchors_fix[(i*4)+1];
			// bbh and bbw is Q15
	    	bbh = ((int) boxes[(i*16)+box_offset_height]);
	    	bbw = ((int) boxes[(i*16)+box_offset_width] );
	    	bboxes[det_bb].h = bbh >> 8;
	    	bboxes[det_bb].w = bbw >> 8;
	    	bboxes[det_bb].xmin = (center_x >> 8) - (bboxes[det_bb].w>>1);
	    	bboxes[det_bb].ymin = (center_y >> 8) - (bboxes[det_bb].h>>1);
	    	//Decoding Keypoints
	    	bboxes[det_bb].k1_x=((boxes[(i*16)+keypoints_coord_offset+0+keypoint_offset_x]  + anchors_fix[(i*4)+0])>>8);
	    	bboxes[det_bb].k1_y=((boxes[(i*16)+keypoints_coord_offset+0+keypoint_offset_y]  + anchors_fix[(i*4)+1])>>8);
	    	bboxes[det_bb].k2_x=((boxes[(i*16)+keypoints_coord_offset+2+keypoint_offset_x]  + anchors_fix[(i*4)+0])>>8);
	    	bboxes[det_bb].k2_y=((boxes[(i*16)+keypoints_coord_offset+2+keypoint_offset_y]  + anchors_fix[(i*4)+1])>>8);
	    	bboxes[det_bb].k3_x=((boxes[(i*16)+keypoints_coord_offset+4+keypoint_offset_x]  + anchors_fix[(i*4)+0])>>8);
	    	bboxes[det_bb].k3_y=((boxes[(i*16)+keypoints_coord_offset+4+keypoint_offset_y]  + anchors_fix[(i*4)+1])>>8);
	    	bboxes[det_bb].k4_x=((boxes[(i*16)+keypoints_coord_offset+6+keypoint_offset_x]  + anchors_fix[(i*4)+0])>>8);
	    	bboxes[det_bb].k4_y=((boxes[(i*16)+keypoints_coord_offset+6+keypoint_offset_y]  + anchors_fix[(i*4)+1])>>8);
	    	bboxes[det_bb].k5_x=((boxes[(i*16)+keypoints_coord_offset+8+keypoint_offset_x]  + anchors_fix[(i*4)+0])>>8);
	    	bboxes[det_bb].k5_y=((boxes[(i*16)+keypoints_coord_offset+8+keypoint_offset_y]  + anchors_fix[(i*4)+1])>>8);
	    	bboxes[det_bb].k6_x=((boxes[(i*16)+keypoints_coord_offset+10+keypoint_offset_x] + anchors_fix[(i*4)+0])>>8);
	    	bboxes[det_bb].k6_y=((boxes[(i*16)+keypoints_coord_offset+10+keypoint_offset_y] + anchors_fix[(i*4)+1])>>8);
	
			bboxes[det_bb].score=FIX2FP(scores[i],15);
	    	bboxes[det_bb++].alive=1;
		}
	}
}

#endif

/*

['center_x', 'center_y', 'width', 'height'])

	From Python Decoder
    def _decode_boxes_and_keypoints(self) -> tuple:
	regressors_output = self.interpreter.get_tensor(self.regressors_output_index)
	if self.is_output_is_reversed:
	    box_offset_y = 1
	    box_offset_x = 0
	    box_offset_height = 3
	    box_offset_width = 2
	    keypoint_offset_y = 1
	    keypoint_offset_x = 0
	else:
	    box_offset_y = 0
	    box_offset_x = 1
	    box_offset_height = 2
	    box_offset_width = 3
	    keypoint_offset_y = 0
	    keypoint_offset_x = 1
	boxes = []
	keypoints = []
	for regression_data, anchor in zip(regressors_output[0], self.anchors):
	    center_y = regression_data[self.box_coordinates_offset + box_offset_y] / self.y_scale * anchor.height + anchor.center_y
	    center_x = regression_data[self.box_coordinates_offset +
				       box_offset_x] / self.x_scale * anchor.width + anchor.center_x
	    height = regression_data[self.box_coordinates_offset +
				     box_offset_height] / self.h_scale * anchor.height
	    width = regression_data[self.box_coordinates_offset +
				    box_offset_width] / self.w_scale * anchor.width

	    min_x = center_x - 0.5 * width
	    min_y = center_y - 0.5 * height
	    boxes.append(Rect(min_x, min_y, width, height))
		
		#number_of_keypoints = 6
		#number_of_values_per_keypoint = 2
	    #keypoints_coordinates_offset =4
	    box_keypoints = []
	    for i in range(self.number_of_keypoints):
		keypoint_idx = i * self.number_of_values_per_keypoint + self.keypoints_coordinates_offset
		keypoint_y = regression_data[keypoint_idx + keypoint_offset_y] / self.y_scale * anchor.height + anchor.center_y
		keypoint_x = regression_data[keypoint_idx + keypoint_offset_x] / self.x_scale * anchor.width + anchor.center_x
		box_keypoints.append(Point(keypoint_x, keypoint_y))
	    keypoints.append(box_keypoints)
	return boxes, keypoints

*/