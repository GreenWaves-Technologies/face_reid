cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME         face_id)
SET(CONFIG_CI yes)
###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

###############################################################################
# Set Mode
###############################################################################

if(CONFIG_ONLY_FACE_ID_INFERENCE)
  set(MAIN                face_id.c)
  set(GRAPH_WARM 0)
  list(APPEND TARGET_PREPROCESSOR -DINPUT_IMAGE_1=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/francesco_1.png_face_crop.ppm
								  -DINPUT_IMAGE_2=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/francesco_2.png_face_crop.ppm
								  -DINPUT_IMAGE_3=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/manuele_1.png_face_crop.ppm
								  -DINPUT_IMAGE_4=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/manuele_2.png_face_crop.ppm)

elseif(CONFIG_FACE_DETECTION_ID_INFERENCE)
  set(MAIN  face_det_id.c)
  set(GRAPH_WARM 3)
  list(APPEND TARGET_PREPROCESSOR -DINPUT_IMAGE_1=${CMAKE_CURRENT_SOURCE_DIR}/vga_samples/test_image_raw_bayer480x480_2.pgm)
  list(APPEND TARGET_SRCS post_process.c)
  list(APPEND TARGET_CFLAGS -DFLOAT_POST_PROCESS )

elseif(CONFIG_FACE_DETECTION_ID_GEN_SIGNATURE)
	set(MAIN  face_det_id_gen_signature.c)
	set(GRAPH_WARM 3)
	list(APPEND TARGET_SRCS post_process.c)
	list(APPEND TARGET_CFLAGS -DFLOAT_POST_PROCESS )
elseif(CONFIG_DEMO)
  	set(MAIN face_det_id_demo.c)
  	set(GRAPH_WARM 3)
	list(APPEND TARGET_SRCS post_process.c)
  	list(APPEND TARGET_CFLAGS -DFLOAT_POST_PROCESS )
	
endif()

###############################################################################
# Autotiler Variables
###############################################################################

math(EXPR FACEID_L3_SIZE "1024*1024*16")
math(EXPR FACEDET_L3_SIZE "1024*1024*16")

### FACE ID Model

set(BUILD_MODEL_FACEID BUILD_MODEL_FACEID)
#set(BUILD_MODEL_FACEID BUILD_MODEL)

set(FACEID_TARGET_NAME       face_id)
set(FACEID_AT_MODEL_DIR      ${CMAKE_BINARY_DIR}/${BUILD_MODEL_FACEID})
set(MODEL_EXPRESSIONS        ${FACEID_AT_MODEL_DIR}/${FACEID_TARGET_NAME}Expression_Kernels.c)

set(FACEID_AT_BIN            ${FACEID_TARGET_NAME}GenTile)
set(FACEID_AT_BIN_PATH       ${FACEID_AT_MODEL_DIR}/${FACEID_TARGET_NAME}GenTile)
set(FACEID_AT_MODEL_PATH     ${FACEID_AT_MODEL_DIR}/${FACEID_TARGET_NAME}model.c)
set(FACEID_AT_TENSORS_DIR    ${FACEID_AT_MODEL_DIR}/tensors)
set(FACEID_NNTOOL_PY         ${CMAKE_CURRENT_SOURCE_DIR}/model/gen_faceid_at_model.py)
set(FACEID_NNTOOL_MODEL      shufflenet.onnx)
set(FACEID_NNTOOL_MODEL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/model/${FACEID_NNTOOL_MODEL})
set(FACEID_AT_MODEL_NAME     ${FACEID_TARGET_NAME}model.c)
set(KERNEL_C                 ${FACEID_TARGET_NAME}Kernels.c)
set(FACEID_KERNEL_C         ${FACEID_AT_MODEL_DIR}/${KERNEL_C})

if(${CONFIG_CI})
  set(NNTOOL_SCRIPT_CI 1)
else()
  set(NNTOOL_SCRIPT_CI 0)
endif()

autotiler_add_model(
  TARGET_NAME ${FACEID_TARGET_NAME}
  MODEL_PATH ${FACEID_AT_MODEL_PATH}
  MODEL_TENSORS_DIR ${FACEID_AT_TENSORS_DIR}
  #APP_NAME face_id 
  MODEL_BUILDDIR ${BUILD_MODEL_FACEID}
  #KERNEL_NAME face_idKernels
  GEN_NAME face_id
  PRI1_USED TRUE
  PRI1_FLASH flash
  #PRI1_FLASH mram
  L1_MEMORY 100000 #${CONFIG_MODEL_L1_MEMORY}
  L2_MEMORY 512000 #${CONFIG_MODEL_L2_MEMORY}
  L3_MEMORY ${FACEID_L3_SIZE}
  #L2STATIC_MEMORY ${CONFIG_MODEL_L2STATIC_MEMORY}
  #L3STATIC_MEMORY ${CONFIG_MODEL_L3STATIC_MEMORY}
)



add_custom_command(
  OUTPUT  ${FACEID_AT_MODEL_PATH} #${MODEL_EXPRESSIONS}
  COMMENT "Generate the autotiler model: ${FACEID_NNTOOL_MODEL}"
  DEPENDS ${FACEID_NNTOOL_PY}
  COMMAND python3 ${FACEID_NNTOOL_PY} --model_path=${FACEID_NNTOOL_MODEL_PATH}
									  --tensors_dir=${FACEID_AT_TENSORS_DIR}
									  --quant_files=${CMAKE_CURRENT_SOURCE_DIR}/quant_dataset/
									  --gen_model_path=${FACEID_AT_MODEL_DIR}
									  --model_file=${FACEID_AT_MODEL_NAME}
									  --gen_name_suffix=${FACEID_TARGET_NAME}
									  --CI=${NNTOOL_SCRIPT_CI}
									  --graph_warm_construct=${GRAPH_WARM}
)

### FACE Detection Model

if(CONFIG_FACE_DETECTION_ID_INFERENCE OR CONFIG_FACE_DETECTION_ID_GEN_SIGNATURE OR CONFIG_DEMO)

  set(BUILD_MODEL_FACEDET BUILD_MODEL_FACEDET)

  set(FACEDET_TARGET_NAME       face_det)
  set(FACEDET_AT_MODEL_DIR      ${CMAKE_CURRENT_BINARY_DIR}/${BUILD_MODEL_FACEDET})
  set(MODEL_EXPRESSIONS         ${FACEDET_AT_MODEL_DIR}/${FACEDET_TARGET_NAME}Expression_Kernels.c)

  set(FACEDET_AT_BIN            ${FACEDET_TARGET_NAME}GenTile)
  set(FACEDET_AT_BIN_PATH       ${FACEDET_AT_MODEL_DIR}/${FACEDET_TARGET_NAME}GenTile)
  set(FACEDET_AT_MODEL_PATH     ${FACEDET_AT_MODEL_DIR}/${FACEDET_TARGET_NAME}model.c)
  set(FACEDET_AT_TENSORS_DIR    ${FACEDET_AT_MODEL_DIR}/tensors)
  set(FACEDET_NNTOOL_PY         ${CMAKE_CURRENT_SOURCE_DIR}/model/gen_facedet_at_model.py)
  set(FACEDET_NNTOOL_MODEL      face_detection_front.tflite)
  set(FACEDET_NNTOOL_MODEL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/model/${FACEDET_NNTOOL_MODEL})
  set(FACEDET_AT_MODEL_NAME     ${FACEDET_TARGET_NAME}model.c)
  set(KERNEL_C                 ${FACEDET_TARGET_NAME}Kernels.c)
  set(FACEDET_KERNEL_C         ${FACEDET_AT_MODEL_DIR}/${KERNEL_C})

  if(${CONFIG_CI})
	set(NNTOOL_SCRIPT_CI 1)
  else()
	set(NNTOOL_SCRIPT_CI 0)
  endif()

  autotiler_add_model(
	TARGET_NAME ${FACEDET_TARGET_NAME}
	MODEL_PATH ${FACEDET_AT_MODEL_PATH}
	MODEL_TENSORS_DIR ${FACEDET_AT_TENSORS_DIR}
	MODEL_BUILDDIR ${BUILD_MODEL_FACEDET}
	GEN_NAME face_det
	PRI1_USED TRUE
	PRI1_FLASH flash
	#PRI1_FLASH mram
	L1_MEMORY 100000 #${CONFIG_MODEL_L1_MEMORY}
	L2_MEMORY 900000 #${CONFIG_MODEL_L2_MEMORY}
	L3_MEMORY ${FACEID_L3_SIZE}
	#L2STATIC_MEMORY ${CONFIG_MODEL_L2STATIC_MEMORY}
	#L3STATIC_MEMORY ${CONFIG_MODEL_L3STATIC_MEMORY}
  )
  


  add_custom_command(
	OUTPUT ${FACEDET_AT_MODEL_PATH} #${MODEL_EXPRESSIONS}
	COMMENT "Generate the autotiler model: ${FACEDET_NNTOOL_MODEL}"
	DEPENDS ${FACEDET_NNTOOL_PY}
	COMMAND python3 ${FACEDET_NNTOOL_PY} --model_path=${FACEDET_NNTOOL_MODEL_PATH}
										--tensors_dir=${FACEDET_AT_TENSORS_DIR}
										--gen_model_path=${FACEDET_AT_MODEL_DIR}
										--model_file=${FACEDET_AT_MODEL_NAME}
										--gen_name_suffix=${FACEDET_TARGET_NAME}
										--graph_warm_construct=3
  )

  list(APPEND TARGET_CFLAGS -DIN_FILE_PATH=\"${CMAKE_SOURCE_DIR}/hd_samples/hd02.pgm\")


  #### ISP MODEL
  
  set(ISP_GEN_PATH $ENV{TILER_ISP_GENERATOR_PATH})
  set(ISP_KER_PATH $ENV{TILER_ISP_KERNEL_PATH})
  set(MODEL_SRC DeMosaicModel.c)
  set(MODEL_GEN DeMosaicKernels)
  set(MODEL_GEN_C ${MODEL_GEN}.c)
  set(MODEL_GEN_H ${MODEL_GEN}.h)
  set(MODEL_GEN_CLEAN ${MODEL_GEN_C} ${MODEL_GEN_H})
  set(ISP_USER_KER_C  ${CMAKE_BINARY_DIR}/${MODEL_GEN_C})

  add_custom_command(
	  OUTPUT  ${CMAKE_BINARY_DIR}/GenTile
	  DEPENDS ${CMAKE_SOURCE_DIR}/${MODEL_SRC}
	  COMMAND           gcc -o ${CMAKE_BINARY_DIR}/GenTile 
	  					-I$ENV{TILER_INC} 
						-I${ISP_GEN_PATH} ${CMAKE_SOURCE_DIR}/${MODEL_SRC} ${ISP_GEN_PATH}/ISP_Generators.c ${TILER_LIB}
  )

  add_custom_command(
	  OUTPUT              ${CMAKE_BINARY_DIR}/DeMosaicKernels.c ${CMAKE_BINARY_DIR}/DeMosaicKernels.h 
	  DEPENDS             ${CMAKE_BINARY_DIR}/GenTile
	  COMMAND             ${CMAKE_BINARY_DIR}/GenTile
  )



endif() #CONFIG_FACE_DETECTION_ID_INFERENCE



###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        ${ISP_USER_KER_C}
						he.c
						${CNN_LIB}
						${KERNEL_C_PATHS}
)

list(APPEND TARGET_INCS -I.
						-I${TILER_INC}
						-I${TILER_EMU_INC}
						-I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
						-I$ENV{GAP_SDK_HOME}/utils/power_meas_utils
						-I${TILER_ISP_KERNEL_PATH}
						-I$ENV{GAP_H_INC}
						${CNN_LIB_INCLUDE}
						-I${CMAKE_CURRENT_BINARY_DIR}/${BUILD_MODEL_FACEDET}
						-I${CMAKE_CURRENT_BINARY_DIR}/${BUILD_MODEL_FACEID}
						-I${CMAKE_CURRENT_SOURCE_DIR}
						-I${CMAKE_CURRENT_SOURCE_DIR}/.)


list(APPEND TARGET_CFLAGS   -O3)
list(APPEND TARGET_CFLAGS -DSTD_FLOAT)
list(APPEND APP_CFLAGS -DSTD_FLOAT)

message(STATUS "Clock speeds - Cluster ${FREQ_CL} FC ${FREQ_FC} Peripheral ${FREQ_PE}")
# math(EXPR FREQ_CL ${CONFIG_FREQUENCY_CLUSTER}/1000000 OUTPUT_FORMAT DECIMAL)
# math(EXPR FREQ_FC ${CONFIG_FREQUENCY_FC}/1000000 OUTPUT_FORMAT DECIMAL)
# math(EXPR FREQ_PE ${CONFIG_FREQUENCY_PERIPH}/1000000 OUTPUT_FORMAT DECIMAL)

list(APPEND TARGET_PREPROCESSOR -DAT_MODEL_PREFIX=${MODEL_PREFIX}
								-DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
								-DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
)

if (CONFIG_CI)
  list(APPEND TARGET_PREPROCESSOR -DCI)
endif()


project(${TARGET_NAME} C ASM)

add_executable(${TARGET_NAME} ${TARGET_SRCS})
target_link_libraries(${TARGET_NAME} PUBLIC m)

target_compile_options(${TARGET_NAME} PUBLIC   ${TARGET_CFLAGS}
												${TARGET_INCS}
												${TARGET_PREPROCESSOR})

###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
