cmake_minimum_required(VERSION 3.16)

set(TARGET_NAME         face_id)
set(MAIN                face_id.c)


###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)

###############################################################################
# Autotiler Variables
###############################################################################

set(CONFIG_MODEL_BUILDDIR BUILD_MODEL)


set(FACEID_TARGET_NAME       face_id)
set(FACEID_AT_MODEL_DIR      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR})
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

add_custom_command(
  OUTPUT ${FACEID_AT_MODEL_PATH} #${MODEL_EXPRESSIONS}
  COMMENT "Generate the autotiler model: ${FACEID_NNTOOL_MODEL}"
  DEPENDS ${FACEID_NNTOOL_PY}
  COMMAND python3 ${FACEID_NNTOOL_PY} --model_path=${FACEID_NNTOOL_MODEL_PATH}
                                      --tensors_dir=${FACEID_AT_TENSORS_DIR}
                                      --quant_files=${CMAKE_CURRENT_SOURCE_DIR}/quant_dataset/
                                      --gen_model_path=${FACEID_AT_MODEL_DIR}
                                      --model_file=${FACEID_AT_MODEL_NAME}
                                      --gen_name_suffix=${FACEID_TARGET_NAME}
                                      --CI=${NNTOOL_SCRIPT_CI}
)
#add_custom_target(${FACEID_TARGET_NAME}at_model_gen DEPENDS ${FACEID_NNTOOL_PY})

add_custom_command(
  OUTPUT ${FACEID_AT_BIN_PATH}
  COMMENT "Generate AT binary for User Kernels generation: ${FACEID_AT_MODEL_NAME}"
  DEPENDS ${FACEID_AT_MODEL_PATH}
  COMMAND gcc -o${FACEID_AT_BIN_PATH}
                  #${FLOAT_PREC} 
                  -I${TILER_INC} 
                  -I${TILER_EMU_INC}
                  -I$ENV{TILER_CNN_GENERATOR_PATH_SQ8}
                  -I$ENV{TILER_CNN_GENERATOR_PATH_FP16}
                  -I$ENV{TILER_CNN_GENERATOR_PATH_NE16}
                  -I$ENV{TILER_CNN_GENERATOR_PATH}
                  -I$ENV{TILER_CNN_KERNEL_PATH}
                  -I$ENV{TILER_CNN_KERNEL_PATH_SQ8}
                  -I$ENV{TILER_CNN_KERNEL_PATH_FP16}
                  -I$ENV{TILER_CNN_KERNEL_PATH_NE16}
                  ${CNN_GEN} ${TILER_LIB}
                  ${MODEL_GEN_SQ8}
                  ${MODEL_GEN_FP16}
                  ${MODEL_GEN_NE16}
                  ${MODEL_GEN}
                  ${FACEID_AT_MODEL_DIR}/${FACEID_AT_MODEL_NAME} 
                  $ENV{TILER_LIB}
)

#add_custom_target(${FACEID_TARGET_NAME}at_model_compile DEPENDS ${FACEID_AT_MODEL_PATH})

add_custom_command( 
  OUTPUT ${FACEID_KERNEL_C}
  COMMENT "Generate Autotiler User Kernels: ${FACEID_AT_MODEL_NAME}"
  DEPENDS ${FACEID_AT_BIN_PATH}
  COMMAND ${FACEID_AT_MODEL_DIR}/${FACEID_AT_BIN} 
          -c${FACEID_AT_TENSORS_DIR} 
          -o${FACEID_AT_MODEL_DIR}
          --L1=110000 
          --L2=1200000
          --L3=8000000
          
)

readfs_add_files(FILES ${FACEID_AT_MODEL_DIR}/${FACEID_TARGET_NAME}_L3_Flash_Const.dat FLASH ${READFS_FLASH})



###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        he.c
                        ${CNN_LIB}
                        ${FACEID_KERNEL_C}
                        ${MODEL_LIB_SQ8}
                        ${MODEL_LIB_FP16}
                        ${MODEL_LIB_NE16}
                        #${MODEL_LIB_DSP}
                        #${FACEDET_KERNEL_C}
                        #${CONFIG_MODEL_BUILDDIR}/${KERNEL_C}
                        #${MODEL_EXPRESSIONS}
)

list(APPEND TARGET_INCS -I.
                        -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                        -I$ENV{GAP_SDK_HOME}/utils/power_meas_utils
                        # ${CNN_LIB_INCLUDE}
                        -I$ENV{GAP_H_INC}
                        -I$ENV{TILER_CNN_KERNEL_PATH}
                        -I$ENV{TILER_CNN_KERNEL_PATH_SQ8}
                        -I$ENV{TILER_CNN_KERNEL_PATH_FP16}
                        -I$ENV{TILER_CNN_KERNEL_PATH_NE16}
                        -I$ENV{TILER_DSP_KERNEL_PATH}
                        -I${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR}/.)


list(APPEND TARGET_CFLAGS   -O3)

message(STATUS "Clock speeds - Cluster ${FREQ_CL} FC ${FREQ_FC} Peripheral ${FREQ_PE}")
# math(EXPR FREQ_CL ${CONFIG_FREQUENCY_CLUSTER}/1000000 OUTPUT_FORMAT DECIMAL)
# math(EXPR FREQ_FC ${CONFIG_FREQUENCY_FC}/1000000 OUTPUT_FORMAT DECIMAL)
# math(EXPR FREQ_PE ${CONFIG_FREQUENCY_PERIPH}/1000000 OUTPUT_FORMAT DECIMAL)

list(APPEND TARGET_PREPROCESSOR -DPERF
                                -DAT_MODEL_PREFIX=${MODEL_PREFIX}
                                -DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
                                -DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
                                -DINPUT_IMAGE_1=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/francesco_1.png_face_crop.ppm
                                -DINPUT_IMAGE_2=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/francesco_2.png_face_crop.ppm
                                -DINPUT_IMAGE_3=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/manuele_1.png_face_crop.ppm
                                -DINPUT_IMAGE_4=${CMAKE_CURRENT_SOURCE_DIR}/cropped_faces/manuele_2.png_face_crop.ppm
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