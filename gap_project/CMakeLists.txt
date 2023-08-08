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

set(MODEL_EXPRESSIONS  ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}/Expression_Kernels.c)
set(AT_MODEL_DIR       ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR})
set(AT_MODEL_PATH       ${AT_MODEL_DIR}/model.c)
set(AT_TENSORS_DIR      ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}/tensors)

autotiler_define_gen_names(${TARGET_NAME})
autotiler_setup(${TARGET_NAME})

add_custom_command(
  OUTPUT ${AT_MODEL_PATH} ${MODEL_EXPRESSIONS}
  COMMENT "Generate the autotiler model: ${CONFIG_TRAINED_MODEL} ${CONFIG_MODEL_FLAG} ${CONFIG_QUANTIZED_TFLITE_MODEL}"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/model/generate_at_model.py
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/model/generate_at_model.py --model_path=${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_NNTOOL_MODEL_PATH}
                                                               --tensors_dir=${AT_TENSORS_DIR}
                                                               --quant_files="/home/francesco/works/machine_learning/face_id/DATASETS/CASIA-WebFace_cropped/"
                                                               --gen_model_path=${AT_MODEL_DIR}
)

###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        he.c
                        ${CNN_LIB}
                        ${CONFIG_MODEL_BUILDDIR}/${KERNEL_C}
                        ${MODEL_EXPRESSIONS}
)

list(APPEND TARGET_INCS -I.
                        -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                        -I$ENV{GAP_SDK_HOME}/utils/power_meas_utils
                          ${CNN_LIB_INCLUDE}
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
                                -DINPUT_IMAGE=${CMAKE_CURRENT_SOURCE_DIR}/${CONFIG_TEST_IMAGE}       
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