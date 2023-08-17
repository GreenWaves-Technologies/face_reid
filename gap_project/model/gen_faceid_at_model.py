import argparse, random, os
from PIL import Image
import numpy as np
from scipy.spatial import distance
from numpy.linalg import norm
from tqdm import tqdm
from nntool.api import NNGraph
from nntool.api.types import ConstantInputNode
from nntool.api.utils import model_settings, quantization_options

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'FaceID Model Generator', description="Face ID AT Model Generation with NNTool")
    
    parser.add_argument("--model_path", default="shufflenet.onnx", type=str, required=True,
                        help="Path to onnx file")
    parser.add_argument("--quant_files", default="None", type=str, required=True,
                        help="Path to images for quantization")
    parser.add_argument("--tensors_dir", default="None", type=str, required=True,
                        help="Path for generated tensors")
    parser.add_argument("--gen_model_path", default="None", type=str, required=True,
                        help="Path for generated Autotiler Model")
    
    parser.add_argument("--l1_size", default=128000, type=int, required=False,
                        help="L1 memory size for AT code generation")
    parser.add_argument("--l2_size", default=1300000, type=int, required=False,
                        help="L2 memory size for AT code generation")
    parser.add_argument("--l3_size", default=8000000, type=int, required=False,
                        help="L3 memory size for AT code generation")
    parser.add_argument("--model_file", default="", type=str, required=False,
                        help="AT model file name")
    parser.add_argument("--gen_name_suffix", default="", type=str, required=False,
                        help="AT generated file suffix")
    parser.add_argument("--CI", default=0, type=int, required=False,
                        help="AT generated file suffix")
    

    args = parser.parse_args()
    
    #Load Graph, adjust and fusions
    G = NNGraph.load_graph(args.model_path)
    G.adjust_order()
    G.fusions('scaled_match_group')
    G.fusions('expression_matcher')
    G.name="face_id"

    #folder_in = "/home/francesco/works/machine_learning/face_id/DATASETS/CASIA-WebFace_cropped/"

    CALIBRATION_IMGS = []

    #init seed to be reproducible choices
    #random.seed(10)

    for root, dirs, files in os.walk(args.quant_files):
        for file in files:
            CALIBRATION_IMGS.append(os.path.join(root, file))

    def representative_dataset():
        #for image in tqdm(random.choices(CALIBRATION_IMGS, k=100)):
        if args.CI:
            for image in CALIBRATION_IMGS:
                img = (np.array(Image.open(image)).astype(np.float32))
                img = img / 256
                img = img.transpose(2, 0, 1)
                #img=img.reshape(3,112,112)
                yield img
        else:
            for image in tqdm(CALIBRATION_IMGS):
                img = (np.array(Image.open(image)).astype(np.float32))
                img = img / 256
                img = img.transpose(2, 0, 1)
                #img=img.reshape(3,112,112)
                yield img


    float_nodes=['_gdc_gdc_0_Conv_fusion_qin0','_gdc_gdc_0_Conv_fusion','_linearconv_Conv_qin0','_linearconv_Conv','_linearconv_Conv_reshape','_Reshape_2','output_1' ]
    
    stats = G.collect_statistics(representative_dataset())
    #np.save("../model/stats",stats)
    #stats = np.load("../model/stats.npy", allow_pickle=True)
    #force_input_size=16,force_output_size=16

    nodeqdict={
        n:quantization_options(scheme="FLOAT",float_type="bfloat16") 
            for n in float_nodes
    }
    #input_qdict={'input_1':quantization_options(bits=8,use_ne16=True,hwc=True,force_input_size=8,force_output_size=16)}
    #nodeqdict.update(input_qdict)


    G.quantize(
        statistics=stats,
        graph_options=quantization_options(bits=8,use_ne16=True,hwc=True),
        # Select specific nodes and move to different quantization Scheme - TOTAL FLEXIBILITY
        node_options=nodeqdict
    )

    # G.quantize(
    #     statistics=stats,
    #     graph_options={
    #         "scheme": "FLOAT",
    #         "float_type": "float16"
    #     })
    
    res = G.gen_at_model(
        settings=model_settings(l1_size=args.l1_size,
                                l2_size=args.l2_size,
                                l3_size=args.l3_size, 
                                tensor_directory=args.tensors_dir,
                                l3_ram_ext_managed=False,
                                l3_flash_ext_managed=False,
                                graph_l1_promotion=False,
                                gen_name_suffix=args.gen_name_suffix,
                                model_file=args.model_file
                                ),
        directory=args.gen_model_path,
        at_loglevel=1
    )

