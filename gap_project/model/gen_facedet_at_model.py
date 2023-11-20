import argparse, random, os
from PIL import Image
import numpy as np
from scipy.spatial import distance
from numpy.linalg import norm
from tqdm import tqdm
from nntool.api import NNGraph
from nntool.api.types import ConstantInputNode
from nntool.api.utils import model_settings, quantization_options, RandomIter
from nntool.quantization.qtype import QType

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'FaceID Model Generator', description="Face ID AT Model Generation with NNTool")
    
    parser.add_argument("--model_path", default="", type=str, required=True,
                        help="Path to onnx file")
    parser.add_argument("--quant_files", default="None", type=str, required=False,
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
    parser.add_argument("--graph_warm_construct", default=0, type=int, required=False,
                        help="Warm Constructor in Autotiler generated code")
    
    args = parser.parse_args()
    
    #Load Graph, adjust and fusions
    G = NNGraph.load_graph(args.model_path)
    G.name='face_det'
    
    G.adjust_order()
    G.fusions('scaled_match_group')
    G.fusions('expression_matcher')
    G.remove_nodes(G["CONCAT_0_163"], leave=False, up=False)
    G.remove_nodes(G["CONCAT_0_162"], leave=False, up=False)
    G.node('output_1').fixed_order = True
    G.node('output_2').fixed_order = True
    G.node('output_3').fixed_order = True
    G.node('output_4').fixed_order = True

    stats = G.collect_statistics(RandomIter.fake(G))
    
    G.quantize(
        stats,
        graph_options=quantization_options( hwc=True, scheme="FLOAT", float_type="float16"),
        node_options={'input_1': {'qtype_ind': QType.from_min_max_sq(-1, 0.991, dtype=np.uint8)}}
    )

    G.insert_resizer(G[0],(480,480),spatial_axes=(0,1),resize_op="bilinear")

    #print(G.show())
    #print(G.qshow())
    """
    #self.G.remove_nodes(node_from, node_to, up=args.up, leave=args.leave, no_check=args.no_check)
    #print(G.show())
    G.remove_nodes(G['CONCAT_0_162'], node_to=None, up=False, leave=False, no_check=False)
    G.remove_nodes(G['CONCAT_0_163'], node_to=None, up=False, leave=False, no_check=False)
    
    setattr(G[3].at_options, "PARALLELFEATURES", 0)
    setattr(G[6].at_options, "PARALLELFEATURES", 0)
    
    # G.insert_resizer(G[0],(1920,1080))
    # int_nodes=['input_1_resizer']
    stats = G.collect_statistics(RandomIter.fake(G))
    
    G.quantize(
        statistics=stats,
        graph_options=quantization_options(scheme="FLOAT",float_type="float16"),
    )
    
    # G.quantize(
    #     statistics=stats,
    #     graph_options={
    #         "scheme": "FLOAT",
    #         "float_type": "float16"
    #     })
    #G.draw(fusions=True,filepath='graph',quant_labels=True,nodes=G.nodes())
    """


    res = G.gen_at_model(
        settings=model_settings(l1_size                           = args.l1_size,
                                l2_size                           = args.l2_size,
                                l3_size                           = args.l3_size, 
                                tensor_directory                  = args.tensors_dir,
                                l3_ram_ext_managed                = True,
                                l3_flash_ext_managed              = False,
                                graph_l1_promotion                = False,
                                gen_name_suffix                   = args.gen_name_suffix,
                                model_file                        = args.model_file,
                                graph_monitor_cycles              = True,
                                graph_monitor_cvar_name           = 'AT_FaceDet_GraphPerf',
                                graph_produce_node_names          = True,
                                graph_produce_node_cvar_name      = 'AT_FaceDet_GraphNodeNames',
                                graph_produce_operinfos           = True,
                                graph_produce_operinfos_cvar_name = 'AT_FaceDet_GraphOperInfosNames',
                                graph_warm_construct              = args.graph_warm_construct,
                                default_input_exec_location       = 'AT_MEM_L3_DEFAULTRAM',
                                default_input_home_location       = 'AT_MEM_L3_DEFAULTRAM',
                                graph_const_exec_from_flash       = False
                                # graph_trace_exec=False, # Set to True for debugging 
                                # graph_checksum=True # Set to True for debugging 
                                ),
        directory=args.gen_model_path,
        at_loglevel=1
    )

