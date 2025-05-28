import os
import sys
import random
import numpy as np
import torch
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from utils.quant_utils import wrap_to_quant_model, init_weight_quantizer, init_input_quantizer, register_online_had, init_k_quantizer, init_v_quantizer
import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from utils.data_utils import test_ppl
from utils.train_utils import load_json_as_namespace,create_logger
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model


torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, tokenizer,prefixed_key_values, args, logger):
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map, skip_keys='past_key_values') # set skip_keys to avoid a bug
    prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)
    results_str=""
    if args.eval_ppl:
        datasets = ["wikitext2", "c4"]
        # datasets = ["wikitext2"]
        try:
            ppl_results = test_ppl(args, model, tokenizer, prefixed_key_values, datasets)
            for dataset in ppl_results:
                logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
                results_str += f"{ppl_results[dataset]:.2f} "
        except Exception as e:
            logger.error(f"Error during perplexity evaluation: {str(e)}")
            logger.info("Continuing with other evaluations...")


    if args.eval_tasks != "":
        try:
            if prefixed_key_values is not None:
                model = model_utils.WrappedPrefixCausalLM(model, prefixed_key_values)
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            from lm_eval.utils import make_table
            task_list = args.eval_tasks.split(',')
            # Create HFLM with explicit tokenizer
            model = HFLM(
                pretrained=model,
                tokenizer=tokenizer,  # Pass tokenizer explicitly
                batch_size=args.eval_batch_size
            )
            task_manager = lm_eval.tasks.TaskManager()
            try:
                results = lm_eval.simple_evaluate(
                model=model,
                tasks=task_list,
                num_fewshot=0,
                task_manager=task_manager,
                )
                logger.info(make_table(results))
                total_acc = 0
                total_acc_with_norm = 0
                for task in ['winogrande','hellaswag','arc_challenge','arc_easy','piqa']:
                    if task in task_list:
                        total_acc += results['results'][task]['acc,none']
                        results_str += f"{results['results'][task]['acc,none']*100:.2f} "
                        if 'acc_norm,none' in results['results'][task]:
                            results_str += f"{results['results'][task]['acc_norm,none']*100:.2f} "
                            total_acc_with_norm += results['results'][task]['acc_norm,none']
                        else:
                            total_acc_with_norm += results['results'][task]['acc,none']
                logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
                logger.info(f'Average Acc (with norm): {total_acc_with_norm/len(task_list)*100:.2f}%')
                logger.info(f'Results string: {results_str.strip()}')
            except RuntimeError as e:
                if "operator torchvision::nms does not exist" in str(e):
                    logger.error(f"Error: Missing torchvision dependency. Please install compatible torchvision:")
                    logger.error(f"Run: pip install -U torchvision")
                    logger.error(f"Then ensure you have the correct version for your PyTorch installation.")
                else:
                    logger.error(f"Runtime error during task evaluation: {str(e)}")
            # remove wrapper
            if prefixed_key_values is not None:
                model = model.model
        except Exception as e:
            logger.error(f"Error during task evaluation: {str(e)}")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")
    parser.add_argument("--original_model_path", type=str, default="", help="original model path")


    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)

    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    # if quant_config['set_prefixed_tokens']:
    if quant_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
    else:
        prefixed_key_values = None


    # init quantized model
    config = AutoConfig.from_pretrained(args.quant_model_path,trust_remote_code=True)
    if not args.original_model_path:
        raise ValueError("--original_model_path must be specified to load the tokenizer")
    logger.info(f"Loading tokenizer from {args.original_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.original_model_path,
        use_fast=False,
        legacy=False,
        trust_remote_code=True,
        local_files_only=False  # Allow downloading from HF if needed
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(args.quant_model_path, config=config, device_map='cpu',torch_dtype=torch.float16,trust_remote_code=True)
    wrap_to_quant_model(model)
    # register on-line hadadamrd transformation
    if quant_config.down_online_had:
        register_online_had(model)
    # wrap rope for online_had and rope output capture
    rope_function_name = model_utils.get_rope_function_name(model)
    layers = model_utils.get_layers(model)
    for layer in layers:
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn, 
                    rope_function_name, 
                    config=model.config,
                    online_had=quant_config.qk_online_had)   

    # init weight quantizer
    if quant_config.wbits < 16:
        logger.info('init weight quantizer')
        init_weight_quantizer(quant_config, model, minmax_init=False)

    # init input quantizer
    if quant_config.input_bits < 16:
        logger.info('init input quantizer')
        init_input_quantizer(quant_config, model,  minmax_init=False)

    # init kv quantizer
    if quant_config.v_bits < 16:
        logger.info('init v quantizer')
        init_v_quantizer(quant_config, model,  minmax_init=False)

    # if True:
    if quant_config.k_bits < 16:
        # consistently init for wrap rope 
        logger.info('init k quantizer')
        init_k_quantizer(quant_config, model,  minmax_init=False)



    # model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=args.quant_model_path,device_map=device_map,dtype=torch.float16)
    model.half()    # to make sure same evaluation results with main
    evaluate(model, tokenizer, prefixed_key_values,  args,logger)



if __name__ == "__main__":
    print(sys.argv)
    main()
