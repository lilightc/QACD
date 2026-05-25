import argparse
import json
import os
from PIL import Image
from tqdm import tqdm

from models.base_models import load_model
from utils.utils import (
    VcdConfig,
    get_last_qid,
    set_seed,
    setup_path
)
setup_path()


def eval_model(args):
    # Print configs
    print('='*79)
    for key, value in vars(args).items():
        print(f'{key:<20}: {value}')
    print('='*79)

    # Data and result file loading
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), 'r')]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Model loading
    cd_config = VcdConfig(
        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        cd_tau=args.cd_tau,
        crop_ratio=args.crop_ratio,
        mask_ratio=args.mask_ratio,
        noise_step=args.noise_step,
        cd_mode=args.cd_mode,
        qacd_layer=args.qacd_layer,
        qacd_thresh_mode=args.qacd_thresh_mode,
        qacd_grow_ratio=args.qacd_grow_ratio,
        qacd_lam=args.qacd_lam,
        qacd_sink_norm=args.qacd_sink_norm,
        qacd_smooth_sigma=args.qacd_smooth_sigma,
        qacd_min_region=args.qacd_min_region,
        qacd_dilate=args.qacd_dilate,
        qacd_region=args.qacd_region,
        qacd_intensity=args.qacd_intensity,
        qacd_ops=tuple(o for o in args.qacd_ops.split(',') if o) if args.qacd_ops else (),
        qacd_prompt=args.qacd_prompt,
        qacd_icl=args.qacd_icl,
        qacd_center_frac=args.qacd_center_frac,
        qacd_debug_dir=args.qacd_debug_dir,
        demo=False
    )
    model = load_model(args.model_id, cd_config)

    # Can save computation by recording SAS results
    if args.sas_path is not None:
        sas = [json.loads(i) for i in open(os.path.expanduser(args.sas_path), 'r')]
    else:
        sas = None

    # Open result file and loop
    if args.limit:
        questions = questions[:args.limit]

    n = parse_fb = region_fb = 0
    ans_file = open(answers_file, 'a')
    for i, line in enumerate(tqdm(questions, ncols=79)):
        idx = line['question_id']

        image_file, query = line['image'], line['text']
        outputs = model.generate_sentence(
            query=query,
            image_path=os.path.join(args.image_folder, image_file),
            append_txt='\nAnswer the question using a single word or phrase',
            mode=sas[i]['applied_aug'] if cd_config.cd_mode == 'selfaug' else cd_config.cd_mode,
            qid=idx,
        )

        qacd_meta = outputs.get('qacd')
        if qacd_meta is not None:
            n += 1
            parse_fb += int(qacd_meta.get('parse_fallback', False))
            region_fb += int(qacd_meta.get('region_fallback', False))

        ans_file.write(json.dumps({'question_id': idx,
                                   'prompt': query,
                                   'text': outputs.get('text'),
                                   'applied_aug': outputs.get('applied_aug'),
                                   'reason': outputs.get('reason'),
                                   'thresholds': outputs.get('thresholds'),
                                   'qacd': qacd_meta,
                                   'model_id': args.model_id,
                                   'image': image_file,
                                   'metadata': {}}) + '\n')
        ans_file.flush()
    ans_file.close()

    if n:
        print('=' * 79)
        print(f'QACD fallback summary ({n} questions)')
        print(f'  parse fallback : {parse_fb}/{n} = {parse_fb / n:.1%}'
              '  (planner output unparseable -> noise/intensity 2)')
        print(f'  region fallback: {region_fb}/{n} = {region_fb / n:.1%}'
              '  (attention grounding failed -> center region)')
        print('=' * 79)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=str, default='llava15_7b')
    parser.add_argument('--image-folder', type=str, default='')
    parser.add_argument('--question-file', type=str, default='tables/question.jsonl')
    parser.add_argument('--answers-file', type=str, default='answer.jsonl')
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    # generation config
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--top_k', type=int, default=None)
    # VCD configs
    parser.add_argument('--crop-ratio', type=float, default=2.0)
    parser.add_argument('--mask-ratio', type=float, default=2.0)
    parser.add_argument('--noise-step', type=int, default=500)
    parser.add_argument('--cd-mode', type=str, default='no_vcd')
    parser.add_argument('--cd-alpha', type=float, default=1)
    parser.add_argument('--cd-beta', type=float, default=0.1)
    parser.add_argument('--cd-tau', type=float, default=None)
    parser.add_argument('--sas-path', type=str, default=None)
    # QACD configs (only used when --cd-mode qacd)
    parser.add_argument('--qacd-layer', type=int, default=16,
                        help='LLM layer index for attention grounding')
    parser.add_argument('--qacd-thresh-mode', type=str, default='std',
                        choices=['std', 'hysteresis'],
                        help='std=threshold+dilate; hysteresis=grow to object extent')
    parser.add_argument('--qacd-grow-ratio', type=float, default=0.5,
                        help='hysteresis low threshold = median+ratio*(high-median)')
    parser.add_argument('--qacd-lam', type=float, default=0.5,
                        help='seed/threshold = mean + lam*std (lower = broader)')
    parser.add_argument('--qacd-sink-norm', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='subtract query-agnostic baseline attention (sink removal)')
    parser.add_argument('--qacd-smooth-sigma', type=float, default=0.6,
                        help='Gaussian smoothing sigma on the patch grid (0=off)')
    parser.add_argument('--qacd-min-region', type=int, default=2,
                        help='drop attention blobs smaller than N grid cells '
                             '(keeps a varying number of regions; 1=keep all)')
    parser.add_argument('--qacd-dilate', type=int, default=0,
                        help='dilate the mask by N grid cells (0=off)')
    parser.add_argument('--qacd-region', type=str, default='attention',
                        choices=['attention', 'center', 'full'])
    parser.add_argument('--qacd-intensity', type=int, default=0,
                        help='0 = use planner-chosen intensity; else force 1/2/3')
    parser.add_argument('--qacd-ops', type=str, default='',
                        help='comma-separated op whitelist; empty = allow all')
    parser.add_argument('--qacd-prompt', type=str, default='adversarial',
                        choices=['adversarial', 'neutral'])
    parser.add_argument('--qacd-icl', action=argparse.BooleanOptionalAction,
                        default=True, help='few-shot exemplars in planner prompt')
    parser.add_argument('--qacd-center-frac', type=float, default=0.5)
    parser.add_argument('--qacd-debug-dir', type=str, default='',
                        help='if set, save per-question overlays + recipes here')
    parser.add_argument('--limit', type=int, default=0,
                        help='process only the first N questions (0 = all)')
    # others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=str, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    set_seed(args.seed)
    eval_model(args)
