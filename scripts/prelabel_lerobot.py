"""One-pass Qwen3-VL-8B → SAM positive-only labels for two-view LeRobot v2 image data.

Run from the repository root with PYTHONPATH=packages/openpi-client/src.
This entry has no dependency on ignored experiments, local services, or private paths.
"""

import argparse
import datetime
import fcntl
import gc
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from openpi_client import image_tools
from PIL import Image
import pyarrow.parquet as pq


ROLES = ('source', 'goal', 'gripper', 'arm')
MODEL = 'Qwen3-VL-8B-Instruct'
REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
PROMPT = Path(__file__).with_name('router_label_prompt.txt')
LIBERO_REVISION = 'a4336d589d589045d1c56423ffdf3b88a0e19b1f'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    part.replace(path)


def save_arrays(path, **arrays):
    part = path.with_suffix('.npz.part')
    with part.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    part.replace(path)


def prompt(row, arrays=False, repair=False):
    text = PROMPT.read_text().replace('{{INSTRUCTION_JSON}}', json.dumps(row['instruction'], ensure_ascii=False))
    if arrays:
        text = text.split('\nReturn ONLY', 1)[0] + (
            '\nReturn ONLY a JSON array of [role,x_min,y_min,x_max,y_max] rows. '
            'Allowed roles: source, goal, gripper, arm. Coordinates are integers normalized '
            'to 0..1000 in this full image. Use separate rows for multiple regions. '
            'Return [] if no region is identifiable. Do not use objects or bbox keys.'
        )
    elif repair:
        text += '\nReturn a complete compact JSON array. Never repeat a bbox key in an object.'
    return text


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_boxes(text, arrays=False):
    text = text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    values = json.loads(text, object_pairs_hook=no_duplicate_keys)
    if not isinstance(values, list):
        raise ValueError('Expected an array')
    objects, adjustments = [], []
    for value in values:
        if arrays:
            if not isinstance(value, list) or len(value) != 5:
                raise ValueError('Expected [role,x_min,y_min,x_max,y_max]')
            value = {'role': value[0], 'bbox': value[1:]}
        if not isinstance(value, dict) or value.get('role') not in ROLES:
            raise ValueError('Unknown or missing role')
        raw = value['bbox']
        if not isinstance(raw, list) or len(raw) != 4 or any(type(x) is not int for x in raw):
            raise ValueError('Expected four integer coordinates')
        if raw == [0, 0, 0, 0]:
            box = None
        else:
            if raw[0] >= raw[2] or raw[1] >= raw[3]:
                raise ValueError('Inverted or zero-area box')
            box = [max(0, min(1000, x)) for x in raw]
            if box[0] >= box[2] or box[1] >= box[3]:
                box = None
        if raw != box:
            adjustments.append({'raw': value, 'visible_bbox': box})
        if box is not None:
            objects.append({**value, 'name': value.get('name', value['role']), 'bbox': box})
    return objects, adjustments


def prepare(args):
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    if not info['codebase_version'].startswith('v2.'):
        raise ValueError('This adapter supports LeRobot v2 image columns, not v3/video storage')
    if any(info['features'][key]['dtype'] != 'image' for key in args.image_keys):
        raise ValueError('Use image columns; video decoding is not provided by this entry')
    tasks = {r['task_index']: r['task'] for r in map(json.loads, (args.dataset / 'meta/tasks.jsonl').read_text().splitlines())}
    episodes = list(map(json.loads, (args.dataset / 'meta/episodes.jsonl').read_text().splitlines()))
    if args.dataset_revision == LIBERO_REVISION:
        if (info['total_frames'], len(episodes), len(tasks)) != (273465, 1693, 40):
            raise ValueError('The declared LIBERO snapshot does not match its metadata')
    selected = episodes if args.episodes is None else [e for e in episodes if e['episode_index'] in args.episodes]
    if not selected or args.episodes is not None and {e['episode_index'] for e in selected} != set(args.episodes):
        raise ValueError('No matching episodes or unknown episode index')
    samples = []
    for episode in selected:
        ep = episode['episode_index']
        data_path = info['data_path'].format(episode_chunk=ep // info['chunks_size'], episode_index=ep)
        columns = ['index', 'episode_index', 'frame_index', 'task_index', *args.image_keys]
        table = pq.read_table(args.dataset / data_path, columns=columns)
        if table.num_rows != episode['length']:
            raise ValueError(f'Episode length mismatch: {ep}')
        for item in table.take(list(range(0, table.num_rows, args.stride))).to_pylist():
            frame = int(item['frame_index'])
            if item['episode_index'] != ep or frame % args.stride:
                raise ValueError('Episode/frame identity mismatch')
            directory = args.output / 'frames' / f'episode_{ep:04d}' / f'{frame:05d}'
            directory.mkdir(parents=True, exist_ok=True)
            hashes = []
            for camera, key in enumerate(args.image_keys):
                image = item[key]
                if image.get('bytes') is not None:
                    data = image['bytes']
                else:
                    path = (args.dataset / image['path']).resolve()
                    if not path.is_relative_to(args.dataset):
                        raise ValueError('Image path escapes the dataset root')
                    data = path.read_bytes()
                path = directory / f'camera{camera}_original.png'
                digest = hashlib.sha256(data).hexdigest()
                if path.exists():
                    if sha256(path) != digest:
                        raise ValueError('Existing source image differs')
                else:
                    path.write_bytes(data)
                hashes.append(digest)
                image = Image.open(io.BytesIO(data)).convert('RGB')
                target = directory / f'camera{camera}.png'
                if not target.exists():
                    Image.fromarray(image_tools.resize_with_pad(np.asarray(image), 224, 224)).save(target)
            samples.append({'dataset_index': int(item['index']), 'episode': ep, 'frame': frame,
                            'task_id': int(item['task_index']), 'instruction': tasks[item['task_index']],
                            'image_sha256': hashes, 'directory': str(directory.relative_to(args.output))})
            if args.max_moments is not None and len(samples) >= args.max_moments:
                break
        if args.max_moments is not None and len(samples) >= args.max_moments:
            break
    if len({r['dataset_index'] for r in samples}) != len(samples):
        raise ValueError('Duplicate dataset index')
    if any(not 0 <= r['dataset_index'] < info['total_frames'] for r in samples):
        raise ValueError('Out-of-range dataset index')
    if args.max_moments is None and len(samples) != sum((e['length']+args.stride-1)//args.stride for e in selected):
        raise ValueError('Incomplete stride sampling')
    return info, samples


def raw_path(output, row, camera):
    return output / 'ground' / f"episode_{row['episode']:04d}" / f"{row['frame']:05d}" / f'qwen8_camera{camera}.json'


def cached_record(args, row, camera):
    path = raw_path(args.output, row, camera)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    arrays = record.get('response_format') == 'role_bbox_arrays'
    repair = bool(record['no_repeat_ngram_size'])
    expected = prompt(row, arrays=arrays, repair=repair)
    if (record['source_sha256'] != row['image_sha256'][camera] or record['model_revision'] != REVISION
            or record['model_name'] != MODEL or record['prompt'] != expected
            or record['dataset_index'] != row['dataset_index'] or record['camera'] != camera
            or sha256(path.with_suffix('.txt')) != record['raw_sha256']):
        raise ValueError('Cached source/prompt/model/raw identity mismatch')
    if record['status'] == 'ok':
        objects, _ = parse_boxes(path.with_suffix('.txt').read_text(), arrays=arrays)
        if objects != record['objects']:
            raise ValueError('Cached objects differ from raw response')
    return record


def ground(args, rows):
    pending = [(r, c) for r in rows for c in (0, 1) if (cached_record(args, r, c) or {}).get('status') != 'ok']
    if not pending:
        return
    if args.finish_existing:
        raise ValueError('--finish-existing requires a valid cached response for every image; no model was loaded')
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen8b, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.qwen8b, local_files_only=True, min_pixels=448*448, max_pixels=672*672)
    processor.tokenizer.padding_side = 'left'
    for size, repair, arrays in ((args.batch_size, False, False), (1, True, False), (1, True, True)):
        failed = []
        for offset in range(0, len(pending), size):
            chunk = pending[offset:offset+size]
            messages = []
            for row, cam in chunk:
                path = args.output / row['directory'] / f'camera{cam}_original.png'
                if sha256(path) != row['image_sha256'][cam]:
                    raise ValueError('Source image changed')
                with Image.open(path) as image:
                    image = image.convert('RGB')
                messages.append([{'role': 'user', 'content': [{'type': 'image', 'image': image},
                    {'type': 'text', 'text': prompt(row, arrays=arrays, repair=repair)}]}])
            inputs = processor.apply_chat_template(messages, tokenize=True, padding=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to(args.device)
            with torch.inference_mode():
                result = model.generate(**inputs, max_new_tokens=1500, do_sample=False,
                    no_repeat_ngram_size=32 if repair else 0)
            texts = processor.batch_decode(result[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for (row, cam), text in zip(chunk, texts, strict=True):
                path = raw_path(args.output, row, cam)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    suffix = '.retry-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    path.rename(path.with_name(path.name + suffix))
                    raw = path.with_suffix('.txt')
                    raw.rename(raw.with_name(raw.name + suffix))
                path.with_suffix('.txt').write_text(text)
                record = {'dataset_index': row['dataset_index'], 'episode': row['episode'], 'frame': row['frame'],
                    'camera': cam, 'source_sha256': row['image_sha256'][cam], 'model_name': MODEL,
                    'model_revision': REVISION, 'prompt': prompt(row, arrays=arrays, repair=repair),
                    'no_repeat_ngram_size': 32 if repair else 0, 'batch_size': size,
                    'raw_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'response_format': 'role_bbox_arrays' if arrays else 'role_bbox_objects'}
                try:
                    objects, adjustments = parse_boxes(text, arrays=arrays)
                    record.update(status='ok', objects=objects, geometry_adjustments=adjustments)
                except (ValueError, KeyError, TypeError) as error:
                    record.update(status='invalid', error=str(error))
                    failed.append((row, cam))
                write_json(path, record)
            print(json.dumps({'stage': 'ground', 'batch_images': len(chunk), 'repair': repair,
                              'arrays': arrays, 'failed_in_round': len(failed)}), flush=True)
        pending = failed
        if not pending:
            break
    del model, processor, inputs, result
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()


def positive_labels(maps):
    pixel = np.zeros((2, 224, 224), np.float32)
    for weight, role in zip((1., .6, .8, 0.), maps, strict=True):
        pixel += weight * role / max(float(role.sum()), 1e-12)
    scores = np.zeros(768, np.float32)
    scores[:512] = pixel.reshape(2, 16, 14, 16, 14).mean(axis=(2, 4)).reshape(-1)
    foreground = (maps[:3].max(axis=0) > 0).reshape(2, 16, 14, 16, 14).mean(axis=(2, 4)).reshape(-1)
    # No filler is exported: pure-arm/background patches cannot become positive targets.
    candidates = np.flatnonzero(foreground >= .25)
    chosen = np.sort(candidates[np.argsort(-scores[candidates], kind='stable')[:64]])
    positive = np.zeros(768, bool)
    positive[chosen] = True
    scores[512:] = 0
    if scores.sum():
        scores /= scores.sum()
    return {'gripper_k64_indices': chosen, 'gripper_k64_positive': positive, 'gripper_k64_scores': scores}


def segment(args, rows):
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    missing = [r for r in rows if not all((args.output / 'labels' / f"episode_{r['episode']:04d}" /
               f"{r['frame']:05d}" / name).exists() for name in ('masks.npz', 'labels.npz', 'quality.json'))]
    if not missing:
        return
    predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=str(args.sam_checkpoint)).to(args.device).eval())
    with torch.inference_mode():
        for row in missing:
            directory = args.output / 'labels' / f"episode_{row['episode']:04d}" / f"{row['frame']:05d}"
            directory.mkdir(parents=True, exist_ok=True)
            records = [cached_record(args, row, camera) for camera in (0, 1)]
            valid = all(r is not None and r['status'] == 'ok' for r in records)
            maps = np.zeros((4, 2, 224, 224), np.float32)
            if valid:
                for camera, record in enumerate(records):
                    source = args.output / row['directory'] / f'camera{camera}_original.png'
                    with Image.open(source) as image:
                        width, height = image.size
                    ratio = max(width/224, height/224)
                    w, h = int(width/ratio), int(height/ratio)
                    offset = np.array([(224-w)//2, (224-h)//2]*2, np.float32)
                    scale = np.array([w, h]*2, np.float32)/1000
                    with Image.open(args.output / row['directory'] / f'camera{camera}.png') as image:
                        predictor.set_image(np.asarray(image.convert('RGB')))
                    for obj in record['objects']:
                        box = np.asarray(obj['bbox'], np.float32)*scale+offset
                        logits, scores, _ = predictor.predict(box=box, multimask_output=False, return_logits=True)
                        if np.isfinite(scores[0]) and scores[0] >= .55:
                            probability = 1/(1+np.exp(-np.clip(logits[0], -30, 30)))
                            pixels = np.where(probability >= .5, probability*scores[0], 0)
                            x0, y0, x1, y1 = np.clip(np.round(box), 0, 224).astype(int)
                            margin = 3 if obj['role'] in ('source', 'goal') else 0
                            bounded = np.zeros_like(pixels)
                            ys = slice(max(0, y0-margin), min(224, y1+margin))
                            xs = slice(max(0, x0-margin), min(224, x1+margin))
                            bounded[ys, xs] = pixels[ys, xs]
                            target = maps[ROLES.index(obj['role']), camera]
                            np.maximum(target, bounded, out=target)
            maps[3][maps[2] > 0] = 0
            save_arrays(directory/'masks.npz', maps=maps)
            save_arrays(directory/'labels.npz', **positive_labels(maps))
            write_json(directory/'quality.json', {'dataset_index': row['dataset_index'], 'quarantined': not valid})
            print(json.dumps({'stage': 'sam', 'dataset_index': row['dataset_index'], 'quarantined': not valid}), flush=True)


def validate(args, info, rows):
    records, quarantined, empty = [], 0, 0
    for row in rows:
        directory = Path('labels') / f"episode_{row['episode']:04d}" / f"{row['frame']:05d}"
        root = args.output / directory
        quality = json.loads((root/'quality.json').read_text())
        with np.load(root/'masks.npz') as data:
            maps = data['maps']
        if maps.shape != (4, 2, 224, 224) or not np.isfinite(maps).all() or (maps < 0).any():
            raise ValueError('Invalid role masks')
        with np.load(root/'labels.npz') as data:
            positive = data['gripper_k64_positive']
            if positive.dtype != bool or positive.shape != (768,) or positive[512:].any() or positive.sum() > 64:
                raise ValueError('Invalid positive-label shape/budget/camera mask')
            if not np.array_equal(positive, positive_labels(maps)['gripper_k64_positive']):
                raise ValueError('Saved positives do not match their masks')
        if quality['dataset_index'] != row['dataset_index']:
            raise ValueError('Label identity mismatch')
        if quality['quarantined'] and (maps.any() or positive.any()):
            raise ValueError('Quarantined moments must have zero Hard supervision')
        quarantined += quality['quarantined']
        empty += not positive.any()
        records.append({'dataset_index': row['dataset_index'], 'episode': row['episode'], 'frame': row['frame'],
                        'task_id': row['task_id'], 'directory': str(directory), 'quarantined': quality['quarantined'],
                        'masks_sha256': sha256(root/'masks.npz'), 'labels_sha256': sha256(root/'labels.npz')})
    if quarantined/len(rows) > .001:
        raise ValueError(f'Quarantine ratio {quarantined}/{len(rows)} exceeds 0.1%; preserve valid results and repair only invalid responses')
    partial = args.episodes is not None or args.max_moments is not None
    manifest = {'status': 'partial' if partial else 'complete', 'dataset_revision': args.dataset_revision,
                'dataset_frames': info['total_frames'], 'stride': args.stride, 'samples': records,
                'roles': ROLES, 'model': MODEL, 'model_revision': REVISION,
                'quarantined_moments': quarantined, 'empty_positive_targets': empty,
                'variant': 'gripper_k64', 'prompt_sha256': sha256(PROMPT),
                'note': 'Positive-only targets, no filler positives; coarse pseudo-labels, not verified object accuracy'}
    write_json(args.output/'manifest.json', manifest)
    (args.output/('preview.complete' if partial else 'annotation.complete')).write_text(sha256(args.output/'manifest.json')+'\n')
    print(json.dumps({'status': manifest['status'], 'moments': len(rows), 'quarantined': quarantined,
                      'empty': empty, 'manifest': str(args.output/'manifest.json')}), flush=True)


def self_check():
    assert parse_boxes('[["gripper",1,2,20,30]]', arrays=True)[0][0]['bbox'] == [1, 2, 20, 30]
    assert parse_boxes('[{"role":"arm","bbox":[0,0,0,0]}]')[0] == []
    for text in ('[{"role":"arm","bbox":[1,2,3,4],"bbox":[2,3,4,5]}]',
                 '[{"role":"background","bbox":[1,2,3,4]}]', '[{"role":"arm","bbox":[true,2,3,4]}]'):
        try:
            parse_boxes(text)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed label accepted')
    maps = np.zeros((4, 2, 224, 224), np.float32)
    maps[0, 0, :14, :14] = 1
    maps[3, 1, :28, :28] = 1
    labels = positive_labels(maps)
    assert labels['gripper_k64_positive'].sum() == 1 and labels['gripper_k64_positive'][0]
    assert not labels['gripper_k64_positive'][256:].any()
    assert not positive_labels(np.zeros_like(maps))['gripper_k64_positive'].any()
    print(json.dumps({'status': 'passed', 'checks': 'strict parsing, arrays, zero-box unknown, positive-only K64, camera zero'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--dataset-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qwen8b', type=Path, required=True)
    parser.add_argument('--sam-checkpoint', type=Path, required=True)
    parser.add_argument('--device', required=True, help='Explicit device, e.g. cuda:0. No automatic CPU fallback.')
    parser.add_argument('--image-keys', nargs=2, default=['image', 'wrist_image'])
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--episodes', nargs='+', type=int)
    parser.add_argument('--max-moments', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--finish-existing', action='store_true', help='Require cached valid Qwen responses; never load Qwen.')
    args = parser.parse_args()
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    if args.stride <= 0 or args.batch_size <= 0 or args.max_moments is not None and args.max_moments <= 0:
        parser.error('stride, batch-size and max-moments must be positive')
    if len(set(args.image_keys)) != 2:
        parser.error('Pass two distinct image columns in base/wrist order')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'run.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signature = {'dataset': str(args.dataset), 'dataset_revision': args.dataset_revision, 'image_keys': args.image_keys,
                     'stride': args.stride, 'episodes': args.episodes, 'max_moments': args.max_moments,
                     'model_config_sha256': sha256(args.qwen8b/'config.json'), 'model_revision': REVISION,
                     'model_weight_files': {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                                            for p in sorted(args.qwen8b.glob('*.safetensors'))},
                     'prompt_sha256': sha256(PROMPT), 'sam_checkpoint': str(args.sam_checkpoint.resolve()),
                     'sam_checkpoint_stat': [args.sam_checkpoint.stat().st_size, args.sam_checkpoint.stat().st_mtime_ns],
                     'sam_quality': .55, 'positive_patch_coverage': .25, 'positive_cap': 64, 'context': 0,
                     'metadata_sha256': {name: sha256(args.dataset/'meta'/name) for name in ('info.json', 'tasks.jsonl', 'episodes.jsonl')}}
        path = args.output/'source_identity.json'
        if path.exists():
            if json.loads(path.read_text()) != signature:
                raise ValueError('Existing output belongs to another dataset/model/prompt/scope; use a new output directory')
        else:
            write_json(path, signature)
        for name in ('annotation.complete', 'preview.complete'):
            marker = args.output/name
            if marker.exists():
                manifest = json.loads((args.output/'manifest.json').read_text())
                if marker.read_text().strip() != sha256(args.output/'manifest.json'):
                    raise ValueError('Completed manifest differs from its receipt')
                print(json.dumps({'status': manifest['status'], 'reused_completed': True,
                                  'moments': len(manifest['samples']), 'no_models_loaded': True}))
                return
        info, rows = prepare(args)
        write_json(args.output/'samples.json', {'dataset_revision': args.dataset_revision, 'samples': rows})
        if args.prepare_only:
            print(json.dumps({'stage': 'prepared', 'moments': len(rows), 'no_models_loaded': True}))
            return
        if not (args.output/'annotation.complete').exists() and not (args.output/'preview.complete').exists():
            ground(args, rows)
            segment(args, rows)
        validate(args, info, rows)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
    else:
        main()
