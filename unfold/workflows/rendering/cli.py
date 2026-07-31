"""CLI helpers for the rendering pipeline."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Replicator multiview RGBD collector v2')
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--out', '--output', dest='out', type=str, default='data/datasets/render_pair_policy')
    parser.add_argument('--task', type=str, default='UnfoldAll-Cloth-Direct-v0')
    parser.add_argument('--config', type=str, default='configs/render_replicator.yaml')
    parser.add_argument('--renderer', dest='pipeline_renderer', type=str, default=None, choices=['RayTracedLighting', 'PathTracing'])
    parser.add_argument('--pt-spp-per-frame', type=int, default=None)
    parser.add_argument('--pt-total-spp', type=int, default=None)
    parser.add_argument('--pt-max-bounces', type=int, default=None)
    parser.add_argument('--pt-denoise', action='store_true', default=None)
    parser.add_argument('--pt-no-denoise', dest='pt_denoise', action='store_false')
    parser.add_argument('--capture-rounds-per-step', type=int, default=None, help='Number of full view sweeps to capture before each env.step(). Independent from renderer SPP controls.')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--manifest', type=str, default='manifest.jsonl')
    parser.add_argument('--render-mode', default=None, choices=['performance', 'balanced', 'quality'])
    parser.add_argument('--aa', default=None, choices=['Off', 'FXAA', 'DLSS', 'TAA', 'DLAA'])
    parser.add_argument('--dlss-mode', type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument('--spp', type=int, default=None)
    parser.add_argument('--denoise', action='store_true', default=None)
    parser.add_argument('--no-denoise', dest='denoise', action='store_false')
    parser.add_argument('--no-dome-bg', action='store_true', default=False, help='Disable dome HDR background randomization')
    parser.add_argument('--no-extra-lights', action='store_true', default=False, help='Disable extra light randomization')
    parser.add_argument('--no-ground-color', action='store_true', default=False, help='Disable ground color randomization')
    parser.add_argument('--no-material-rand', action='store_true', default=False, help='Disable cloth material randomization')
    parser.add_argument('--no-cam-intrinsics', action='store_true', default=False, help='Disable camera intrinsics randomization')
    parser.add_argument('--disable-rp-between-captures', action='store_true', default=False, help='Disable render products between captures for performance')
    parser.add_argument('--no-fabric-texture', action='store_true', default=False, help='Disable Fabric texture randomization')
    parser.add_argument('--fabric-texture-prob', type=float, default=0.5, help='Probability of using external texture (default 0.5)')
    parser.add_argument('--fabric-physical-size', type=float, default=None, help='Deprecated alias of --patch-size-m.')
    parser.add_argument('--patch-size-m', type=float, default=None, help='Physical size of one external texture patch in meters (default 0.2).')
    parser.add_argument('--cloth-root', type=str, default=None, help='Root prim path of cloth environments (default /World/Cloth).')
    parser.add_argument('--max-samples', '--num_samples', dest='max_samples', type=int, default=None, help='Stop collection once this many samples are saved (for stable bounded runs).')
    parser.add_argument('--samples-per-asset', type=int, default=None, help='Target number of saved samples to collect for each asset before switching to the next asset.')
    parser.add_argument('--barycentric-weight-dtype', type=str, default='float32', choices=['float16', 'float32'], help='Output dtype for barycentric_weights.npy')
    parser.add_argument('--projection-overlap-threshold', type=float, default=0.5, help='Minimum overlap ratio against legacy projection consistency check (0~1).')
    AppLauncher.add_app_launcher_args(parser)
    return parser
