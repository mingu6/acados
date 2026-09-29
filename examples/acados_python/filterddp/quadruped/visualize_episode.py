#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

"""
Render closed-loop episodes of mujoco_closed_loop.py to a video, one panel per controller, all on the
same episode (same seed, so the same disturbance sequence). Overlays: ground reaction forces
planned by the MPC for the first stage (green arrows at the feet), the external force (red) and
torque (purple, along its axis) on the base, both drawn from a point above the base, the MPC's
swing-foot landing points (yellow dots), and a label with the controller, time and disturbance. The
camera tracks the base and turns with its heading. A panel freezes when its robot falls.

Uses MuJoCo's offscreen renderer: set MUJOCO_GL=egl on a machine without a display (glfw, the
default, needs one).

    visualize_episode.py --controllers STOCK FILTERDDP_GN:15 SQP_GN:3 --wrench 7.5 --episode 21 --out output/videos/w7.5_ep21.mp4
"""

import argparse
import os

import numpy as np

from closed_loop_core import HERE, setup, run_episode

GRF_SCALE = 0.004       # m per N (0.6 m for the body weight); smaller arrows disappear inside the shins
PUSH_SCALE = 0.05       # m per N, force on the base (red)
TORQUE_SCALE = 0.05     # m per N m, torque on the base (purple), along its axis
WRENCH_ORIGIN = 0.3     # m above the base centre where both disturbance arrows start


def add_arrow(scene, p_from, p_to, rgba, width):
    import mujoco
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.asarray(p_to) - np.asarray(p_from)) < 1e-4:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(p_from, dtype=float), np.asarray(p_to, dtype=float))
    scene.ngeom += 1


def add_sphere(scene, pos, radius, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def label(frame, lines, fell):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    y = 8
    for text in lines:
        draw.text((10, y), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += 22
    if fell:
        draw.text((10, y + 6), 'FELL', fill=(255, 60, 60), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def render_controller(spec, args):
    import mujoco
    # same lock as mujoco_closed_loop.py, so rendering can run beside a grid
    ctx = setup(spec, args, env_lock_path=os.path.join(HERE, '.quadruped_env.lock'))
    env = ctx.env
    renderer = mujoco.Renderer(env.mjModel, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = ctx.base_body
    cam.distance, cam.elevation, cam.azimuth = args.cam_distance, args.cam_elevation, args.cam_azimuth
    every = max(1, int(round(1.0/(args.fps*ctx.sim_dt))))
    frames = []

    def on_step(ctx, step, t, wrench, terminated):
        if (step + 1) % every != 0 and not terminated:
            return
        # follow the heading: the controller tracks yaw rate, not yaw, so pushes turn the robot
        cam.azimuth = args.cam_azimuth + np.degrees(float(env.base_ori_euler_xyz[2]))
        renderer.update_scene(env.mjData, camera=cam)
        scene = renderer.scene
        # shadows and reflections streak with some GL drivers; upstream's simulator turns them off too
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        feet = env.feet_pos(frame='world')
        grfs = ctx.wrapper.nmpc_GRFs
        footholds = ctx.wrapper.nmpc_footholds
        for leg in ctx.legs_order:
            f = np.asarray(grfs[leg]).flatten()
            if np.linalg.norm(f) > 1.0:
                p = np.asarray(feet[leg]).flatten()
                add_arrow(scene, p, p + GRF_SCALE*f, (0.1, 0.9, 0.2, 0.9), 0.012)
            fh = np.asarray(footholds[leg]).flatten()
            if fh.size == 3 and np.linalg.norm(fh) > 0:
                add_sphere(scene, fh, 0.015, (1.0, 0.85, 0.1, 0.8))
        base = np.asarray(env.base_pos).flatten()
        f, tq = wrench[:3], wrench[3:]
        origin = base + np.array([0.0, 0.0, WRENCH_ORIGIN])
        if np.linalg.norm(f) > 0.5:
            add_arrow(scene, origin, origin + PUSH_SCALE*f, (0.95, 0.15, 0.1, 0.95), 0.03)
        if np.linalg.norm(tq) > 0.5:
            add_arrow(scene, origin, origin + TORQUE_SCALE*tq, (0.6, 0.2, 0.9, 0.95), 0.025)
        if np.linalg.norm(wrench) > 0.5:
            add_sphere(scene, origin, 0.02, (0.2, 0.2, 0.2, 0.9))
        img = renderer.render()
        lines = [spec, f't = {t:5.2f} s',
                 f'push F = ({f[0]:5.1f}, {f[1]:5.1f}, {f[2]:5.1f}) N',
                 f'torque = ({tq[0]:5.1f}, {tq[1]:5.1f}, {tq[2]:5.1f}) N m']
        frames.append(label(img, lines, terminated))

    result, failed = run_episode(ctx, args.episode, args, on_step=on_step)
    renderer.close()
    env.close()
    return frames, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controllers', nargs='+', default=['STOCK', 'FILTERDDP_GN:15'])
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--friction', type=float, default=0.8)
    parser.add_argument('--scene', default='flat')
    parser.add_argument('--gait', default='trot')
    parser.add_argument('--wrench', type=float, default=5.0)
    parser.add_argument('--wrench_period', type=float, default=2.0)
    parser.add_argument('--wrench_start', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--fps', type=float, default=25.0, help='frames per simulated second')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--cam_distance', type=float, default=1.4)
    parser.add_argument('--cam_elevation', type=float, default=-18.0)
    parser.add_argument('--cam_azimuth', type=float, default=135.0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    # one fresh process per controller, as in mujoco_closed_loop.py, rendered in parallel
    import multiprocessing as mp
    with mp.get_context('fork').Pool(len(args.controllers), maxtasksperchild=1) as pool:
        out = pool.starmap(render_controller, [(spec, args) for spec in args.controllers])
    panels = [frames for frames, _ in out]
    for spec, (_, result) in zip(args.controllers, out):
        print(f"{spec}: success {result['success']} t_end {result['t_end']:.3f} cost {result['cost']:.5g} "
              f"iters_mean {result.get('iters_mean', 0):.2f}", flush=True)
    n = max(len(p) for p in panels)
    panels = [p + [p[-1]]*(n - len(p)) for p in panels]
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with imageio.get_writer(args.out, fps=args.fps, codec='libx264', quality=8, macro_block_size=8) as w:
        for i in range(n):
            w.append_data(np.hstack([p[i] for p in panels]))
    print(f'wrote {args.out}: {n} frames, {len(panels)} panels')


if __name__ == '__main__':
    main()
