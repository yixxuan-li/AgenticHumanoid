# Gaussian rendering in Isaac Sim — figure revision 6

The office now renders as its original two-million-Gaussian reconstruction in
**Isaac Sim 5.1 RTX / NuRec**. It is not the extracted textured mesh, a browser
render, a screen-space background composite, or newly generated artwork.
The teaser, six skill close-ups, and two onboard views were rendered natively
with the robot geometry in the same Isaac stage.

## Outputs

- `outputs/figures_v6/teaser_v6.{png,pdf,svg}`
- `outputs/figures_v6/system_v6.{png,pdf,svg}`
- `outputs/figures_v6/assets/`: native RGB renders.
- `outputs/figures_v6/scene/room_nurec.usda` and `room_nurec.nurec`: converted
  Gaussian volume and payload.
- `outputs/figures_v6/scene/*_foreground.usdc`: static URDF-derived robot/prop
  snapshots for each authored pose.
- `outputs/figures_v6/scene/*_isaac.usda`: composed native Isaac stages.
- `outputs/figures_v6/*_render.json`: camera and renderer records per view.
- `outputs/figures_v6/provenance.json` and `validation.json`: provenance and checks.
- `deliverables/figures_v6.zip`: rendered figures, native assets, code, and notes.

The teaser retains the requested HITCH-inspired layout and short English labels.
The visible note says **Isaac Sim · 3DGS · authored poses**. Previous figure
versions and their mesh rendering paths remain available.

## Source and conversion

The source is `outputs/worldlabs/splats/refinedSplats_2000000.spz`, SHA-256:

```
578508d8e7236c4e81fa44602041f9968ec9879e752c20da619b3af6073f153d
```

This is the existing Atlas reconstruction, including its previously documented
inferred surfaces. No reconstruction job, Gaussian training, or GPT call was
submitted for this revision.

The SPZ header specifies version 2, 2,000,000 Gaussians, and spherical harmonics
degree zero. Conversion preserves the Gaussian positions, anisotropic scales,
rotations, opacity, and DC color. The adapter rejects higher SH degrees rather
than silently discarding coefficients. Quantized opacity endpoints are clamped
by `1e-6` for finite logits; NuRec tensors use float16. The source SPZ is unchanged.

The converter follows Niantic's published SPZ v2 layout and loads the two pure
NuRec template functions from NVIDIA's
[3DGRUT](https://github.com/nv-tlabs/3dgrut), commit
`dfd3d54c8c612003fb7948443edf7778f2bcd5ec`. It authors the matching NuRec Volume
and field references. It does not run the 3DGRUT trainer or substitute a mesh.
One recorded native-to-simulator transform from the existing room manifest is
applied to positions, rotations, and scales.

The robot is the same official Unitree G1 URDF used in version 5, pinned to
`unitree_ros` commit `157ff950351e5f535e03c5d85a16b7d594b3dbcc`.
The figure adapter exports its visual triangles after forward kinematics into
USD snapshots. Isaac Sim renders those snapshots beside the Gaussian volume.
These snapshots are not imported PhysX articulations; the simulation controller
and research results have not changed.

## Occlusion and limitations

The selected views use one approximate tabletop matte proxy at the previously
assumed 0.74 m table height. It supplies a native occlusion boundary; it does not
supply the room's color or texture. This is an authored approximation, not
recovered dense depth or validated collision geometry.

A full extracted-mesh matte proxy was tested and rejected because geometry
mismatch caused visible artifacts. The selected final views use the tabletop
proxy instead. The initial mesh-figure camera was also rejected because it lay
outside a Gaussian wall. The selected hero camera is inside the room at
`[5.0, 2.3, 2.0]`, looking at `[3.5, 3.9, 0.7]`.

The four temporal robot exposures are opaque to avoid legacy NuRec/matte
transparency artifacts. Poses, orange cube, drawer articulation, task target,
route, and torso camera mounting remain authored illustrations. These are not
successful grasps, acquired skills, physics rollouts, or policy execution.
Lighting, contact shadows, and occlusion are not a calibrated real-world model.

## Reproduce

Conversion/layout use the figure Python environment. Native rendering uses the
separate existing Isaac Sim environment through `scripts/run_isaac.sh`.
No API credentials are required.

```bash
# Optional: rebuild the NuRec format from the existing SPZ.
uv pip install --python .venv-skill-lab/bin/python -r requirements-gaussian-figures.txt
.venv-skill-lab/bin/python scripts/convert_atlas_spz_nurec.py

# Optional: re-export authored poses from version 5's saved models/states.
# This step requires the version 5 project assets/bundle.
.venv-skill-lab/bin/python scripts/export_figure_foregrounds_usd.py

# The delivered native foreground USD files can be rendered directly.
scripts/run_isaac.sh scripts/render_gaussian_isaac.py \
  --view hero --name hero_sequence --proxy \
  --foreground outputs/figures_v6/scene/hero_foreground.usdc \
  --width 2200 --height 1350
.venv-skill-lab/bin/python scripts/render_gaussian_figure_views.py
.venv-skill-lab/bin/python scripts/draw_research_figures_v6.py
.venv-skill-lab/bin/python scripts/validate_research_figures_v6.py
```

To reconstruct the converter's vendor dependency if absent:

```bash
git clone https://github.com/nv-tlabs/3dgrut.git vendor/3dgrut_native
git -C vendor/3dgrut_native checkout dfd3d54c8c612003fb7948443edf7778f2bcd5ec
```

The delivered scene/payload can also be opened directly in Isaac Sim. Use
`hero_sequence_isaac.usda` to view the composed scene. The source RGB captured
frame required by the system diagram is retained with the package.

NVIDIA documents native NuRec volumes in
[Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_nurec.html)
and the behavior of
[NuRec proxy compositing](https://docs-prod.omniverse.nvidia.com/materials-and-rendering/latest/neural-rendering.html).

## Figure caption

AgenticHumanoid in a Gaussian reconstruction of the captured office. The room's
appearance is rendered natively in Isaac Sim RTX / NuRec from the existing Atlas
SPZ, with official G1 URDF-derived robot geometry and an approximate tabletop
occlusion proxy. Four authored exposures and supporting skill views illustrate
the intended repertoire; the task props, poses, and camera mounting are
illustrative and do not establish learned behavior or controller success.
