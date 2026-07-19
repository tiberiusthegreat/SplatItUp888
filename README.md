# SplatItUp888

SplatItUp888 is a native Windows drag-and-drop pipeline that turns an orbit video into a verified 3D Gaussian Splat PLY.

![SplatItUp888 desktop app](docs/splatitup888.png)

## What It Does

1. Probes and decodes phone or camera footage with FFmpeg.
2. Selects sharp, well-exposed frames from evenly distributed time bins.
3. Reconstructs camera poses with COLMAP 4 view-graph calibration and global mapping.
4. Trains a full Gaussian Splat with Brush, or optionally NVIDIA 3DGRUT.
5. Verifies required 3DGS properties such as `f_dc_0`, opacity, scale, and rotation.
6. Opens the result in a local SuperSplat build or the hosted SuperSplat editor.

## Upgrades

- Preview mode: 7,000 training steps, 150 selected frames, and adaptive 1600px extraction.
- Final mode: 30,000 training steps and native-resolution frame candidates.
- Explicit phone-video autorotation control.
- Temporal frame coverage with global blur-percentile avoidance.
- Reconstruction gates: at least 80% image registration, 5,000 sparse points, and no more than 1.5px mean reprojection error.
- Per-stage resume markers that invalidate stale extraction, image selection, or training results when settings change.
- Optional `3DGUT` and `3DGUT-MCMC` trainer backends.

## Requirements

- Windows 10 or 11
- PowerShell 5.1+
- NVIDIA GPU recommended
- Python with Pillow and NumPy
- FFmpeg and FFprobe
- COLMAP 4.x with `view_graph_calibrator` and `global_mapper`
- Brush for the stable training backend
- Optional local SuperSplat build
- Optional [NVIDIA 3DGRUT](https://github.com/nv-tlabs/3dgrut)

## Setup

1. Copy `splatitup.config.example.psd1` to `splatitup.local.psd1`.
2. Add paths for tools that are not available on `PATH`.
3. Run the diagnostic:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\doctor.ps1
```

4. Launch `Launch SplatItUp888.cmd` and drop in a video.

The local configuration, tools, and reconstruction runs are intentionally ignored by Git.

## Command Line

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\pipeline\run_video_to_splat.ps1 `
  -VideoPath "C:\captures\orbit.mov" `
  -RunName "atv-orbit-01" `
  -SelectedFrames 180 `
  -TrainingSteps 30000 `
  -Trainer Brush
```

Fast preview:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\pipeline\run_video_to_splat.ps1 `
  -VideoPath "C:\captures\orbit.mov" `
  -SelectedFrames 150 `
  -TrainingSteps 7000 `
  -AdaptiveExtraction `
  -MaxLongSide 1600
```

## Trainer Notes

Brush remains the tested default. The 3DGUT backends require a separate 3DGRUT installation and configured `ThreeDGRUT.Repo` and `ThreeDGRUT.Python` paths. SplatItUp888 undistorts the solved COLMAP dataset before invoking 3DGRUT and verifies the resulting PLY with the same Gaussian checks used for Brush.

3DGRT is deliberately not exposed in the desktop app because it is slower and its reflection/refraction benefits are not the normal video-to-splat requirement. The faster rasterized 3DGUT and its MCMC strategy are the useful alternatives here.

## Output

Each run contains:

- `selected_frames_contact_sheet.jpg`
- `frame_quality.json` and CSV
- `reconstruction_report.json`
- COLMAP database, images, and sparse model
- training logs and stage markers
- `gaussian_ply_report.json`
- `final/<run-name>.ply`

The final PLY is a trained Gaussian file, not a sparse COLMAP point cloud.

## Acknowledgements

The preview and alternate-trainer ideas were informed by [QuickSplat](https://github.com/abhinow03/quicksplat). SplatItUp888 keeps its own Windows-native pipeline, global reconstruction, quality selection, validation, editor, and delivery workflow.

## License

MIT
