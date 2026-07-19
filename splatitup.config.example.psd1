@{
    # Leave blank to store runs inside this repository's runs folder.
    OutputRoot = ""

    Tools = @{
        # Tools found on PATH do not need explicit paths.
        Python = ""
        Ffmpeg = ""
        Ffprobe = ""
        Colmap = ""

        # Brush is the stable default trainer.
        Brush = ""

        # Optional local build of the SuperSplat editor.
        SuperSplatDist = ""
    }

    # Optional experimental NVIDIA trainer.
    ThreeDGRUT = @{
        Repo = ""
        Python = ""
    }
}
