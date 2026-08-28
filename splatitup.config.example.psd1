@{
    # Leave blank to store runs inside this repository's runs folder.
    OutputRoot = ""

    Tools = @{
        # Command-line tools found on PATH do not need explicit paths. Set explicit
        # Brush and Blender paths so the validated versions are selected.
        Python = ""
        Ffmpeg = ""
        Ffprobe = ""
        Colmap = ""
        # Official COLMAP 32K FAISS vocabulary tree, required for house loop closure.
        VocabTree = ""

        # Brush is the stable default trainer.
        Brush = ""

        # Spirula Studio is called as a separate GPL executable; its source is not copied into this MIT app.
        Spirula = ""

        # Build with Blender 5.0 + KIRI 3DGS Render 4.1.5, then open the self-contained file in Blender 5.2.
        BlenderBuilder = ""
        BlenderOpen = ""

        # Optional local build of the SuperSplat editor.
        SuperSplatDist = ""
    }

    Spirula = @{
        FloaterSuppression = "mild"
        DistractionRobustness = "mild"
        # Disabled by default: v2026.8.28 can leave a truncated large checkpoint on constrained drives.
        SaveFullCheckpoint = $false
    }

    Production = @{
        # A single job is refused below this free-space floor. Batch queues also
        # reserve PerQueuedJobReserveGB for every pending video.
        MinimumFreeSpaceGB = 20
        PerQueuedJobReserveGB = 12
        PlannedQueueSize = 10
    }

    # Optional experimental NVIDIA trainer.
    ThreeDGRUT = @{
        Repo = ""
        Python = ""
        # Official CUDA 12.8 toolkit root containing bin\nvcc.exe.
        CudaHome = ""
        # x64 Visual Studio environment script and the CUDA-compatible toolset selected from it.
        VcVars = ""
        VcVarsVersion = "14.29"
        # Stable writable caches for Warp, JIT extensions, and Torch Hub/LPIPS weights.
        RuntimeCacheRoot = ""
        TorchCudaArchList = "8.6"
        # Official 3DGUT-MCMC validation uses 1M; the runner binds this exact cap into CLI and provenance.
        McmcMaxSplats = 1000000
    }
}
