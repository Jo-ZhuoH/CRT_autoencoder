# Autoencoder for the polarmaps and FFT polarmaps.
## Data
### Image data
1. Pre-CRT data (64x64 array, perfusion, systolicPhase, wallthk)
   - GUIDE n=137
   - GUIDE II n=46
   - Taiwan n=29
   - VISION n=187
2. Post-CRT data (64x64 array, perfusion, systolicPhase, wallthk)
   - GUIDE n=52
   - Taiwan n=28
   - VISION n=155
### Tabular data


## Preprocessing
1. Generate FFT images
    - Run `core/fft/exe_fft.sh` to generate FFT images from 3 centers.
    - Save all FFT images to `data/FFT/pre` and `data/FFT/post`
2. Get the systolicPhase image of MPI & FFT data
    - Run `/autoencoder/exe_preprocess.sh` to split data into training and test sets.
       - Save all systolicPhase MPI images to `data/saved_polarmaps_gray/train`
       - Save all systolicPhase FFT images to `data/fft/train`

## FFT & MPI Autoencoder
Run `AE_train.py` to train the AE model and statistical analysis.

    