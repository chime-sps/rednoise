This repo contains code to analyze rednoise across the CHAMPSS system. The actual rednoise files are too large to stay on GitHub and so will not be uploaded.


# `./scripts/combine_rednoise_medians.py`
This program searches a directory structure like the one we use for CHAMPSS storage and combines individual medians.npz files into one VERY large `hdf5` file that can then be referenced more easily for analysis. This will definitely need fiddling with for your specific directory structure.

Example test run:
`python3 combine_rednoise_medians.py /mnt/beegfs-client/processed/ -o test_run.h5 --n-files 10`

Test that the resume feature works:
`python3 combine_rednoise_medians.py /mnt/beegfs-client/processed/ -o test_run.h5 --n-files`

If this works, go for the full thing...
`python3 combine_rednoise_medians.py /mnt/beegfs-client/processed/ -o omg_its_the_real_run.h5`
...and you can resume it where it left off if it dropped, aside from the directory searching, which does have to run every time.
