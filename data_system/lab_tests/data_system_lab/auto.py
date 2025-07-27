# auto.py
import os
import config
import shutil
import subprocess


def main():
    i = 0
    num_iterations = 3  # Set the number of times you want to call the function
    while i <= num_iterations:
        subprocess.run(["python", "run.py", str(i)])

        # check how many photometry batch files were saved to disk
        out_path = config.out_path + '_scene_' + str(i)
        for d,s,f in os.walk(out_path + '\photometry'):
            photometry_files = [file for file in f if 'photometry' in file]

        # if the number of photometry files is equal to the burn in condition, delete the results directory
        if len(photometry_files) <= config.burn_in + 1:
            print('Found only %d photometry files...' % len(photometry_files))
            print('Deleting directory:', out_path)
            shutil.rmtree(out_path)
        else:
            i += 1

if __name__ == "__main__":
    main()
