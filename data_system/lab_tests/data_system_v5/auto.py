# auto.py
import os
import shutil
import subprocess
import time

def load_config(path="config.txt"):
    config = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and full-line comments

            # Remove inline comment
            line = line.split("#", 1)[0].strip()

            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip().strip('"').strip("'")  # remove quotes
    return config

config = load_config()

def main():
    i = 0
    num_iterations = 100  # Set the number of times you want to call the function
    while i <= num_iterations:

        # call the function
        print('RUN:', N)
        print('SCENE:', i)
        subprocess.run(["python", "run.py", str(N), str(i)])

        # check how many photometry batch files were saved to disk
        out_path = os.path.join(config['out_path'] + 'RUN_' + str(N), 'SCENE_' + str(i))
        for d,s,f in os.walk(out_path + '/photometry'):
            photometry_files = [file for file in f if 'photometry' in file]

        # if the number of photometry files is equal to the burn in condition, delete the results directory
        if len(photometry_files) <= int(config['burn_in']) + 1:
            print('Found only %d photometry files...' % len(photometry_files))
            print('Deleting directory:', out_path)
            shutil.rmtree(out_path)
        else:
            i += 1

        # hang for X minutes before relaunching (associated with swivelling to a new target, as want stable reference)
        time.sleep(int(config['hang_time']))


if __name__ == "__main__":

    # check run number
    print('Results path:', config['out_path'])
    print('Path exists:', os.path.exists(config['out_path']))
    result_dirs = []
    for d,s,f in os.walk(config['out_path']):
        if 'RUN' in d:
            result_dirs.append(d)

    # if no priors runs exist, initialise a new one
    if len(result_dirs) == 0:
        N = 0
    else:
        # find most recent run number, and initialise a new one
        N = max([int(d.split('RUN_')[-1].split('\\')[0]) for d in result_dirs]) + 1
    main()
