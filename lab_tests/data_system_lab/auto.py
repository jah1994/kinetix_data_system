# auto.py
import subprocess


def main():
    num_iterations = 3  # Set the number of times you want to call the function
    for i in range(num_iterations):
        subprocess.run(["python", "run.py", str(i)])

if __name__ == "__main__":
    main()
