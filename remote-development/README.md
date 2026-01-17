# Remote Development for AI

## Overview
AI development often requires powerful hardware (high-end GPUs) that may not be available on a local laptop. Remote development allows you to write code on your local machine while executing it on a powerful remote server or cloud instance.

## What is Remote Development on Cloud?
Remote Development on Cloud involves using cloud-based virtual machines (VMs) or containers to run your development environment. Instead of relying on your personal computer's CPU and GPU, you connect to a server hosted by providers like **AWS**, **Azure**, **Google Cloud**, or specialized GPU clouds (e.g., **InferenceAI**).

**Key Benefits:**
- **Scalability**: Instantly access massive computational power and specialized hardware (like NVIDIA A100/H100 GPUs).
- **Cost-Effectiveness**: Pay only for the compute time you use (pay-as-you-go) rather than buying expensive hardware.
- **Persistence**: Your environment can be saved as a disk image (snapshot) and paused when not in use.

## Key Components

### 1. Linux (or WSL for Windows)
**Why?**
- **Standard Environment**: The vast majority of AI/ML libraries, frameworks ([PyTorch](https://pytorch.org/get-started/locally/), [TensorFlow](https://www.tensorflow.org/install)), and deployment servers are built for and tested primarily on Linux.
- **Terminal Efficiency**: Command-line tools and scripting are integral to managing data pipelines and servers.
- **WSL (Windows Subsystem for Linux)**: Allows Windows users to run a full Linux kernel and user space, providing the best of both worlds—Windows for productivity apps and Linux for development without dual-booting.
  - *Reference*: [How to Install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

**Health Check & Resource Monitoring:**
- **Check OS Version**: Run `uname -a` or `cat /etc/os-release` to verify your Linux distribution.
  - *Example Output*: `Linux my-server 5.4.0-1045-aws #47-Ubuntu SMP ... x86_64 GNU/Linux`
  - *WSL Users*: Run `wsl --list --verbose` in PowerShell to check if your distro is running (Version 2 is recommended).
- **Check CPU/RAM**: Run `htop` (or `top`) in the terminal to see active processes and memory usage.
  - *Reference*: [htop - an interactive process viewer](https://htop.dev/)
  - *Instruction*: Look at the bars at the top for CPU cores and Memory usage. Press `F10` to quit.
- **Check Disk Space**: Run `df -h` to ensure you have enough space for datasets.
  - *Example Output*: `Filesystem Size Used Avail Use% Mounted on` ... `/dev/root 100G 45G 55G 45% /`
  - *Reference*: [Linux df command help](https://man7.org/linux/man-pages/man1/df.1.html)
- **Check GPU**: Run `nvidia-smi` to verify drivers are loaded and to see VRAM usage.
  - *Example Output*: Shows a table with "GPU Name", "Persistence-M", "Fan", "Temp", "Perf", "Pwr:Usage/Cap", "Memory-Usage".
  - *Reference*: [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### 2. IDE (VS Code)
**Why?**
- **Remote Development**: Extensions like "Remote - SSH" allow you to code on your local laptop while the code resides and runs on a powerful remote server, feeling exactly like a local environment.
  - *Tutorial*: [Remote Development using SSH in VS Code](https://code.visualstudio.com/docs/remote/ssh)
- **Productivity**: Features like IntelliSense (code completion), linting, and integrated debugging save hours of troubleshooting.
- **Ecosystem**: A massive library of extensions for Python, Git, Docker, and AI tools helps streamline workflows.

**Recommended Extensions:**
- **[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)**: Essential for Python development, including linting, debugging, and environment selection.
- **[Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)**: Provides fast, feature-rich language support for Python.
- **[Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)**: Run Jupyter Notebooks directly within VS Code.
- **[GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)**: Visualizes code authorship and history line-by-line.
- **[Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)**: Easily build, manage, and deploy containerized applications.

**Health Check & Resource Monitoring:**
- **Verify Connection**: Look for the green remote indicator in the bottom-left corner of VS Code (e.g., `SSH: <hostname>`).
  - *Troubleshooting*: Use `Ctrl+Shift+P` (Cmd+Shift+P on Mac) and run `Remote-SSH: Show Log` to debug connection issues.
- **Check Extension Status**: Open the "Remote Explorer" sidebar to verify the connection status.
  - *Command Line Check*: Run `code --list-extensions --show-versions` in the integrated terminal to confirm remote extensions are installed.
- **Monitor Editor Resources**: Go to **Help > Open Process Explorer** to see how much memory VS Code extensions are consuming on both local and remote machines.
  - *Reference*: [VS Code Process Explorer](https://github.com/microsoft/vscode/wiki/Performance-Issues#the-process-explorer)

### 3. Jupyter Notebooks
**Why?**
- **Interactivity**: Perfect for data exploration and experimentation. You can run code in small blocks (cells) and see output immediately.
  - *Reference*: [Project Jupyter Documentation](https://docs.jupyter.org/en/latest/)
- **Visualization**: Plots, charts, and images are rendered directly inline, making it easier to analyze datasets and model performance.
- **Documentation**: Markdown cells allow you to mix rich text, math equations, and code.

**What are Kernels?**
A **Kernel** is the computational engine that executes the code contained in a Notebook document. Each notebook is associated with a specific kernel (e.g., Python 3.8, Python 3.10 with PyTorch).
- **Managing Kernels**: You can switch kernels to change environments (e.g., from a CPU-only environment to a GPU-enabled one) without restarting the notebook server.

**Health Check & Resource Monitoring:**
- **Check Running Servers**: Run `jupyter notebook list` in the terminal to see active notebook servers and their tokens.
  - *Example Output*: `http://localhost:8888/?token=... :: /home/user/projects`
- **Check Kernel Status**: Ensure the kernel indicator (top right) is "Idle" (filled circle) or "Busy" (solid circle), not "Disconnected".
- **Test Execution**: Run a simple cell: `print("Hello World")`.
  - *Expected Output*: `Hello World` appears immediately below the cell.
- **Check GPU from Notebook**: Execute `!nvidia-smi` in a code cell to check GPU availability directly from the notebook.
- **Check RAM**: Use magic commands like `%load_ext memory_profiler` (requires installation via `pip install memory_profiler`) to profile memory usage of specific cells.
  - *Reference*: [IPython Magic Commands](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

### 4. Containers (Docker)
**Why?**
- **Reproducibility**: Containers package your code with all its dependencies. If it runs in your container, it will run anywhere.
  - *Guide*: [Get Started with Docker](https://www.docker.com/get-started/)
- **Isolation**: Avoid "dependency hell" by keeping projects separate. One project can use `PyTorch 1.10` and another `PyTorch 2.0` without conflict.
- **Portability**: Makes moving from development (local/remote) to production (cloud/cluster) seamless.

**Example Dockerfile:**
Here is a simple `Dockerfile` for an AI project using PyTorch:
```dockerfile
# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Command to run when starting the container
CMD ["python", "main.py"]
```
  - *Reference*: [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)

**Health Check & Resource Monitoring:**
- **Verify Docker Daemon**: Run `docker info` to ensure the Docker daemon is running and reachable.
  - *Test Run*: Run `docker run --rm hello-world` to confirm containers can start.
  - *Expected Output*: "Hello from Docker! This message shows that your installation appears to be working correctly."
- **List Running Containers**: Run `docker ps` to see currently active containers and their port mappings.
  - *Example Output*: `CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES`
  - *Reference*: [Docker CLI Reference](https://docs.docker.com/reference/cli/docker/)
- **Monitor Container Resources**: Run `docker stats` to see a live stream of CPU, Memory, and Network usage for each running container.
  - *Instruction*: Press `Ctrl+C` to exit the live stream.
