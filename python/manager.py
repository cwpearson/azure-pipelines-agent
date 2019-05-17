import docker
import signal
import sys
import time
import socket
import argparse
import platform
import subprocess
import re
import pathlib




AZP_TOKEN = None
AZP_URL = "https://dev.azure.com/c3srdev"
AZP_POOL = "amd64-ubuntu1604-cuda100"
AZP_AGENT_NAME_BASE = socket.gethostname()
AGENT_NAME = "cwpearson/azp-cuda-agent:amd64-ubuntu1604-cuda100"
OOM_SCORE_ADJ = 1000 # make containers get killed first in a low-memory situation

CHECK_WAIT_SECONDS = 60
DOCKER_CLIENT_TIMEOUT = 10


parser = argparse.ArgumentParser()
parser.add_argument("PAT", help="Azure Pipelines personal access token")
parser.add_argument("URL", help="Azure Pipelines url (https://dev.azure.com/<project>)")
parser.add_argument("POOL", type=str, help="Azure Pipelines pool")
parser.add_argument("-n", help="number of agents (default = 2)", default=2)
parser.add_argument("--name", help="agent name (default = {})".format(socket.gethostname()), default=socket.gethostname())
parser.add_argument('-d', "--docker", help="docker image for agent")
parser.add_argument("--timeout", type=int, help="docker client timeout (s)")
parser.add_argument("--poll-time", type=int, help="time between checking agent status", default=CHECK_WAIT_SECONDS)
parser.add_argument("--volume", type=str, nargs="+", help="volume in host:container:var format")

args = parser.parse_args()


volumeSpecs = []
# try to parse volumes
for volumeStr in args.volume:
    fields = volumeStr.split(":")
    if len(fields) != 3:
        print(f"expected host:container:var format for argument {volumeStr}")
        sys.exit(-1)

    # get the absolute host path
    hostPath = pathlib.Path(fields[0])
    hostPath = hostPath.resolve()

    if not hostPath.is_dir():
        print(f"{hostPath} is not a directory on the host")
        sys.exit(-1)

    containerPath = fields[1]
    envVar = fields[2]
    volumeSpecs += [(str(hostPath), containerPath, envVar)]

AZP_TOKEN = args.PAT
AZP_URL = args.URL
AZP_POOL = args.POOL
AZP_AGENT_NAME_BASE = args.name
NUM = args.n

def get_arch():
    machine = platform.machine()
    if not machine:
        return None
    elif machine == "x86_64":
        return "amd64"
    else:
        return machine
    
def get_cuda_version():

    # try nvidia-smi
    # nvidia-smi in 9.2 doesn't report cuda version possibly?
    raw = subprocess.check_output('nvidia-smi')
    print(raw)
    if type(raw) == bytes:
        raw = str(raw)
    matches = re.findall(r"CUDA Version: (\d+).(\d+)", raw)
    print(matches)
    if matches:
        return matches[0]

    # try nvcc
    raw = subprocess.check_output(['nvcc', "--version"]) 

    if type(raw) == bytes:
        raw = str(raw)
    matches = re.findall(r"V(\d+)\.(\d+)\.(\d*)", raw)
    if matches:
        return matches[0]
    return []


client = docker.from_env(timeout=DOCKER_CLIENT_TIMEOUT) 

# Get host CUDA version


# Test for nvidia-docker
NVIDIA_DOCKER_TEST_IMAGE = "nvidia/cuda:9.0-base"
NVIDIA_DOCKER_TEST_COMMAND = "nvidia-smi"
NVIDIA_DOCKER_RUNTIME = "nvidia"
print("testing for nvidia-docker")
try:
    client.containers.run(NVIDIA_DOCKER_TEST_IMAGE, runtime=NVIDIA_DOCKER_RUNTIME, command=NVIDIA_DOCKER_TEST_COMMAND, auto_remove=True)
# except docker.errors.ImageNotFound as e:
except docker.errors.APIError as e:
    print(e)
    print("Please make sure nvidia-docker is working right")
    print("docker run --runtime={} --rm {} {}".format(NVIDIA_DOCKER_RUNTIME, NVIDIA_DOCKER_TEST_IMAGE, NVIDIA_DOCKER_TEST_COMMAND))
    sys.exit(-1)

print("nvidia-docker looks good")


if args.docker:
    DOCKER_IMAGE = args.docker
else:
    DOCKER_IMAGE  = "cwpearson/azp-cuda-agent:"

    machine = get_arch()
    if not machine:
        print("unable to detect machine")
        sys.exit(1)
    # [9, 2, 88]
    # [10, 1]
    versionStr = "".join(get_cuda_version()[0:2])
    if not versionStr:
        print("unable to detect installed cuda")
        sys.exit(1)

    tag = "{}-ubuntu1604-cuda{}".format(machine, versionStr)

    DOCKER_IMAGE += tag

    print("autodetected docker image {}".format(DOCKER_IMAGE))





# try to pull the requested image
print("pulling {}".format(DOCKER_IMAGE))
client.images.pull(DOCKER_IMAGE)

agents = {}

for agentID in range(NUM):
    agents[agentID] = None


def cleanup():
    print('Cleaning up...')
    scan_agents()

    for agentID, containerID in agents.items():
        if containerID is not None:
            try:
                c = client.containers.get(containerID)
                print("removing {} ({})".format(c.name, c.short_id))
                c.remove(force=True)
            except docker.errors.NotFound as e:
                print("couldn't find {}".format(containerID))
        


def signal_handler(sig, frame):
    # restore the original signal handler as otherwise evil things will happen
    # in raw_input when CTRL+C is pressed, and our signal handler is not re-entrant
    signal.signal(signal.SIGINT, original_sigint)

    try:
        cleanup()
        sys.exit(0)

    except KeyboardInterrupt:
        print("Ok ok, quitting")
        sys.exit(1)

    # restore the exit gracefully handler here    
    signal.signal(signal.SIGINT, signal_handler)



# Test for image
print("looking for {} image".format(DOCKER_IMAGE))
try:
    image = client.images.get(DOCKER_IMAGE)
except docker.errors.ImageNotFound as e:
    print(e)
    print("the requested image was not found")
    sys.exit(1)
print("found image")




print("registering interrupt handler")
original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, signal_handler)

def launch_agent(agentID, volumeSpecs):

    # build environment
    AZP_AGENT_NAME = AZP_AGENT_NAME_BASE + "_{}".format(agentID)
    environment = {
        "AZP_URL": AZP_URL,
        "AZP_TOKEN": AZP_TOKEN,
        "AZP_AGENT_NAME": AZP_AGENT_NAME,
        "AZP_POOL": AZP_POOL,
    }

    volumes = {}
    for vs in volumeSpecs:
        hostPath, containerPath, var = vs
        if var not in environment:
            environment[var] = containerPath
        else:
            print(f"var requested by volumeSpec {vs} is already in environment {environment}")
            sys.exit(1)
        volumes[hostPath] = {
            "bind": containerPath,
            "mode": "ro",
        }

    newContainer = client.containers.run(DOCKER_IMAGE, 
        runtime=NVIDIA_DOCKER_RUNTIME,
        detach=True, 
        auto_remove=True, 
        name=AZP_AGENT_NAME, 
        environment=environment,
        oom_score_adj=OOM_SCORE_ADJ,
        volumes=volumes,
    )
    agents[agentID] = newContainer.id
    print(f"launched {newContainer.short_id} as {newContainer.name}")

    return newContainer

def scan_agents():
    # check whether our agents are running
    for agentID, containerID in agents.items():
        if containerID is not None:
            print("looking for container {}".format(containerID))
            try:
                c = client.containers.get(containerID)
                print(c.short_id, "is found. status {}".format(c.status))
            except docker.errors.NotFound:
                print(containerID, "not found")
                agents[agentID] = None

def replenish_agents():
    # figure out who is running
    scan_agents()

    # relaunch all not-running agents
    for agentID, containerID in agents.items():
        if containerID is None:
            try:
                launch_agent(agentID)
            except docker.errors.APIError as e:
                print(e)
                print("is the manager already running on this system?")


while True:
    
    # look for running containers
    replenish_agents()

    # wait a bit
    time.sleep(CHECK_WAIT_SECONDS)

cleanup()
