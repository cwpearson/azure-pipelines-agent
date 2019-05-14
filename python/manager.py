import docker
import signal
import sys
import time
import socket


NUM = 2

agents = {}

for agentID in range(NUM):
    agents[agentID] = None

AZP_TOKEN = sys.argv[1]
AZP_URL = "https://dev.azure.com/c3srdev"
AZP_POOL = "amd64-ubuntu1604-cuda100"
AZP_AGENT_NAME_BASE = socket.gethostname()

CHECK_WAIT_SECONDS = 60
DOCKER_CLIENT_TIMEOUT = 10

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






client = docker.from_env(timeout=DOCKER_CLIENT_TIMEOUT)

# Test for image
print("looking for {} image".format("dockeragent"))
try:
    image = client.images.get("dockeragent")
except docker.errors.ImageNotFound as e:
    print(e)
    print("please build the dockeragent image")
    sys.exit(1)
print("found image")


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

print("registering interrupt handler")
original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, signal_handler)

def launch_agent(agentID):
    AZP_AGENT_NAME = AZP_AGENT_NAME_BASE + "_{}".format(agentID)
    environment = {
        "AZP_URL": AZP_URL,
        "AZP_TOKEN": AZP_TOKEN,
        "AZP_AGENT_NAME": AZP_AGENT_NAME,
        "AZP_POOL": AZP_POOL,
        "NVIDIA_VISIBLE_DEVICES": "all",
    }
    newContainer = client.containers.run("dockeragent", 
        runtime=NVIDIA_DOCKER_RUNTIME,
        detach=True, 
        auto_remove=True, 
        name=AZP_AGENT_NAME, 
        environment=environment
    )
    agents[agentID] = newContainer.id
    print("launched {} as {}".format(newContainer.short_id, newContainer.name))

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
            launch_agent(agentID)


while True:
    
    # look for running containers
    replenish_agents()

    # wait a bit
    time.sleep(CHECK_WAIT_SECONDS)

cleanup()
