import docker
import signal
import sys
import time
import socket


NUM = 2

launchedIDs = []

AZP_TOKEN = sys.argv[1]
AZP_URL = "https://dev.azure.com/c3srdev"
AZP_POOL = "amd64-ubuntu1604-cuda100"
AZP_AGENT_NAME_BASE = socket.gethostname()

CHECK_WAIT_SECONDS = 60
DOCKER_CLIENT_TIMEOUT = 10

def cleanup():
    print('Cleaning up...')
    for c in client.containers.list():
        if c.id in launchedIDs:
            print("removing {}".format(c.short_id))
            c.remove(force=True)

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

# Test for images
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


while True:
    numRunning = 0
    for c in client.containers.list():
        if c.attrs['Config']['Image'] == 'dockeragent':
            numRunning += 1
            print("found {} ({}/{})".format(c.short_id, numRunning, NUM))

    while numRunning < NUM:
        AZP_AGENT_NAME = AZP_AGENT_NAME_BASE + "_{}".format(numRunning)
        environment = {
            "AZP_URL": AZP_URL,
            "AZP_TOKEN": AZP_TOKEN,
            "AZP_AGENT_NAME": AZP_AGENT_NAME,
            "AZP_POOL": AZP_POOL,
        }
        newContainer = client.containers.run("dockeragent", 
            runtime=NVIDIA_DOCKER_RUNTIME,
            detach=True, 
            auto_remove=True, 
            name=AZP_AGENT_NAME, 
            environment=environment
        )
        print("launched {} as {}".format(newContainer.short_id, newContainer.name))
        numRunning += 1
        launchedIDs += [newContainer.id]

    time.sleep(CHECK_WAIT_SECONDS)

cleanup()