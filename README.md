# azure-pipelines-agent

Self-hosted GPU agents for Azure Pipelines

https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops

The following steps are marked with (HOST) or (WEBSITE), suggesting whether the step should be carried out on the machine intened to be the self-hosted agent, or on the Azure Pipelines website.

## (WEBSITE) set up a Personal Access Token

Go to https://dev.azure.com/c3srdev

profile >> security >> personal access tokens

new token

more scopes: agent pools read/manage

token is saved in bitwarden under azure-pipelines-agent token

## (WEBSITE) Create an agent pool

You must tell Azure Pipelines what pool your agents work in.
You can assign each job to a unique pool, so in C3SR we organize pools according to the capabilities of the underlying host.
For example, an amd64 host with ubuntu 1604 and cuda 10.0 would be `amd64-ubuntu1604-cuda100`.

To create a new agent pool do

https://dev.azure.com/c3srdev/_settings/agentpools >> new agent pool

You will use the name of the agent pool in your `azure-pipelines.yml`.

## (HOST) Install CUDA, docker, nvidia-docker, and python3

The Azure pipelines agent runs inside Docker, to create a fresh environment for each job.
The manager is written in python.

* [Install CUDA](https://developer.nvidia.com/cuda-downloads)
* [Install Docker](https://docs.docker.com/)
* [Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* Install python3. How you do this probably depends on your host system.


## (HOST) Run the agents using Docker

The docker agent is configured to accept a single job and then exit.
This ensures that each job will have a fresh environment.
`python/manager.py` is responsible for making sure new agents are created whenever the number of active agents falls below a threshold.
The manager will run forever.
When it is interrupted, it will try to clean up any dangling containers that it created.

1. Start the manager `python3 python/manager.py`

```
python3 python/manager.py <PAT> <URL> <POOL>
```

The manager needs to be passed the Personal Access Token you created earlier, as well as the Azure Pipelines project URL, and the name of the pool the agent should be registered to.

for example

```
python3 python/manager.py [long string of letters and numbers] https://dev.azure.com/c3srdev amd64-ubuntu1604-cuda100
```



The manager will query the host system and try to determine the agent to run with the most CUDA support.
These agent Docker images are hosted on the Docker Hub, and defined in the [dockeragent](dockeragent) directory of this repository.
* `cwpearson/azp-cuda-agent:amd64-ubuntu1604-cuda92`
* `cwpearson/azp-cuda-agent:amd64-ubuntu1604-cuda100`
* `cwpearson/azp-cuda-agent:amd64-ubuntu1604-cuda101`


If the manager fails to understand your system, or your system is not supported by one of those images, you can supply your own docker image with the `-d` flag.

If you want to make host directories available to the containers, you can use the `--volume` option, like so

* `--volume hostpath:containerpath:envvar`

this will map `hostpath` into the container at `containerpath` and execute the container with the environment variable `envvar` set to `containerpath`.
`--volume` may be specified more than once.


2. If you want to build your own agent:
    1. Define a Docker image compatible with your system

        Use the Dockerfiles in [dockeragent](dockeragent) as an example.
        You will probably need to change the downloaded Azure Pipelines agent binary, as well as the nvidia CUDA base image.

    2. Build the docker image yourself

        ```
        cd dockeragent
        docker build -f <your docker file> -t myazpagent .
        ```

    3. Run `python/manager.py`

        ```
        cd python
        python manager.py <PAT> <URL> <POOL> -d myazpagent
        ```
