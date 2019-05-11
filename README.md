# azure-pipelines-agent
experiments with an azure pipelines agent

https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops

## Agent pools

You must tell Azure Pipelines what pool your agents work in.

https://dev.azure.com/trekinator/_settings/agentpools >> new agent pool

I created `azure-pipelines-agent` for the vagrant box and `amd64-cuda-10-1` for AMD64 CUDA 10.1.
In the azure-pipelines.yml, use this string for the pool on the job.

## Setting up the virtual agent

Uses Vagrant

```
vagrant up
```

## Setting up the real CUDA agent

Install CUDA in Ubuntu 18.04

### set up PAT

Go to https://dev.azure.com/trekinator

profile >> security >> personal access tokens

new token

more scopes: agent pools read/manage

token is saved in bitwarden under azure-pipelines-agent token



### Download and configure agent


https://dev.azure.com/trekinator/_settings/agentpools >> download agent

```
~/$ mkdir myagent && cd myagent
~/myagent$ tar zxvf ~/Downloads/vsts-agent-linux-x64-2.148.0.tar.gz
```

```
bin/installdependencies.sh
```

```
~/myagent$ ./config.sh
```

server URL: 
https://dev.azure.com/trekinator

authentication type:
PAT

agent pool:
pick a name (I picked azure-pipelines-agent)

agent name:
vagrant0

work folder:
_work

# Run the agent interactively

If you have been running as a service, uninstall

https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops#service_uninstall

sudo ./svc.sh uninstall

Then do

```
~/myagent$ ./run.sh
```

# Run the agent as a service

After configuring, run 

sudo ./svc.sh install