# azure-pipelines-agent
Some notes on Azure Pipeline self-hosted agent setup.

https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops

The followin steps are marked with (HOST) or (WEBSITE), suggesting whether the step should be carried out on the machine intened to be the self-hosted agent, or on the Azure Pipelines website.

## (WEBSITE) Agent pools

You must tell Azure Pipelines what pool your agents work in.
You can assign each job to a unique pool, so in C3SR we organize pools according to the capabilities of the underlying host.
For example, an amd64 host with ubuntu 1604 and cuda 10.0 would be `amd64-ubuntu1604-cuda100`.

To create a new agent pool do

https://dev.azure.com/c3srdev/_settings/agentpools >> new agent pool

You will use the name of the agent pool in your `azure-pipelines.yml`.

## (HOST) Setting up the virtual agent

Uses Vagrant

```
vagrant up
```

## (HOST) Setting up the real CUDA agent

1. Install the desired version of CUDA on your system
2. Create a `sudo` user to run the Azure pipelines agent.
This user will run the Azure Pipelines agent service.
This user should not need to type in their password to get sudo access, so that `sudo` can be used in the `azure-pipelines.yml` file.
You can do this with `sudo visudo` and then add a line like

```
<user> ALL=(ALL) NOPASSWD: ALL
```

for example

```
azure-pipelines ALL=(ALL) NOPASSWD: ALL
```

### (WEBSITE) set up PAT

Go to https://dev.azure.com/c3srdev

profile >> security >> personal access tokens

new token

more scopes: agent pools read/manage

token is saved in bitwarden under azure-pipelines-agent token



### (HOST) Download and configure agent

1. ssh into the host and switch to the user you created for the Azure Pipelines agent.

2. Download the agent software.

https://dev.azure.com/trekinator/_settings/agentpools >> download agent

```
~/$ mkdir myagent && cd myagent
~/myagent$ tar zxvf ~/Downloads/vsts-agent-linux-x64-2.148.0.tar.gz
```

3. Install the dependencies required for the agent.

```
bin/installdependencies.sh
```

4. Configure the agent.

```
~/myagent$ ./config.sh
```

server URL: 
https://dev.azure.com/c3srdev

authentication type:
PAT

agent pool:
Use the corresponding agent pool you created on the website.

agent name:
the hostname of the system is a good choice

work folder:
_work

# (HOST) Run the agent interactively

If you have been running as a service, uninstall

https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops#service_uninstall

```
sudo ./svc.sh uninstall
```

Then do

```
./run.sh
```

# (HOST) Run the agent as a service

After configuring, run 

```
sudo ./svc.sh install
sudo ./svc.sh start
```