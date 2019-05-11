# azure-pipelines-agent
experiments with an azure pipelines agent

https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/v2-linux?view=azure-devops

# set up PAT

Go to https://dev.azure.com/{your_organization}

profile >> security >> personal access tokens

new token

more scopes: agent pools read/manage

token is saved in bitwarden under azure-pipelines-agent token

# Create an agent pool

https://dev.azure.com/trekinator/_settings/agentpools >> new agent pool
azure-pipelines-agent

# Download and configure agent


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

# Run the agent

```
~/myagent$ ./run.sh
```