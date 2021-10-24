# Creating a TDSP Project

This clip will walk you through the process of creating a TDSP project, an Azure Machine Learning workspace, utilizing the SDK for Python, and provisioning compute resources.

## Notice

In this clip you will provision GPU supported compute resources.  If you do not decommission these instances (which will also be covered in this clip), you could incur a substantial cost.

## Preparation

To prepare for this clip, you will need to have a browser open with the following two tabs:

* [TDSP Project Template](https://github.com/Azure/Azure-TDSP-ProjectTemplate) (in Github)
* [Azure Portal](https://portal.azure.com/) (logged in to your account)

In addition, you need to have [Anaconda](https://www.anaconda.com/distribution/#download-section), [Git](https://git-scm.com/), and [VS Code](https://code.visualstudio.com/) already installed on your computer.

## Creating an Azure Machine Learning Workspace

The first step is to create a new Azure Machine Learning workspace.  In the tab that you have open to the Azure portal, enter `Machine Learning` in the search bar.  Select the Machine Learning Service. Click the `Add` button.

Give your workspace a name of `ps-ml-workspace` (or whatever you prefer).  Select the subscription, resource group, and location that you desire.  For this effort, we can utilize the `Basic` workspace edition.

Next, complete the creation process.

## Installing the Python SDK

Next, open an Anaconda prompt on your machine. If you haven't worked with Anaconda environments, you can follow the instructions here to create and setup a new Anaconda environment:

* [Setting Up Local Dev Environment](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#local)

Once you have your environment, you will need to activate the it:

```
conda activate aml
```

Once that is in place, you can then install the needed dependencies for working with Azure Machine Learning:

```
pip install azureml-sdk[notebooks,automl]
```

## Cloning the TDSP Project Template

Next, we will clone the TDSP project template. Navigate to the parent directory of where you will want your project directory and enter the following:

```
git clone https://github.com/Azure/Azure-TDSP-ProjectTemplate azml-project
```

This will clone the project in a directory `azml-project` within that parent directory.

## Leveraging VS Code

From the same command prompt, change into the directory for your recently cloned project.  Next, enter the following to launch VS Code for that project:

```
code .
```

Next, you will need to install the `Azure Machine Learning` extension for VS Code.  Navigate to the extensions tab and search for `Azure Machine Learning`.  Install the extension.

Once this is complete, navigate to the `Code\Data_Acquisition_and_Understanding` directory. Create a new file called `provisionCompute.py`.  Insert the following contents (be sure to update your workspace name, subscription id, and resource group):

```
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Get a reference to our workspace
ws = Workspace.get(name='ps-ml-workspace',
					subscription_id='<INSERT SUBSCRIPTION ID>',
					resource_group='<INSERT RESOURCE GROUP>')

# Create a name for our new cluster
cpu_cluster_name = 'tdsp-cluster'

# Verify that cluster does not exist already
try:
    cpu_cluster = AmlCompute(workspace=ws, name=cpu_cluster_name)
    print('Cluster already exists.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6',
                                                           max_nodes=4)
    cpu_cluster = AmlCompute.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
```

This will create a cluster with up to 4 ND-series instances (which utilize 1 NVIDIA Tesla K80 GPU). 

Next, be sure that the python interpreter that is selected for this project is the one that corresponds to the correct Anaconda environment. We can now run this script, and we can also see the results in the Azure Machine Learning tab under compute.

We can now add another script, `decommissionCompute.py` in the same directory:

```
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Get a reference to our workspace
ws = Workspace.get(name='ps-ml-workspace',
					subscription_id='<INSERT SUBSCRIPTION ID>',
					resource_group='<INSERT RESOURCE GROUP>')

# Enter the name of our cluster
cpu_cluster_name = 'tdsp-cluster'

# Attempt to delete compute resources
try:
    cpu_cluster = AmlCompute(workspace=ws, name=cpu_cluster_name)
    cpu_cluster.delete()
    print('Deleting cluster...')
except ComputeTargetException:
    print('Cluster does not exist on workspace...')
```

Now, we can run this script to decommission the compute resources for our workspace.


 