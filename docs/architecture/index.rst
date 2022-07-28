.. _det-system-architecture:

.. _system-architecture:

#####################
 System Architecture
#####################

**********
 Overview
**********

This document describes the structural and behavioral aspects of the Determined system architecture.
It presents the architecture using different views, which provide context for both system
administrators and machine learning engineers.

The following image provides a high-level view of the components that comprise the Determined
platform:

.. image:: /assets/images/arch11.png

TBD

The following figure depicts a generalized machine learning workflow:

.. image:: /assets/images/arch12.png

Determined provides the training functionality represented by the **Run Training** section of the
figure with built-in support for data, checkpoint, and metric storage.

************
 Components
************

Master and Agent Nodes
======================

Determined comprises a single *master* and one or more *agents* in a cluster environment. The master
node is a single, non-GPU instance that manages the cluster.

The following figure shows the key architectural components. The diagram shows master and agent
instances on separate machines but a single machine can have both a master and one agent instance.

.. image:: /assets/images/arch00.png

The master keeps experiment metadata in a PostgreSQL database, which can be queried with the WebUI
or CLI.

Users interact with the master using a port configured at installation time and do not interact with
agents, directly.

Master Node Scope of Responsibility
-----------------------------------

Master node responsibilities:

-  Stores experiment, trial, and workload metadata.
-  Schedules and dispatch experiments to agents.
-  Dynamically provisions and deprovisions on-premises and cloud agents.
-  Advances experiment, trial, and workload state machines.
-  Responds to CLI commands.
-  Hosts the WebUI, which is primarily used to monitor experiments.
-  Serves the REST API.

The master is also responsible for the following training-specific searcher, metrics, checkpointing,
and scheduling functionality:

+---------------+----------------------------------------------------------------------+
| Function      | Description                                                          |
+===============+======================================================================+
| Searcher      | The *Searcher* implements a hyperparameter search algorithm and is   |
|               | responsible for coordinating the work of all experiment trials.      |
+---------------+----------------------------------------------------------------------+
| Metrics       | *Metrics* provides persistent storage of metrics reported by Trials. |
+---------------+----------------------------------------------------------------------+
| Checkpointing | *Checkpointing* captures the training state at a given time in the   |
|               | the *model registry*.                                                |
+---------------+----------------------------------------------------------------------+
| Scheduling    | *Scheduling* schedules jobs to run, ensuring that all of the compute |
|               | resources required for a job are available before the job launches.  |
+---------------+----------------------------------------------------------------------+

Agent Node Scope of Responsibility
----------------------------------

Agent node responsibilities:

-  Discovers local computing devices and sends device/slot metadata to the master.
-  Runs workloads at the request of the master.
-  Monitors containers and sends container information to the master.
-  For a *trial runner*, which runs a trial in a containerized environment, reports trial runner
   states to the master.

The agent manages a number of GPU or CPU devices, which are referred to as *slots*. There is
typically one agent per compute server and the active experiment volume dictates the number of
agents needed.

Agents communicate only with the master and state information is not maintained on the agent.

PostgreSQL Database
===================

Each cluster requires access to a `PostgreSQL <https://www.postgresql.org/>`_ database to store
experiment and trial metadata. Although not required, the database usually resides on the master
node. If you use the ``det deploy`` command to launch Determined on a cloud provider or on-premises,
PostgreSQL preinstalls a network file system that is shared by all agent nodes. Otherwise, you need
to manually install PostgreSQL.

Docker Images
=============

Determined launches workloads using `Docker <https://www.docker.com/>`_ containers. By default,
workloads execute inside a Determined-provided container that includes common deep learning
libraries and frameworks.

Default Docker images are provided to launch containers for experiments, commands, and other
workflows. All trial runner containers are launched with additional Determined-specific harness
code, which orchestrates model training and evaluation in the container. Trial runner containers are
also loaded with the experiment model definition and hyperparameter values for the current trial.
GPU-specific versions of each library are automatically selected when running on agents with GPUs.

If your model code has additional dependencies, you can specify a startup hook to load the
additional dependencies. If a startup hook exists, the file automatically runs with every Docker
container startup.

If you use the ``det deploy`` command on a cloud provider, Docker is preinstalled. For a manual or
on-premises deployment using ``det deploy``, you also need to install Docker.

Additional Core Cloud Resources
===============================

+----------+---------------------------------------------+
| Provider | Core Resource                               |
+==========+=============================================+
| AWS      | -  AWS Identity and Access Management (IAM) |
|          | -  Security Groups                          |
+----------+---------------------------------------------+
| GCP      | -  Service Account                          |
|          | -  Firewall Rules                           |
+----------+---------------------------------------------+

Additional Peripheral Cloud Resources
=====================================

+----------+----------------------------------------------+
| Provider | Peripheral Resource                          |
+==========+==============================================+
| AWS      | -  Network/Subnetwork                        |
|          | -  Elastic IP                                |
|          | -  Amazon Simple Storage Service (S3) Bucket |
+----------+----------------------------------------------+
| GCP      | -  Network/Subnetwork                        |
|          | -  Static IP                                 |
|          | -  Google Filestore                          |
|          | -  Google Cloud Storage (GCS) bucket         |
|          | -  AWS Identity and Access Management (IAM)  |
|          | -  Security Groups                           |
+----------+----------------------------------------------+

*******************
 Design Principles
*******************

The Determined platform is designed with the following principles.

Concurrency
===========

Determined provides three types of concurrent processing that take advantage of a mult-GPU
environment:

-  Parallelism across experiments. Schedule multiple experiments to run concurrently on the number
   of available GPUs.

-  Parallelism within an experiment. Schedule multiple experiment trials. A hyperparameter search
   can train multiple trials simultaneously, each of which on its own GPU.

-  Parallelism within a trial. Use multiple GPUs to speed up the training of a single trial, which
   is called distributed training. Determined can coordinate across multiple GPUs on a single
   machine or across multiple GPUs on multiple machines to improve single-trial training
   performance.

Reproducibility
===============

Determined supports reproducible machine learning experiments such that the result of running a
Determined experiment is deterministic. Rerunning a previous experiment should produce an identical
model. This ensures that if the model produced from an experiment is ever lost, it can be recovered
by rerunning the experiment that produced it.

Determined can control and reproduce the following sources of randomness:

-  Hyperparameter sampling decisions.
-  The initial weights for a given hyperparameter configuration.
-  Shuffling of training data in a trial.
-  Dropout or other random layers.

Determined does not currently offer support for controlling non-determinism in floating-point
operations.

Configuration
=============

Determined is a deep learning training platform that simplifies infrastructure management for domain
experts while enabling configuration-based deep learning functionality. Determined uses YAML for
configuring tasks.

At a typical organization, many Determined configuration files will contain similar settings. For
example, all of the training workloads run at a given organization might use the same checkpoint
storage configuration. One way to reduce this redundancy is to use *configuration templates*. With
this feature, users can move settings that are shared by many experiments into a single YAML file
that can then be referenced by configurations that require those settings.

Provisioning and Deprovisioning
===============================

A cluster is managed by the master node, which provisions and deprovisions agent nodes depending on
the current volume of experiments being run on the cluster.

Scheduling
==========

The Determined master takes care of scheduling distributed training jobs automatically, ensuring
that all of the compute resources required for a job are available before the job is launched.

TBD: from Scheduling

Job queue management extends scheduler functionality to offer better visibility and control over
scheduling decisions. It does this by using the *job queue*, which provides information about job
ordering and which jobs are queued, and by permitting dynamic job modification. Job queue management
shows all submitted jobs and job states and lets you modify configuration options, including
priority, queue position, and resource pool membership.

Queue management is available to the fair share, priority, and Kubernetes preemption schedulers.

By default, the Kubernetes scheduler does not support gang scheduling or preemption. This can be
problematic for deep learning workloads that require multiple pods to be scheduled before execution
starts (e.g., for distributed training). Determined includes built-in support for the `lightweight
coscheduling plugin
<https://github.com/kubernetes-sigs/scheduler-plugins/tree/release-1.18/pkg/coscheduling>`__, which
extends the default Kubernetes scheduler to support gang scheduling. Determined also includes
support for priority-based scheduling with preemption. Neither are enabled by default. For more
details and instructions on how to enable the coscheduling plugin, refer to
:ref:`gang-scheduling-on-kubernetes` and :ref:`priority-scheduling-on-kubernetes`.

********************
 Training Framework
********************

Training Scenarios
==================

You have the option of using trial-based training or accessing Core API directly to run your
training logic. Trial-based training hooks into the Determined framework to run the training loop,
whereas, Core API does not hook into the framework.

The following figure compares ``Trial``-based training to using the Core API directly:

.. image:: /assets/images/arch03.png

You run an experiment by providing a *launcher*. Launcher options are:

-  legacy bare-Trial-class

   In general, you convert existing training code by subclassing a ``Trial`` class and implementing
   methods that advertise components of your model, such as model architecture, data loader,
   optimizer, learning rate scheduler, and callbacks. Your ``Trial`` class inherits from Determined
   classes provided for PyTorch, PyTorch Lightning, Keras, or Estimator, depending on your
   framework. This is called the trial definition and by structuring your code in this way,
   Determined can run the training loop, providing advanced training and model management
   capabilities.

-  Determined predefined launchers:

   +---------------------+-------------------------------------------------------------------+
   | Launcher            | Description                                                       |
   +=====================+===================================================================+
   | Horovod             | The horovod launcher is a wrapper around `horovodrun              |
   |                     | <https://horovod.readthedocs.io/en/stable/summary_include.html>`_ |
   |                     | which automatically configures the workers for the trial.         |
   +---------------------+-------------------------------------------------------------------+
   | PyTorch Distributed | This launcher is a Determined wrapper around the PyTorch native   |
   |                     | distributed training launcher, ``torch.distributed.run.``         |
   +---------------------+-------------------------------------------------------------------+
   | DeepSpeed           | The DeepSpeed launcher launches a training script under           |
   |                     | ``deepspeed`` with automatic handling of IP addresses, sshd       |
   |                     | containers, and shutdown.                                         |
   +---------------------+-------------------------------------------------------------------+

-  custom launcher or use one of the Determined predefined launchers

-  a command with arguments, which is run in the container

The distributed training launcher must implement the following logic:

-  Launch all of the workers you want, passing any required peer info, such as rank or chief ip
   address, to each worker.
-  Monitor workers and handle worker termination.

Trial-based Distributed Training
--------------------------------

In trial-based distributed training, Determined starts multiple workers with a Determined-provided
*launcher*. With ``Trial``-based training, you specify a ``Trial`` class as your ``entrypoint``
instead of an entire command. A Determined-provided training script loads the *user trial* and
starts a Determined-provided *trial logic* training loop. The training loop makes Core API calls on
your behalf, transparently. Each worker runs the same trial logic, which is coordinated across many
workers.

Non-distributed Training using the Core API
-------------------------------------------

In Core API-based training, there is no framework or plugins and you interact directly with the
Determined platform to run the following tasks:

-  report metrics and checkpoints
-  check for preemption signals
-  do hyperparameter searches

The following figure shows the software logic you need to provide when using the Core API, directly:

.. image:: /assets/images/arch01.png

The Determined master launches a single container, which calls the *training script* specified in
the experiment configuration file. The launcher, which is not shown, starts a single worker with the
training script.

The training script has complete flexibility in how it defines and trains the model.

After initialization is completed, distributed training in Determined runs the following loop:

#. Every worker performs a forward and backward pass on a unique mini-batch of data.
#. As the result of the backward pass, every worker generates a set of updates to the model
   parameters based on the data it processed.
#. The workers communicate their updates to each other, so that all the workers see all the updates
   made during that batch.
#. Every worker averages the updates by the number of workers.
#. Every worker applies the updates to its copy of the model parameters, resulting in all the
   workers having identical solution states.
#. Return to the first step.

Distributed Training using the Core API
---------------------------------------

The following figure shows multiple agents in a distributed training scenario using the Core API:

.. image:: /assets/images/arch02.png

The Determined master launches a single container with multiple *slots* attached or multiple
containers that each have one or more slots. The training script is called once in each container.

It is advisable to implement training functionality in separate launcher and training scripts:

-  The launcher is responsible for launching multiple workers according to the distributed training
   configuration, with each worker running the training script.
-  The training script should execute training with the number of available peer workers.

If both the launcher and the training script are able to handle non-distributed training, where the
launcher launches only one worker and the worker can operate without peer workers, switching between
distributed training and non-distributed training requires only changing the ``slots_per_trial``
configuration parameter. This is the recommended strategy for using Determined and is how
Trial-based training works.

Workflow
========

Startup Sequence
----------------

#. Each agent notifies the master of the number of resident GPUs.
#. The master provisions agent nodes according to the volume of experiments.
#. For agent-based installations, excluding `Kubernetes <https://kubernetes.io/>`_ and `Slurm
   <https://www.schedmd.com/>`_, the master process requests agents to launch containers.
#. When an experiment starts, the master creates agent instances.

Basic Determined Training Sequence
----------------------------------

TBD: training run overview

#. Prepare data

   Data plays a fundamental role in machine learning model development. The best way to load data
   into your ML models depends on several factors, including whether you are running on-premise or
   in the cloud, the size of your data sets, and your security requirements. Accordingly, Determined
   supports a variety of methods for accessing data.

#. Configure a Launcher

#. Create an Experiment

#. Pre-training Setup

#. Pause and Activate

Or,

#. Submit an experiment to the master.
#. If the agent does not already exist, the master creates one or more agents, depending on
   experiment requirements.
#. The agent accesses the data required to run the experiment.
#. On experiment completion, the agent communicates completion to the master.
#. The master shuts down agents that are no longer needed.

Or,

#. Downloading Data

#. Loading Data

#. Defining Training Loop

   -  Initializing Objects Optimization Step/Using Optimizer Using Learning Rate Scheduler
   -  Checkpointing

#. Defining Validation Loop

Or ...

#. build data set
#. build ``Trial`` class
#. build config file that tells Det how to run experiment

-  How do you load your data; how to pull the data into python: ``build_training_data_loader`` and
   ``build_validation_data_loader``

-  How do you perform training: ``train_batch``

   -  Find best set of parameters to get what you want.

   -  Do it repetitively to jiggle parameters

      -  loss = how well we're doing

      -  Define the training backward pass and step the optimizer (jiggling):

         -  self.context.backward(loss)
         -  self.context.step_optimizer(self.optimizer)

-  How do you perform validation: ``evaluate_batch``

   -  checks results against new data (cat image)

-  checkpointing step

   A checkpoint includes the model definition (Python source code), experiment configuration file,
   network architecture, and the values of the model's parameters (i.e., weights) and
   hyperparameters. When using a stateful optimizer during training, checkpoints will also include
   the state of the optimizer (i.e., learning rate). Users can also embed arbitrary metadata in
   checkpoints

   The *model registry* is a way to group together conceptually related checkpoints (including ones
   across different experiments), storing metadata and longform notes about a model, and retrieving
   the latest version of a model for use or futher development. The model registry can be accessed
   using the WebUI, Python API, REST API, or CLI.

Programming View
================

When you use a ``Trial`` class for training, the ``Trial`` class handles the Core API entirely but
you still have access to the underlying framework to build your model and dataset by directly using
PyTorch or TensorFlow, for example. The following figure shows the relationship of user code to
``PyTorchTrial`` and supported frameworks:

.. image:: /assets/images/arch09.png

When you use ``PyTorchTrial``, you use PyTorch or TensorFlow to define the model, dataset, optimzer,
and other trial-specific objects. ``PyTorchTrial`` handles the Core API details and the PyTorch or
TensorFlow details needed to run the training loop.

When you use Core API directly, you can train using the framework of your choice, and you use the
TBD. The following figure shows user code has direct access to the Core API and supported
frameworks:

.. image:: /assets/images/arch10.png

Launcher
--------

A model definition ``Trial`` class specification or Python launcher script, which is the model
processing entrypoint.

The launcher specification can have the following formats.

-  Arbitrary Script: An arbitrary entrypoint script name.
-  Preconfigured Launch Module with Script: The name of a preconfigured launch module and script
   name.
-  Preconfigured Launch Module with Legacy Trial Definition: The name of a preconfigured launcher
   and legacy ``Trial`` class specification.
-  Legacy Trial Definition: A legacy ``Trial`` class specification.

Core API Primitives
-------------------

.. image:: /assets/images/arch04.png

The Core API exposes mechanisms to integrate your code with the Determined platform. Each
``core_context`` component corresponds to a Determined platform component, as described in the
following sections.

The ClusterInfo API provides master information about the currently-running task and is available
only to tasks running on the cluster. ``ClusterInfo`` exposes various properties that are set for
tasks while running on the cluster, such as ``container_addrs``, which contains the IP addresses of
all containers participating in a distributed task. The ClusterInfo API is intended to be most
useful for implementing custom launchers.

The following sections describe the Core API component interfaces in more detail.

Metrics
^^^^^^^

The master *metrics* storage is the persistent storage of metrics reported by all Trials. WebUI
graphs are rendered from the data in this store. Operations such as "top-N checkpoints" read metrics
storage to determine which checkpoints correspond to the best searcher metric.

The ``core_context.train`` component reports metrics to be stored in metric stroage, with
``.report_training_metrics()`` or ``.report_validation_metrics()``.

Searcher
^^^^^^^^

There is a single *searcher* for each experiment, which implements a hyperparameter search algorithm
and is responsible for coordinating the work of all of the trials in an experiment.

The ``core_context.searcher`` component enables code to integrate with the searcher for an
experiment. You can use the ``core_context.searcher`` class for your trial to participate in the
hyperparameter search for an experiment.

The role of each trial in the hyperparameter search is to iterate through the ``SearcherOperation``
objects from the ``core_context.searcher.operations`` method. Each ``SearcherOperation`` has a
``.length`` that describes how long the trial should train for. The trial evaluates the searcher
metric at that point and reports the metric using the ``op.report_completed(metric_value)`` method.
Traditionally, evaluating the searcher metric means running through the validation dataset and
reporting the metric specified by the ``searcher.metric`` field in the experiment configuration
file. TBD: Is this tradition now dead?

Optionally, each trial can report training progress using the ``op.report_progress`` method. The
searcher collects all reported progress from all trials in the experiment and reflects the
aggregated progress in the WebUI.

Checkpoint
^^^^^^^^^^

A *checkpoint* contains the training state at a point in time. Checkpoints are key to persisting
your trained model after training completes by providing the ability to pause and continue training
without losing progress. The master stores metadata about each checkpoint in external storage.

TBD: I see the programming view diagram is missing a Checkpoint Storage block, which is outside of
the Determined-master.

The ``core_context.checkpoint`` component enables your code to upload and download checkpoint
contents from checkpoint storage and to fetch and store metadata from the master. The ``.upload()``
method takes a directory to upload to external storage with the checkpoint metadata you want to set
with the master. You can fetch the metadata using the ``.get_metadata()`` method and the file
contents using the ``.download()`` method.

Scheduler
^^^^^^^^^

The *scheduler* decides which jobs are allocated time on the scheduler and can preempt running jobs.
Preemption can occur if a higher-priority job arrives or because of user actions, such as clicking
the pause button in the WebUI. Preemption is participatory, so running jobs save a checkpoint and
shut down cleanly.

The ``core_context.preempt`` component enables your code to participate in the preemption process by
periodically calling the ``.should_preempt()`` method and taking the appropriate action if it
indicates your job is preempted. Usually, the only action needed is to save a checkpoint and exit.

If you choose not to participate in preemption, your code continues to run to completion. This has
the effect of degrading overall cluster responsiveness and features such as the WebUI pause button
will not work.

**********
 See Also
**********

Setup:

-  :doc:`/cluster-setup-guide/basic`
-  :doc:`/cluster-setup-guide/deploy-cluster/sysadmin-deploy-on-prem/overview`
-  :doc:`/cluster-setup-guide/deploy-cluster/sysadmin-deploy-on-aws/overview`
-  :doc:`/cluster-setup-guide/deploy-cluster/sysadmin-deploy-on-gcp/overview`

Training:

-  :doc:`/training/setup-guide/overview`
-  :doc:`/training/dtrain-introduction`

Interface:

-  :doc:`/interfaces/commands-and-shells`
-  :doc:`/interfaces/notebooks`
