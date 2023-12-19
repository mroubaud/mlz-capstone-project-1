# ML Zoompamp Capstone Project 1: Horses or Humans

## Problem description

In this project we want to clasify images of humans and horses. The model will classify any input image as a human or as a hourse.
For doing this, we gonna use Deep Learning (DL), a ML technique that uses deep neural networks (DNN's) as models.

<p align="center">
  <img alt="A horse." width="300" src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Zaniskari_Horse_in_Ladakh.jpg">
  <img alt="A human." width="200" src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Marina_person_tvbrasil.jpg">
</p>

We gonna train our model and test it using a tensor flow [data set](https://www.tensorflow.org/datasets/catalog/horses_or_humans)

We gonna devlop a cluster using kind and kubernetesv. The structure of the cluster can be summarise as follows:

- **Gateway Service**: Handle external request to Gateway POD
- **Gateway POD**: For preprocessing and postprocessing
- **Model Service**: Handle internal request to Model POD
- **Model POD**: For predict if image is human or horse

The cluster can be upload to AWS following the instructions given in **Cloud deployment** section.

## Exploratory data Analysis

Exploratory data analysis can be found in the jupyter notebook `Exploratory Data Analysis.ipynb`

## Model training

Model was trained was trained using the following hyperparameters:

- **learning_rate**
- **size_inner**
- **dropout**

For each hyperparameter, we get a different version of the trained model (v1, v2, v3, v4 and v5). v5 is the one used for the final model.
All the models obteined can be found under the **models** folder and can be generated using `training.py` script.

Best parameters found were:

- **learning_rate = 0.025**
- **size_inner = 150**
- **dropout = 0.2**

## Train and Test the Model Locally

For testing the model locally you should follow this steps:

0. For testing the model locally you will need to have dokcer and kind installed in your PC
1. Download the dataset and paste it in the folder of the project. You can download it by openning a Junyper notebook and running the following code:
   ```
   #install tensor flow datasets
   !pip install tensorflow
   !pip install tensorflow-datasets
   #download dataset
   ds, info = tfds.load('horses_or_humans', with_info=True)
   ```
   You will see two zip files in this location: user/tensorflow_datasets/downloads, one for training set and the other for test set, unzip both.
   You should change folders names and location (we want both in the project folder). Folders and location of the train and test images should look like this: **mlz-capstone-project-1/datasets/train** and **mlz-capstone-project-1/datasets/test**
2. Open a terminal in project folders directory
3. Execute `python3 training.py`. After that, tranining process should start. A folder with the name **"models"** should be created. Inside **"models"** folder, go to **"v5"** folder, you will be using the model with the best accuracy (last obtained). Copy the name of the file of that model.
4. Paste the file name with it path inside the **"save_tf_model.py"**. Code is comment so you will note where to paste the code. After that, execute the script with
   ```
   python3 save_tf_model.py
   ```
   A new folder named **"horses_vs_humans"** should be created. Is the folder with the DNN model saved in tensorflow format.
5. From project directory, execute the following command:
   ```
   saved_model_cli show --dir horses_vs_humans --all
   ```
6. From the output of th last command, you will need to keep:
   - **signature_def['serving_default']:**
   - **inputs['conv2d_KEEP-THIS-NUMBER-INPUT_input']**
   - **outputs['dense_KEEP-THIS-NUMBER-OUTPUT']**
7. Inside the gateway folder, you will find the gateway.py script. Open it and fill the fiels of **pb_response.outputs['dense_OUTPUT_NUMBER']** and **pb_request.inputs['input_INPUT_NUMBER'].CopyFrom(np_tp_protobuf(X))** with the values you get in the previous step.
8. Create kind cluster with the following command:
   ```
   kind create cluster
   ```
9. Give context to the cluster with the command:
   ```
   kubectl cluster-info --context kind-kind
   ```
10. Create docker image for the model and laod it to the cluster
    First lets create our docker image of the model. In the same directory of the image-model.dockerfile execute:
    ```
    docker build -t horses_vs_humans:v1 -f image-model.dockerfile .
    ```
    Second, we load the image of the model to the created cluster:
    ```
    kind load docker-image horses_vs_humans:v1
    ```
11. Create deployment for the model and model service in the cluster. For this, your terminal needs to be in the kube-config folder. Once you are there execute:
    ```
    kubectl apply -f model-deployment.yaml
    ```
    ```
    kubectl apply -f model-service.yaml
    ```
12. Port mapping for the model service. Execute:
    ```
    kubectl port-forward pod-name-taken-from-get-pod 9696:9696
    ```
13. Open a new terminal in project directory
14. Create docker image for the gateway and laod it to the cluster
    First lets create our docker image of the gateway. In the same directory of the image-gateway.dockerfile (this file is inside the gateway folder) execute:
    ```
    docker build -t horses_vs_humans-gateway:v1 -f image-gateway.dockerfile .
    ```
    Second, we load the gateway image to the created cluster:
    ```
    kind load docker-image horses_vs_humans-gateway:v1
    ```
15. Create deployment for the gateway and gateway service in the cluster. For this, your terminal should be located in the kube-config folder. Once you are there execute:
    ```
    kubectl apply -f gateway-deployment.yaml
    ```
    ```
    kubectl apply -f gateway-service.yaml
    ```
16. Port mapping for the gateway service. Execute:
    ```
    kubectl port-forward service/gateway 8080:80
    ```
17. Open a new terminal in project directory and execute the test script:
    ```
    python3 test.py.
    ```
    You should get the result of the classification

## Cloud Deployment

We can deploy our model to the cloud. For this case, we gonna summarize the instructions in order to deploy the model to AWS by using EKS and ECR.

In the `eks-config.yaml` you can find the code to create the cluster. You can run `eksctl create cluster -f eks-config.yml` in order to execute `the eks-config.yaml` code.

For deployng the code in a ECR machine of AWS we should do the following:

- Create an ECR repository and login to it.
- Create the remote URIs for the model and gateway images.
- URI prefix is the repo URI.
- URI suffix will be the names of the images but substituting the colons with dashes:
  - `horses_vs_humans-model:v1` becomes `horses_vs_humans-model-v1`
  - `horses_vs_humans-gateway:v1` becomes `horses_vs_humans-gateway_v1`
- Tag the latest versions of your images with the remote URIs.
- Push the images to ECR.

## Files Summary and Description

- **datasets**: Folder with the datasets for training and testing the model
- **gateway**: Folder with the file used for the gateway in the kubernetes cluster. it includes:
  - `gateway.py`: For preprocess and postprocess the input image
  - `image-gateway.docker`: docker image of the gateway
  - `Pipfile`: For the virtual env for the gateway
  - `Pipefile.lock`: For the virtual env for the gateway
  - `proto.py`: Aux program in order use pb communication more efficeincy
- **horses_vs_humans**: Folder with the tf serving model already trained
- **kube-config**: Folder with yaml files used for deployment of the cluster (gateway and model)
- **models**: Folder with all the models obtained during hyperparameter tuning.
- **commands kind and docker.txt**: txt file with important commands for docker and kind
- **eks-config.yaml**: yaml file with eks code to create the cluster using EKS in order to deploy it in AWS
- **Exploratory Data Analysis.ipynb**: Junyper notebook with EDA and Hyperparameter tuning
- **image-model.dockerfile**: Docker file used to create image of the model
- **save-tf_model.py**: Python script used to save the trained model in tf format
- **test.py**: Python script used to test the model
- **training.py**: Python script used to train the model
