# Explaining Competency with Counterfactuals

This is the codebase for the paper titled "Explaining Low Perception Model Competency with High-Competency Counterfactuals," which was submitted to the 3rd World Conference on eXplainable Artificial Intelligence (xAI). This README describes how to reproduce the results achieved in this paper. 

## 0) Set Up Codebase

### 0a. Clone this repo

Clone this repository:

```
git clone https://github.com/sarapohland/competency-counterfactuals.git
```

### 0b. Set up the source directory

It is recommended that you create an environment with Python 3.8:

```
conda create -n compcount python=3.8
```

```
conda activate compcount
```

Then, in the main folder (`competency-counterfactuals`), run the following command:

```
pip install -e .
```

## 1) Setup Training Dataset

### 1a. Download the dataset files

To replicate the results presented in the paper, download the lunar and speed dataset files from the `data` folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link). Create a folder called `data` in the  main directory (`competency-counterfactuals`) and subfolders called `lunar` and `speed`. Place the dataset files you downloaded into the corresponding subfolders. If you simply want to use the default datasets, you can skip to step 2. If you want to create a new dataset, proceed through the remaining substeps in this section.

### 1b. Set up directory structure

By default, datasets are assumed to be saved in the following structure:

|-- data  
&emsp;|-- dataset1  
&emsp;&emsp;|-- dataset.npz  
&emsp;&emsp;|-- images  
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted  
&emsp;|-- dataset2  
&emsp;&emsp;|-- dataset.npz  
&emsp;&emsp;|-- images   
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted 

The unsorted folder should contain in-distribution training images that have not been labeled, while the ID folder contains all labeled in-distribution images organized by their class labels. If you already have a labeled dataset, you can organize them in the ID folder and skip to step 1d. If you only have unlabeled data, you can place it all in the unsorted folder and proceed to step 1c. The OOD folder should contain all out-of-distribution images. If this data is labeled, it can be orgnized into its class labels. If it is unlabeled, you can place it all into the same subfolder within the OOD folder. A dataset that has already been set up (following step 1d) will be saved in a compressed NumPy file called dataset.npz in the main dataset folder.

### 1c. Cluster unlabeled data

If you have labeled data, skip to the next step. If you have unlabeled in-distribution data saved in the unsorted directory, you can cluster these images using the create_dataset script:

```
python src/datasets/create_dataset.py <path_to_dataset> --cluster_data
```

This command will cluster the unsorted images and save them in subfolders within the ID folder.

### 1d. Save custom dataset

Once you have existing classes of in-distribution data, you can save a dataset of training, test, and ood data using the create_dataset script:

```
python src/datasets/create_dataset.py <path_to_dataset> --save_data
```

Note that this step can be combined with the previous one. By separating these two steps, you can validate the generated clusters before saving your dataset. You can also use to height and width arguments to resize your images if desired. This script will save a compressed NumPy file called dataset.npz in your dataset directory.

### 1e. Update dataloader setup script

Use the existing cases in the setup_dataloader script to enable the use of your custom dataset. You will need to add a section to the get_class_names, get_num_classes, and the setup_loader functions.

## 2) Generate Classification Model

### 2a. Download the classification model files

To replicate the results presented in the paper, download the lunar and speed classification models from the models folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place them in a folder called `models` in the main directory (`competency-counterfactuals`) with the same folder structure provided in the Drive. (Note that you only need to first model.pth file.) If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the classification model, see substep 2e. Otherwise, you can skip to step 3. 

### 2b. Define the classification model architecture

Create a JSON file defining your model architecture using the example given in `src/networks/classification/layers.json`. Currently, you can define simple model architectures composed of convolutional, pooling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add a flattening layer in between other layers. For convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and amount of zero padding. For pooling layers, you must specify the pooling function (max or average) and the kernel size. Finally, for fully-connected layers, you must specify the number of input and output nodes.

### 2c. Define the classification training parameters

Create a configuration file defining your training parameters using the example given in `src/networks/classification/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 2d. Train the classification model

You can train your model using the train script in the networks classification folder:

```
python src/networks/classification/train.py --train_data <dataset> --output_dir models/<dataset>/classify/
```

The argument train_data is used to indicate which dataset should be used to train your classification model, which should be lunar or speed if you are using the default training datasets. The argument output_dir is used to define where your trained classification model will be saved. (This is `models/<dataset>/classify` for the default models downloaded in 2a.) The arguments network_file and train_config can be used to specify the location of your model architecture JSON file (created in 2b) and training parameter config file (created in 2c) if you are not using ones contained in output_dir. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 2e. Evaluate the classification model

You can evaluate your model using the test script in the networks classification folder:

```
python src/networks/classification/test.py --test_data <dataset> --model_dir models/<dataset>/classify/
```

The argument test_data is used to indicate which dataset should be used to evaluate your classification model, which should be lunar or speed if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 2d. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save a confusion matrix to the model_dir directory.

## 3) Design Probabilistic Competency Estimator

### 3a. Download the competency model files

If you have not done so already, download the reconstruction models from the models folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link) and place them in the appropriate folders in the main directory (`competency-counterfactuals`). The trained competency estimators used in the paper are also contained in the reconstruction folders (as `parce.p`). If you want to modify the configurations to train new models, go through the remaining steps in this section. To evaluate the reconstruction model, see substep 3e. To evaluate the performance of the competency estimator, see 3g. To visualize examples of model competency estimates, see substep 3h. Otherwise, you can skip to step 4. 

### 3b. Define the reconstruction model architecture

Create a JSON file defining your model architecture using the example given in `src/networks/reconstruction/layers.json`. The reconstruction model used by the competency estimator is meant to reconstruct the input image. Currently, you can define simple model architectures composed of convolutional, pooling, transposed convolutional, unsampling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also perform 1D and 2D batch normalization and add an unflattening layer in between other layers. For transposed convolutional layers, you must specify the number of input and output channels and the kernel size. You can optionally provide the stride length and the input/output zero padding. For unsampling layers, you must specify the scale factor or the target output size. If the unsampling mode is not specified, then the 'nearest' unsampling technique will be used. For fully-connected layers, you must specify the number of input and output nodes. Finally, for unflattening, the number of output channels, as well as the resulting height and width, must be provided.

### 3c. Define the reconstruction training parameters

Create a configuration file defining your training parameters using the example given in `src/networks/reconstruction/train.config`. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 3d. Train the reconstruction model

To train the image reconstruction model, you can use the train script in the networks reconstruction folder:

```
python src/networks/reconstruction/train.py reconstruct --architecture autoencoder --train_data <dataset> --model_dir models/<dataset>/classify/ --output_dir models/<dataset>/reconstruct/
```

The argument train_data is used to indicate which dataset should be used to train your reconstruction model, which should be lunar or speed if you are using the default training datasets. The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 2d. The argument output_dir is used to define where your trained reconstruction model will be saved. (This is `models/<dataset>/reconstruct` for the default models.) The arguments network_file and train_config can be used to specify the location of your model architecture JSON file (created in 3b) and training parameter config file (created in 3c) if you are not using ones contained in output_dir. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 3e. Evaluate the reconstruction model

To evaluate the image reconstruction model, you can use the test script in the networks reconstruction folder:

```
python src/networks/reconstruction/test.py reconstruct --architecture autoencoder --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument test_data is used to indicate which dataset should be used to evaluate your reconstruction model, which should be lunar or speed if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will save several figures (displaying the original and reconstructed images, along with the reconstruction loss) to a folder called `reconstruction` in decoder_dir.

### 3f. Train the competency estimator

You can train a competency estimator for your model using the train script in the competency folder:

```
python src/competency/train.py overall --train_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument train_data is used to indicate which dataset should be used to train the competency estimator, which should be lunar or speed if you are using the default training datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to train your model using a GPU.

### 3g. Evaluate the competency estimator 

You can evaluate your competency estimator using the test script in the competency folder:

```
python src/competency/test.py overall --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument test_data is used to indicate which dataset should be used to evaluate the competency estimator, which should be lunar or speed if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to evaluate your model using a GPU. This script will generate plots of the reconstruction loss distributions and probabilistic competency estimates for the correctly classified and misclassified in-distribution data, as well as the out-of-distribution data, and save them to the decoder_dir directory.

### 3h. Visualize the competency estimates

You can visualize the probabilisitic competency estimates for each test image using the visualize script in the competency folder:

```
python src/competency/visualize.py overall --test_data <dataset> --model_dir models/<dataset>/classify/ --decoder_dir models/<dataset>/reconstruct/
```

The argument test_data is used to indicate which dataset should be used for visualization, which should be lunar or speed if you are using the default evaluation datasets. The argument model_dir is used to specify where your trained classification model was saved, and decoder_dir is used to specify where your trained reconstruction model was saved. You can optionally use the use_gpu flag if you want to visualize the model estimates using a GPU. This script will save figures of the input image and estimated competency score to subfolders (correct, incorrect, and ood) in a folder called `competency` in decoder_dir.

## 4) Generate High-Competency Counterfactual Images

### 4a. Select example images for evaluation

If you have not done so already, download the lunar and speed examples files from the `data` folder available [here](https://drive.google.com/drive/folders/1_oob1W8P_NH8YmVQNqvRDHeDdVz_FVWv?usp=share_link). If you simply wish to reproduce the results from our work using the provided examples, you can skip to step 4b. First confirm that you downloaded the desired examples.npz files and placed them in the appropriate locations. If you wish to work with a new dataset or generate additional example images from the provided datasets, proceed through the remainder of this step.

We focus on six causes of low model competency: spatial, brightness, contrast, saturation, pixelation, and noise. (Note that the list of available image modifications are provided in the `mods` list within the utils script in the counterfactual folder.) You can select example images with spatial anomalies using the select_examples script in the datasets folder with the `ood` data type:

```
python src/datasets/select_examples.py <dataset> --decoder_dir models/<dataset>/reconstruct/ --data_type ood
```

Each image in the OOD set of the chosen dataset will be displayed if it results in low model competency (until you reach the number of desired examples, which is defined as NUM_EXAMPLES in this script). (Note that the threshold on competency is defined as COMP_THRESH within the utils script in the counterfactual folder.) You will then see the prompt: "Do you want to save this image?" Answering yes (y) will save this image to the folder `data/<dataset>/spatial/`. Note that you can change the location from which you begin reviewing the OOD set using the start_idx parameter. You can also use a GPU for competency estimation using the use_gpu flag.

You can create example images with the other five sources of low model competency (brightness, contrast, saturation, pixelation, and noise) using the same script with the `test` data type:

```
python src/datasets/select_examples.py <dataset> --decoder_dir models/<dataset>/reconstruct/ --data_type test
```

Now you will be shown images from the ID test set with high model competency. For each image, you will be asked: "Do you want to save this image?" Answering yes (y) will save this image to the folder `data/<dataset>/none/` and then allow you to generate new images from this one with modified properties. For the first image property, brightness, you will be asked: "Select brightness factor." You should enter a floating point number indicating the brightness value. If the competency of this modified image does not cause the model competency to drop below the threshold, you will see: "This image does not result in low levels of competency." You can then try inputting another brightness value. If the competency of this modified image does cause the model competency to drop below the threshold, you will see the modified image and will be asked: "Do you want to save this image?" Answering yes (y) will save this image to the folder `data/<dataset>/brightness/`. If you do not wish to generate an image with modified brightness, simply input a non-numerical character when asked to select a brightness factor. You will see the same prompts for the remaining four image properties (contrast, saturation, pixelation, and noise). You will stop being prompted about certain image properties once you have generated the desired number of images (as defined by NUM_EXAMPLES in this script). Note that you can change the location from which you begin reviewing the ID test set using the start_idx parameter. You can also use a GPU for competency estimation using the use_gpu flag.

You can review the generated images in the subfolders of `data/<dataset>/`. Once you have generated all of the desired examples, you can create an examples dataset using the create_examples script in the datasets folder:

```
python src/datasets/create_examples.py <path_to_dataset>
```

You can optionally use the height and width arguments to resize your images if desired. This script will save a compressed NumPy file called examples.npz in your dataset directory.

## 4b. Evaluate counterfactual generation methods

The methods for generating high-competency counterfactual images are implemented in the methods script in the counterfactual folder. Currently implemented methods are: Image Gradient Descent (IGD), Feature Gradient Descent (FGD), Latent Gradient Descent (LGD), and Latent Nearest Neighbors (LNN), along with the reconstructed image generated by the competency autoencoder (Reco). The list of available methods is also provided in the utils script in the counterfactual folder. 

To generate counterfactual images for all of the examples using a given method, you can use the evaluate script in the counterfactual folder:

```
python src/counterfactual/evaluate.py --method <counterfactual_method> --test_data <dataset> --decoder_dir models/<dataset>/reconstruct/ --output_dir results/<dataset>/
``` 
Evaluation results will be saved to a pickle file with the name of the counterfactual method within a folder called `data` in the output_dir. To save the images generated by this method, use the save_images flag. Images will be saved to a subfolder named by the counterfactual method within a folder called `images` in the output_dir. To run evaluations using your GPU, use the use_gpu flag. Note that you can also change the distance metric used by FGD, LGD, and LNN using the metric argument. The available distance metrics are the l2 norm (l2), the l1 norm (l1), and cosine similarity (cos).

To run the evaluations for all of the implemented counterfactual generation methods, you can use the evaluation bash script in the counterfactual folder:

```
./src/counterfactual/evaluate.sh <dataset>
```

Note that you must ensure this script is executable on your machine. If it is not, you can use the command: `chmod +x ./src/counterfactual/evaluate.sh`. This script will save the results to pickle files in the `data` folder in output_dir and the generated images to subfolders within the `images` folder in output_dir.

To generate a table summarizing these results, you can use the analyze script in the counterfactual folder:

```
python src/counterfactual/analyze.py --output_dir results/<dataset>/
```

Note that output_dir should be the same one chosen when running the evaluations. This script will save a summary CSV file comparing the validity, proximity, similarity, and speed of all currentlly implemented methods to a folder called `results` in output_dir.

To analyze the realism of generated images (assuming you saved the images for the methods of interest), you can run the evaluate_imgs script in the counterfactual folder:

```
python src/counterfactual/evaluate_imgs.py --output_dir results/<dataset>/ --decoder_dir models/<dataset>/reconstruct/
```

Again, output_dir should be the same one chosen when running the evaluations (and saving generated images). This script will compute two realism metrics (KID and FID) for each method across sources of low model competency and save the `results` in two CSV files within the results folder in output_dir.

## 5) Explain High-Competency Counterfactual Images

### 5a. Get access to the LLaMA 3.2: 11B model

To generate language explanations from the high-competency counterfactual images, we employ the LLaMA 3.2 model (in the 11B size). Review the LLama 3.2 community license agreement on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) and request access to the model. Once you receive an email confirming your access, you should be able to successfully use the LLaMA model implemented in the llama script in the explanation model. If not, you need to first provide your Hugging Face credentials using the following command:

```
huggingface-cli login
```

Note that the first time you run the llama script will take a bit of time because it needs to download several gigabytes of data.

### 5b. Generate language explanations

To obtain a language explanation for why an image is associated with low levels of competency, you can use the explain script in the explanation folder:

```
python src/explanation/explain.py <dataset> --counterfactual <counterfactual_method> --output_dir results/<dataset>/
```

The counterfactual method can be one of the five methods evaluated in the previous step (`igd`, `fgd`, `reco`, `lgd`, or `lnn`) or `none` if you would like to generate an explanation without the aid of a counterfactual image. (Note that we do not evaluate the language explanations with the use of counterfactuals obtained by `igd` or `fgd` in our work.) If you are not using a counterfactual image, each of the original images in the `images` folder in the output_dir will be given to the LLaMA model, along with a query about the cause of low model competency. The responses will be saved to six JSON files (corresponding to each of the six true causes of low model competency) in a folder called `explanations` in the output_dir. If you are using a counterfactual image, LLaMA will receive each of the original images, along with each image that was generated using the specified counterfactual method in the `images` folder in the output_dir. The language model will again be asked to explain why the perception model lacked confidence in the original image, and the responses will be saved in JSON files within the `explanations` folder.

To generate all of the language explanations across the counterfactual methods we evaluate for a given dataset, you can use the explain bash script in the explanation folder:

```
./src/explanation/explain.sh <dataset>
```

Note that you must ensure this script is executable on your machine. If it is not, you can use the command: `chmod +x ./src/explanation/explain.sh`. This script will save the explanations to JSON files in the `explanations` folder in the output_dir.

### 5c. Evaluate language explanations

We recommend manually evaluating the language explanations using the evaluate script in the explanations folder:

```
python src/explanation/evaluate.py <counterfactual_method> --modification <low_comp_cause> --output_dir results/<dataset>/
```

This script will allow you to manually evaluate the language explanations generated with the help of the specified counterfactual method (none, igd, fgd, reco, lgd, or lnn) for images with the known cause of low model competency specified by modification (spatial, brightness, contrast, saturation, noise, or pixelate). This script will search for keywords (depending on the true cause of low model competency) and identify language explanations that contain these keywords. For the identified language explanations, it will display the explanation underneath the original low-competency image and the high-competency counterfactual (if one is available). It will then ask: "Is this an accurate explanation for this image pair?" To which you can respond yes (y) or no (n). The script will save your responses to compressed NumPy files in the `explanations` folder.

<!-- Note that you can also automatically evaluate language explanation using the evaluate-auto script, but this is not recommended:

```
python src/explanation/evaluate-auto.py --output_dir results/<dataset>/
``` -->

After you run the evaluations for each combination of counterfactual method and true cause for low model competency, you can analyze the accuracy of the explanations using the analyze script in the explanations folder:

```
python src/explanation/analyze.py --output_dir results/<dataset>/
```

This script will read the labels you assigned to the language explanations in the previous step and compute the accuracy of each counterfactual method for each cause of low model competency and display those accuracies, along with the average accuracy, in a table. It will also save that same table to a CSV file called `explanations.csv` in the `results` folder in the output_dir.

To visualize particular counterfactuals and explanations of interest, you can use the visualize script in the explanation folder:

```
python src/explanation/visualize.py <counterfactual_method> --modification <low_comp_cause> --output_dir results/<dataset>/ --index <image_id>
```

This script will generate a figure of the specified counterfactual and explanation and save it to a folder called `examples` in output_dir.

### 5d. Finetune language model

To finetune the LLaMA model for a particular dataset, you should begin by creating a dataset for finetuning using the create_finetune script in the datasets folder:

```
python src/datasets/create_examples.py <dataset> --decoder_dir models/<dataset>/reconstruct/
```

You can optionally use the use_gpu flag if you want to generate this dataset using a GPU. This script will save a compressed NumPy file called finetune.npz in your dataset directory.

You can then finetune the LLaMA model with this dataset (and sample explanations provided in src/explanation/utils.py) using the finetune script in the explanation folder:

```
python src/explanation/finetune.py <dataset> --counterfactual <counterfactual_method> --output_dir results/<dataset>/explainability/  --decoder_dir models/<dataset>/reconstruct/
```

This script will finetune the language model and save the trained model to a folder called llama_\<dataset\>_\<counterfactual_method\>.

To finetune a model for all the counterfactual methods we evaluate for a given dataset, you can use the finetune bash script in the explanation folder:

```
./src/explanation/finetune.sh <dataset>
```

Note that you must ensure this script is executable on your machine. If it is not, you can use the command: `chmod +x ./src/explanation/finetune.sh`. This script will save each of the finetuned models to a unique folder named llama_\<dataset\>_\<counterfactual_method\>.

To generate language explanations using these finetuned models, follow step 5b. To evaluate these explanations, you should follow step 5c.