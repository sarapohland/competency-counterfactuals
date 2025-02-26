import os
import torch
import argparse
import numpy as np

from trl import SFTTrainer, SFTConfig

from unsloth import FastVisionModel
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

from src.utils.visualize import numpy_to_pil, tensor_to_pil
from src.datasets.custom_dataset import FinetuneDataset
from src.datasets.setup_dataloader import setup_loader
from src.counterfactual.utils import *
from src.counterfactual.methods import *
from src.explanation.utils import *
from src.explanation.llama import Llama

# Set device to GPU 0 or 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('dataset', type=str)
    parser.add_argument('--counterfactual', type=str, default='none')
    parser.add_argument('--response_type', type=str, default='free')
    parser.add_argument('--decoder_dir', type=str, default='models/lunar/reconstruct/')
    parser.add_argument('--output_dir', type=str, default='results/lunar/explainability/')
    args = parser.parse_args()

    # Set location to store fine-tuned model
    new_model_name = f"llama_{args.dataset}_{args.counterfactual}"

    # Generate counterfactual images (if applicable)
    if not args.counterfactual == 'none':

        # Load trained competency estimator
        file = os.path.join(args.decoder_dir, 'parce.p')
        estimator = pickle.load(open(file, 'rb'))
        # estimator.set_device(device)

        # Create object for generating counterfactuals (if applicable)
        if args.counterfactual == 'igd':
            gen = ImageGradientDescent(estimator, thresh=COMP_THRESH)
        elif args.counterfactual == 'fgd':
            gen = FeatureGradientDescent(estimator, thresh=COMP_THRESH)
        elif args.counterfactual == 'lgd':
            gen = LatentGradientDescent(estimator, thresh=COMP_THRESH)
        elif args.counterfactual == 'lnn':
            gen = LatentNearestNeighbors(estimator, thresh=COMP_THRESH)

        # Set up data loader for finetuning dataset
        dataloader = setup_loader(args.dataset, batch_size=50, finetune=True)

        # Generate counterfactuals for each batch
        counterfactuals = []
        for X in dataloader:
            # X = X.to(device)

            # Compute losses and competency scores for original images
            output = estimator.model(X).cpu()
            scores = estimator.comp_scores(X, output.detach().numpy())

            # Generate counterfactual images with desired method
            X_new = X.clone()
            X_low = X[scores < COMP_THRESH]
            
            if args.counterfactual == 'reco':
                X_new[scores < COMP_THRESH] = estimator.decoder(X_low)
            else:
                X_new[scores < COMP_THRESH] = gen.get_counterfactual(X_low)
            counterfactuals.append(X_new)
        counterfactuals = torch.vstack(counterfactuals).detach().cpu()

    # Generate finetuning dataset
    dataset = FinetuneDataset(os.path.join('./data/', args.dataset))
    images, responses = [], []
    for idx, (data, label) in enumerate(zip(dataset.data, dataset.labels)):
        if args.counterfactual == 'none':
            images.append(numpy_to_pil(data))
            responses.append(label)

        else:
            orig_img = numpy_to_pil(data)
            gen_img = tensor_to_pil(counterfactuals[idx])
            image = combine_images(orig_img, gen_img)
            images.append(image)
            responses.append(label)

    # Load the pre-trained Llama 3.2 Vision model
    model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"
    llama = Llama(model_name)

    # Get instructions for Llama model
    instructions = get_llama_instructions(args.dataset, args.response_type, args.counterfactual)

    # Convert the finetuning dataset to the appropriate format
    converted_dataset = [llama.convert_response(image, instructions, response) \
                            for image, response in zip(images, responses)]

    # Fine-tune the model parameters
    FastVisionModel.for_training(llama.model)

    trainer = SFTTrainer(
        model=llama.model,
        tokenizer=llama.tokenizer,
        data_collator=UnslothVisionDataCollator(llama.model, llama.tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # max_steps=30,
            num_train_epochs=10,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # For Weights and Biases
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        ),
    )

    print(f'\nFine-tuning model with {args.counterfactual} counterfactual...')
    trainer_stats = trainer.train()

    # Save the model and tokenizer
    llama.model.save_pretrained(new_model_name)
    llama.tokenizer.save_pretrained(new_model_name)

    # # Load the fine-tuned model
    # llama = Llama(new_model_name)

if __name__=="__main__":
    main()