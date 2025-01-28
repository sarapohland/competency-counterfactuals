import torch
import numpy as np

from lpips import LPIPS
from torch.nn.functional import cosine_similarity

class ImageGradientDescent():

    def __init__(self, estimator, thresh=0.9, max_iter=100, gamma=0.1):
        self.estimator = estimator
        self.max_iter = max_iter
        self.thresh = thresh
        self.gamma = gamma

        self.lpips_loss = LPIPS(net='alex')

    def get_counterfactual(self, X):
        # Get initial counterfactual image
        X_new = X.clone().detach().requires_grad_(True)

        # Define optimizer
        optimizer = torch.optim.Adam([X_new], lr=0.001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        
        for iter in range(self.max_iter):
            # Compute competency scores for current image
            outputs = self.estimator.model(X_new)
            scores = self.estimator.comp_scores_torch(X_new, outputs)

            # If all scores are above threshold, break
            if torch.all(scores >= self.thresh):
                break

            # Otherwise, update counterfactual image
            loss = torch.mean(-scores.cpu() + self.gamma * self.lpips_loss(2*X.cpu()-1, 2*X_new.cpu()-1))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        return X_new
    

class FeatureGradientDescent():

    def __init__(self, estimator, thresh=0.9, metric='cos', max_iter=100, gamma=0.1):
        if not metric in ['l2', 'l1', 'cos']:
            raise NotImplementedError('Unkown distance metric {} for FGD.'.format(metric))

        self.estimator = estimator
        self.max_iter = max_iter
        self.thresh = thresh
        self.metric = metric
        self.gamma = gamma

    def get_counterfactual(self, X):
        # Get initial feature vector
        z = self.estimator.model.get_feature_vector(X)
        X_new = X.clone().detach().requires_grad_(True)
        z_new = self.estimator.model.get_feature_vector(X_new)

        # Define optimizer
        optimizer = torch.optim.Adam([X_new], lr=0.001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        
        for iter in range(self.max_iter):
            # Compute competency scores for current image
            outputs = self.estimator.model(X_new)
            scores = self.estimator.comp_scores_torch(X_new, outputs)

            # If all scores are above threshold, break
            if torch.all(scores >= self.thresh):
                break

            # Otherwise, update counterfactual image
            if self.metric == 'l2':
                loss = torch.mean(-scores + self.gamma * torch.norm(z - z_new, dim=1, p=2))
            elif self.metric == 'l1':
                loss = torch.mean(-scores + self.gamma * torch.norm(z - z_new, dim=1, p=1))
            elif self.metric == 'cos':
                loss = torch.mean(-scores - self.gamma * cosine_similarity(z, z_new, dim=1))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # And corresponding feature vector
            z_new = self.estimator.model.get_feature_vector(X_new)

        return X_new
    

class LatentGradientDescent():

    def __init__(self, estimator, thresh=0.9, metric='cos', max_iter=100, gamma=0.1):
        if not metric in ['l2', 'l1', 'cos']:
            raise NotImplementedError('Unkown distance metric {} for LGD.'.format(metric))

        self.estimator = estimator
        self.max_iter = max_iter
        self.thresh = thresh
        self.metric = metric
        self.gamma = gamma

    def get_counterfactual(self, X):
        # Get initial latent vector
        z = self.estimator.decoder.encode_image(X)
        z_new = z.clone().detach().requires_grad_(True)
        X_new = X.clone()

        # Define optimizer
        optimizer = torch.optim.Adam([z_new], lr=0.001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        
        for iter in range(self.max_iter):
            # Compute competency scores for current image
            outputs = self.estimator.model(X_new)
            scores = self.estimator.comp_scores_torch(X_new, outputs)

            # If all scores are above threshold, break
            if torch.all(scores >= self.thresh):
                break

            # Otherwise, update latent vectors
            if self.metric == 'l2':
                loss = torch.mean(-scores + self.gamma * torch.norm(z - z_new, dim=1, p=2))
            elif self.metric == 'l1':
                loss = torch.mean(-scores + self.gamma * torch.norm(z - z_new, dim=1, p=1))
            elif self.metric == 'cos':
                loss = torch.mean(-scores - self.gamma * cosine_similarity(z, z_new, dim=1))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # And get image decodings from new latent vectors
            X_new = self.estimator.decoder.decode_latent(z_new)

        return X_new
    

class LatentNearestNeighbors():

    def __init__(self, estimator, thresh=0.9, metric='l1'):
        if not metric in ['l2', 'l1', 'cos']:
            raise NotImplementedError('Unkown distance metric {} for LNN.'.format(metric))

        self.estimator = estimator
        self.thresh = thresh
        self.metric = metric
        self.latents = self._save_latents()

    def _save_latents(self):
        # Save latent representation of calibration data
        all_latents = []
        for X, y in self.estimator.dataloader:
            # Get input from calibration data
            if self.estimator.method == 'overall':
                inputs = X.clone()
            elif self.estimator.method == 'regional':
                inputs, _, _, _ = self._mask_images(X, y)

            # Compute competency scores for input data 
            outputs = self.estimator.model(inputs).detach().numpy()
            scores = self.estimator.comp_scores(inputs, outputs)
            
            # Save latent representations for images with high competency
            high_inputs = inputs[scores > self.thresh]
            latents = self.estimator.decoder.encode_image(high_inputs)
            all_latents.append(latents)

        return torch.vstack(all_latents)
    
    def _closest_vectors_l1(self, query_vectors, candidate_vectors):
        # Determine indices of the closest vectors in terms of l1 norm
        distances = torch.norm(query_vectors[:, None, :] - candidate_vectors, dim=2, p=1)
        closest_indices = torch.argmin(distances, dim=1)
        return closest_indices
    
    def _closest_vectors_l2(self, query_vectors, candidate_vectors):
        # Determine indices of the closest vectors in terms of l2 norm
        distances = torch.norm(query_vectors[:, None, :] - candidate_vectors, dim=2, p=2)
        closest_indices = torch.argmin(distances, dim=1)
        return closest_indices
    
    def _closest_vectors_cos(self, query_vectors, candidate_vectors):
        # Normalize both sets of vectors
        query_norm = torch.nn.functional.normalize(query_vectors, p=2, dim=1)
        candidate_norm = torch.nn.functional.normalize(candidate_vectors, p=2, dim=1)
        
        # Determine indices of the closest vectors in terms of cosine similarity
        cosine_similarities = torch.mm(query_norm, candidate_norm.T)
        closest_indices = torch.argmax(cosine_similarities, dim=1)
        return closest_indices
    
    def get_counterfactual(self, X):
        if len(X) < 1:
            return X

        # Get latent representation of original images
        z = self.estimator.decoder.encode_image(X).detach()

        # Find nearest high-competency neighbors in latent space
        if self.metric == 'l2':
            idxs = self._closest_vectors_l2(z, self.latents)
        elif self.metric == 'l1':
            idxs = self._closest_vectors_l1(z, self.latents)
        elif self.metric == 'cos':
            idxs = self._closest_vectors_cos(z, self.latents)
        z_new = self.latents[idxs]

        # Get image decodings from new latent vectors
        X_new = self.estimator.decoder.decode_latent(z_new)

        return X_new