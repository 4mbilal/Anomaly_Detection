function [mu, Xform] = Get_PCA_TrainingFeatures(Features)
%GET_PCA_TRAININGFEATURES Fit ONE global PCA transform across all spatial
%   locations and all training images pooled together.
%
%   [mu, Xform] = GET_PCA_TRAININGFEATURES(Features), Features is
%   H-by-W-by-C-by-B (spatial height, width, channels, batch of images).
%   Every (spatial location, image) pair is treated as one observation of
%   a C-dimensional feature vector; mu and Xform (the full set of
%   eigenvectors, descending by eigenvalue) come from a single PCA fit
%   over all H*W*B of them combined.
%
%   STATUS: superseded / not used by the current pipeline -- see
%   pca_custom.m's header for the full explanation. In short: this global
%   approach assumes every spatial location shares the same feature
%   distribution, which getFeaturesPCA.m's per-location PCA does not
%   assume (and the paper's results are from the per-location version).

Features = gather(extractdata(Features));
s = size(Features);
Features = reshape(Features, [s(1) * s(2), s(3), s(4)]);

Features_PCA = zeros(s(1) * s(2) * s(4), s(3));
for i = 1:s(3)   % channels
    for j = 1:s(4)   % images
        idx_img = (j - 1) * s(1) * s(2) + 1;
        Features_PCA(idx_img:idx_img + s(1) * s(2) - 1, i) = Features(:, i, j);
    end
end

[mu, eigvec, ~] = pca_custom(Features_PCA);   % rows of Features_PCA are observations
Xform = eigvec(:, 1:end);

% Reconstruction-error sanity check considered during development (not
% part of the fitting logic -- purely diagnostic):
%   eigval_energy = cumsum(eigval) / sum(eigval);
%   plot(eigval_energy); title("Eigen Values Energy")
%   Features_PCA_scaled = Features_PCA - mu;
%   Features_PCA_hat = Xform' * Features_PCA_scaled';
%   Features_PCA_recovered = Xform * Features_PCA_hat + mu';
%   reconstruction_error = sqrt(mean((Features_PCA_recovered' - Features_PCA).^2, "all"))

end
