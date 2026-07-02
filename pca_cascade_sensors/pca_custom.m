function [mu, eigvec, eigval] = pca_custom(X)
%PCA_CUSTOM Standard PCA via eigendecomposition of the covariance matrix.
%
%   [mu, eigvec, eigval] = PCA_CUSTOM(X), X is n-by-d (rows = observations,
%   columns = features/channels). Returns the mean, eigenvectors sorted
%   DESCENDING by eigenvalue (highest-variance first), and the eigenvalues
%   themselves.
%
%   STATUS: superseded / not used by the current pipeline. This is the
%   PCA primitive behind the pipeline's *first* algorithmic approach: one
%   global PCA fit across all spatial locations pooled together (see
%   Get_PCA_TrainingFeatures.m / ApplyPCATransform.m in this same folder).
%   That approach was superseded by getFeaturesPCA.m's per-spatial-location
%   PCA (a separate covariance eigendecomposition for each grid position,
%   matching PaDiM's per-patch-position philosophy), which is what the
%   published paper actually uses. Kept here for reference / ablation
%   comparison between "one global null-subspace" vs "one null-subspace
%   per spatial location" -- not because it's still in use.
%
%   Note eigenvectors here are sorted highest-variance-first (descending),
%   the opposite convention from getFeaturesPCA.m's PCA_helper, which
%   relies on MATLAB eig()'s natural ascending order to cheaply grab the
%   lowest-variance (near-null-space) components. Don't mix the two
%   conventions if reusing code between them.

mu = mean(X);
X_scaled = X - mu;
S = (X_scaled' * X_scaled) / (length(X) - 1);
[eigvec, eigval] = eig(S);
eigvec = fliplr(eigvec);          % re-sort descending (eig() default is ascending)
eigval = flipud(diag(eigval));

% Alternative SVD-based formulation considered (mathematically equivalent
% up to sign/scale, more numerically stable for wide/ill-conditioned X,
% but never used for any reported result):
%   [~, SS, V] = svd(X_scaled * X_scaled');
%   eigval = diag(SS);
%   eigvec = X_scaled' * V * diag(1 ./ sqrt(eigval));

end
