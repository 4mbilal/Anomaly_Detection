function PCAdata = getFeaturesPCA(Embeddings, Options)
%GETFEATURESPCA Fit the per-spatial-location null-subspace PCA for every
%   cascade stage's feature map. This is the training-time half of the
%   paper's core method (Section 3.1) -- detectAnomalies.m is the
%   corresponding scoring-time half.
%
%   PCAdata = GETFEATURESPCA(Embeddings, Options), Embeddings is a struct
%   with fields f1, f2, ... (one per cascade stage, from getFeatures.m).
%   Returns PCAdata, a 1-by-K struct array (K = Options.No_of_output_layers)
%   with fields .means, .xforms, .eigval for each stage -- see PCA_helper
%   below for exactly what these contain.
%
%   REFACTOR NOTE: the original implementation used an explicit switch(K)
%   block (cases 1 through 8, each hand-listing K near-identical
%   PCA_helper calls) to work around not knowing K until runtime, and
%   returned up to 8 separate output arguments (PCAdata1, PCAdata2, ...)
%   instead of one array. Replaced here with a loop over a struct array,
%   which is behaviorally equivalent for K <= 8 (the only case ever
%   exercised -- the paper uses K=4) and removes the hardcoded ceiling.
%   Call sites now do PCAdata(k).means / .xforms / .eigval instead of
%   PCAdata1.means, PCAdata2.means, etc. -- see detectAnomalies.m.

K = Options.No_of_output_layers;
PCAdata = struct('means', {}, 'xforms', {}, 'eigval', {});

for k = 1:K
    fieldName = sprintf('f%d', k);
    N = round(Options.feature_maps_pc(k) * size(Embeddings.(fieldName), 3));
    stageEmbeddings = gather(extractdata(Embeddings.(fieldName)));
    [means, xforms, eigval] = PCA_helper(stageEmbeddings, N);
    PCAdata(k).means = means;
    PCAdata(k).xforms = xforms;
    PCAdata(k).eigval = eigval;
end

end

function [means, xforms, eigval] = PCA_helper(Embeddings, N)
%PCA_HELPER Fit ONE PCA transform PER SPATIAL LOCATION, keeping the N
%   eigenvectors with the SMALLEST eigenvalues (the approximate null
%   subspace -- paper Section 3.1, Eq. 11-12). This is the paper's core
%   algorithmic contribution: unlike a global PCA, every (i,j) grid
%   position gets its own mean and covariance eigendecomposition, fit
%   only from that position's activations across the training batch.
%
%   [means, xforms, eigval] = PCA_HELPER(Embeddings, N), Embeddings is
%   H-by-W-by-C-by-B. Returns, per spatial location (H*W of them):
%     means(idx,:)      -- C-dim mean vector
%     xforms(idx,:,:)   -- C-by-N matrix, the N lowest-eigenvalue
%                          eigenvectors (ascending order, since MATLAB's
%                          eig() returns them ascending by default -- this
%                          is what lets N smallest simply be eigvec(:,1:N)
%                          with no extra sorting)
%     eigval(idx,:)     -- the N corresponding eigenvalues
%
%   Deliberately eigendecomposes the C-by-C covariance matrix directly
%   (not the data matrix via SVD), so it always returns a full-rank C-dim
%   eigenbasis even when the number of training images B is smaller than
%   the channel count C (e.g. MVTec's "toothbrush" class, 60 training
%   images against 384+ channels) -- a data-matrix SVD would truncate to
%   rank B-1 and silently discard exactly the null-space directions this
%   method depends on.

[H, W, C, B] = size(Embeddings);
XTrainEmbeddings = reshape(Embeddings, [H * W, C, B]);
means = mean(XTrainEmbeddings, 3);
xforms = zeros([H * W, C, N]);
eigval = zeros([H * W, N]);

identityMatrix = eye(C);

for idx = 1:H*W
    channel_data = squeeze(XTrainEmbeddings(idx, :, :))';
    channel_data_zero_mean = channel_data - means(idx, :);
    S = (channel_data_zero_mean' * channel_data_zero_mean) / (length(channel_data_zero_mean) - 1);

    try
        [eigvec, eigval_] = eig(S);
    catch
        % S can be (near-)singular when B is small relative to C (see
        % PCA_HELPER's header). Regularize and retry rather than fail the
        % whole training run over one degenerate spatial location.
        warning('getFeaturesPCA:regularizedEig', ...
            'eig() failed at spatial location %d (B=%d, C=%d) -- retrying with S + eps*I.', idx, B, C);
        [eigvec, eigval_] = eig(S + eps * identityMatrix);
    end

    eigval_ = diag(eigval_);
    Xform = eigvec(:, 1:N);        % ascending order -> smallest N eigenvalues
    eigval_ = eigval_(1:N);

    eigval(idx, :) = eigval_';
    xforms(idx, :, :) = Xform;
end

end
