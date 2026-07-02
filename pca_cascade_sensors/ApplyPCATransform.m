function Embeddings_out = ApplyPCATransform(Embeddings_in, mu, Xform)
%APPLYPCATRANSFORM Project every spatial location's feature vector through
%   ONE global (mu, Xform) pair -- the scoring counterpart to
%   Get_PCA_TrainingFeatures.m.
%
%   Embeddings_out = APPLYPCATRANSFORM(Embeddings_in, mu, Xform),
%   Embeddings_in is H-by-W-by-C-by-B. The same mu/Xform is applied at
%   every (i,j) spatial location -- unlike getFeaturesPCA.m's per-location
%   scoring, which uses a different transform at each location.
%
%   STATUS: superseded / not used by the current pipeline -- see
%   pca_custom.m's header for the full explanation.

s = size(Embeddings_in);
Embeddings_in = gather(extractdata(Embeddings_in));
Embeddings_out = Embeddings_in;

for b = 1:s(4)
    for i = 1:s(1)
        for j = 1:s(2)
            channels = squeeze(Embeddings_in(i, j, :, b));
            channels = channels - mu';
            channels_pca = Xform' * channels;
            Embeddings_out(i, j, :, b) = channels_pca;
        end
    end
end

end
