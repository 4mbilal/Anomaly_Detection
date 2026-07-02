function [Features, Labels] = getFeatures(dataStore, Options)
%GETFEATURES Run the (possibly multi-output) backbone over a datastore and
%   collect each cascade stage's feature maps into Features.f1, .f2, ...
%
%   [Features, Labels] = GETFEATURES(dataStore, Options) requires
%   Options.net to already be the compiled multi-output dlnetwork from
%   getEmbeddingsModel.m, and Options.No_of_output_layers to match the
%   number of outputs that network produces.
%
%   Labels is true for anomalous images, false for "good" -- derived from
%   folder names, so it depends on datastore order being preserved
%   (shuffling is intentionally never enabled here).
%
%   REFACTOR NOTE: the original implementation used an explicit
%   switch(N) block (cases 1 through 8, each hand-unpacking N `predict`
%   outputs) to work around MATLAB's fixed-arity multiple-return-value
%   syntax. That is replaced here with dynamic output capture
%   (`[Y{:}] = predict(net, X)`), which is behaviorally identical for
%   N <= 8 and additionally removes the previous hard 8-layer ceiling.
%
%   INPUT SCALING: no explicit pixel rescaling happens here (or anywhere
%   else in this pipeline) -- see scaleX() below, defined but never
%   called. Whatever normalization is built into the backbone's own input
%   layer is what actually gets applied.

Labels = dataStore.UnderlyingDatastores{1}.Labels ~= "good";
minibatchSize = 16;
trainQueue = minibatchqueue(dataStore, ...
    PartialMiniBatch="return", ...
    MiniBatchFormat=["SSCB", "CB"], ...
    MiniBatchSize=minibatchSize);

net = Options.net;
N = Options.No_of_output_layers;

featureBatches = cell(1, N);

reset(trainQueue);
% Never shuffle: Labels above was derived assuming datastore order is
% preserved through iteration.

while hasdata(trainQueue)
    X = next(trainQueue);
    Y = cell(1, N);
    [Y{:}] = predict(net, X);
    for i = 1:N
        featureBatches{i} = cat(4, featureBatches{i}, Y{i});
    end
end

Features = struct();
for i = 1:N
    Features.(sprintf('f%d', i)) = featureBatches{i};
end

end

function X = scaleX(X)
%SCALEX Pixel rescaling helper. NOT CALLED ANYWHERE in the current
%   pipeline (the call site above is intentionally absent -- it used to
%   be present, commented out, as `% X = scaleX(X);`). Kept for
%   documentation: it shows what an explicit ImageNet-style rescale would
%   look like, in case the backbone's own input-layer normalization is
%   ever swapped out and this needs reinstating. The mean/std values below
%   are the standard ImageNet per-channel statistics in [0,255] scale.
X = X / 255;
% means = [123.6750 116.28 103.53];
% stds = [58.3950 57.1200 57.3750];
% means = reshape(means, [1, 1, 3, 1]);
% stds = reshape(stds, [1, 1, 3, 1]);
% X = (single(X) - means) ./ stds;
end
