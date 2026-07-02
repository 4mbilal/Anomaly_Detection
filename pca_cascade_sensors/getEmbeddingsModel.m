function Options = getEmbeddingsModel(Options)
%GETEMBEDDINGSMODEL Truncate the backbone and expose each cascade stage's
%   feature map as a separate dlnetwork output.
%
%   Options = GETEMBEDDINGSMODEL(Options) takes Options.net (a layerGraph)
%   and Options.feature_maps (layer names, ordered DEEPEST FIRST -- see
%   note below) and returns Options.net as a compiled dlnetwork with one
%   output per requested layer.
%
%   How it works:
%     1. The network is truncated so its final layer is
%        Options.feature_maps(1) -- i.e. feature_maps(1) must be the
%        DEEPEST of the requested layers, since everything after it in
%        the original graph gets removed.
%     2. For every other requested layer (feature_maps(2), (3), ...,
%        shallower layers), a trivial leakyReluLayer with slope 1 (an
%        identity pass-through -- it doesn't change values, it just gives
%        dlnetwork a named additional output branch) is grafted on, so
%        the compiled network returns multiple outputs in one forward
%        pass instead of requiring one truncated network per layer.
%
%   ORDERING WARNING: Options.feature_maps(1) must be the deepest stage.
%   The default config (getOptions_pca_cascade.m) lists
%   ["block_11_add", "block_8_depthwise_BN", "block_7_depthwise_BN",
%   "block_3_depthwise_BN"] -- deepest to shallowest -- to satisfy this.
%   detectAnomalies.m then reverses this back to shallow-to-deep stage
%   order for the actual cascade; see that file's header comment.
%
%   INPUT NORMALIZATION: this function does not modify the network's
%   input layer, so whatever normalization is baked into the pretrained
%   network's imageInputLayer (e.g. MobileNetV2's default) is what
%   actually gets applied -- no explicit rescaling is done elsewhere in
%   this pipeline. An earlier attempt to override this with a custom
%   z-score normalization is preserved below, disabled, for reference:
%   note its [512 512 3] input size doesn't match this pipeline's actual
%   224x224 target size, which suggests it was copied over from a
%   different (higher-resolution) experiment and never adapted -- treat
%   it as a historical note, not a ready-to-use option.
%
%     % m1 = net.Layers(1).Mean;
%     % s1 = net.Layers(1).StandardDeviation;
%     % il = imageInputLayer([512 512 3], Normalization="zscore");
%     % il.Mean = m1;
%     % il.StandardDeviation = s1;
%     % net = replaceLayer(net, net.Layers(1).Name, il);

net = Options.net;
Options.No_of_output_layers = numel(Options.feature_maps);

while net.Layers(end).Name ~= Options.feature_maps(1)
    net = removeLayers(net, net.Layers(end).Name);
end

for i = 2:Options.No_of_output_layers
    lr = leakyReluLayer(1, "Name", strcat("LR_", num2str(i - 1)));
    net = addLayers(net, lr);
    net = connectLayers(net, Options.feature_maps(i), strcat("LR_", num2str(i - 1)));
end

Options.net = dlnetwork(net);

end
