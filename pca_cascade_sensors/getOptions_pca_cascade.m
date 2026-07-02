function Options = getOptions_pca_cascade(varargin)
%GETOPTIONS_PCA_CASCADE Configuration for the cascaded null-subspace PCA
%   anomaly detector (Bilal & Hanif, "Fast Anomaly Detection for
%   Vision-Based Industrial Inspection Using Cascades of Null Subspace PCA
%   Detectors", Sensors 2025).
%
%   Options = GETOPTIONS_PCA_CASCADE() returns the default configuration
%   that reproduces the paper's results: MobileNetV2 backbone, 4-stage
%   cascade (block_3/7/8 depthwise-BN + block_11 residual-add, shallow to
%   deep), 80% null subspace per layer, full 15-class MVTec AD sweep.
%
%   Options = GETOPTIONS_PCA_CASCADE('Name', Value, ...) overrides
%   defaults. Supported name-value pairs:
%
%     'backbone'    'mobilenetv2' (default) | 'resnet50' | 'squeezenet'
%                   Alternative backbones evaluated during the backbone
%                   ablation (paper Section 4.4.1 / Table 4). MobileNetV2
%                   is the paper's final choice -- best AUROC/speed
%                   trade-off among the lightweight options tried. The
%                   others are kept for reproducibility, not as
%                   recommendations.
%
%     'quick_test'  false (default) | true
%                   true runs a fast single-class, single-layer smoke
%                   test (SqueezeNet, 'transistor' only) instead of the
%                   full evaluation sweep. Useful for confirming the
%                   pipeline runs end-to-end before committing to a full
%                   15-class run.
%
%                   NOTE ON HISTORY: this quick-test configuration used
%                   to be silently hardcoded as the *only* configuration,
%                   embedded directly in main_anomaly_pca_cascade.m as a
%                   local getOptions2() function. That meant running the
%                   main script out of the box did NOT reproduce the
%                   paper -- it ran the debug config with no indication
%                   anything was different. See version_history/CHANGELOG.md.
%
%     'dataDir'     Path to the MVTec AD dataset root. Defaults to the
%                   original development machine's path; you will need to
%                   override this.
%
%   Example:
%     Options = getOptions_pca_cascade();                       % paper config
%     Options = getOptions_pca_cascade('quick_test', true);      % fast smoke test
%     Options = getOptions_pca_cascade('backbone', 'resnet50');  % ablation

p = inputParser;
addParameter(p, 'backbone', 'mobilenetv2');
addParameter(p, 'quick_test', false);
addParameter(p, 'dataDir', 'D:\RnD\Frameworks\Datasets\anomaly\mvtec_anomaly_detection\');
parse(p, varargin{:});
opt = p.Results;

Options = struct();
Options.dataDir = opt.dataDir;

if opt.quick_test
    Options.classes = "transistor";
else
    Options.classes = ["carpet" "grid" "leather" "tile" "wood" "bottle" "cable" ...
        "capsule" "hazelnut" "metal_nut" "pill" "screw" "toothbrush" "transistor" "zipper"];
end

% --- Backbone + cascade layer selection -------------------------------
% Stage order is shallow -> deep. See detectAnomalies.m for how
% the cascade uses this order.
switch opt.backbone
    case 'mobilenetv2'
        Options.net = layerGraph(mobilenetv2);
        Options.feature_maps = ["block_11_add", "block_8_depthwise_BN", ...
                                 "block_7_depthwise_BN", "block_3_depthwise_BN"];
        Options.feature_maps_pc = [0.8 0.8 0.8 0.8];

    case 'resnet50'
        % Alternative tried during backbone ablation. Never used for any
        % reported result; kept only for reproducibility of that ablation.
        Options.net = layerGraph(resnet50);
        Options.feature_maps = ["add_3", "add_2", "add_1"];
        Options.feature_maps_pc = [0.8 0.8 0.8];

    case 'squeezenet'
        % The quick_test smoke-test backbone: single layer, fast to run,
        % NOT part of the paper's cascade ablation.
        Options.net = layerGraph(squeezenet);
        Options.feature_maps = "fire8-concat";
        Options.feature_maps_pc = 1.0;

    otherwise
        error('getOptions_pca_cascade:unknownBackbone', ...
            ['Unknown backbone "%s". Other backbones (resnet18, darknet53, ' ...
             'yolov2-tiny) were referenced in early experiments but never had ' ...
             'a complete configuration (feature_maps_pc was never set for ' ...
             'them in the original code) -- add one here if you need them.'], ...
            opt.backbone);
end

% --- Image preprocessing -----------------------------------------------
% Pad to 1024x1024 (if smaller) -> resize to 256x256 -> center-crop 224x224.
% See prepareData.m for the actual implementation and the per-class
% padding rationale (paper Section 3.3).
Options.resizeImageSize = [256 256];
Options.targetImageSize = [224 224];

% --- Training-time augmentation -----------------------------------------
% 10% zoom-in + +-2.5 degree rotation, applied train_aug_times passes over
% the training set. Matters most for classes with few training images
% (e.g. toothbrush, 60 images) -- see paper Section 4.4.4.
Options.train_aug = true;
Options.train_aug_times = 3;
Options.test_aug = false;

end
