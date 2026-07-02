%MAIN_ANOMALY_PCA_CASCADE Entry point: trains and evaluates the cascaded
%   null-subspace PCA anomaly detector across all configured MVTec AD
%   classes (Bilal & Hanif, Sensors 2025).
%
%   Produces, per class, image-level and pixel-level AUROC, and prints the
%   average across all classes at the end.
%
%   BUGS FIXED DURING REFACTOR (all three silently affected results, not
%   just style):
%
%   1. CONFIG WIRING: this script used to call a local getOptions2()
%      function containing a leftover debug configuration (SqueezeNet
%      backbone, single feature map, only the "transistor" class) instead
%      of getOptions_pca_cascade.m's paper-reproducing MobileNetV2
%      4-layer-cascade config. That means running this script out of the
%      box did NOT reproduce the paper's results, with no indication
%      anything was different. Fixed: now calls getOptions_pca_cascade()
%      directly; the old debug config is still available via
%      getOptions_pca_cascade('quick_test', true).
%
%   2. TRAINING AUGMENTATION: the multi-pass augmentation concatenation
%      loop only handled feature fields f1, f2, f3 (hardcoded), silently
%      dropping augmentation passes for f4 even though the default
%      config uses a 4-layer cascade -- meaning the deepest cascade
%      layer's PCA was fit on only ONE augmentation pass of training data
%      while the other three layers got Options.train_aug_times passes.
%      Fixed: now loops generically over Options.No_of_output_layers.
%
%   3. UNSEEDED RNG: `rng_seed = randi(1000);` drew a random integer but
%      never actually applied it via rng() -- it had zero effect on
%      anything, and re-running this script (especially repeatedly within
%      the same MATLAB session, since MATLAB does not reset its RNG
%      stream between script runs) produced genuinely different data
%      augmentation, and hence a different average AUROC, every time.
%      Fixed: rng(rng_seed) is now actually called, and the seed is
%      printed so a specific run can be reproduced later. Note rng_seed
%      itself is still drawn freshly each run (so back-to-back runs won't
%      be identical by default) -- hardcode a fixed integer instead if
%      you want every run to match exactly.

clear all
close all
clc
addpath(pwd);

gpuDevice(1);

% BUG FIXED DURING REFACTOR: this used to be `rng_seed = randi(1000);`
% with no following rng() call -- it drew a random integer (itself using
% whatever the current unseeded RNG state happened to be) and then never
% applied it to anything. Every run's data augmentation (random rotation/
% zoom in resizeAndCropImage.m) was therefore genuinely different from
% every other run, with no way to reproduce a specific result. Fixed:
% rng_seed is now actually applied via rng(). Change the value below (or
% set it from a script argument) to reproduce a specific prior run, or
% leave it as-is and note whatever value gets printed if you want to
% reproduce *this* run later.
rng_seed = randi(1000);   % set to a fixed integer instead (e.g. 42) for identical results every run
rng(rng_seed);
fprintf('Using rng_seed = %d (rng(%d) to reproduce this run)\n', rng_seed, rng_seed);

Options = getOptions_pca_cascade();
Options = getEmbeddingsModel(Options);

doTrain = true;             % false requires cache_pca_data = true and a previously-saved cache
cache_pca_data = false;     % if true, save/load each class's PCAdata to/from <class>_PCAdata.mat
                             % instead of always recomputing -- useful when iterating on
                             % detection/evaluation logic without re-fitting PCA every time.
plot_eigen_diagnostics = false;   % true reproduces the paper's Figure 4/5-style plot per class

auc_array = [];
TestScores = {};

for i = 1:numel(Options.classes)
    className = Options.classes(i);
    fprintf('%-12s.... ', className);
    [tdsTrain, tdsTest, tdsMasks] = prepareData(fullfile(Options.dataDir, className), Options);
    figure

    cacheFile = strcat(className, "_PCAdata.mat");

    if doTrain
        % tic
        [XTrainEmbeddings, ~] = getFeatures(tdsTrain, Options);
        % toc

        for k = 1:(Options.train_aug_times - 1)
            [XTrainEmbeddings_, ~] = getFeatures(tdsTrain, Options);
            for f = 1:Options.No_of_output_layers
                fieldName = sprintf('f%d', f);
                XTrainEmbeddings.(fieldName) = cat(4, XTrainEmbeddings.(fieldName), XTrainEmbeddings_.(fieldName));
            end
        end

        % tic
        PCAdata = getFeaturesPCA(XTrainEmbeddings, Options);
        % toc

        if cache_pca_data
            save(cacheFile, "PCAdata");
        end
    elseif cache_pca_data
        load(cacheFile, "PCAdata");
    else
        error('main_anomaly_pca_cascade:noTrainingData', ...
            'doTrain is false but cache_pca_data is also false -- no source for PCAdata for class %s.', className);
    end

    clear XTrainEmbeddings   % free memory before test-time feature extraction

    [XTestEmbeddings, Testlabels] = getFeatures(tdsTest, Options);

    diagOptions = struct('plot_eigen_diagnostics', plot_eigen_diagnostics, ...
        'diagnostic_stage', 2, 'class_name', className);
    [auc_i, auc_p, ts] = detectAnomalies(XTestEmbeddings, Testlabels, Options, PCAdata, tdsMasks, diagOptions);

    auc_array = [auc_array; [auc_i, auc_p]];
    fprintf('image AUROC=%.4f  pixel AUROC=%.4f\n', auc_i, auc_p);
    TestScores{i} = ts;
end

m_auc_i_p = mean(auc_array, 1);
fprintf('%-12s image AUROC=%.4f  pixel AUROC=%.4f\n', 'AVERAGE', m_auc_i_p(1), m_auc_i_p(2));
