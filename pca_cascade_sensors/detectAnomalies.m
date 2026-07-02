function [auc_i, auc_p, TestScores] = detectAnomalies(XTestEmbeddings, Testlabels, Options, PCAdata, tdsMasks, diagOptions)
%DETECTANOMALIES Cascaded anomaly scoring + evaluation (paper Section 3.2,
%   Eq. 13). This is the scoring-time counterpart to getFeaturesPCA.m.
%
%   [auc_i, auc_p, TestScores] = DETECTANOMALIES(XTestEmbeddings, Testlabels,
%       Options, PCAdata, tdsMasks)
%
%   XTestEmbeddings: struct with fields f1, f2, ... (from getFeatures.m).
%   PCAdata: struct array from getFeaturesPCA.m, PCAdata(k) = {.means,
%       .xforms, .eigval} for cascade layer k.
%   Testlabels: true for anomalous test images.
%   tdsMasks: ground-truth mask datastore for pixel-level AUROC.
%
%   STAGE ORDERING: Options.feature_maps is DEEPEST-FIRST (see
%   getEmbeddingsModel.m's header), so XTestEmbeddings.f1 / PCAdata(1) is
%   the deepest layer. The cascade itself runs SHALLOWEST FIRST (paper
%   Fig. 1: Stage 1 uses the earliest layer), so stage s in the cascade
%   uses embeddings/PCAdata index (K + 1 - s).
%
%   CASCADE GATING RULE (Eq. 13): each stage's raw score is
%   max(anomaly map). A stage's dual thresholds are
%       th_max = max(score | label==normal) + eps
%       th_min = min(score | label==anomalous) - eps
%   (computed on this same test set -- this is the paper's stated
%   validation-style threshold selection, not a leak introduced by this
%   refactor). When advancing to the next stage, samples the current
%   stage was confident about (score > th_max or < th_min) have their
%   NEXT stage's score overwritten to exactly 1 or 0; ambiguous samples
%   keep their own next-stage score. The final stage's (partially
%   overwritten) scores are the reported TestScores.
%
%   PIXEL-LEVEL SCORE: unlike the image-level cascade, the pixel-level
%   anomaly map is simply the SUM of all K stages' (already normalized)
%   per-pixel maps -- not gated by the cascade decision.
%
%   REFACTOR NOTE: the original implementation had a separate, almost
%   entirely duplicated code block for every possible cascade depth K=1
%   through K=8 (over 150 lines of copy-pasted assignments). Replaced
%   with a loop -- behaviorally identical for K<=8 (the paper always uses
%   K=4) and works for any K.
%
%   diagOptions (optional) enables the paper's Figure 4/5-style eigenvalue
%   diagnostic plot for one cascade stage:
%     diagOptions.plot_eigen_diagnostics   false (default) | true
%     diagOptions.diagnostic_stage         which cascade stage to plot (default 2)
%     diagOptions.class_name               class name for the plot title
%   NOTE ON HISTORY: this diagnostic used to run UNCONDITIONALLY on every
%   call, with a hardcoded "Carpet" title regardless of the actual class,
%   and ended with an unconditional `keyboard` breakpoint that halted
%   execution every time -- making an automated multi-class sweep
%   impossible without manually continuing at every single class. Both
%   are fixed: it's now opt-in, takes the class name as a parameter, and
%   never blocks execution. See plotEigenDiagnostics.m.

if nargin < 6
    diagOptions = struct();
end
if ~isfield(diagOptions, 'plot_eigen_diagnostics'); diagOptions.plot_eigen_diagnostics = false; end
if ~isfield(diagOptions, 'diagnostic_stage');        diagOptions.diagnostic_stage = 2;           end
if ~isfield(diagOptions, 'class_name');              diagOptions.class_name = '';                end

K = Options.No_of_output_layers;

if diagOptions.plot_eigen_diagnostics
    k = diagOptions.diagnostic_stage;
    stageField = sprintf('f%d', k);
    plotEigenDiagnostics(XTestEmbeddings.(stageField), PCAdata(k).means, ...
        PCAdata(k).xforms, PCAdata(k).eigval, Testlabels, diagOptions.class_name);
end

% --- Per-stage anomaly maps, shallowest (cascade stage 1) to deepest ---
stageScoreMaps = cell(1, K);
for s = 1:K
    embIdx = K + 1 - s;   % see STAGE ORDERING note above
    embeddingField = sprintf('f%d', embIdx);
    stageScoreMaps{s} = computeStageAnomalyMap(XTestEmbeddings.(embeddingField), ...
        PCAdata(embIdx).means, PCAdata(embIdx).xforms, PCAdata(embIdx).eigval, Options);
end

% --- Per-stage raw scores + thresholds ---
stageTestScores = cell(1, K);
stageThMax = zeros(1, K);
stageThMin = zeros(1, K);
for s = 1:K
    [stageTestScores{s}, stageThMax(s), stageThMin(s)] = cascadeWorker(stageScoreMaps{s}, Testlabels);
end

% --- Gate: freeze confident decisions from stage s-1 into stage s ---
for s = 2:K
    stageTestScores{s}(stageTestScores{s-1} > stageThMax(s-1)) = 1;
    stageTestScores{s}(stageTestScores{s-1} < stageThMin(s-1)) = 0;
end
TestScores = stageTestScores{K};

% --- Pixel-level map: plain sum across stages, no gating ---
anomalyScoreMap = stageScoreMaps{1};
for s = 2:K
    anomalyScoreMap = anomalyScoreMap + stageScoreMaps{s};
end

if K > 1
    total = numel(stageTestScores{1});
    ambiguous_after_stage = zeros(1, K - 1);
    for s = 2:K
        ambiguous_after_stage(s - 1) = total - sum(stageTestScores{s} == 1) - sum(stageTestScores{s} == 0);
    end
    fprintf('Cascade(%d): %s\n', total, strjoin(string([total, ambiguous_after_stage]), '->'));
end

close all

[auc_i, ~, ~] = getROC(Testlabels, TestScores);
[auc_p, ~] = getAnomalyMaps(tdsMasks, anomalyScoreMap, Testlabels);

end

function [TestScores, th_max, th_min] = cascadeWorker(anomalyScoreMap, Testlabels)
%CASCADEWORKER Per-image max score for one cascade stage, plus the dual
%   thresholds (Eq. 13) used to gate the next stage.
TestScores = squeeze(max(anomalyScoreMap, [], [1 2 3]));
th_max = max(TestScores(Testlabels == 0)) + eps;
th_min = min(TestScores(Testlabels == 1)) - eps;
end

function anomalyScoreMap = computeStageAnomalyMap(Embeddings, means, xforms, eigval, Options)
%COMPUTESTAGEANOMALYMAP Full per-stage pipeline: null-subspace distance
%   at every pixel -> upsample -> blur -> GLOBAL min-max normalization
%   ACROSS THE ENTIRE BATCH passed in (all images, all pixels, at once --
%   NOT per-image). Per-image normalization would force every image's max
%   pixel to exactly 1.0 regardless of how anomalous it actually is,
%   which destroys the inter-image score ranking that image-level AUROC
%   depends on. Call this once with the FULL test set for a category as
%   the batch, since the normalization constant is shared across
%   everything passed in.
Embeddings = gather(extractdata(Embeddings));
[H, W, C, B] = size(Embeddings);
Embeddings = reshape(Embeddings, [H * W, C, B]);
distances = calculateDistance(Embeddings, H, W, B, means, eigval, xforms, Options);
anomalyScoreMap = createAnomalyScoreMap(distances, H, W, B, Options.targetImageSize, Options);
max_overall = max(anomalyScoreMap, [], [1 2 3 4]);
min_overall = min(anomalyScoreMap, [], [1 2 3 4]);
anomalyScoreMap = (anomalyScoreMap - min_overall) / (max_overall - min_overall);
end

function distances = calculateDistance(XEmbeddings, H, W, B, means, eigval, xforms, Options)
%CALCULATEDISTANCE Null-subspace anomaly distance at every spatial
%   location, for every image in the batch.
%
%   Options.distance_method selects the scoring formula. Default
%   'unweighted_l2' is the paper's method (Eq. 12:
%   Snull(x) = ||W_low^T (x - mu)||_2, i.e. plain L2 norm of the
%   null-subspace projection, no eigenvalue weighting). Every other case
%   below is a variant that was tried during development and is NOT used
%   for any reported result -- preserved here (rather than left as inert
%   comments) in case they're useful for a future ablation.
if ~isfield(Options, 'distance_method')
    Options.distance_method = 'unweighted_l2';
end

distances = zeros([H * W, 1, B]);

for dIdx = 1:H*W
    channel_data = squeeze(XEmbeddings(dIdx, :, :))';
    channel_data_zero_mean = channel_data - means(dIdx, :);
    Xform = squeeze(xforms(dIdx, :, :));
    channel_data_pca = (Xform' * channel_data_zero_mean')';
    S = eigval(dIdx, :)';

    for b = 1:B
        Y = squeeze(channel_data_pca(b, :))';
        distances(dIdx, 1, b) = distanceFromProjection(Y, S, Options.distance_method);
    end
end

% An adaptive, energy-based null-subspace-size exploration was also found
% partially implemented in this function (accumulating per-pixel
% cumulative-eigenvalue-energy thresholds to pick a per-location subspace
% size, instead of the fixed feature_maps_pc fraction used everywhere
% else). It referenced an undefined `covars` variable (likely copy-pasted
% from the separate PaDiM Gaussian-model baseline in
% baseline_comparisons/, which does have a `covars` variable, and was
% never finished being adapted here) and was never in a working state.
% Not preserved as a runnable option since completing it would mean
% guessing at unwritten logic; noted here so the idea isn't lost if you
% want to revisit adaptive null-subspace sizing.

end

function d = distanceFromProjection(Y, S, method)
%DISTANCEFROMPROJECTION One anomaly-score formula given a null-subspace
%   projection Y (column vector) and its corresponding eigenvalues S.
switch method
    case 'unweighted_l2'
        % Paper Eq. 12. Default; only method used for any reported result.
        d = sqrt(sum(Y.^2, "all"));
    case 'l1'
        d = sum(abs(Y), "all");
    case 'l_inf'
        d = max(abs(Y), [], "all");
    case 'inverse_sqrt_eigenvalue_weighted'
        d = sqrt(sum((Y.^2) ./ (abs(S).^0.5 + 1e-4), "all"));
    case 'inverse_eigenvalue_weighted'
        d = sqrt(sum((Y.^2) ./ (abs(S) + 1e-4), "all"));
    case 'inverse_eigenvalue_power_1p5_weighted'
        % Marked "current best on mobilenetv2" at some point during
        % development in the original comments -- superseded by
        % 'unweighted_l2' for the final reported results.
        d = sqrt(sum((Y.^2) ./ abs(1 + S).^1.5, "all"));
    case 'inverse_eigenvalue_power_6_weighted'
        d = sqrt(sum(abs(Y) ./ abs(S).^6, "all"));
    case 'inverse_eigenvalue_squared_weighted'
        d = sqrt(sum((Y.^2) ./ (abs(S).^2 + 1e-4), "all"));
    case 'truncated_50_inverse_weighted'
        % Only the first 50 (lowest-eigenvalue) components.
        d = sqrt(sum((Y(1:50).^2) ./ (abs(S(1:50)) + 1e-4), "all"));
    case 'min_subtracted_l2'
        m1 = min(abs(Y));
        d = sqrt(sum(Y.^2 - m1.^2, "all"));
    case 'max_eigenvalue_weighted_component'
        % Marked "close to best" in the original comments.
        d = sqrt(max(Y.^2 ./ S, [], "all"));
    case 'max_inverse_sqrt_weighted_component'
        % Marked "close to best" in the original comments.
        d = sqrt(max(abs(Y) ./ sqrt(S), [], "all"));
    otherwise
        error('calculateDistance:unknownMethod', 'Unknown distance_method: %s', method);
end
end

function anomalyScoreMap = createAnomalyScoreMap(distances, H, W, B, targetImageSize, Options)
%CREATEANOMALYSCOREMAP Reshape per-location distances into a coarse map,
%   upsample to image resolution, and Gaussian-blur (sigma=4, 33x33
%   kernel) for smoother anomaly boundaries (paper Section 3.3).
if ~isfield(Options, 'resize_interp'); Options.resize_interp = 'bilinear'; end

anomalyScoreMap = reshape(distances, [H, W, 1, B]);
anomalyScoreMap = imresize(anomalyScoreMap, targetImageSize, Options.resize_interp);
for mIdx = 1:size(anomalyScoreMap, 4)
    anomalyScoreMap(:, :, 1, mIdx) = imgaussfilt(anomalyScoreMap(:, :, 1, mIdx), 4, FilterSize=33);
end
end
