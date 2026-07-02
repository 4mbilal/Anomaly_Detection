function [auc_i, acc, anomalyThreshold] = getROC(Testlabels, TestScores, options)
%GETROC Image-level ROC curve, AUROC, and decision threshold, plus a
%   normal-vs-anomaly score histogram and confusion matrix (matches the
%   evaluation methodology described in paper Section 4.5 / Figure 6).
%
%   [auc_i, acc, anomalyThreshold] = GETROC(Testlabels, TestScores)
%   Testlabels: true for anomalous, false for normal.
%   TestScores: higher = more anomalous (see 'invert_convention' below if
%               your scores use the opposite convention).
%
%   Threshold is chosen by Youden's J statistic (max(TPR - FPR)), i.e. the
%   ROC point furthest from the diagonal -- matches the paper's stated
%   methodology of selecting a threshold from relative score statistics
%   on the test set.
%
%   [...] = GETROC(..., options) accepts a struct to select alternative
%   threshold strategies that were tried during development but are not
%   the default:
%
%     options.invert_convention   false (default) | true
%         If true, flips both TestScores and Testlabels' polarity before
%         scoring -- for score conventions where LOWER means more
%         anomalous.
%
%     options.threshold_method    'youden' (default) | 'min_anomalous'
%                                  | 'youden_min_anomalous_average'
%         'min_anomalous': use the minimum score among true anomalies
%           (minus eps) as the threshold instead of the Youden-optimal
%           point.
%         'youden_min_anomalous_average': average the Youden-optimal and
%           min-anomalous thresholds.
%         Neither alternative was used for any reported result.

if nargin < 3
    options = struct();
end
if ~isfield(options, 'invert_convention'); options.invert_convention = false; end
if ~isfield(options, 'threshold_method');  options.threshold_method = 'youden'; end

if options.invert_convention
    Testlabels = logical(1 - Testlabels);
    TestScores = -TestScores;
end

figure
[~, edges] = histcounts(TestScores, 100);
hGood = histogram(TestScores(Testlabels == 0), edges);
hold on
hBad = histogram(TestScores(Testlabels == 1), edges);
hold off
legend([hGood, hBad], "Normal (Negative)", "Anomaly (Positive)")
xlabel("Mean Anomaly Score");
ylabel("Counts");

[xroc, yroc, troc, auc_i] = perfcurve(Testlabels, TestScores, true);
figure
lroc = plot(xroc, yroc);
hold on
lchance = plot([0 1], [0 1], "r--");
hold off
xlabel("False Positive Rate")
ylabel("True Positive Rate")
title("ROC Curve AUC: " + auc_i);
legend([lroc, lchance], "ROC curve", "Random Chance")

[~, ind] = max(yroc - xroc);
anomalyThreshold_youden = troc(ind) - eps;
anomalyThreshold_min_anomalous = min(TestScores(Testlabels == 1)) - eps;

switch options.threshold_method
    case 'youden'
        anomalyThreshold = anomalyThreshold_youden;
    case 'min_anomalous'
        anomalyThreshold = anomalyThreshold_min_anomalous;
    case 'youden_min_anomalous_average'
        anomalyThreshold = (anomalyThreshold_youden + anomalyThreshold_min_anomalous) / 2;
    otherwise
        error('getROC:unknownThresholdMethod', 'Unknown threshold_method: %s', options.threshold_method);
end

figure
predictedLabels = TestScores >= anomalyThreshold;
targetLabels = logical(Testlabels);
M = confusionmat(targetLabels, predictedLabels);
confusionchart(M, ["Negative", "Positive"])
acc = sum(diag(M)) / sum(M, "all");
title("Accuracy: " + acc);

end
