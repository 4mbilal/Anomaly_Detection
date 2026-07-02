function [auc_p, anomalyThreshold] = getAnomalyMaps(dataStore, anomalyScoreMap, Testlabels)
%GETANOMALYMAPS Pixel-level ROC curve and AUROC.
%
%   [auc_p, anomalyThreshold] = GETANOMALYMAPS(dataStore, anomalyScoreMap, Testlabels)
%   dataStore: the ground-truth mask datastore (tdsMasks from prepareData.m),
%       containing masks for the ANOMALOUS test images only.
%   anomalyScoreMap: predicted per-pixel anomaly score maps for ALL test
%       images (normal and anomalous).
%   Testlabels: true for anomalous test images, false for normal --
%       used to align anomalyScoreMap's images with dataStore's masks,
%       and to synthesize all-zero "masks" for the normal test images.
%
%   Evaluation protocol: every pixel from every anomalous test image
%   (labeled by its real ground-truth mask) AND every pixel from every
%   normal test image (labeled entirely negative) are pooled into one
%   pixel-level ROC curve. This is deliberately more strict than scoring
%   only the anomalous images' pixels -- a false-positive patch on an
%   otherwise-normal test image counts against the score here.

minibatchSize = 16;
trainQueue = minibatchqueue(dataStore, ...
    PartialMiniBatch="return", ...
    MiniBatchFormat=["SSCB", "CB"], ...
    MiniBatchSize=minibatchSize);

reset(trainQueue);
% Never shuffle: Testlabels' alignment with anomalyScoreMap depends on
% preserved datastore order.

Maps = [];
while hasdata(trainQueue)
    X = next(trainQueue);
    Maps = cat(4, Maps, X);
end
Maps = squeeze(gather(extractdata(Maps)));
Maps = Maps(:, :, 1, :);

actual_anomalies = anomalyScoreMap(:, :, :, Testlabels);
false_anomalies = anomalyScoreMap(:, :, :, logical(1 - Testlabels));

detections = [actual_anomalies(:); false_anomalies(:)];
false_anomalies_labels = false_anomalies * 0;   % normal test images -> all-negative ground truth
labels = logical([logical(Maps(:)); false_anomalies_labels(:)]);

% Alternative, less strict protocol considered during development: score
% ONLY the anomalous test images' pixels against their real masks,
% ignoring normal test images' pixels entirely. Not used for any reported
% result, but noted here since it would give a different (typically
% higher) pixel AUROC than the protocol actually used above.
%   [xroc, yroc, troc, auc_p] = perfcurve(logical(Maps(:)), actual_anomalies(:), true);

[xroc, yroc, troc, auc_p] = perfcurve(labels, detections, true);
figure
lroc = plot(xroc, yroc);
hold on
lchance = plot([0 1], [0 1], "r--");
hold off
xlabel("False Positive Rate")
ylabel("True Positive Rate")
title("ROC Curve AUC (Pixel-wise): " + auc_p);
legend([lroc, lchance], "ROC curve", "Random Chance")

[~, ind] = max(yroc - xroc);
anomalyThreshold = troc(ind);

end
