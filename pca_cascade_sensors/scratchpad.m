%SCRATCHPAD Layer-selection / score-fusion exploration script.
%
%   Loads pre-computed per-image anomaly scores for six individual
%   MobileNetV2 blocks (3, 4, 7, 8, 9, 10 -- note this is a DIFFERENT and
%   larger set than the final published 4-layer cascade, which uses
%   blocks 3, 7, 8, 11), sequentially applies each layer's own
%   Youden-optimal threshold to zero out confidently-normal samples (the
%   same gating idea as the real cascade in detectAnomalies.m,
%   but done here as a one-off manual sequence rather than the general
%   loop), then compares two ways of fusing all six layers' surviving
%   scores into one final score:
%     - additive fusion (commented out below)
%     - multiplicative fusion (active)
%
%   STATUS: exploratory / historical. Requires labels.mat and
%   ts_3.mat/ts_4.mat/ts_7.mat/ts_8.mat/ts_9.mat/ts_10.mat (per-layer
%   TestScores saved from earlier individual-layer runs) which are NOT
%   included in this repository -- this script is not runnable as-is,
%   kept for reference on what was tried before settling on the final
%   4-layer cascade with additive pixel-map fusion (see
%   detectAnomalies.m).

clear all
close all
clc

load("labels.mat", "Testlabels");

load("ts_3.mat", "TestScores");
ts_3 = TestScores;
anomalyThreshold = getROC(Testlabels, ts_3);
ts_3(ts_3 <= anomalyThreshold) = 0;

load("ts_4.mat", "TestScores");
ts_4 = TestScores;
ts_4(ts_3 <= anomalyThreshold) = 0;
anomalyThreshold = getROC(Testlabels, ts_4);

load("ts_7.mat", "TestScores");
ts_7 = TestScores;
ts_7(ts_4 <= anomalyThreshold) = 0;
anomalyThreshold = getROC(Testlabels, ts_7);

load("ts_8.mat", "TestScores");
ts_8 = TestScores;
ts_8(ts_7 <= anomalyThreshold) = 0;
anomalyThreshold = getROC(Testlabels, ts_8);

load("ts_9.mat", "TestScores");
ts_9 = TestScores;
ts_9(ts_8 <= anomalyThreshold) = 0;
anomalyThreshold = getROC(Testlabels, ts_9);

load("ts_10.mat", "TestScores");
ts_10 = TestScores;
ts_10(ts_9 <= anomalyThreshold) = 0;
anomalyThreshold = getROC(Testlabels, ts_10);

% Additive fusion alternative (not used for the comparison actually run):
%   ts = ts_3 + ts_4 + ts_7 + ts_8 + ts_9 + ts_10;
ts = ts_3 .* ts_4 .* ts_7 .* ts_8 .* ts_9 .* ts_10;   % multiplicative fusion
anomalyThreshold = getROC(Testlabels, ts);
