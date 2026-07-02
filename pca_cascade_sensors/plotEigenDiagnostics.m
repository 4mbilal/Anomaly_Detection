function plotEigenDiagnostics(Embeddings, means, xforms, eigval, Testlabels, className)
%PLOTEIGENDIAGNOSTICS Reproduces the paper's Figure 4/5-style diagnostic:
%   training-time eigenvalues vs. test-time null-subspace projections for
%   normal and anomalous test images, side by side, to visually confirm
%   that anomalies activate the near-zero-eigenvalue subspace more than
%   normal images do.
%
%   PLOTEIGENDIAGNOSTICS(Embeddings, means, xforms, eigval, Testlabels, className)
%   Embeddings: one cascade stage's test-set feature maps (H-by-W-by-C-by-B).
%   means, xforms, eigval: that same stage's PCAdata(k) fields from
%       getFeaturesPCA.m.
%   Testlabels: true for anomalous.
%   className: used in subplot titles (e.g. "carpet").
%
%   Call this from detectAnomalies.m via its diagOptions
%   parameter -- it is NOT called automatically.
%
%   HISTORY: this used to be called unconditionally on every single
%   detectAnomalies() invocation, with the class name hardcoded to
%   "Carpet" regardless of which class was actually being processed, and
%   ended with an unconditional `keyboard` breakpoint -- meaning a full
%   multi-class sweep would halt execution at every single class and
%   require manually typing `dbcont` to continue. Both fixed here.

if nargin < 6
    className = '';
end

Embeddings = gather(extractdata(Embeddings));
[H, W, C, B] = size(Embeddings);
Embeddings = reshape(Embeddings, [H * W, C, B]);

anomaly_count = sum(double(Testlabels));
normal_count = numel(Testlabels) - anomaly_count;
eigen_data_anomaly = zeros([H * W * anomaly_count, size(eigval, 2)]);
eigen_data_normal = zeros([H * W * normal_count, size(eigval, 2)]);
eig_idx_a = 1;
eig_idx_n = 1;

for dIdx = 1:H*W
    channel_data = squeeze(Embeddings(dIdx, :, :))';
    channel_data_zero_mean = channel_data - means(dIdx, :);
    Xform = squeeze(xforms(dIdx, :, :));
    channel_data_pca = (Xform' * channel_data_zero_mean')';

    for b = 1:B
        Y = squeeze(channel_data_pca(b, :))';
        if Testlabels(b) == 1
            eigen_data_anomaly(eig_idx_a, :) = Y;
            eig_idx_a = eig_idx_a + 1;
        else
            eigen_data_normal(eig_idx_n, :) = Y;
            eig_idx_n = eig_idx_n + 1;
        end
    end
end

figure
subplot(1, 3, 1)
plot(abs(eigval)')
title(sprintf('%s (Train-Normal)', className), 'FontSize', 28, 'FontWeight', 'normal', 'FontName', 'Times New Roman')
xlabel('Channels', 'FontSize', 28, 'FontAngle', 'italic', 'FontName', 'Times')
ylabel('Eigen Values', 'FontSize', 28, 'FontWeight', 'normal', 'FontName', 'Times')

subplot(1, 3, 2)
plot(abs(eigen_data_normal)')
title(sprintf('%s (Test-Normal)', className), 'FontSize', 28, 'FontWeight', 'normal', 'FontName', 'Times New Roman')
xlabel('Channels', 'FontSize', 28, 'FontAngle', 'italic', 'FontName', 'Times')
ylabel('Eigen Values', 'FontSize', 28, 'FontWeight', 'normal', 'FontName', 'Times')

subplot(1, 3, 3)
plot(abs(eigen_data_anomaly)')
title(sprintf('%s (Test-Anomalous)', className), 'FontSize', 28, 'FontWeight', 'normal', 'FontName', 'Times New Roman')
xlabel('Channels', 'FontSize', 28, 'FontAngle', 'italic', 'FontName', 'Times')
ylabel('Eigen Values', 'FontSize', 28, 'FontWeight', 'normal', 'FontName', 'Times')

end
