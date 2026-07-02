function fps = measureInferenceSpeed(Options, imageFolder)
%MEASUREINFERENCESPEED Full-cascade inference throughput benchmark --
%   this is how the paper's Table 3 "20.1 fps on RTX 3050" figure was
%   produced (paper Section 4.3: "corresponds to the worst-case scenario
%   where every cascade stage is executed").
%
%   fps = MEASUREINFERENCESPEED(Options, imageFolder) runs every PNG in
%   imageFolder through Options.net (already compiled via
%   getEmbeddingsModel.m) and reports images/second.
%
%   REFACTOR NOTE: previously named rt_test(), a local function inside
%   main_anomaly_pca_cascade.m with its own hardcoded image folder path
%   and its own near-duplicate copy of the resize/crop preprocessing
%   (now resizeAndCropImage.m). Extracted here so it can be called
%   independently of the training/evaluation sweep, and so it shares the
%   same preprocessing implementation as the rest of the pipeline instead
%   of a separately-maintained copy.
%
%   Example:
%     Options = getOptions_pca_cascade();
%     Options = getEmbeddingsModel(Options);
%     fps = measureInferenceSpeed(Options, fullfile(Options.dataDir, "leather", "test", "color"));

imageFiles = dir(fullfile(imageFolder, '*.png'));
if isempty(imageFiles)
    error('measureInferenceSpeed:noImages', 'No .png files found in %s', imageFolder);
end

t = 0;
for i = 1:length(imageFiles)
    imagePath = fullfile(imageFolder, imageFiles(i).name);
    img = imread(imagePath);

    tic
    img = resizeAndCropImage(img, Options.resizeImageSize, Options.targetImageSize, false);
    Y = cell(1, Options.No_of_output_layers);
    [Y{:}] = predict(Options.net, dlarray(single(img), 'SSCB'));
    t = t + toc;

    % Live preview of the deepest stage's raw activations (summed over
    % channels), matching the original's diagnostic display.
    Y1_ = gather(extractdata(squeeze(sum(Y{1}, [3 4]))));
    imagesc(Y1_)
    drawnow
end

fps = length(imageFiles) / t;
fprintf('measureInferenceSpeed: %.1f fps over %d images (%s)\n', fps, length(imageFiles), imageFolder);

end
