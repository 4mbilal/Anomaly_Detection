function [tdsTrain, tdsTest, tdsMask] = prepareData(dataDir, Options)
%PREPAREDATA Build train/test/mask datastores for one MVTec AD category.
%
%   [tdsTrain, tdsTest, tdsMask] = PREPAREDATA(dataDir, Options) expects
%   dataDir to contain the standard MVTec AD category layout:
%     dataDir/train/good/*.png
%     dataDir/test/<defect_type or "good">/*.png
%     dataDir/ground_truth/<defect_type>/*.png
%
%   Each datastore is wrapped with the pad -> resize -> center-crop
%   preprocessing in resizeAndCropImage.m (see that file for the
%   preprocessing options struct, e.g. denoising/interpolation
%   alternatives). Training images are augmented per Options.train_aug;
%   test images and masks never are (Options.test_aug should stay false
%   for masks regardless, since augmenting a mask independently of its
%   image would misalign them).
%
%   Displays a montage of one preprocessed sample from each datastore as
%   a quick visual sanity check.

imdsTrain = imageDatastore(fullfile(dataDir, "train"), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest = imageDatastore(fullfile(dataDir, "test"), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsMasks = imageDatastore(fullfile(dataDir, "ground_truth"), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

addLabelFcn = @(x, info) deal({x, onehotencode(info.Label, 1)}, info);
tdsTrain = transform(imdsTrain, addLabelFcn, IncludeInfo=true);
tdsTest = transform(imdsTest, addLabelFcn, IncludeInfo=true);
tdsMask = transform(imdsMasks, addLabelFcn, IncludeInfo=true);

resizeAndCropImageFcnTrain = @(x, info) deal({resizeAndCropImage(x{1}, Options.resizeImageSize, Options.targetImageSize, Options.train_aug), x{2}});
resizeAndCropImageFcnTest = @(x, info) deal({resizeAndCropImage(x{1}, Options.resizeImageSize, Options.targetImageSize, Options.test_aug), x{2}});
tdsTrain = transform(tdsTrain, resizeAndCropImageFcnTrain);
tdsTest = transform(tdsTest, resizeAndCropImageFcnTest);
tdsMask = transform(tdsMask, resizeAndCropImageFcnTest);

sampleTrain = read(tdsTrain);
sampleTrain = sampleTrain{1};
sampleTest = preview(tdsTest);
sampleTest = sampleTest{1};
sampleMask = preview(tdsMask);
sampleMask = sampleMask{1};
montage({sampleTrain, sampleTest, sampleMask})
title('Preprocessed Data (train sample / test sample / mask sample)')

end
