function imageOut = resizeAndCropImage(imageIn, resizeSize, targetSize, augment, options)
%RESIZEANDCROPIMAGE Pad -> resize -> center-crop pipeline shared by the
%   training/test datastore transform (prepareData.m) and the standalone
%   inference-speed benchmark (measureInferenceSpeed.m).
%
%   imageOut = RESIZEANDCROPIMAGE(imageIn, resizeSize, targetSize, augment)
%   applies, in order:
%     1. (if augment) random rotation (+-2.5 deg) + up to 10% zoom-in,
%        center-cropped back to the original size
%     2. zero-padding to 1024x1024 if either dimension is smaller (avoids
%        distorting small-resolution MVTec classes like metal_nut/tile/
%        bottle/pill when they get downsized -- paper Section 3.3)
%     3. resize to resizeSize (bilinear)
%     4. center-crop to targetSize
%
%   imageOut = RESIZEANDCROPIMAGE(..., options) accepts a struct to select
%   preprocessing alternatives that were evaluated during development but
%   are not the paper's default pipeline:
%
%     options.resize_interp   'bilinear' (default) | 'lanczos2'
%     options.denoise         'none' (default) | 'bilateral' | 'gaussian'
%                              | 'nlmeans' | 'diffusion' | 'median'
%                              Applied after resize, before crop. None of
%                              these denoising variants were used for any
%                              reported result; they were an exploration
%                              of whether pre-filtering noise helped before
%                              feature extraction. Preserved here (rather
%                              than left as inert comments) so they can
%                              still be run if you want to revisit that
%                              question.

if nargin < 5
    options = struct();
end
if ~isfield(options, 'resize_interp'); options.resize_interp = 'bilinear'; end
if ~isfield(options, 'denoise');       options.denoise = 'none';           end

if augment
    angle = randn * 2.5;
    imageIn = imrotate(imageIn, angle, "bicubic", "crop");
    [h, w, ~] = size(imageIn);
    s = 1 + rand * 0.1;
    scaled_img = imresize(imageIn, s);
    crop_x = round((size(scaled_img, 2) - w) / 2);
    crop_y = round((size(scaled_img, 1) - h) / 2);
    imageIn = imcrop(scaled_img, [crop_x, crop_y, w - 1, h - 1]);
end

is_grayscale = size(imageIn, 3) == 1;
if is_grayscale
    imageIn = cat(3, imageIn, imageIn, imageIn);
end
imageOut = zeros([targetSize, size(imageIn, [3 4])], 'like', imageIn);

for idx = 1:size(imageIn, 4)
    s = size(imageIn);
    if s(1) < 1024
        imageIn = padarray(imageIn, [1024 - s(1), 1024 - s(1)] * 0.5, 0, 'both');
    end

    imageTemp = imresize(uint8(imageIn(:, :, :, idx)), resizeSize, options.resize_interp);

    switch options.denoise
        case 'none'
            % default -- no denoising
        case 'bilateral'
            imageTemp = imbilatfilt(imageTemp, 300, 3);   % (DegreeOfSmoothing, SpatialSigma)
        case 'gaussian'
            imageTemp = imgaussfilt(imageTemp, 3);        % sigma=3
        case 'nlmeans'
            imageTemp = imnlmfilt(imageTemp);
        case 'diffusion'
            for k = 1:3
                imageTemp(:, :, k) = imdiffusefilt(imageTemp(:, :, k));
            end
        case 'median'
            for k = 1:3
                imageTemp(:, :, k) = medfilt2(imageTemp(:, :, k), [5 5]);
            end
        otherwise
            error('resizeAndCropImage:unknownDenoise', 'Unknown denoise option: %s', options.denoise);
    end

    win = centerCropWindow2d(size(imageTemp), targetSize);
    [r, c] = deal(win.YLimits(1):win.YLimits(2), win.XLimits(1):win.XLimits(2));
    imageOut(:, :, :, idx) = imageTemp(r, c, :);
end

end
