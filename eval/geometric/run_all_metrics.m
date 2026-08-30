clc; clear; close all;

base_dir = fileparts(mfilename('fullpath'));
data_dir = fileparts(base_dir);

addpath(genpath(fullfile(base_dir, 'SIFTflow')));

gt_folder = fullfile(data_dir, 'scan');
pred_folder = fullfile(data_dir, 'eval_img');

target_area = 598400;

test_files = [dir(fullfile(pred_folder, '*.png')); dir(fullfile(pred_folder, '*.jpg'))];

tasks = struct('test_path', {}, 'basename', {}, 'gt_path', {});
for img_idx = 1:numel(test_files)
    [~, basename, ~] = fileparts(test_files(img_idx).name);
    gt_path = fullfile(gt_folder, sprintf('%s.png', basename));
    if ~exist(gt_path, 'file')
        fprintf('Warning: GT image not found for %s\n', basename);
        continue;
    end
    tasks(end+1) = struct( ...
        'test_path', fullfile(pred_folder, test_files(img_idx).name), ...
        'basename', basename, ...
        'gt_path', gt_path);
end
num_tasks = numel(tasks);
fprintf('Found %d image(s) to evaluate\n', num_tasks);

if num_tasks == 0
    error('No images found in %s', pred_folder);
end

result_file = fullfile(base_dir, 'result.txt');
fileID = fopen(result_file, 'w');
fprintf(fileID, 'Image\tMSSIM\tLD\tAD\n');
fclose(fileID);

tic;
parfor t = 1:num_tasks
    task = tasks(t);
    try
        gt_img = imread(task.gt_path);
        test_img = imread(task.test_path);

        [ms_value, ld_value, ad_value] = evalAllMetrics(test_img, gt_img, target_area);

        result_line = sprintf('%s\t%g\t%g\t%g', ...
            task.basename, ms_value, ld_value, ad_value);

        fid = fopen(result_file, 'a');
        fprintf(fid, '%s\n', result_line);
        fclose(fid);
    catch ME
        fprintf('Error processing %s: %s\n', task.basename, ME.message);
    end
end
elapsed_time = toc;

fprintf('\n=== Evaluation Completed ===\n');
fprintf('Total processing time: %.2f seconds\n', elapsed_time);
fprintf('Results saved to: %s\n', result_file);

data = readtable(result_file, 'Delimiter', '\t');
fprintf('\nNum_Images: %d\n', height(data));
fprintf('%-8s %12s %12s\n', 'Metric', 'Mean', 'Std');
fprintf('%s\n', repmat('-', 1, 34));
fprintf('%-8s %12.6f %12.6f\n', 'MSSIM', mean(data.MSSIM), std(data.MSSIM));
fprintf('%-8s %12.6f %12.6f\n', 'LD', mean(data.LD), std(data.LD));
fprintf('%-8s %12.6f %12.6f\n', 'AD', mean(data.AD), std(data.AD));


function [ms, ld, ad] = evalAllMetrics(test_img, gt_img, target_area)

gt_gray = rgb2gray(gt_img);
test_gray = rgb2gray(test_img);

scale = sqrt(target_area / size(gt_gray, 1) / size(gt_gray, 2));
y = imresize(gt_gray, scale);
x = imresize(test_gray, size(y));

wt = [0.0448 0.2856 0.3001 0.2363 0.1333];
ss = zeros(5, 1);
xa = x; ya = y;
for s = 1 : 5
    ss(s) = ssim(xa, ya);
    xa = impyramid(xa, 'reduce');
    ya = impyramid(ya, 'reduce');
end
ms = wt * ss;

[vx, vy] = siftFlow(y, x);

ld = mean(sqrt(vx(:).^2 + vy(:).^2));

g = imgradient(y);
g = g / max(g(:));
[T, ~] = alignLD(g, vx, vy);
[xx, yy] = meshgrid(1 : size(vx, 2), 1 : size(vy, 1));
vxa = T(1, 1) .* (xx + vx) + T(3, 1) - xx;
vya = T(2, 2) .* (yy + vy) + T(3, 2) - yy;
ga = imresize(g, size(vxa));
vxa = ga .* vxa;
vya = ga .* vya;
ad = mean(sqrt(vxa(:).^2 + vya(:).^2));

end


function [T, relres] = alignLD(g, vx, vy)
g = imresize(g, size(vx)) > 0.5;
[xx1, yy1] = meshgrid(1 : size(g, 2), 1 : size(g, 1));
xx2 = xx1 + vx;
yy2 = yy1 + vy;
xx1 = xx1(g);
yy1 = yy1(g);
xx2 = xx2(g);
yy2 = yy2(g);
t = [xx2, zeros(size(xx2, 1), 1)];
A1 = reshape(t', 1, [])';
t = [zeros(size(yy2, 1), 1), yy2];
A2 = reshape(t', 1, [])';
A3 = repmat([1; 0], size(xx2, 1), 1);
A4 = repmat([0; 1], size(xx2, 1), 1);
A = [A1, A2, A3, A4];
B = reshape([xx1, yy1]', 1, [])';
[x, ~, relres] = lsqr(A, B);
T = zeros(3);
T(1) = x(1);
T(5) = x(2);
T(3) = x(3);
T(6) = x(4);
T(end) = 1;

end


function [vx, vy] = siftFlow(im1, im2)
im1 = imresize(imfilter(im1, fspecial('gaussian', 7, 1.), 'same', 'replicate'), 0.5, 'bicubic');
im2 = imresize(imfilter(im2, fspecial('gaussian', 7, 1.), 'same', 'replicate'), 0.5, 'bicubic');

im1 = im2double(im1);
im2 = im2double(im2);

cellsize = 3;
gridspacing = 1;

sift1 = mexDenseSIFT(im1, cellsize, gridspacing);
sift2 = mexDenseSIFT(im2, cellsize, gridspacing);

SIFTflowpara.alpha = 2 * 255;
SIFTflowpara.d = 40 * 255;
SIFTflowpara.gamma = 0.005 * 255;
SIFTflowpara.nlevels = 4;
SIFTflowpara.wsize = 2;
SIFTflowpara.topwsize = 10;
SIFTflowpara.nTopIterations = 60;
SIFTflowpara.nIterations = 30;

[vx, vy, ~] = SIFTflowc2f(sift1, sift2, SIFTflowpara);
end