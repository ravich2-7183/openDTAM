%% Test the output costvolume from test-depth-estimation.cpp

%% default dir and file names
img_dirname = '../../../../../data/rgb_images/';
depth_img_dirname = '../../../../../data/depth_images/';

poses_pathname = '../../../../../data/';
poses_filename = 'camera_poses.csv';

yaml_filename = '../../../../../data/openDTAM_settings_blender_matlab.yaml';
cpp_yaml_filename = '../../../../../data/openDTAM_settings_blender.yaml';

alpha_G       = 0; % 100.0;
beta_G        = 1.6 ;
theta_start   = 0.2;
theta_min     = 1.0e-4;
huber_epsilon = 0.02; % 0.00147;
lambda        = 0.20; % 0.80;

regularize_p = 0;
disp(['Regularization set to ', num2str(regularize_p)]);

%% set settings file, dir names of rgb images and depth images
if(~exist(cpp_yaml_filename, 'file'))
    cpp_yaml_filename = uigetfile({'*.yaml'}, ['Select camera properties ' ...
                        'yaml file']);
end

if(~exist(img_dirname, 'dir'))
    img_dirname = uigetdir('~/', 'Pick rendered rgb images directory');
end

rgb_images = dir(img_dirname);

if(~exist(depth_img_dirname, 'dir'))
    depth_img_dirname = uigetdir('~/', 'Pick rendered rgb images directory');
end

if(~exist(fullfile(poses_pathname, poses_filename), 'file'))
    [poses_filename, poses_pathname] = ...
        uigetfile({'*.csv'}, 'Select camera poses file');
end

output_dirname = './';

%% run test-multiple-images.cpp for left, right and full sequence
seq = {'left', 'right', 'full'};

% keyframe and other frames
costvolume_num_frames = randi([5, 15]);
disp(['costvolume_num_frames = ', num2str(costvolume_num_frames)]);
s = randi([3, length(rgb_images)-costvolume_num_frames-1]);
e = s + costvolume_num_frames;
r = uint32((s+e)/2);

disp(['other_start = ', rgb_images(s).name]);
disp(['other_end = ', rgb_images(e).name]);
disp(['reference = ', rgb_images(r).name]);

sm = 0; em = 0;

d_fig_l = figure(81);
d_fig_r = figure(82);
d_fig_f = figure(83);

d_abs_err_fig_l = figure(84);
d_abs_err_fig_r = figure(85);
d_abs_err_fig_f = figure(86);

d_err_distribution_fig = figure(88);

for i = 1:length(seq)
    disp(['Running test-depth-estimation.cpp for ', seq{i}, ' sequence']);
    
    switch seq{i}
      case 'left'
        sm = s;
        em = r-1;
      case 'right'
        sm = r+1;
        em = e;
      case 'full'
        sm = s;
        em = e;
    end
    disp(['start frame sm = ', num2str(sm)]);
    disp(['end frame em = ', num2str(em)]);

    tic;
    system(['../build/test-multiple-images', ' ', cpp_yaml_filename, ' ', img_dirname, ' ', ...
            depth_img_dirname, ' ', fullfile(poses_pathname, poses_filename), ...
            ' ', rgb_images(r).name(1:4), ' ',  rgb_images(sm).name(1:4), ...
           ' ', rgb_images(em).name(1:4), ' ', num2str(alpha_G), ' ', ...
            num2str(beta_G), ' ', ...
            num2str(theta_start), ' ', ...
            num2str(theta_min), ' ', ...
            num2str(huber_epsilon), ' ', ...
            num2str(lambda), ' ', ...
            num2str(regularize_p)]);
    toc;

    % read ground truth depth
    gd_info_fn = fullfile(output_dirname, 'ground_depth_multiple.bin.info');
    gd_info_f = fopen(gd_info_fn);
    gd_rows = sscanf(fgetl(gd_info_f), ['%d']);
    gd_cols = sscanf(fgetl(gd_info_f), ['%d']);
    fclose(gd_info_f);

    gd_fn = fullfile(output_dirname, 'ground_depth_multiple.bin');
    gd_f = fopen(gd_fn);
    % matlab/fortran are column major, while c/c++ are row major
    gd = single(fread(gd_f, [gd_cols, gd_rows], 'single')); 
    gd = gd';
    fclose(gd_f);

    % read inverse depth output from test-multiple-images.cpp
    id_info_fn = fullfile(output_dirname, 'inv_depth_multiple.bin.info');
    id_info_f = fopen(id_info_fn);
    id_rows = sscanf(fgetl(id_info_f), ['%d']);
    id_cols = sscanf(fgetl(id_info_f), ['%d']);
    fclose(id_info_f);

    id_fn = fullfile(output_dirname, 'inv_depth_multiple.bin');
    id_f = fopen(id_fn);
    % matlab/fortran are column major, while c/c++ are row major
    id = single(fread(id_f, [id_cols, id_rows], 'float32'));
    id = id';
    fclose(id_f);

    % show estimated depth
    d = 1 ./ id;
    switch seq{i}
      case 'left'
        figure(81);
      case 'right'
        figure(82);
      case 'full'
        figure(83);
    end
    hold on;
    imshow(mat2gray(d));
    title(['Estimated Depth: ', seq{i}]);
    hold off;

    % show depth error as a color mapped image
    d_err = (d-gd);
    d_abs_err = abs(d_err);
    mean_d_abs_err = mean(mean(d_abs_err));
    disp(['Mean abs depth error: ', seq{i}, ' = ', num2str(mean_d_abs_err), 'm']);
    switch seq{i}
      case 'left'
        figure(84);
      case 'right'
        figure(85);
      case 'full'
        figure(86);
    end
    hold on;
    imshow(uint8(floor(mat2gray(d_abs_err)*255)), jet(255));
    title(['Mean abs depth error for ', seq{i}, ' sequence = ', num2str(mean_d_abs_err), 'm']);
    colorbar;
    hold off;

    % show depth error distribution
    figure(88);
    hold on;
    histogram(d_err(:));
    hold off;
end

figure(88);
hold on;
title('Depth error (d-gd) distribution (units: m)');
legend(seq{1}, seq{2}, seq{3});
hold off;

gd_fig = figure;
hold on;
imshow(mat2gray(gd));
title('Ground Depth');
hold off;
