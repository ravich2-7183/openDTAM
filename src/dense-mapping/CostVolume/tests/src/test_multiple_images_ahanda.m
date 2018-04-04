%% Test the output costvolume from test-depth-estimation.cpp

default_cpp_yaml_filename = '../../../../../data/openDTAM_settings_ahanda.yaml';
default_img_dirname = '~/research/vslam/datasets/ahanda_60fps/scene_00_images';
default_depth_img_dirname = '~/research/vslam/datasets/ahanda_60fps/scene_00_ground_depth';
default_poses_pathname = '~/research/vslam/datasets/ahanda_60fps/scene_00_poses';
default_poses_filename = 'camera_poses_sxyz.csv';

%% set settings file, dir names of rgb images and depth images
if(exist(default_cpp_yaml_filename, 'file'))
    cpp_yaml_filename = default_cpp_yaml_filename;
else
    cpp_yaml_filename = uigetfile({'*.yaml'}, ['Select camera properties ' ...
                        'yaml file']);
end

if(exist(default_img_dirname, 'dir'))
    img_dirname = default_img_dirname;
else
    img_dirname = uigetdir('~/', 'Pick rendered rgb images directory');
end

rgb_images = dir(img_dirname);

if(exist(default_depth_img_dirname, 'dir'))
    depth_img_dirname = default_depth_img_dirname;
else
    depth_img_dirname = uigetdir('~/', 'Pick rendered rgb images directory');
end

if(exist(fullfile(default_poses_pathname, default_poses_filename), 'file'))
    poses_pathname = default_poses_pathname;
    poses_filename = default_poses_filename;
else
    [poses_filename, poses_pathname] = ...
        uigetfile({'*.csv'}, 'Select camera poses file');
end

output_dirname = './';

%% run test-multiple-images.cpp for left, right and full sequence
seq = {'left', 'right', 'full'};

% select keyframe and other frames
costvolume_num_frames = randi([50, 100]);
s = randi([3, length(rgb_images)-costvolume_num_frames-1]);
e = s + costvolume_num_frames;
r = uint32((s+e)/2);

disp(['other_start = ', rgb_images(s).name]);
disp(['other_end = ', rgb_images(e).name]);
disp(['reference = ', rgb_images(r).name]);

sm = 0; em = 0;

d_err_distribution_fig = figure(88);

d_abs_err_fig_l = figure(81);
d_abs_err_fig_r = figure(82);
d_abs_err_fig_f = figure(83);

% read ground truth depth
gd_dir = '~/research/vslam/datasets/ahanda_60fps/';
txt_file = fullfile(gd_dir, 'scene_00_poses_txt', ...
                    [rgb_images(r).name(1:4), '.txt']);
depth_file = fullfile(gd_dir, 'scene_00_ground_depth', ...
                      [rgb_images(r).name(1:4), '.depth']);
gd = computeAhandaZ(txt_file, depth_file);

% run test-multiple-images.cpp for left, right and full sequence
for i = 1:length(seq)
    disp(['Running test-depth-estimation.cpp for ', seq{i}, ' sequence']);
    
    switch seq{i}
      case 'left'
        sm = s
        em = r-1
      case 'right'
        sm = r+1
        em = e
      case 'full'
        sm = s
        em = e
    end
    
    tic;
    system(['../build/test-multiple-images-ahanda', ' ', cpp_yaml_filename, ' ', img_dirname, ' ', ...
            depth_img_dirname, ' ', fullfile(poses_pathname, poses_filename), ...
            ' ', rgb_images(r).name(1:4), ' ',  rgb_images(sm).name(1:4), ...
           ' ', rgb_images(em).name(1:4)]);
    toc;

    % read inverse depth output from test-multiple-images-ahanda.cpp
    id_info_fn = fullfile(output_dirname, 'inv_depth_multiple_ahanda.bin.info');
    id_info_f = fopen(id_info_fn);
    id_rows = sscanf(fgetl(id_info_f), ['%d']);
    id_cols = sscanf(fgetl(id_info_f), ['%d']);
    fclose(id_info_f);

    id_fn = fullfile(output_dirname, 'inv_depth_multiple_ahanda.bin');
    id_f = fopen(id_fn);
    % matlab/fortran are column major, while c/c++ are row major
    id = single(fread(id_f, [id_cols, id_rows], 'float32'));
    id = id';
    fclose(id_f);

    d = 1 ./ id;

    % show depth error as a color mapped image
    d_err = (d-gd);
    d_abs_err = abs(d_err);
    mean_d_abs_err = mean(mean(d_abs_err));

    switch seq{i}
      case 'left'
        figure(81);
      case 'right'
        figure(82);
      case 'full'
        figure(83);
    end
    hold on;
    imshow(uint8(floor(mat2gray(d_abs_err)*255)), jet(255));
    title(['Mean abs depth error:', num2str(mean_d_abs_err), ' mm', ...
          ' (', seq{i}, ' sequence)']);
    colorbar;
    hold off;

    % show depth error distribution
    figure(88);
    hold on;
    histogram(d_err(:));
    title('Depth error (d-gd) distribution (units: mm)');
    hold off;
end

figure(88);
hold on;
legend('left half sequence', 'right half sequence', 'full sequence');
hold off;