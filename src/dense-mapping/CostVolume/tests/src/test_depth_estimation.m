%% Test the output costvolume from test-depth-estimation.cpp

%% Read camera image names
if(exist('../blender_model/rgb_images', 'dir'))
    img_dirname = '../blender_model/rgb_images';
else
    img_dirname = uigetdir('~/', 'Pick rendered rgb images directory');
end

%% Read camera poses
if(exist('../blender_model/camera_poses.csv', 'file'))
    poses_pathname = '../blender_model/';
    poses_filename = 'camera_poses.csv';
else
    [poses_filename, poses_pathname] = ...
        uigetfile({'*.csv'}, 'Select camera poses file');
end

poses_mat = csvread(fullfile(poses_pathname, poses_filename));

poses = containers.Map('KeyType', 'int32', 'ValueType', 'any');
for i = 1:length(poses_mat)
    poses(int32(poses_mat(i,1))) = poses_mat(i, 2:end);
end

%% keyframe and other frames
rgb_images = dir(img_dirname);
keyframe_n = randi([3, length(rgb_images)-5]);
keyframe = rgb_images(keyframe_n).name
otherframe_n = keyframe_n+randi([1,5]);
if(~(otherframe_n <= length(rgb_images)) )
    otherframe_n = length(rgb_images);
end
otherframe = {rgb_images(otherframe_n).name}

%% read camera matrix and costvolume properties
addpath(genpath('./'));

if(exist('../../../../../data/openDTAM_settings_blender_matlab.yaml', 'file'))
    yaml_filename = '../../../../../data/openDTAM_settings_blender_matlab.yaml';
else
    yaml_filename = uigetfile({'*.yaml'}, ['Select camera properties ' ...
                        'yaml file']);
end

y = ReadYaml(yaml_filename);

fx = y.camera0x2Efx;
fy = y.camera0x2Efy;
cx = y.camera0x2Ecx;
cy = y.camera0x2Ecy;

K = [fx, 0, cx;
     0, fy, cy;
     0, 0, 1];

% costvolume properties
c_layers = y.costvolume0x2Elayers;
c_near = y.costvolume0x2Enear_inverse_distance;
c_far  = y.costvolume0x2Efar_inverse_distance;

c_depthStep = (c_near - c_far)/(c_layers-1);

%% camera transforms
pose_wr = poses(str2num(keyframe(1:4))); % pose of camera in world coordinates
t_wr = pose_wr(1:3)';
R = eul2rotm(pose_wr(4:6), 'ZYX'); % this assumes rzyx 
% hack to convert R generated above from rzyx to required form with sxyz euler angles 
R_wr = zeros(3);
R_wr(1,1) = R(3,3); R_wr(1,2) = R(2,3); R_wr(1,3) = R(1,3);
R_wr(2,1) = R(3,2); R_wr(2,2) = R(2,2); R_wr(2,3) = R(1,2);
R_wr(3,1) = R(3,1); R_wr(3,2) = R(2,1); R_wr(3,3) = R(1,1);
T_wr = [R_wr, t_wr; 0, 0, 0, 1];
T_rw = inv(T_wr)

pose_wm = poses(str2num(otherframe{1}(1:4))); % pose of camera in world coordinates
t_wm = pose_wm(1:3)';
R = eul2rotm(pose_wm(4:6), 'ZYX'); % this assumes rzyx 
% hack to convert R generated above from rzyx to required form with sxyz euler angles 
R_wm = zeros(3);
R_wm(1,1) = R(3,3); R_wm(1,2) = R(2,3); R_wm(1,3) = R(1,3);
R_wm(2,1) = R(3,2); R_wm(2,2) = R(2,2); R_wm(2,3) = R(1,2);
R_wm(3,1) = R(3,1); R_wm(3,2) = R(2,1); R_wm(3,3) = R(1,1);
T_wm = [R_wm, t_wm; 0, 0, 0, 1];
T_mw = inv(T_wm)

T_mr = T_mw*T_wr;

%% run test-depth-estimation.cpp
disp('Running test-depth-estimation.cpp');

if(exist('../../../../../data/openDTAM_settings_blender.yaml', 'file'))
    cpp_yaml_filename = '../../../../../data/openDTAM_settings_blender.yaml';
else
    cpp_yaml_filename = uigetfile({'*.yaml'}, ['Select camera properties ' ...
                        'yaml file']);
end

if(exist('../blender_model/depth-render-600-frames/', 'dir'))
    depth_img_dirname = '../blender_model/depth-render-600-frames/';
else
    depth_img_dirname = uigetdir('~/', 'Pick rendered depth images directory');
end

tic;
system(['../build/test-dense-mapping', ' ', cpp_yaml_filename, ' ', img_dirname, ' ', ...
 depth_img_dirname, ' ', fullfile(poses_pathname, poses_filename), ...
 ' ', keyframe(1:4), ' ',  otherframe{1}(1:4)]);
toc;

%% read costvolume output by test-depth-estimation.cpp
costvolume_dirname = './';

c_fn = fullfile(costvolume_dirname, 'costvolume.bin');
c_info_fn = fullfile(costvolume_dirname, 'costvolume.bin.info');

c_info_f = fopen(c_info_fn);
c_rows = sscanf(fgetl(c_info_f), ['%d']);
c_cols = sscanf(fgetl(c_info_f), ['%d']);
fclose(c_info_f);

c_f = fopen(c_fn);
% matlab/fortran are column major, while c/c++ are row major
c = single(fread(c_f, [c_cols, c_rows], 'float32'));
c = c';
fclose(c_f);

%% read ground truth depth
gd_info_fn = fullfile(costvolume_dirname, 'ground_depth.bin.info');
gd_info_f = fopen(gd_info_fn);
gd_rows = sscanf(fgetl(gd_info_f), ['%d']);
gd_cols = sscanf(fgetl(gd_info_f), ['%d']);
fclose(gd_info_f);

gd_fn = fullfile(costvolume_dirname, 'ground_depth.bin');
gd_f = fopen(gd_fn);
% matlab/fortran are column major, while c/c++ are row major
gd = single(fread(gd_f, [gd_cols, gd_rows], 'single')); 
gd = gd';
fclose(gd_f);

% ground_depth_fig = figure(70);
% imshow(mat2gray(gd));

%% read inverse depth output from test-depth-estimation.cpp
id_info_fn = fullfile(costvolume_dirname, 'inv_depth.bin.info');
id_info_f = fopen(id_info_fn);
id_rows = sscanf(fgetl(id_info_f), ['%d']);
id_cols = sscanf(fgetl(id_info_f), ['%d']);
fclose(id_info_f);

id_fn = fullfile(costvolume_dirname, 'inv_depth.bin');
id_f = fopen(id_fn);
% matlab/fortran are column major, while c/c++ are row major
id = single(fread(id_f, [id_cols, id_rows], 'float32'));
id = id';
fclose(id_f);

d = 1 ./ id;

% cuda_depth_fig = figure(71);
% imshow(mat2gray(d));

%% show depth error 
% d_err = (d-gd);
% d_abs_err = abs(d_err);
% mean_d_abs_err = mean(mean(d_abs_err));

% d_abs_err_fig = figure(71);
% hold on;
% imshow(uint8(floor(mat2gray(d_abs_err)*255)), jet(255));
% colorbar;
% title(['Mean abs depth error:', num2str(mean_d_abs_err), ' m']);
% hold off;

% % compute and show error distribution
% d_err_distribution = figure(70);
% hold on;
% histogram(d_err(:));
% title('Depth error (d-gd) distribution (units: m)');
% hold off;

%% show keyframe image and set up datacursormode that allows pixels to be picked
Ir   = imread(fullfile(img_dirname, keyframe));
Irf  = im2single(Ir);

[rows, cols, channels] = size(Ir);

ref_fig = figure(1);
imshow(Irf);
datacursormode on;
dcm_obj = datacursormode(ref_fig);

%% show the other image 
other_fig = figure(2);
Im   = imread(fullfile(img_dirname, otherframe{1}));
Imf  = im2single(Im);

ImF = griddedInterpolant({1:rows, 1:cols, 1:3}, Imf, 'linear', 'none');

imshow(Imf);

% %% read texture output from test-depth-estimation.cpp
% tx_info_fn = fullfile(costvolume_dirname, 'texture_output_r.bin.info');
% tx_info_f = fopen(tx_info_fn);
% tx_rows = sscanf(fgetl(tx_info_f), ['%d']);
% tx_cols = sscanf(fgetl(tx_info_f), ['%d']);
% fclose(tx_info_f);

% tx_fns = {'texture_output_r.bin', 'texture_output_g.bin', 'texture_output_b.bin'};
% Im_tex = zeros(tx_rows, tx_cols, 3);
% for i = 1:length(tx_fns)
%     tx_fn = fullfile(costvolume_dirname, tx_fns{i});
%     tx_f = fopen(tx_fn);
%     % matlab/fortran are column major, while c/c++ are row major
%     tx = fread(tx_f, [tx_cols, tx_rows], 'float32');
%     tx = tx';
%     Im_tex(:,:,i) = tx;
%     fclose(tx_f);
% end

% tex_fig = figure(4);
% hold on;
% grid on;
% plot([1:640], Im_tex(240,:,1), 'g-o', 'DisplayName', 'CUDA');
% plot([1:640], ImF(240.5*ones(1,640), [1:640]+.5, 1*ones(1,640)), 'b-o', 'DisplayName', 'Matlab');
% legend('show');
% hold off;

%% Plot costs for the picked pixel and epipolar line on other frame

cost_fig = figure(3);

ur = rows/2 - 1; vr = cols/2 - 1;

l = c_layers-1 : -1 : 0;
inv_depths = c_far + l*c_depthStep;
zr = -1 ./ inv_depths; % goes from near to far

Kinv = inv(K);

itr = 1;
while itr < 200 && (ur ~= 1 && vr ~= 1)
    ur_prev = ur;
    vr_prev = vr;
    try
        c_info = getCursorInfo(dcm_obj);
        point = c_info.Position(1,:)
        ur = point(1);
        vr = point(2);
    catch
        ur = cols/2;
        vr = rows/2;
    end

    if ~(ur_prev == ur && vr_prev == vr)
        % plot epipolar line on other image, Imf
        figure(2);
        cla;
        imshow(Imf);
        hold on;
        
        % TODO need to ensure that zr is -ve before back projecting
        xr = (Kinv(1,1)*ur + Kinv(1,3)) * abs(zr);
        yr = (Kinv(2,2)*vr + Kinv(2,3)) * abs(zr);
        
        xm = T_mr(1,1)*xr + T_mr(1,2)*yr + T_mr(1,3)*zr + T_mr(1,4);
        ym = T_mr(2,1)*xr + T_mr(2,2)*yr + T_mr(2,3)*zr + T_mr(2,4);
        zm = T_mr(3,1)*xr + T_mr(3,2)*yr + T_mr(3,3)*zr + T_mr(3,4);
        
        % TODO need to ensure that zm is -ve before projecting
        um = K(1,1)*(xm ./ abs(zm)) + K(1,3);
        vm = K(2,2)*(ym ./ abs(zm)) + K(2,3);
        
        plot(um, vm, 'go-', 'DisplayName', 'epipolar-line');
        % for k=1:numel(um)
        %     text(um(k),vm(k)+10, num2str(zr(k)));
        % end
        
        hold off;
        
        % plot cost: computed by CostVolume_update.cu
        figure(3); hold on;
        cla;
        grid on;
        title('Cost along depth: CUDA vs Matlab');
        xlabel('inverse depth');
        ylabel('cost: reprojected pixel difference');

        c_col = ur + cols*vr;
        plot(-1./zr, c(end:-1:1, c_col), '-o', 'DisplayName', 'cuda cost');
        hold off;

        % plot cost: recomputed in matlab
        cost_rc = ones(1, c_layers);
        for i = 1:c_layers
            cost_rc(i) = abs(ImF(vm(i), um(i), 1) - Irf(vr, ur, 1)) + ... 
                         abs(ImF(vm(i), um(i), 2) - Irf(vr, ur, 2)) + ...
                         abs(ImF(vm(i), um(i), 3) - Irf(vr, ur, 3));
        end
        
        figure(3); hold on;
        plot(-1./zr, cost_rc, 'r-x', 'DisplayName', 'matlab cost');
        plot([1./gd(vr, ur), 1./gd(vr, ur)], ylim, 'b', 'LineWidth', 2, 'DisplayName', 'ground depth');
        plot([1./d(vr, ur), 1./d(vr, ur)], ylim, 'g', 'LineWidth', 2, 'DisplayName', 'cuda depth');

        % % plot cost: adding parabolic correction to matlab recomputed
        % [min_c, min_l] = min(cost_rc)
        % zr_min = 0;
        % if (min_l == 1 || min_l == c_layers)
        %     zr_min = 1 / (c_far + min_l*c_depthStep);
        % else
        %     A = cost_rc(1, min_l-1);
        %     B = min_c;
        %     C = cost_rc(1, min_l+1);
        %     delta = ~((A+C)==2*B) * ((A-C)*c_depthStep)/(2*(A-2*B+C));
        %     delta = ~(abs(delta) > c_depthStep) * delta;
        %     zr_min = 1 / (c_far + min_l*c_depthStep + delta);
        % end
        
        % plot(-zr, um, 'r-o', 'DisplayName', 'Matlab');
        % plot(-zr, ImF(vm(1:64), um(1:64), 1*ones(1,64)), 'r-x', 'DisplayName', 'Matlab');
        
        % plot(-zr, cost_rc, 'r-x', 'DisplayName', 'matlab cost');
        % plot([gd(vr, ur), gd(vr, ur)], ylim, 'b', 'LineWidth', 2, 'DisplayName', 'ground depth');
        % plot([d(vr, ur), d(vr, ur)], ylim, 'g', 'LineWidth', 2, 'DisplayName', 'cuda depth');

        % plot([zr_min, zr_min], ylim, 'r', 'LineWidth', 2, 'DisplayName', 'matlab depth');

        legend('show');
        hold off;
        
        % plot actual corresponding point
        figure(2);  hold on;
        
        zr_a = -gd(vr, ur); % actual or ground truth depth
        
        xr_a = (Kinv(1,1)*ur + Kinv(1,3)) * abs(zr_a);
        yr_a = (Kinv(2,2)*vr + Kinv(2,3)) * abs(zr_a);
        
        xm_a = T_mr(1,1)*xr_a + T_mr(1,2)*yr_a + T_mr(1,3)*zr_a  + T_mr(1,4);
        ym_a = T_mr(2,1)*xr_a + T_mr(2,2)*yr_a + T_mr(2,3)*zr_a  + T_mr(2,4);
        zm_a = T_mr(3,1)*xr_a + T_mr(3,2)*yr_a + T_mr(3,3)*zr_a  + T_mr(3,4);
        
        um_a = K(1,1)*(xm_a ./ abs(zm_a)) + K(1,3);
        vm_a = K(2,2)*(ym_a ./ abs(zm_a)) + K(2,3);
        
        plot(um_a, vm_a, 'yo', 'LineWidth', 4, 'DisplayName', 'actual');
        text(double(um_a), double(vm_a-10), num2str(zr_a));
        legend('show');
        hold off;
        
        % plot predicted point
        figure(2);  hold on;
        
        zr_p = -d(vr, ur); % estimated depth
        
        xr_p = (Kinv(1,1)*ur + Kinv(1,3)) * abs(zr_p);
        yr_p = (Kinv(2,2)*vr + Kinv(2,3)) * abs(zr_p);
        
        xm_p = T_mr(1,1)*xr_p + T_mr(1,2)*yr_p + T_mr(1,3)*zr_p  + T_mr(1,4);
        ym_p = T_mr(2,1)*xr_p + T_mr(2,2)*yr_p + T_mr(2,3)*zr_p  + T_mr(2,4);
        zm_p = T_mr(3,1)*xr_p + T_mr(3,2)*yr_p + T_mr(3,3)*zr_p  + T_mr(3,4);
        
        um_p = K(1,1)*(xm_p ./ abs(zm_p)) + K(1,3);
        vm_p = K(2,2)*(ym_p ./ abs(zm_p)) + K(2,3);
        
        plot(um_p, vm_p, 'ro', 'LineWidth', 4, 'DisplayName', 'predicted');
        legend('show');
        hold off;

        % % plot texture sample points cuda vs. matlab
        % figure(4);
        % cla;
        % title('Texture fetch on Im: CUDA vs Matlab');
        % hold on;
        % grid on;
        % plot([1:640], Im_tex(vr,:,2), 'g-o', 'DisplayName', 'CUDA');
        % plot([1:640], ImF((vr+.5)*ones(1,640), [1:640]+.5, 2*ones(1,640)), 'b-x', 'DisplayName', 'Matlab');
        % % plot([1:640], Imf(vr,:,2), 'r-x', 'DisplayName', 'NoIntrp');
        % legend('show');
        % hold off;

        % itr increment
        itr = itr + 1;
    end
    
    % pause loop for 2 sec
    pause(2);
end

