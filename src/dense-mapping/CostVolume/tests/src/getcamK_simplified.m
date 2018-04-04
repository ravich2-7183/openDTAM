function K = getcamK_simplified(cam_file)

f = fopen(cam_file, 'r') ;
if f ~= -1
  script=char(fread(f, inf, 'uchar')) ;
  eval(script) ;
end
fclose(f);

F  = norm(cam_dir) ;
angle  = 2*atan( norm(cam_right)/2 / F ); % field of view
aspect = norm(cam_right) / norm(cam_up); % aspect ratio

M = 480; % cam_height in pixels
N = 640; % cam_width in pixels

width  = 2*F*tan(0.5*angle);
height = width/aspect;

% pixels per unit (actual) length (e.g. pixels/mm)
sx = N / width;
sy = M / height;

Ox = 0.5*(N+1);
Oy = 0.5*(M+1);

fx = sx*F;
fy = sy*F;

K = [fx     0     Ox;
      0     fy    Oy;
      0     0     1];
        
K(2,2) = -K(2,2);        

end
