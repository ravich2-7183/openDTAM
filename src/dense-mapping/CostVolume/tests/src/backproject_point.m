function Xw = backproject_point(Trw, K, U, zr)
Kinv = inv(K);
Twr  = inv(Trw);

ur = U(1);
vr = U(2);

assert(zr < 0);

xr = (Kinv(1,1)*ur + Kinv(1,3)) * abs(zr);
yr = (Kinv(2,2)*vr + Kinv(2,3)) * abs(zr);

Xw = Twr*[xr; yr; zr; 1];
Xw = Xw(1:3);

end
